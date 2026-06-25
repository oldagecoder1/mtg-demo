"""
MTG PCM Master Document Parser
================================
A professional document parser for Physics, Chemistry & Math textbook pages.
Accepts: JPEG/PNG images (scanned pages), PDF files.

Key design decisions:
- Images: crop with PIL at exact VLM pixel coordinates (NO fitz coordinate mismatch).
- PDFs:   render each page to a PIL image at 200 DPI, then crop with PIL.
- Column-aware reading order: detect 2-column layout and sort left col → right col.
- PaddleOCRVL block access: use .label, .content, .bbox attributes (not .get()).
- save_to_markdown() and save_to_json() are called on restructured result for clean output.
"""

import os
import json
import time
import uuid
import re
from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF – only used to render PDF pages to PIL images
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from paddleocr import PaddleOCRVL

# ==================== APP SETUP ====================
app = FastAPI(title="MTG PCM Master Document Parser")

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static_outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="static_outputs")

print("--> Initializing PaddleOCRVL pipeline...")

pipeline = None
def get_pipeline():
    global pipeline

    if pipeline is None:
        print("Loading PaddleOCRVL...")

        pipeline = PaddleOCRVL(
            pipeline_version="v1"
        )

    return pipeline


# ==================== LATEX CLEANER ====================
def fix_latex(text: str) -> str:
    """Post-process VLM text to ensure LaTeX delimiters are correct."""
    if not text:
        return ""
    # Fix broken scientific notation tokens: 10 $ ^{5} $ → $10^{5}$
    text = re.sub(r'10\s*\$\s*\^\{([-+]?\d+)\}\s*\$', r'$10^{\1}$', text)
    # Fix bare scientific notation without $ delimiters
    text = re.sub(r'(?<!\$)(?:\d+(?:\.\d+)?\s*[xX×]\s*)?10\^\{[-+]?\d+\}(?!\$)',
                  lambda m: '$' + m.group(0).replace('×', r'\times ').replace('x', r'\times ') + '$', text)
    # Fix chemical subscripts/superscripts not already wrapped: KClO_{3} → $KClO_{3}$
    text = re.sub(r'(?<!\$)\b([A-Z][A-Za-z0-9]*[_\^]\{[a-zA-Z0-9+\-]+\}[A-Za-z0-9]*)(?!\$)',
                  lambda m: '$' + m.group(0) + '$', text)
    # Replace unicode × with LaTeX \times inside math
    text = text.replace('×', r'\times ')
    return text


# ==================== COLUMN-AWARE READING ORDER ====================
def column_aware_sort(blocks, page_width: float):
    """
    Sorts blocks into reading order for a 2-column textbook layout.
    - Detects the column midpoint from the distribution of block x-centers.
    - Groups blocks into left and right columns.
    - Within each column, sorts top-to-bottom.
    - Returns: left column blocks (top→bottom) + right column blocks (top→bottom).
    """
    if not blocks:
        return []

    def get_bbox(b):
        bbox = b.bbox if hasattr(b, 'bbox') else b.get('block_bbox', [0, 0, 0, 0])
        return list(map(float, bbox))

    # Estimate column boundary: use the median x-center of all blocks
    x_centers = [(get_bbox(b)[0] + get_bbox(b)[2]) / 2.0 for b in blocks]
    x_centers_sorted = sorted(x_centers)
    mid_x = x_centers_sorted[len(x_centers_sorted) // 2]

    # Use page_width/2 as fallback if median is too close to edges
    col_boundary = mid_x if (0.2 * page_width < mid_x < 0.8 * page_width) else page_width / 2.0

    left_col, right_col, full_width = [], [], []

    for b in blocks:
        bbox = get_bbox(b)
        x_center = (bbox[0] + bbox[2]) / 2.0
        block_width = bbox[2] - bbox[0]

        # Full-width blocks (spanning >70% of page) go in reading order between columns
        if block_width > 0.65 * page_width:
            full_width.append(b)
        elif x_center < col_boundary:
            left_col.append(b)
        else:
            right_col.append(b)

    def top_sort(b):
        bbox = get_bbox(b)
        return bbox[1]  # y1

    left_col.sort(key=top_sort)
    right_col.sort(key=top_sort)
    full_width.sort(key=top_sort)

    # Merge: full-width blocks act as section dividers; insert them by y-position
    merged = []
    fw_idx = 0
    all_col = list(left_col) + list(right_col)

    # Simple merge: full-width blocks first if they appear before the columns
    # For simplicity: left col → right col, with full-width blocks inserted by y
    result = []
    li, ri, fi = 0, 0, 0
    combined = sorted(left_col, key=top_sort) + sorted(right_col, key=top_sort)
    # Re-sort into: full_width blocks at their natural y position, interspersed
    all_blocks_sorted = sorted(blocks, key=lambda b: (get_bbox(b)[1], get_bbox(b)[0]))

    # Use a simple 2-pass: left column top→bottom, then right column top→bottom
    return sorted(left_col, key=top_sort) + sorted(right_col, key=top_sort)


# ==================== PIL IMAGE CROPPER ====================
def crop_block_from_image(pil_image: Image.Image, bbox: list, out_path: Path) -> bool:
    """
    Crops a region from a PIL image using VLM bbox coordinates (pixel space).
    Returns True if successful.
    """
    try:
        img_w, img_h = pil_image.size
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Clamp to image bounds
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))

        if x2 <= x1 or y2 <= y1:
            return False

        cropped = pil_image.crop((x1, y1, x2, y2))
        cropped.save(str(out_path))
        return True
    except Exception as e:
        print(f"   -> Crop failed for bbox {bbox}: {e}")
        return False


# ==================== LOAD UPLOADED FILE AS PIL IMAGES ====================
def load_as_pil_pages(file_path: str) -> list:
    """
    Returns a list of PIL Images, one per page.
    For PDFs: renders at 200 DPI.
    For images: returns single-element list.
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        pages = []
        doc = fitz.open(file_path)
        for page in doc:
            mat = fitz.Matrix(200 / 72, 200 / 72)  # 200 DPI
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append(img)
        doc.close()
        return pages
    else:
        return [Image.open(file_path).convert("RGB")]


# ==================== MAIN PARSING ENDPOINT ====================
@app.post("/parse-document/")
async def parse_document(file: UploadFile = File(...)):
    from paddleocr import PaddleOCRVL

    print("Creating NEW pipeline...")

    pipeline = PaddleOCRVL(
        device="gpu",
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_layout_detection=True,
    )

    print("New pipeline created.")
    print("--> PaddleOCRVL Engine Ready. 🚀")
    filename = file.filename
    unique_run_id = uuid.uuid4().hex[:6]
    temp_path = f"incoming_{unique_run_id}_{filename}"

    print(f"\n{'='*65}")
    print(f"  NEW PARSING REQUEST: {filename}")
    print(f"{'='*65}")

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        # ----- STEP 1: Run VLM pipeline -----
        print("--> Step 1: Running PaddleOCRVL...")
        import paddle

        print("=" * 60)
        print("CUDA Compiled :", paddle.is_compiled_with_cuda())
        print("Current Device:", paddle.device.get_device())
        print("=" * 60)
        import time

        start = time.perf_counter()
        print("temp_path =", temp_path)
        print("type =", type(temp_path))
        output = pipeline.predict(input=temp_path)
        print("=" * 60)
        print(f"Prediction Time: {time.perf_counter() - start:.2f} seconds")
        print("=" * 60)
        pages_res = list(output)
        print(f"    Pages detected: {len(pages_res)}")

        # ----- STEP 2: Create run session folder -----
        run_folder_name = f"run_{unique_run_id}_{uuid.uuid4().hex[:4]}"
        current_run_dir = OUTPUT_DIR / run_folder_name
        current_run_dir.mkdir(parents=True, exist_ok=True)
        run_imgs_dir = current_run_dir / "imgs"
        run_imgs_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*65}")
        print(f"  📁 RUN FOLDER : {run_folder_name}")
        print(f"  📂 FULL PATH  : {current_run_dir.resolve()}")
        print(f"{'='*65}\n")

        # ----- STEP 3: Save official paddle outputs (md + json) -----
        print("--> Step 3: Saving official PaddleOCR markdown & JSON...")
        restructured = pipeline.restructure_pages(
            pages_res,
            merge_tables=True,
            relevel_titles=True,
            concatenate_pages=True
        )
        for res in restructured:
            res.save_to_markdown(save_path=current_run_dir)
            res.save_to_json(save_path=current_run_dir)

        # ----- STEP 4: Load original file as PIL pages for cropping -----
        print("--> Step 4: Loading pages as PIL images for precise cropping...")
        pil_pages = load_as_pil_pages(temp_path)

        # ----- STEP 5: Build clean custom markdown + extract images -----
        print("--> Step 5: Building structured markdown with cropped assets...")
        compiled_md_lines = []
        structured_layout = []

        for page_idx, page_res in enumerate(pages_res):
            page_number = page_idx + 1
            pil_image = pil_pages[page_idx] if page_idx < len(pil_pages) else None
            img_w = pil_image.width if pil_image else 1

            compiled_md_lines.append(f"\n---\n## Page {page_number}\n")

            # Retrieve parsing_res_list (PaddleOCRVLBlock objects)
            blocks = []
            try:
                blocks = page_res.get("parsing_res_list", [])
            except Exception:
                blocks = getattr(page_res, "parsing_res_list", [])

            if not blocks:
                print(f"    [Page {page_number}] No blocks detected.")
                structured_layout.append({"page": page_number, "elements": []})
                continue

            # Sort blocks with column awareness
            sorted_blocks = column_aware_sort(blocks, img_w)
            print(f"    [Page {page_number}] {len(sorted_blocks)} blocks, image size: {img_w}px wide")

            page_elements = []

            for elem_idx, block in enumerate(sorted_blocks):
                # --- Read block attributes safely from PaddleOCRVLBlock (always a class instance) ---
                label = str(getattr(block, 'label', 'text') or 'text').lower()
                content = str(getattr(block, 'content', '') or '').strip()
                bbox = list(getattr(block, 'bbox', [0, 0, 0, 0]) or [0, 0, 0, 0])
                bbox = [float(v) for v in bbox]

                elem_meta = {
                    "block_id": elem_idx,
                    "label": label,
                    "bbox": bbox,
                    "content": content,
                    "cropped_image": None
                }

                crop_filename = None

                # === VISUAL BLOCKS: image, chart (crop with PIL) ===
                if label in ["image", "chart", "figure"] and pil_image:
                    crop_filename = f"img_p{page_number}_b{elem_idx}.png"
                    crop_path = run_imgs_dir / crop_filename
                    success = crop_block_from_image(pil_image, bbox, crop_path)
                    if success:
                        rel_path = f"imgs/{crop_filename}"
                        elem_meta["cropped_image"] = rel_path
                        compiled_md_lines.append(f"\n![{label} bbox={bbox}]({rel_path})\n")
                        print(f"    ✅ Cropped {label}: {crop_filename}")
                    else:
                        print(f"    ⚠️  Crop failed for {label} at {bbox}")

                # === TABLE BLOCKS: use text content (markdown) whenever available ===
                elif label == "table":
                    if content:
                        # VLM already transcribed the table — write it as text
                        fixed = fix_latex(content)
                        compiled_md_lines.append(f"\n{fixed}\n")
                        elem_meta["content"] = fixed
                        print(f"    📋 Table written as text (block {elem_idx})")
                    elif pil_image:
                        # No text content at all — last resort: crop as image
                        crop_filename = f"table_p{page_number}_b{elem_idx}.png"
                        crop_path = run_imgs_dir / crop_filename
                        success = crop_block_from_image(pil_image, bbox, crop_path)
                        if success:
                            rel_path = f"imgs/{crop_filename}"
                            elem_meta["cropped_image"] = rel_path
                            compiled_md_lines.append(f"\n![table bbox={bbox}]({rel_path})\n")
                            print(f"    🖼️  Table had no text — cropped as image: {crop_filename}")

                # === DISPLAY FORMULA ===
                elif label in ["display_formula", "formula"]:
                    inner = content.strip().strip("$").strip()
                    fixed = fix_latex(inner)
                    compiled_md_lines.append(f"\n$$\n{fixed}\n$$\n")
                    elem_meta["content"] = f"$$\n{fixed}\n$$"

                # === INLINE FORMULA ===
                elif label == "inline_formula":
                    inner = content.strip().strip("$").strip()
                    fixed = fix_latex(inner)
                    compiled_md_lines.append(f" ${fixed}$ ")
                    elem_meta["content"] = f"${fixed}$"

                # === DOCUMENT TITLE ===
                elif label == "doc_title":
                    compiled_md_lines.append(f"\n# {content}\n")

                # === PARAGRAPH / SECTION TITLE ===
                elif label in ["paragraph_title", "figure_title"]:
                    compiled_md_lines.append(f"\n### {content}\n")

                # === REGULAR TEXT (paragraphs, list items, footnotes) ===
                elif label in ["text", "paragraph", "list_item", "footnote", "number",
                               "aside_text", "header", "footer", "vision_footnote"]:
                    if content:
                        fixed = fix_latex(content)
                        compiled_md_lines.append(f"\n{fixed}\n")
                        elem_meta["content"] = fixed

                # === UNKNOWN LABELS: print but still add content ===
                else:
                    if content:
                        fixed = fix_latex(content)
                        compiled_md_lines.append(f"\n{fixed}\n")
                        elem_meta["content"] = fixed

                page_elements.append(elem_meta)

            structured_layout.append({"page": page_number, "elements": page_elements})

        # ----- STEP 6: Write custom output.md and output.json -----
        final_md = "".join(compiled_md_lines)
        custom_md_path = current_run_dir / "custom_output.md"
        with open(custom_md_path, "w", encoding="utf-8") as f:
            f.write(final_md)

        custom_json_path = current_run_dir / "custom_output.json"
        json_payload = {
            "file": filename,
            "run_folder": run_folder_name,
            "pages": len(pages_res),
            "layout": structured_layout
        }
        with open(custom_json_path, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=2, ensure_ascii=False)

        print(f"\n{'-'*65}")
        print(f"  📄 Custom Markdown : {custom_md_path.resolve()}")
        print(f"  📊 Custom JSON     : {custom_json_path.resolve()}")
        print(f"  🖼️  Images folder   : {run_imgs_dir.resolve()}")
        print(f"  📋 Paddle MD/JSON  : see run folder for official paddle outputs")
        print(f"{'-'*65}\n")
        print("--> Workflow Complete ✅")

        return JSONResponse(content={
            "file_parsed": filename,
            "run_folder": run_folder_name,
            "pages": len(pages_res),
            "markdown_content": final_md,
            "msg": f"All outputs in ./output/{run_folder_name}/"
        })

    except Exception as e:
        import traceback
        print(f"--> ERROR: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print("==================== END ====================\n")


# ==================== LIVE DASHBOARD ====================
@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>PCM Master Parser</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script>
        window.MathJax = {
            tex: { inlineMath: [['$','$']], displayMath: [['$$','$$']], processEscapes: true },
            options: { processHtmlClass: 'tex2jax_process' }
        };
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Inter', sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; }

        .hero {
            background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
            border-bottom: 1px solid #1e293b;
            padding: 40px 24px 32px;
            text-align: center;
        }
        .hero h1 { font-size: 2rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.5px; }
        .hero h1 span { background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .hero p { color: #94a3b8; margin-top: 8px; font-size: 0.95rem; }

        .upload-card {
            max-width: 680px; margin: 32px auto; padding: 32px;
            background: #1e293b; border: 1px solid #334155; border-radius: 16px;
        }
        .drop-zone {
            border: 2px dashed #334155; border-radius: 12px;
            padding: 40px 24px; text-align: center; cursor: pointer;
            transition: all 0.2s; position: relative;
        }
        .drop-zone:hover, .drop-zone.drag-over { border-color: #38bdf8; background: rgba(56, 189, 248, 0.05); }
        .drop-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
        .drop-zone .icon { font-size: 2.5rem; margin-bottom: 12px; }
        .drop-zone p { color: #94a3b8; font-size: 0.9rem; }
        .drop-zone p span { color: #38bdf8; font-weight: 600; }
        #file-name { margin-top: 12px; font-size: 0.85rem; color: #64748b; min-height: 20px; }

        .parse-btn {
            display: block; width: 100%; margin-top: 20px;
            padding: 14px; border: none; border-radius: 10px;
            background: linear-gradient(90deg, #0ea5e9, #6366f1);
            color: white; font-size: 1rem; font-weight: 600;
            cursor: pointer; transition: opacity 0.2s;
        }
        .parse-btn:hover { opacity: 0.9; }
        .parse-btn:disabled { opacity: 0.5; cursor: not-allowed; }

        .progress {
            display: none; margin-top: 16px; padding: 14px 18px;
            background: #0f172a; border: 1px solid #1e293b; border-radius: 10px;
            font-size: 0.875rem; color: #38bdf8;
        }
        .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid #1e3a5f; border-top-color: #38bdf8; border-radius: 50%; animation: spin 0.7s linear infinite; vertical-align: middle; margin-right: 8px; }
        @keyframes spin { to { transform: rotate(360deg); } }

        .output-section { max-width: 900px; margin: 0 auto 60px; padding: 0 24px; }

        .book-container {
            display: none; margin-top: 32px;
            background: #ffffff; color: #1a202c;
            border-radius: 16px; padding: 56px 64px;
            box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
        }
        .book-container h1 { font-size: 1.8rem; color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; margin: 32px 0 16px; }
        .book-container h2 { font-size: 1.4rem; color: #1e40af; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; margin: 28px 0 12px; }
        .book-container h3 { font-size: 1.1rem; color: #0f172a; margin: 20px 0 10px; font-weight: 600; }
        .book-container p { font-size: 0.95rem; line-height: 1.8; margin: 8px 0; color: #374151; }
        .book-container ul, .book-container ol { padding-left: 24px; margin: 8px 0; }
        .book-container li { font-size: 0.95rem; line-height: 1.7; color: #374151; }
        .book-container img { display: block; max-width: 100%; height: auto; margin: 20px auto; border: 1px solid #e5e7eb; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
        .book-container table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.875rem; }
        .book-container th, .book-container td { border: 1px solid #d1d5db; padding: 8px 12px; text-align: left; }
        .book-container th { background: #f3f4f6; font-weight: 600; }
        .book-container hr { border: none; border-top: 2px solid #e2e8f0; margin: 24px 0; }
        .book-container code { background: #f3f4f6; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; color: #db2777; }
    </style>
</head>
<body>
    <div class="hero">
        <h1>PCM <span>Master Parser</span></h1>
        <p>Physics · Chemistry · Math — Professional document intelligence engine</p>
    </div>

    <div class="upload-card">
        <div class="drop-zone" id="dropZone">
            <input type="file" id="docFile" accept=".pdf,image/*">
            <div class="icon">📄</div>
            <p><span>Click to browse</span> or drag & drop</p>
            <p style="margin-top:4px; font-size:0.8rem;">Supports PDF, JPG, PNG</p>
        </div>
        <div id="file-name">No file selected</div>
        <button class="parse-btn" id="parseBtn" onclick="parseDocument()">Upload & Parse</button>
        <div class="progress" id="progress">
            <span class="spinner"></span>
            Running VLM layout detection and formula extraction... ⏳
        </div>
    </div>

    <div class="output-section">
        <div class="book-container tex2jax_process" id="outputBook"></div>
    </div>

    <script>
        const fileInput = document.getElementById('docFile');
        const fileNameEl = document.getElementById('file-name');

        fileInput.addEventListener('change', () => {
            fileNameEl.textContent = fileInput.files[0] ? '📎 ' + fileInput.files[0].name : 'No file selected';
        });

        async function parseDocument() {
            const file = fileInput.files[0];
            if (!file) { alert('Please select a file first.'); return; }

            const btn = document.getElementById('parseBtn');
            const progress = document.getElementById('progress');
            const outputBook = document.getElementById('outputBook');

            btn.disabled = true;
            progress.style.display = 'block';
            outputBook.style.display = 'none';
            outputBook.innerHTML = '';

            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('/parse-document/', { method: 'POST', body: formData });
                const data = await res.json();
                progress.style.display = 'none';
                btn.disabled = false;

                if (data.markdown_content) {
                    outputBook.style.display = 'block';
                    let md = data.markdown_content;

                    // Rewrite relative image paths to the static server route
                    md = md.replace(/\]\(imgs\//g, `](/static_outputs/${data.run_folder}/imgs/`);

                    // Shield math from marked.js interference
                    const mathBlocks = [];
                    md = md.replace(/\$\$([\s\S]*?)\$\$/g, (m, c) => { mathBlocks.push('$$' + c + '$$'); return `___MB_${mathBlocks.length-1}___`; });
                    md = md.replace(/\$((?:[^$]|\\.)+?)\$/g, (m, c) => { mathBlocks.push('$' + c + '$'); return `___MI_${mathBlocks.length-1}___`; });

                    let html = marked.parse(md);

                    mathBlocks.forEach((b, i) => {
                        html = html.replace(`___MB_${i}___`, b).replace(`___MI_${i}___`, b);
                    });

                    outputBook.innerHTML = html;

                    if (window.MathJax?.typesetPromise) {
                        setTimeout(() => MathJax.typesetPromise([outputBook]), 100);
                    }
                } else {
                    alert('Error: ' + (data.error || 'Unknown error'));
                }
            } catch (e) {
                progress.style.display = 'none';
                btn.disabled = false;
                alert('Server error: ' + e.message);
            }
        }
    </script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_pcm_master_parser:app", host="0.0.0.0", port=8000, reload=False)
