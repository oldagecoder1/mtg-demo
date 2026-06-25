"""
Handwritten Formula & Notes Parser
====================================
Dedicated script for handwritten Physics / Chemistry / Math notes.

How it works (mirrors the HuggingFace demo's "Element-level Recognition"):
  - Layout detection is DISABLED  →  the whole image (or a crop you provide) is
    passed directly to the PaddleOCR-VL vision-language model.
  - A task-specific prompt is injected:  "formula", "ocr", "table", or "chart"
  - This is the SAME approach used internally by the official online demo and
    gives best results for element-level handwritten content.

Supported recognition modes (choose one per request):
  • formula  – handwritten / printed math / chemistry formulas  → clean LaTeX
  • ocr      – handwritten / printed plain text                 → plain text
  • table    – handwritten / printed table                      → markdown table
  • chart    – hand-drawn chart / graph                         → description

Output per run (saved in ./output/hw_run_<uuid>/):
  input_preview.png   – copy of the uploaded image
  result.md           – recognised content as Markdown / LaTeX
  result.json         – structured JSON with all metadata
"""

import os
import json
import uuid
import re
from pathlib import Path
from PIL import Image
import io

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from paddleocr import PaddleOCRVL

# ==================== SETUP ====================
app = FastAPI(title="Handwritten Formula Parser")

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static_hw", StaticFiles(directory=str(OUTPUT_DIR)), name="static_hw")

# ==================== ENGINE INIT ====================
# Two lightweight pipelines – initialised once at startup.
# Both use the same underlying VLM but with different configs.

print("=" * 60)
print("  Initialising PaddleOCR-VL  •  Element-level mode")
print("  (formula / ocr / table / chart  –  no layout detection)")
print("=" * 60)

# Element-level pipeline  –  layout detection OFF
# The VLM receives the raw image + a task prompt (the "magic" the demo uses)
element_pipeline = PaddleOCRVL(
    device="cpu",
    use_layout_detection=False,          # ← KEY: skip layout detection entirely
    use_doc_orientation_classify=False,  # not needed for element crops
    use_doc_unwarping=False,             # not needed for element crops
)

print("--> Element-level pipeline ready ✅")
print("=" * 60)

# ==================== PROMPT MAPPING ====================
# These are the exact prompt labels the VLM understands.
# Mirror of the mapping in the official HuggingFace demo source.
PROMPT_LABEL_MAP = {
    "formula": "formula",      # math / chemistry equations  → LaTeX
    "ocr":     "ocr",          # plain text / handwriting     → text
    "table":   "table",        # tables                       → markdown table
    "chart":   "chart",        # charts / graphs              → description
}

# Human-readable display names
MODE_DISPLAY = {
    "formula": "🧮 Formula Recognition (LaTeX)",
    "ocr":     "📝 Text / Handwriting Recognition",
    "table":   "📊 Table Recognition (Markdown)",
    "chart":   "📈 Chart / Graph Recognition",
}

# ==================== LATEX POST-PROCESSOR ====================
def clean_latex(text: str, mode: str) -> str:
    """
    Post-process the raw VLM output:
    - For formula mode: wrap in $$ ... $$ if not already wrapped.
    - For all modes: fix common OCR artefacts in math notation.
    """
    if not text:
        return ""

    text = text.strip()

    if mode == "formula":
        # Remove spurious leading/trailing punctuation
        text = text.strip(".,;:")

        # If the VLM returned raw LaTeX without delimiters, wrap it
        already_wrapped = (
            text.startswith("$$") or text.startswith("$") or
            text.startswith(r"\[") or text.startswith(r"\(")
        )
        if not already_wrapped:
            # Detect if it looks like inline vs display formula
            # (display: has \frac, \sum, \int, multi-line etc.)
            display_indicators = [r"\frac", r"\sum", r"\int", r"\prod",
                                  r"\begin", r"\end", "\\\\", r"\matrix"]
            is_display = any(ind in text for ind in display_indicators)
            if is_display:
                text = f"$$\n{text}\n$$"
            else:
                text = f"${text}$"

        # Fix broken scientific notation:  10 $ ^{5} $  →  10^{5}
        text = re.sub(r'10\s*\$\s*\^\{([-+]?\d+)\}\s*\$', r'10^{\1}', text)

        # Replace unicode × with LaTeX \times
        text = text.replace("×", r"\times ")

        # Replace ≤ ≥ with \le \ge inside math
        text = re.sub(r'(?<=\$)([^$]*?)(?=\$)',
                      lambda m: m.group(0).replace("≤", r"\le ").replace("≥", r"\ge "),
                      text)

    return text


# ==================== CORE RECOGNITION FUNCTION ====================
def run_element_recognition(image_path: str, mode: str) -> dict:
    """
    Run element-level VLM recognition on a single image.
    Returns dict with: raw_text, cleaned_text, mode, prompt_used.
    """
    prompt_label = PROMPT_LABEL_MAP.get(mode, "formula")

    print(f"\n  Mode      : {MODE_DISPLAY.get(mode, mode)}")
    print(f"  Prompt    : {prompt_label}")
    print(f"  Image     : {image_path}")
    print("  Running VLM inference...")

    # The pipeline.predict() for element-level mode:
    # - pass the image path directly
    # - set prompt= to the task label (e.g. "formula")
    results = list(element_pipeline.predict(
        input=image_path,
        prompt=prompt_label,         # task-specific prompt sent to VLM
    ))

    # Save the raw pipeline results directly to the run folder
    out_dir = str(Path(image_path).parent)
    for res in results:
        try:
            if hasattr(res, 'save_to_json'):
                res.save_to_json(save_path=out_dir)
                print(f"  ✅ Saved raw pipeline result to {out_dir}")
            else:
                print(f"  ⚠️  Pipeline result object has no 'save_to_json' method. Cannot save raw file.")
            
            if hasattr(res, 'save_to_markdown'):
                res.save_to_markdown(save_path=out_dir)
                print(f"  ✅ Saved raw pipeline markdown to {out_dir}")
            else:
                print(f"  ⚠️  Pipeline result object has no 'save_to_markdown' method. Cannot save markdown file.")
        except Exception as e:
            print(f"  ❌ Error saving pipeline raw json: {e}")

    raw_text = ""
    for res in results:
        # 1. Try accessing the markdown property of the Paddlex result object
        if hasattr(res, "markdown"):
            try:
                md = res.markdown
                if isinstance(md, dict) and "markdown_texts" in md:
                    raw_text += md["markdown_texts"].strip() + "\n"
                    continue
            except Exception as e:
                print(f"  Warning: Failed to extract text via .markdown property: {e}")

        # 2. Fallback to iterating block elements in parsing_res_list
        try:
            blocks = res.get("parsing_res_list", []) or getattr(res, "parsing_res_list", [])
            for block in blocks:
                content = getattr(block, 'content', None)
                if content:
                    raw_text += str(content).strip() + "\n"
        except Exception as e:
            print(f"  Warning: Failed fallback block extraction: {e}")

    raw_text = raw_text.strip()
    cleaned = clean_latex(raw_text, mode)

    print(f"  Raw output length: {len(raw_text)} chars")
    print(f"  Done ✅")

    return {
        "mode": mode,
        "display_mode": MODE_DISPLAY.get(mode, mode),
        "prompt_used": prompt_label,
        "raw_text": raw_text,
        "cleaned_text": cleaned,
    }


# ==================== API ENDPOINT ====================
@app.post("/recognize/")
async def recognize(
    file: UploadFile = File(...),
    mode: str = Form(default="formula"),
):
    """
    Endpoint: POST /recognize/
    Body (multipart):
      file  - image file (JPEG / PNG)
      mode  - one of: formula | ocr | table | chart  (default: formula)
    """
    if mode not in PROMPT_LABEL_MAP:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid mode '{mode}'. Choose from: {list(PROMPT_LABEL_MAP.keys())}"}
        )

    # ── Create run folder ──────────────────────────────────────────
    run_id = uuid.uuid4().hex[:8]
    run_name = f"hw_run_{run_id}"
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  📁 NEW RUN : {run_name}")
    print(f"  📂 PATH    : {run_dir.resolve()}")
    print(f"{'='*60}")

    # ── Save uploaded image ────────────────────────────────────────
    raw_bytes = await file.read()
    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    input_path = run_dir / "input_preview.png"
    pil_img.save(str(input_path))
    temp_path = str(input_path)   # use the saved PNG as VLM input

    try:
        # ── Run VLM recognition ────────────────────────────────────
        result = run_element_recognition(temp_path, mode)

        # ── Build markdown output ──────────────────────────────────
        md_lines = [
            f"# Handwritten Recognition Result\n",
            f"**Mode:** {result['display_mode']}\n",
            f"**File:** {file.filename}\n",
            f"**Run:** `{run_name}`\n",
            f"\n---\n",
            f"## Recognised Content\n",
            f"\n{result['cleaned_text']}\n",
        ]
        if result['raw_text'] != result['cleaned_text']:
            md_lines += [
                f"\n---\n",
                f"## Raw VLM Output\n",
                f"```\n{result['raw_text']}\n```\n",
            ]

        md_content = "".join(md_lines)

        # ── Save files ─────────────────────────────────────────────
        md_path = run_dir / "result.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        json_payload = {
            "run": run_name,
            "file": file.filename,
            **result,
            "output_dir": str(run_dir.resolve()),
        }
        json_path = run_dir / "result.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=2, ensure_ascii=False)

        print(f"\n  📄 result.md   → {md_path.resolve()}")
        print(f"  📊 result.json → {json_path.resolve()}")
        print(f"{'='*60}\n")

        return JSONResponse(content={
            "run": run_name,
            "mode": mode,
            "cleaned_text": result["cleaned_text"],
            "raw_text": result["raw_text"],
            "markdown": md_content,
            "preview_image": f"/static_hw/{run_name}/input_preview.png",
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# ==================== BEAUTIFUL FRONTEND ====================
# Served from a separate HTML file to avoid Python string escaping
# corrupting JavaScript regex literals and causing browser SyntaxErrors.
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = Path("static/hw_formula.html")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ==================== ENTRYPOINT ====================
if __name__ == "__main__":
    import uvicorn
    print("\n🚀  Starting Handwritten Formula Parser")
    print("   URL: http://127.0.0.1:8001\n")
    uvicorn.run("app_handwritten_formula:app", host="127.0.0.1", port=8001, reload=False)
