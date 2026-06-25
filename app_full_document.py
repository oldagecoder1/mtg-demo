"""
Handwritten Full Document Parser
================================
Dedicated script for processing entire document pages at once.
Uses layout detection (use_layout_detection=True) to partition the page,
and passes each identified element to the VLM, stitching them into Markdown.
"""

import os
import json
import uuid
import re
from pathlib import Path
from PIL import Image
import io
import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from paddleocr import PaddleOCRVL

app = FastAPI(title="Full Document Parser")

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static_full", StaticFiles(directory=str(OUTPUT_DIR)), name="static_full")

# Initialize VLM Pipeline with Layout Detection ON
print("=" * 60)
print("  Initialising PaddleOCR-VL (Document Layout Parsing mode)")
print("=" * 60)

doc_pipeline = PaddleOCRVL(
    device="cpu",
    use_layout_detection=True,
    use_doc_orientation_classify=True,   # ← Pre-initialize preprocessor models
    use_doc_unwarping=True,              # ← Pre-initialize preprocessor models
)

print("--> Document Parsing VLM pipeline ready ✅")
print("=" * 60)


# ==================== LATEX POST-PROCESSOR ====================
def post_process_markdown(md_text: str) -> str:
    """
    Cleans up common OCR symbol replacements in the generated markdown.
    """
    if not md_text:
        return ""
    
    # Escape inequalities for safe markdown rendering
    md_text = md_text.replace("<=", r" \le ").replace(">=", r" \ge ")
    md_text = md_text.replace("≤", r" \le ").replace("≥", r" \ge ")
    
    return md_text


# ==================== API ENDPOINTS ====================
@app.post("/parse/")
async def parse_endpoint(
    file: UploadFile = File(...),
    unwarping: bool = Form(default=False),
    orientation: bool = Form(default=False),
):
    """
    Upload a full image, run layout parsing, and output structured markdown + visualization.
    """
    run_id = uuid.uuid4().hex[:8]
    run_name = f"doc_run_{run_id}"
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  📁 DOCUMENT PARSE RUN : {run_name}")
    print(f"  📂 PATH                 : {run_dir.resolve()}")
    print(f"{'='*60}")

    # Save uploaded image
    raw_bytes = await file.read()
    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    input_path = run_dir / "input_preview.png"
    pil_img.save(str(input_path))

    try:
        # Run Document Parsing VLM pipeline
        results = list(doc_pipeline.predict(
            input=str(input_path.resolve()),
            use_doc_unwarping=unwarping,
            use_doc_orientation_classify=orientation,
        ))

        md_text = ""
        vis_saved = False

        for idx, res in enumerate(results):
            # 1. Extract markdown text
            md = getattr(res, "markdown", None) or {}
            texts = md.get("markdown_texts", "")
            if texts:
                md_text += texts + "\n\n"

            # 2. Extract and save the layout visualization image
            try:
                img_dict = getattr(res, "img", None) or {}
                # Look for layout_det_res key in the image dictionary
                vis_img = img_dict.get("layout_det_res")
                if vis_img is not None:
                    vis_path = run_dir / "visualization.png"
                    if isinstance(vis_img, np.ndarray):
                        cv2.imwrite(str(vis_path), vis_img)
                    else:
                        vis_img.save(str(vis_path))
                    vis_saved = True
            except Exception as e:
                print(f"  Warning: Failed to save visualization image: {e}")

        # Post-process formatting
        cleaned_md = post_process_markdown(md_text)

        # Save result Markdown
        md_output_path = run_dir / "result.md"
        with open(md_output_path, "w", encoding="utf-8") as f:
            f.write(cleaned_md)

        # Save result metadata JSON
        payload = {
            "run": run_name,
            "unwarping_applied": unwarping,
            "orientation_applied": orientation,
            "markdown": cleaned_md,
            "visualization": f"/static_full/{run_name}/visualization.png" if vis_saved else None,
            "preview_image": f"/static_full/{run_name}/input_preview.png"
        }
        with open(run_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"  📄 result.md      → {md_output_path.resolve()}")
        print(f"  ✅ Done")
        print(f"{'='*60}\n")

        return JSONResponse(content=payload)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = Path("static/full_document.html")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Full Document Parser")
    print("   URL: http://127.0.0.1:8003\n")
    uvicorn.run("app_full_document:app", host="127.0.0.1", port=8003, reload=False)
