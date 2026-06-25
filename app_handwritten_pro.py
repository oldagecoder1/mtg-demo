"""
Handwritten Note Parser Pro
===========================
Interactive crop-and-parse interface for handwritten formulas and notes.
Uses OpenCV for pre-processing (Adaptive Thresholding, CLAHE, Denoising)
and PaddleOCR-VL for element-level transcription.
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

app = FastAPI(title="Handwritten Note Parser Pro")

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static_pro", StaticFiles(directory=str(OUTPUT_DIR)), name="static_pro")

# Initialize VLM Pipeline
print("=" * 60)
print("  Initialising PaddleOCR-VL (Element-level mode)")
print("=" * 60)

element_pipeline = PaddleOCRVL(
    device="cpu",
    use_layout_detection=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
)

print("--> Element-level VLM pipeline ready ✅")
print("=" * 60)

PROMPT_LABEL_MAP = {
    "formula": "formula",
    "ocr": "ocr",
    "table": "table",
    "chart": "chart",
}

# ==================== IMAGE PRE-PROCESSING ====================
def apply_preprocessing(img_bytes: bytes, steps: list) -> tuple[np.ndarray, bytes]:
    """
    Applies selected OpenCV preprocessing filters to the image bytes.
    Returns: (OpenCV image array, processed PNG bytes)
    """
    # Load image using OpenCV
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes.")

    out = img.copy()

    # 1. CLAHE (Contrast Booster)
    if "clahe" in steps:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        out = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 2. Denoise (Gaussian Blur)
    if "denoise" in steps:
        out = cv2.GaussianBlur(out, (3, 3), 0)

    # 3. Adaptive Binarization (B&W Binarize)
    if "binarize" in steps:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        # Convert back to BGR so we write a 3-channel PNG
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    # Encode back to PNG bytes
    _, encoded = cv2.imencode(".png", out)
    return out, encoded.tobytes()


# ==================== LATEX POST-PROCESSOR ====================
def clean_latex(text: str, mode: str) -> str:
    """
    Post-process raw LaTeX outputs to clean up OCR errors.
    """
    if not text:
        return ""
    text = text.strip()

    if mode == "formula":
        # Remove common tailing symbols
        text = text.strip(".,;:")
        
        # Wrap if it doesn't have math delimiters
        already_wrapped = (
            text.startswith("$$") or text.startswith("$") or
            text.startswith(r"\[") or text.startswith(r"\(")
        )
        if not already_wrapped:
            display_indicators = [r"\frac", r"\sum", r"\int", r"\prod",
                                  r"\begin", r"\end", "\\\\", r"\matrix"]
            is_display = any(ind in text for ind in display_indicators)
            if is_display:
                text = f"$$\n{text}\n$$"
            else:
                text = f"${text}$"

        # Fix specific sci-notation bugs: 10 $ ^{x} $ -> 10^{x}
        text = re.sub(r'10\s*\$\s*\^\{([-+]?\d+)\}\s*\$', r'10^{\1}', text)
        text = text.replace("×", r"\times ")

    return text


# ==================== API ENDPOINTS ====================
@app.post("/preprocess/")
async def preprocess_endpoint(
    file: UploadFile = File(...),
    steps: str = Form(default="[]"), # JSON array of strings e.g. '["clahe", "binarize"]'
):
    """
    Applies preprocessing to the full image and returns it as base64 so the
    frontend can display/crop on top of the preprocessed image.
    """
    try:
        steps_list = json.loads(steps)
        raw_bytes = await file.read()
        _, proc_bytes = apply_preprocessing(raw_bytes, steps_list)
        
        import base64
        b64_str = base64.b64encode(proc_bytes).decode("utf-8")
        return JSONResponse(content={"preprocessed_image": f"data:image/png;base64,{b64_str}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/recognize/")
async def recognize_endpoint(
    file: UploadFile = File(...),
    mode: str = Form(default="formula"),
    steps: str = Form(default="[]"),
):
    """
    Accepts a cropped image, applies optional pre-processing steps,
    runs VLM inference, and returns recognized LaTeX / markdown.
    """
    if mode not in PROMPT_LABEL_MAP:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid mode '{mode}'. Use one of: {list(PROMPT_LABEL_MAP.keys())}"}
        )

    run_id = uuid.uuid4().hex[:8]
    run_name = f"pro_run_{run_id}"
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load cropped image
        crop_bytes = await file.read()
        steps_list = json.loads(steps)

        # Apply pre-processing to the crop
        proc_img, proc_bytes = apply_preprocessing(crop_bytes, steps_list)

        # Save processed crop & original crop for debugging
        original_crop_path = run_dir / "crop_original.png"
        processed_crop_path = run_dir / "crop_processed.png"
        
        with open(original_crop_path, "wb") as f:
            f.write(crop_bytes)
        with open(processed_crop_path, "wb") as f:
            f.write(proc_bytes)

        # Run PaddleOCR-VL inference on the preprocessed crop
        prompt_label = PROMPT_LABEL_MAP[mode]
        results = list(element_pipeline.predict(
            input=str(processed_crop_path.resolve()),
            prompt=prompt_label,
        ))

        raw_text = ""
        for res in results:
            if hasattr(res, "markdown"):
                try:
                    md = res.markdown
                    if isinstance(md, dict) and "markdown_texts" in md:
                        raw_text += md["markdown_texts"].strip() + "\n"
                        continue
                except Exception as e:
                    print(f"Warning: .markdown property extract failed: {e}")

            # Fallback to blocks
            try:
                blocks = res.get("parsing_res_list", []) or getattr(res, "parsing_res_list", [])
                for block in blocks:
                    content = getattr(block, 'content', None)
                    if content:
                        raw_text += str(content).strip() + "\n"
            except Exception as e:
                print(f"Warning: block extract failed: {e}")

        raw_text = raw_text.strip()
        cleaned = clean_latex(raw_text, mode)

        # Build run metadata
        payload = {
            "run": run_name,
            "mode": mode,
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "original_crop": f"/static_pro/{run_name}/crop_original.png",
            "processed_crop": f"/static_pro/{run_name}/crop_processed.png",
        }

        # Save to JSON
        with open(run_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return JSONResponse(content=payload)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = Path("static/hw_pro.html")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Handwritten Note Parser Pro")
    print("   URL: http://127.0.0.1:8002\n")
    uvicorn.run("app_handwritten_pro:app", host="127.0.0.1", port=8002, reload=False)
