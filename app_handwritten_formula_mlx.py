import os
import json
import uuid
import re
from pathlib import Path
from io import BytesIO
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Apple Native MLX Import
import mlx_vlm

app = FastAPI(title="Handwritten Formula Parser - Apple MLX Native")

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static_hw", StaticFiles(directory=str(OUTPUT_DIR)), name="static_hw")

STATIC_WEB_DIR = Path("./static")
STATIC_WEB_DIR.mkdir(parents=True, exist_ok=True)

# Automatically create static dashboard asset if missing to prevent file crash
HTML_DASHBOARD_FILE = STATIC_WEB_DIR / "hw_formula.html"
if not HTML_DASHBOARD_FILE.exists():
    HTML_DASHBOARD_FILE.write_text("""<!DOCTYPE html>
<html>
<head>
    <title>MTG AI - MLX Element Parser</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script>
        window.MathJax = { tex: { inlineMath: [['$', '$']], displayMath: [['$$', '$$']] } };
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body { font-family: system-ui, sans-serif; background: #f4f6f9; padding: 20px; color: #2c3e50; }
        .box { max-width: 800px; margin: 20px auto; background: white; padding: 25px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); text-align: center; }
        select, input, button { padding: 10px; margin: 10px 0; border-radius: 4px; border: 1px solid #ccc; font-size: 15px; }
        button { background: #004D40; color: white; cursor: pointer; font-weight: bold; width: 100%; border: none;}
        #preview { max-width: 50%; display: block; margin: 15px auto; }
        #output { display: none; margin-top: 25px; padding: 20px; border-top: 2px solid #004D40; text-align: left; }
    </style>
</head>
<body>
    <div class="box">
        <h2>MTG Element-Level Apple MLX Engine 🍏</h2>
        <label>Select Recognition Mode:</label>
        <select id="modeSelect">
            <option value="formula">🧮 Formula Recognition (LaTeX)</option>
            <option value="ocr">📝 Text / Handwriting Recognition</option>
            <option value="table">📊 Table Recognition (Markdown)</option>
            <option value="chart">📈 Chart / Graph Recognition</option>
        </select>
        <input type="file" id="fileInput" accept="image/*" />
        <button onclick="uploadAndParse()">Run MLX Inference on Mac GPU</button>
        <div id="status" style="color: #ff9800; font-weight: bold; margin: 10px 0;"></div>
        <img id="preview" />
        <div id="output"></div>
    </div>
    <script>
        async function uploadAndParse() {
            const file = document.getElementById('fileInput').files[0];
            const mode = document.getElementById('modeSelect').value;
            if(!file) { alert("File select karo bhai!"); return; }
            
            document.getElementById('status').innerText = "VLM Processing on Apple Silicon Metal GPU... ⏳";
            const formData = new FormData();
            formData.append("file", file);
            formData.append("mode", mode);
            
            try {
                const res = await fetch("/recognize/", { method: "POST", body: formData });
                const data = await res.json();
                document.getElementById('status').innerText = "";
                if(data.markdown) {
                    document.getElementById('preview').src = data.preview_image;
                    const outDiv = document.getElementById('output');
                    outDiv.style.display = "block";
                    outDiv.innerHTML = marked.parse(data.markdown);
                    if(window.MathJax) MathJax.typesetPromise([outDiv]);
                }
            } catch(e) { document.getElementById('status').innerText = "Error Occurred!"; console.error(e); }
        }
    </script>
</body>
</html>""", encoding="utf-8")

MODEL_PATH = "./mlx_paddleocr_vl"

print("=" * 60)
print("  🍏 INITIALISING NATIVE APPLE MLX VLM ENGINE 🍏")
print(f"  Loading Local Weights From: {MODEL_PATH}")
print("=" * 60)

# Load the model directly into Mac's Unified Memory GPU Space
model, processor = mlx_vlm.load(MODEL_PATH)

print("--> Apple Silicon Metal GPU Pipeline Active & Ready! 🚀")
print("=" * 60)

PROMPT_LABEL_MAP = {
    "formula": "Render the exact mathematical expressions found in this image into clean standalone LaTeX markdown blocks using $$...$$. Do not classify the subject, do not explain, and do not output any Chinese text.",
    "ocr":     "Transcribe the handwritten text from this image into clean plain text. Do not comment or classify.",
    "table":   "Convert the table from this image into a clean Markdown table format.",
    "chart":   "Describe the chart data trends found in this image format.",
}

MODE_DISPLAY = {
    "formula": "🧮 Formula Recognition (LaTeX)",
    "ocr":     "📝 Text / Handwriting Recognition",
    "table":   "📊 Table Recognition (Markdown)",
    "chart":   "📈 Chart / Graph Recognition",
}

def clean_latex(text: str, mode: str) -> str:
    if not text: return ""
    text = text.strip()
    if mode == "formula":
        text = text.strip(".,;:")
        already_wrapped = text.startswith("$$") or text.startswith("$") or text.startswith(r"\[") or text.startswith(r"\(")
        if not already_wrapped:
            display_indicators = [r"\frac", r"\sum", r"\int", r"\prod", r"\begin", r"\end", "\\\\", r"\matrix"]
            if any(ind in text for ind in display_indicators):
                text = f"$$\n{text}\n$$"
            else:
                text = f"${text}$"
        text = re.sub(r'10\s*\$\s*\^\{([-+]?\d+)\}\s*\$', r'10^{\1}', text)
        text = text.replace("×", r"\times ")
    return text

@app.post("/recognize/")
async def recognize(file: UploadFile = File(...), mode: str = Form(default="formula")):
    if mode not in PROMPT_LABEL_MAP:
        return JSONResponse(status_code=400, content={"error": f"Invalid mode."})

    prompt_label = PROMPT_LABEL_MAP[mode]
    
    run_id = uuid.uuid4().hex[:8]
    run_name = f"mlx_run_{run_id}"
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # 🎯 [DIRECT PRINT]: Clear visibility for current_run_dir
    print("\n" + "="*60)
    print(f"📁 ACTIVE DIRECTORY FOR THIS MLX INFERENCE RUN:")
    print(f"👉 current_run_dir: {run_dir}")
    print(f"👉 Full Path: {run_dir.resolve()}")
    print("="*60 + "\n")
    # ==================================================================

    raw_bytes = await file.read()
    pil_img = Image.open(BytesIO(raw_bytes)).convert("RGB")
    input_path = run_dir / "input_preview.png"
    pil_img.save(str(input_path))

    try:
        # Native MLX generate call (Returns GenerationResult Object)
        mlx_result = mlx_vlm.generate(
            model=model,
            processor=processor,
            image=pil_img,
            prompt=prompt_label,
            max_tokens=2048,
            temperature=0.0
        )
        
        # 🎯 [THE FIX]: Extracting the actual string from the object
        formatted_output = mlx_result.text if hasattr(mlx_result, "text") else str(mlx_result)
        
        cleaned = clean_latex(formatted_output, mode)

        md_lines = [
            f"# Handwritten MLX Recognition Result\n",
            f"**Mode:** {MODE_DISPLAY.get(mode, mode)}\n",
            f"**File:** {file.filename}\n",
            f"**Run Folder ID:** `{run_name}`\n",
            f"\n---\n",
            f"## Recognised Content\n",
            f"\n{cleaned}\n",
        ]
        md_content = "".join(md_lines)

        with open(run_dir / "result.md", "w", encoding="utf-8") as f:
            f.write(md_content)

        json_payload = {
            "run": run_name,
            "file": file.filename,
            "mode": mode,
            "prompt_used": prompt_label,
            "raw_text": formatted_output,
            "cleaned_text": cleaned,
            "output_dir": str(run_dir.resolve()),
        }
        with open(run_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=2, ensure_ascii=False)

        return JSONResponse(content={
            "run": run_name,
            "mode": mode,
            "cleaned_text": cleaned,
            "raw_text": formatted_output,
            "markdown": md_content,
            "preview_image": f"/static_hw/{run_name}/input_preview.png",
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content=HTML_DASHBOARD_FILE.read_text(encoding="utf-8"))

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting MLX Handwritten Formula Parser Engine")
    print("   URL Portal Access: http://127.0.0.1:8005\n")
    uvicorn.run("app_handwritten_formula_mlx:app", host="127.0.0.1", port=8005, reload=False)