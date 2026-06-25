import os
import json
import uuid
import re
import sys
import inspect
from pathlib import Path
from io import BytesIO
from PIL import Image
import torch

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Native Transformers Imports
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

print(f"--> [STARTUP] Current Working Directory: {os.getcwd()} 📂")

# ==================== THE AUTOMATIC MONKEYPATCH SHIELD ====================
def ultimate_causal_mask_patch():
    """
    Python ke memory canvas me chal rahe saare modules aur classes ko scan karke
    'create_causal_mask' se unexpected inputs_embeds argument ko drop karne ka system.
    PyTorch JIT/C++ classes ke crash ko rokne ke liye isme safe try-except check lagaya hai.
    """
    for mod_name, module in list(sys.modules.items()):
        if not module: continue
        
        # 1. Patch module-level standalone functions
        if hasattr(module, "create_causal_mask"):
            try:
                orig = getattr(module, "create_causal_mask")
                is_patched = False
                try:
                    # PyTorch C++ custom classes ke RuntimeError bypass karne ke liye direct getattr check
                    is_patched = getattr(orig, "_patched", False)
                except Exception:
                    pass
                
                if not is_patched:
                    def make_safe_func(f):
                        def safe_func(*args, **kwargs):
                            kwargs.pop("inputs_embeds", None)
                            return f(*args, **kwargs)
                        safe_func._patched = True
                        return safe_func
                    setattr(module, "create_causal_mask", make_safe_func(orig))
            except Exception:
                pass
        
        # 2. Patch deep class-level methods inside transformers and paddle cache spaces
        if "transformers" in mod_name.lower() or "paddle" in mod_name.lower():
            try:
                for attr_name in dir(module):
                    try:
                        attr = getattr(module, attr_name)
                        if inspect.isclass(attr) and hasattr(attr, "create_causal_mask"):
                            orig = getattr(attr, "create_causal_mask")
                            is_patched = False
                            try:
                                is_patched = getattr(orig, "_patched", False)
                            except Exception:
                                pass
                            
                            if not is_patched:
                                def make_safe_method(m):
                                    def safe_method(*args, **kwargs):
                                        kwargs.pop("inputs_embeds", None)
                                        return m(*args, **kwargs)
                                    safe_method._patched = True
                                    return safe_method
                                setattr(attr, "create_causal_mask", make_safe_method(orig))
                    except Exception:
                        pass
            except Exception: 
                pass

# Initial run on startup
ultimate_causal_mask_patch()
# ==========================================================================

app = FastAPI(title="Handwritten Formula Parser - Mac GPU Native")

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
    <title>MTG AI - Element Parser Dashboard</title>
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
        #output { display: none; margin-top: 25px; padding: 20px; border-top: 2px solid #004D40; }
    </style>
</head>
<body>
    <div class="box">
        <h2>MTG Element-Level Native Parser Dashboard 🧪</h2>
        <label>Select Recognition Mode:</label>
        <select id="modeSelect">
            <option value="formula">🧮 Formula Recognition (LaTeX)</option>
            <option value="ocr">📝 Text / Handwriting Recognition</option>
            <option value="table">📊 Table Recognition (Markdown)</option>
            <option value="chart">📈 Chart / Graph Recognition</option>
        </select>
        <input type="file" id="fileInput" accept="image/*" />
        <button onclick="uploadAndParse()">Run Inference on Mac GPU</button>
        <div id="status" style="color: #ff9800; font-weight: bold; margin: 10px 0;"></div>
        <img id="preview" />
        <div id="output"></div>
    </div>
    <script>
        async function uploadAndParse() {
            const file = document.getElementById('fileInput').files[0];
            const mode = document.getElementById('modeSelect').value;
            if(!file) { alert("File select karo bhai!"); return; }
            
            document.getElementById('status').innerText = "VLM Processing on Apple Silicon GPU... ⏳";
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

# Setup hardware accelerator (Forces Mac GPU execution natively via MPS)
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
model_path = "PaddlePaddle/PaddleOCR-VL"

print("=" * 60)
print("  Initialising Native Transformers Engine  •  Element-level Mode")
print(f"  Target Accelerator Node: [{DEVICE}]")
print("=" * 60)

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
if hasattr(config, "rope_scaling") and config.rope_scaling:
    if isinstance(config.rope_scaling, dict) and config.rope_scaling.get("type") == "default":
        config.rope_scaling["type"] = "linear"
if hasattr(config, "rope_type") and config.rope_type == "default":
    config.rope_type = "linear"

model = AutoModelForCausalLM.from_pretrained(
    model_path, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16
).to(DEVICE).eval()

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
ultimate_causal_mask_patch()

print("--> Mac Native GPU VLM Pipeline Ready ✅")
print("=" * 60)

# The Exact Task Token Matrix mapping the VLM understands
PROMPT_LABEL_MAP = {
    "formula": "Formula Recognition:",
    "ocr":     "OCR:",
    "table":   "Table Recognition:",
    "chart":   "Chart Recognition:",
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

def run_element_recognition(image_path: str, mode: str) -> dict:
    prompt_label = PROMPT_LABEL_MAP.get(mode, "Formula Recognition:")
    print(f"\n  Mode      : {MODE_DISPLAY.get(mode, mode)}")
    print(f"  Prompt Token: {prompt_label}")
    print(f"  Image Engine Target: {image_path}")
    
    ultimate_causal_mask_patch()
    image = Image.open(image_path).convert("RGB")
    
    messages = [
        {
            "role": "user",         
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_label},
            ]
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False
        )
        
    raw_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    cleaned = clean_latex(raw_text, mode)
    
    return {
        "mode": mode,
        "display_mode": MODE_DISPLAY.get(mode, mode),
        "prompt_used": prompt_label,
        "raw_text": raw_text,
        "cleaned_text": cleaned,
    }

@app.post("/recognize/")
async def recognize(file: UploadFile = File(...), mode: str = Form(default="formula")):
    if mode not in PROMPT_LABEL_MAP:
        return JSONResponse(status_code=400, content={"error": f"Invalid mode. Choose from: {list(PROMPT_LABEL_MAP.keys())}"})

    run_id = uuid.uuid4().hex[:8]
    run_name = f"hw_run_{run_id}"
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # 🎯 [DIRECT PRINT FOR USER]: Active Directory Logging
    print("\n" + "="*60)
    print(f"📁 ACTIVE RUN LAYER : {run_name}")
    print(f"📁 current_run_dir  : {run_dir.resolve()}")
    print("="*60 + "\n")
    # ==================================================================

    raw_bytes = await file.read()
    pil_img = Image.open(BytesIO(raw_bytes)).convert("RGB")
    input_path = run_dir / "input_preview.png"
    pil_img.save(str(input_path))

    try:
        result = run_element_recognition(str(input_path), mode)

        md_lines = [
            f"# Handwritten Recognition Result\n",
            f"**Mode:** {result['display_mode']}\n",
            f"**File:** {file.filename}\n",
            f"**Run Folder ID:** `{run_name}`\n",
            f"\n---\n",
            f"## Recognised Content\n",
            f"\n{result['cleaned_text']}\n",
        ]
        md_content = "".join(md_lines)

        with open(run_dir / "result.md", "w", encoding="utf-8") as f:
            f.write(md_content)

        json_payload = {
            "run": run_name,
            "file": file.filename,
            **result,
            "output_dir": str(run_dir.resolve()),
        }
        with open(run_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=2, ensure_ascii=False)

        return JSONResponse(content={
            "run": run_name,
            "mode": mode,
            "cleaned_text": result["cleaned_text"],
            "raw_text": result["raw_text"],
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
    print("\n🚀 Starting Handwritten Formula Parser Engine on Mac GPU Architecture")
    print("   URL Portal Access: http://127.0.0.1:8004\n")
    uvicorn.run("app_handwritten_formula_with_mps:app", host="127.0.0.1", port=8004, reload=False)