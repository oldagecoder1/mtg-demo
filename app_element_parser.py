import os
import json
import uuid
import glob
import sys
import inspect
from pathlib import Path
from io import BytesIO
from PIL import Image
import torch

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Native Transformers Imports
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

print(f"--> [STARTUP] Current Working Directory: {os.getcwd()} 📂")

# ==================== THE DEEP SWEEPING MONKEYPATCH SHIELD ====================
def ultimate_causal_mask_patch():
    """
    Python ke memory canvas me chal rahe saare modules aur classes ko scan karke
    'create_causal_mask' se unexpected inputs_embeds argument ko complete drop karne ka universal system.
    """
    for mod_name, module in list(sys.modules.items()):
        if not module:
            continue
        
        # 1. Patch module-level standalone functions
        if hasattr(module, "create_causal_mask"):
            orig = getattr(module, "create_causal_mask")
            if not hasattr(orig, "_patched"):
                def make_safe_func(f):
                    def safe_func(*args, **kwargs):
                        kwargs.pop("inputs_embeds", None)
                        return f(*args, **kwargs)
                    safe_func._patched = True
                    return safe_func
                setattr(module, "create_causal_mask", make_safe_func(orig))
        
        # 2. Patch deep class-level methods inside transformers and paddle cache spaces
        if "transformers" in mod_name.lower() or "paddle" in mod_name.lower():
            try:
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if inspect.isclass(attr) and hasattr(attr, "create_causal_mask"):
                        orig = getattr(attr, "create_causal_mask")
                        if not hasattr(orig, "_patched"):
                            def make_safe_method(m):
                                def safe_method(*args, **kwargs):
                                    kwargs.pop("inputs_embeds", None)
                                    return m(*args, **kwargs)
                                safe_method._patched = True
                                return safe_method
                            setattr(attr, "create_causal_mask", make_safe_method(orig))
            except Exception:
                pass

# Initial run on startup
ultimate_causal_mask_patch()
# ==============================================================================

app = FastAPI(title="MTG POC - Universal Style Transformers VLM Engine")

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static_outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="static_outputs")

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model_path = "PaddlePaddle/PaddleOCR-VL"

print("--> Loading Universal Style VLM Config & Weights... Please wait ⏳")

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

if hasattr(config, "rope_scaling") and config.rope_scaling:
    if isinstance(config.rope_scaling, dict) and config.rope_scaling.get("type") == "default":
        config.rope_scaling["type"] = "linear"
if hasattr(config, "rope_type") and config.rope_type == "default":
    config.rope_type = "linear"

model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    config=config,
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16
).to(DEVICE).eval()

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# Run post-load class patch to clean freshly initialized instances
ultimate_causal_mask_patch()

print(f"--> Universal Textbook Style Transformers Engine Online on [{DEVICE}]! 🚀")

@app.post("/parse-document/")
async def parse_document(file: UploadFile = File(...)):
    filename = file.filename
    
    # Executing deep sweep once again before generation phase to remain bulletproof
    ultimate_causal_mask_patch()
    
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        prompt_text = (
            "OCR Task: Convert this image into high-quality structured Markdown. "
            "Identify all mathematical expressions, rules, derivations, and definitions. "
            "Format them into clean standalone LaTeX blocks using $$ ... $$. "
            "CRITICAL STYLE RULES:\n"
            "1. Whenever there is a division or fraction of words or numbers, strictly use the standard LaTeX '\\frac{numerator}{denominator}' structure.\n"
            "2. Always wrap text labels, descriptions, or words inside math blocks using '\\text{...}' to preserve font formatting.\n"
            "3. Do not use complex matrix arrays or tables to align simple formulas; output them as clean, sequential independent equations.\n"
            "4. Keep the output beautifully typeset, mirroring the exact logical steps of the input image but in professional print layout."
        )
        
        messages = [
            {
                "role": "user",         
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ]
            }
        ]
        
        inputs = processor.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True, 	
            return_dict=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                temperature=0.0
            )
            
        generated_markdown = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        unique_id = uuid.uuid4().hex[:6]
        run_folder_name = f"run_{unique_id}_{uuid.uuid4().hex[:4]}"
        current_run_dir = OUTPUT_DIR / run_folder_name
        current_run_dir.mkdir(parents=True, exist_ok=True)
        
        # ==================================================================
        # 🎯 [DIRECT PRINT FOR USER]: Active Directory Logging
        print("\n" + "="*60)
        print(f"📁 ACTIVE DIRECTORY FOR THIS INFERENCE RUN:")
        print(f"👉 current_run_dir: {current_run_dir}")
        print(f"👉 Full Path: {current_run_dir.resolve()}")
        print("="*60 + "\n")
        # ==================================================================
        
        md_file_path = current_run_dir / f"output_{unique_id}.md"
        with open(md_file_path, "w", encoding="utf-8") as f_md:
            f_md.write(generated_markdown)
            
        json_file_path = current_run_dir / f"output_{unique_id}.json"
        structured_json_log = {
            "file_parsed": filename,
            "run_session_id": run_folder_name,
            "extracted_markdown_raw": generated_markdown
        }
        with open(json_file_path, "w", encoding="utf-8") as f_js:
            json.dump(structured_json_log, f_js, indent=4, ensure_ascii=False)
            
        return JSONResponse(content={
            "file_parsed": filename,
            "run_folder": run_folder_name,
            "markdown_content": generated_markdown
        })
        
    except Exception as e:
        print(f"Transformers Model Inference Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ==================== LIVE VISUAL PREVIEW DASHBOARD ====================
@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MTG AI - Textbook Standard Engine</title>
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        
        <script>
            window.MathJax = {
                tex: {
                    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                    processEscapes: true
                },
                options: {
                    ignoreHtmlClass: 'noscript',
                    processHtmlClass: 'tex2jax_process'
                }
            };
        </script>
        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        
        <style>
            body { font-family: system-ui, -apple-system, sans-serif; background-color: #f4f6f9; margin: 0; padding: 20px; color: #2c3e50; }
            .uploader-section { max-width: 850px; margin: 20px auto; background: white; padding: 25px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); text-align: center; }
            .file-input { padding: 10px; margin-right: 10px; border: 1px solid #ccc; border-radius: 4px; }
            .upload-btn { background: #004D40; color: white; border: none; padding: 11px 25px; font-weight: bold; border-radius: 4px; cursor: pointer; }
            .upload-btn:hover { background: #00332c; }
            #loading-status { display: none; font-weight: bold; color: #ff9800; margin: 15px; }
            
            .book-container { max-width: 850px; background: white; margin: 30px auto; padding: 50px; box-shadow: 0 4px 25px rgba(0,0,0,0.1); border-radius: 8px; display: none; }
            .book-container p { font-size: 16px; line-height: 1.8; text-align: justify; }
            .mcq-options-line { background: #fafafa; border-left: 4px solid #004D40; padding: 12px 18px; margin: 15px 0; border-radius: 4px; }
            .option-tag { color: #004D40; font-weight: bold; margin-right: 5px; }
        </style>
    </head>
    <body>

        <div class="uploader-section">
            <h2>MTG Universal Textbook Preserver Engine v5.4 🧪</h2>
            <p style="color: #666;">Upload any image or handwritten sheet to parse mirror markdown & latex natively.</p>
            <input type="file" id="pdfFile" class="file-input" accept="image/*, .pdf" />
            <button onclick="processPDF()" class="upload-btn">Upload & Process</button>
            <div id="loading-status">Direct VLM Parsing into Textbook Style Layout... Please wait ⏳</div>
        </div>

        <div class="book-container tex2jax_process" id="outputBook"></div>

        <script>
            async function processPDF() {
                const fileInput = document.getElementById('pdfFile');
                const loadingStatus = document.getElementById('loading-status');
                const outputBook = document.getElementById('outputBook');
                
                if (!fileInput.files[0]) { alert("Bhai file select kijiye!"); return; }
                
                loadingStatus.style.display = "block"; outputBook.style.display = "none"; outputBook.innerHTML = ""; 
                const formData = new FormData(); formData.append("file", fileInput.files[0]);
                
                try {
                    const response = await fetch("/parse-document/", { method: "POST", body: formData });
                    const data = await response.json(); loadingStatus.style.display = "none";
                    
                    if (data.markdown_content) {
                        outputBook.style.display = "block";
                        
                        let rawContent = data.markdown_content;
                        let mathBlocks = [];
                        
                        rawContent = rawContent.replace(/\$\$\s*([\s\S]*?)\s*\$\$/g, function(match, capture) {
                            mathBlocks.push('$$' + capture + '$$');
                            return '___MATH_SHIELD_DB_' + (mathBlocks.length - 1) + '___';
                        });
                        
                        rawContent = rawContent.replace(/\$\s*([\s\S]*?)\s*\$/g, function(match, capture) {
                            mathBlocks.push('$' + capture + '$');
                            return '___MATH_SHIELD_IL_' + (mathBlocks.length - 1) + '___';
                        });
                        
                        let htmlOutput = marked.parse(rawContent);
                        
                        mathBlocks.forEach((block, index) => {
                            htmlOutput = htmlOutput.replace('___MATH_SHIELD_DB_' + index + '___', block);
                            htmlOutput = htmlOutput.replace('___MATH_SHIELD_IL_' + index + '___', block);
                        });
                        
                        outputBook.innerHTML = htmlOutput;
                        postProcessBookLayout();
                        
                        if (window.MathJax && typeof MathJax.typesetPromise === 'function') {
                            setTimeout(() => {
                                MathJax.typesetPromise([outputBook]).catch(err => console.log(err));
                            }, 50);
                        }
                    } else { alert("Processing error occurred."); }
                } catch (error) { loadingStatus.style.display = "none"; console.error(error); alert("Server Error!"); }
            }

            function postProcessBookLayout() {
                document.querySelectorAll('#outputBook p').forEach(p => {
                    const html = p.innerHTML;
                    if (html.includes('(a)') || html.includes('(b)') || html.includes('(c)')) {
                        p.classList.add('mcq-options-line');
                        p.innerHTML = html.replace(/(\([a-d]\))/g, '<span class="option-tag">$1</span>');
                    }
                });
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_element_parser:app", host="127.0.0.1", port=8000, reload=False)