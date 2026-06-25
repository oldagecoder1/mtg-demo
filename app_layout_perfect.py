import os
import json
import uuid
import glob
import re
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Latest Stable Import
from paddleocr import PaddleOCRVL

app = FastAPI(title="MTG POC - Perfected LaTeX & Storage Engine")

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Static file mount for assets access
app.mount("/static_outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="static_outputs")

# Initialize Engine
pipeline = PaddleOCRVL(
    device="cpu",
    use_formula_recognition=True
)

# SMART BACKEND LATEX FIXER ENGINE
def auto_fix_latex_delimiters(text: str) -> str:
    # 1. Clean messy split tokens from legacy text parser (e.g., "10 $ ^{5} $" -> "$10^{5}$")
    text = re.sub(r'10\s*\$\s*\^\{([-+]?\d+)\}\s*\$', r'$10^{\1}$', text)
    text = re.sub(r'([0-9.]+)\s*\$\s*[xX×]\s*\$\s*10', r'\1 \\times 10', text)
    
    # 2. Fix Scientific Notations without delimiters like: 4.8 × 10^{-3} or 5.00 × 10^{2}
    text = re.sub(r'(?<!\$)(?:\d+(?:\.\d+)?\s*[xX×]\s*)?10\^\{[-+]?\d+\}(?!\$)', 
                  lambda m: '$' + m.group(0).replace('×', '\\times ').replace('x', '\\times ').replace('X', '\\times ') + '$', text)
    
    # 3. Fix Chemical Formulas like: KClO_{3}, O_{2}, CaCO_{3}, CO_{2}, N_{2}
    text = re.sub(r'(?<!\$)\b([A-Z][A-Za-z0-9]*_\{[a-zA-Z0-9+-]+\}[A-Za-z0-9]*)(?!\w)(?!\$)', 
                  lambda m: '$' + m.group(0) + '$', text)
    
    # 4. Fix Physics Units like: m s^{-2} or kg m^{-3}
    text = re.sub(r'(?<!\$)\b([a-zA-Z]+\s+[a-zA-Z]+\^\{[-+]?\d+\})(?!\w)(?!\$)', 
                  lambda m: '$' + m.group(0) + '$', text)
    
    # 5. Global internal clean for un-escaped multiplication signs inside existing math blocks
    text = text.replace('×', '\\times ')
    
    return text

print("--> Ultimate LaTeX + JSON Session Engine Armed! 🚀")

@app.post("/parse-document/")
async def parse_document(file: UploadFile = File(...)):
    filename = file.filename
    unique_id = uuid.uuid4().hex[:6]
    temp_path = f"incoming_{unique_id}_{filename}"
    
    with open(temp_path, "wb") as f:
        f.write(await file.read())
        
    try:
        # Step 1: Run VLM Predictor
        output = pipeline.predict(input=temp_path)
        pages_res = list(output)
        
        # Step 2: Create isolated session directory
        run_folder_name = f"run_{unique_id}_{uuid.uuid4().hex[:4]}"
        current_run_dir = OUTPUT_DIR / run_folder_name
        current_run_dir.mkdir(parents=True, exist_ok=True)
        
        # [FIXED] Save BOTH markdown AND json inside the isolated folder
        restructured = pipeline.restructure_pages(pages_res, merge_tables=True, relevel_titles=True, concatenate_pages=True)
        for res in restructured:
            res.save_to_markdown(save_path=current_run_dir)
            res.save_to_json(save_path=current_run_dir) # <-- FIXED: JSON file restored successfully!
            
        # Step 3: Find generated .md file
        md_files = glob.glob(os.path.join(str(current_run_dir), "*.md"))
        if not md_files:
            raise Exception("Markdown file generation failed.")
            
        with open(md_files[0], 'r', encoding='utf-8') as f_md:
            markdown_content = f_md.read()
            
        # Step 4: Run Auto-LaTeX Wrap & Asset URL Correction
        markdown_content = auto_fix_latex_delimiters(markdown_content)
        markdown_content = markdown_content.replace('src="imgs/', f'src="/static_outputs/{run_folder_name}/imgs/')
        
        return JSONResponse(content={
            "file_parsed": filename,
            "run_folder": run_folder_name,
            "markdown_content": markdown_content
        })
        
    except Exception as e:
        print(f"Pipeline Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

# ==================== LIVE VISUAL PREVIEW DASHBOARD ====================
@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MTG AI - Layout & LaTeX Perfection Engine</title>
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
            
            .book-container h1 { color: #004D40; font-size: 26px; border-bottom: 2px solid #004D40; padding-bottom: 8px; margin-top: 35px; font-weight: bold; }
            .book-container h2 { color: #00796B; font-size: 20px; margin-top: 25px; }
            .book-container h3, .book-container h4 { color: #111; font-size: 16px; margin-top: 20px; font-weight: 600; }
            .book-container p { font-size: 15px; line-height: 1.7; text-align: justify; }
            
            .mcq-options-line { background: #fafafa; border-left: 4px solid #004D40; padding: 12px 18px; margin: 15px 0; border-radius: 4px; font-weight: 500; }
            .option-tag { color: #004D40; font-weight: bold; margin-right: 5px; }
            
            .book-container table { width: 100%; border-collapse: collapse; margin: 25px 0; font-size: 14px; }
            .book-container th, .book-container td { border: 1px solid #B2DFDB; padding: 10px; text-align: center; }
            .book-container tr:nth-child(even) { background-color: #E0F2F1; }
            .book-container tr:first-child { background: #004D40; color: white; font-weight: bold; }
            
            .book-container img { display: block; max-width: 60%; height: auto; margin: 25px auto; border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
        </style>
    </head>
    <body>

        <div class="uploader-section">
            <h2>MTG Dynamic Layout + LaTeX Perfectionist Engine v4.2 🧪</h2>
            <input type="file" id="pdfFile" class="file-input" accept=".pdf, ".image/*" />
            <button onclick="processPDF()" class="upload-btn">Upload & Parse Document</button>
            <div id="loading-status">Processing VLM Layers & Formulating Protected LaTeX Expressions... ⏳</div>
        </div>

        <div class="book-container tex2jax_process" id="outputBook"></div>

        <script>
            async function processPDF() {
                const fileInput = document.getElementById('pdfFile');
                const loadingStatus = document.getElementById('loading-status');
                const outputBook = document.getElementById('outputBook');
                
                if (!fileInput.files[0]) { alert("Bhai pahle ek PDF file select karo!"); return; }
                
                loadingStatus.style.display = "block";
                outputBook.style.display = "none";
                outputBook.innerHTML = ""; 
                
                const formData = new FormData();
                formData.append("file", fileInput.files[0]);
                
                try {
                    const response = await fetch("/parse-document/", { method: "POST", body: formData });
                    const data = await response.json();
                    loadingStatus.style.display = "none";
                    
                    if (data.markdown_content) {
                        outputBook.style.display = "block";
                        
                        // [THE BULLETPROOF FIX]: Protect Math blocks before Markdown compile
                        let rawContent = data.markdown_content;
                        let mathBlocks = [];
                        
                        // 1. Protect Display Math ($$...$$)
                        rawContent = rawContent.replace(/\$\$\s*([\s\S]*?)\s*\$\$/g, function(match, capture) {
                            mathBlocks.push('$$' + capture + '$$');
                            return '___MATH_DISPLAY_BLOCK_' + (mathBlocks.length - 1) + '___';
                        });
                        
                        // 2. Protect Inline Math ($...$)
                        rawContent = rawContent.replace(/\$\s*([\s\S]*?)\s*\$/g, function(match, capture) {
                            mathBlocks.push('$' + capture + '$');
                            return '___MATH_INLINE_BLOCK_' + (mathBlocks.length - 1) + '___';
                        });
                        
                        // 3. Compile protected plain markdown safely
                        let htmlOutput = marked.parse(rawContent);
                        
                        // 4. Inject original crisp math expressions back into final HTML
                        mathBlocks.forEach((block, index) => {
                            htmlOutput = htmlOutput.replace('___MATH_DISPLAY_BLOCK_' + index + '___', block);
                            htmlOutput = htmlOutput.replace('___MATH_INLINE_BLOCK_' + index + '___', block);
                        });
                        
                        outputBook.innerHTML = htmlOutput;
                        
                        // Format MCQ horizontal spacing
                        postProcessBookLayout();
                        
                        // Force MathJax to re-scan the pristine HTML string
                        if (window.MathJax && typeof MathJax.typesetPromise === 'function') {
                            setTimeout(() => {
                                MathJax.typesetPromise([outputBook]).catch(err => console.log(err));
                            }, 50);
                        }
                    } else {
                        alert("Parsing Error!");
                    }
                } catch (error) {
                    loadingStatus.style.display = "none";
                    console.error(error);
                    alert("Server Error!");
                }
            }

            function postProcessBookLayout() {
                document.querySelectorAll('#outputBook p').forEach(p => {
                    const html = p.innerHTML;
                    if (html.includes('(a)') || html.includes('(b)') || html.includes('(c)')) {
                        p.classList.add('mcq-options-line');
                        let formattedHtml = html.replace(/(\([a-d]\))/g, '<span class="option-tag">$1</span>');
                        p.innerHTML = formattedHtml;
                    }
                });
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_layout_perfect:app", host="127.0.0.1", port=8000, reload=False)