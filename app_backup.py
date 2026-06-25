import os
import json
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from llama_parse import LlamaParse
import ollama

# MTG POC App Initialize
app = FastAPI(title="MTG POC - LlamaParse + Local Ollama Pure JSON Extractor")

IMG_OUTPUT_DIR = "extracted_images"
os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)

# 1. CONFIGURE LLAMAPARSE API KEY
# Apni LlamaParse API Key yahan paste karein ya system environment variable me set karein
LLAMA_CLOUD_API_KEY = os.environ.get("LLAMA_CLOUD_API_KEY", "llx-zeVdvnspQRp6AzIRKxKrrmcs6pAbk1EtbwrV7wP1JWxyByxY")

# Initialize LlamaParse - Yeh background me markdown format me text aur formulas nikalega
parser = LlamaParse(
    api_key=LLAMA_CLOUD_API_KEY,
    result_type="markdown",
    num_workers=4,
    use_vendor_multimodal_model=True,     # Naya strict syntax parameter
    vendor_multimodal_model_name="openai-gpt-4o-mini" # Accurate layout parsing
)

print("--> System Status: LlamaParse Cloud + Local Ollama Engine Connected & Ready!")

# STRICT SYSTEM PROMPT FOR ONLY QUESTIONS
# Yeh prompt Ollama ko baaki sab (overview, tables, intros) delete karne par majboor karega
STRICT_JSON_PROMPT = """
You are a strict data extractor. Analyze the provided textbook text and extract ONLY the multiple-choice questions (MCQs).

CRITICAL INSTRUCTIONS:
- Extract ONLY actual questions that have options like (a, b, c, d) or (A, B, C, D).
- Completely IGNORE paragraph texts, intros, assessment notes, instructions, and analysis tables.
- Convert all math symbols, units, and chemical equations into standard LaTeX format enclosed in single '$'.

You MUST return a JSON object with a single key named "questions" containing the array of objects. Follow this exact structure:
{
  "questions": [
    {
      "question_number": 1,
      "question_text": "Which of the following is a chemical fertilizer?",
      "options": {"A": "Urea", "B": "Sodium nitrate", "C": "Ammonium sulphate", "D": "All of these"},
      "correct_answer": null
    }
  ]
}
"""

@app.post("/parse-document/")
async def parse_document(file: UploadFile = File(...)):
    filename = file.filename
    temp_path = f"incoming_{uuid.uuid4().hex}_{filename}"
    print(f"\n==================== NEW PURGE REQUEST: {filename} ====================")
    
    with open(temp_path, "wb") as f:
        f.write(await file.read())
        
    try:
        print("--> Step 1: LlamaParse extracting raw text/formulas from document...")
        extra_info = {"target_dir": IMG_OUTPUT_DIR}
        parsed_pages = await parser.aload_data(temp_path, extra_info=extra_info)
        
        final_questions_array = []
        total_pages = len(parsed_pages)
        
        print("--> Step 2: Local Ollama filtering data to keep ONLY questions JSON...")
        for idx, doc in enumerate(parsed_pages):
            page_markdown = doc.text
            
            if not page_markdown.strip():
                print(f"[PAGE {idx + 1}/{total_pages}] - Empty raw text. Skipping.")
                continue

            # --- LIVE LOG DEBUG PRINTS ---
            print(f"\n--- [RAW TEXT FROM LLAMAPARSE - PAGE {idx + 1}] ---")
            print(page_markdown[:300] + "\n... [TRUNCATED] ...")
            print("--------------------------------------------------\n")

            # Local Ollama API Call (Using llama3.1)
            response = ollama.generate(
                model='llama3.1', 
                prompt=f"System Prompt:\n{STRICT_JSON_PROMPT}\n\nContent to extract from:\n{page_markdown}",
                options={"temperature": 0.1}, # Output strict aur uniform rakhne ke liye
                format="json" # Ollama native JSON enforcer mode active
            )
            
            try:
                # Response parse karna
                page_data = json.loads(response['response'])
                
                # Model ne "questions" key ke sath array diya hai ya nahi check karna
                page_questions = page_data.get("questions", [])
                
                if isinstance(page_questions, list) and len(page_questions) > 0:
                    for q in page_questions:
                        q["extracted_from_page"] = idx + 1
                        final_questions_array.append(q)
                    print(f"[PAGE {idx + 1}/{total_pages}] - Extracted {len(page_questions)} clean questions via Ollama.")
                else:
                    print(f"[PAGE {idx + 1}/{total_pages}] - Overview/Tables filtered out (No questions found).")
            except Exception as parse_err:
                print(f"[PAGE {idx + 1}/{total_pages}] - JSON Parsing error on Ollama response: {parse_err}")
                print(f"Raw Model response was: {response['response'][:200]}")
                
    except Exception as e:
        print(f"--> ERROR: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)
        print(f"==================== WORKFLOW COMPLETE ====================\n")
        
    # Final output me aapko sirf flat array of questions milega
    return JSONResponse(content={
        "file_parsed": filename, 
        "total_questions_extracted": len(final_questions_array), 
        "questions": final_questions_array
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)