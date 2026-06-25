import os
import json
import uuid
import fitz  # PyMuPDF
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types

# FastAPI Application Initialize
app = FastAPI(title="MTG POC - Gemini Modern Spatial Parser")

IMG_OUTPUT_DIR = "extracted_images"
os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)

# GEMINI MODERN CLIENT INITIALIZATION
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=GEMINI_API_KEY)

# LIFESPAN STARTUP EVENT - TO PREVENT DUPLICATE LOGS PRINTING
@app.on_event("startup")
async def startup_event():
    print("\n--> System Status: Gemini Modern Visual Coordinator Active & Connected! 🚀")
    print("--> Ready to accept payload hit on: http://127.0.0.1:8000/parse-document/\n")

# STRICT SPATIAL EXTRACTION PROMPT FOR GEMINI
STRICT_JSON_PROMPT = """
You are a highly precise document spatial analyzer. Scan the attached document pages and extract objective multiple-choice questions (MCQs).

CRITICAL EXTRACTION LAWS:
1. Extract ONLY valid questions that contain option blocks like (a, b, c, d) or (A, B, C, D).
2. Completely IGNORE general tables, timeline trends, or introductory text headers.
3. Convert all math formulas and chemical compounds into standard LaTeX format enclosed in single '$'.

IMAGE BOUNDING BOX DETECTION RULES:
- If a question has an embedded diagram, chemical structure, or graph, you MUST locate the full visual area including its core identification labels (e.g., internal text like "Diagram I", "I", "II", "III", axis units, or chemical names that describe the drawing). Return its normalized coordinates `[ymin, xmin, ymax, xmax]` on a scale of 0-1000 inside "question_image_box".
- CRITICAL EXCLUSION: Do NOT include any separate MCQ options text sentences, or option markers (like external standalone "(a)", "(b)", or "Option A") that belong to the test structure itself. Cut the box precisely right after the diagram labels end and before the actual option choices begin.
- If an option (A, B, C, or D) contains a visual structure/diagram instead of text, set its text as "IMAGE_CONTAINED" and provide its coordinates inside "option_image_boxes" mapped to that option key.
- If there are no images, set the boxes to null.

Return a valid JSON object matching this exact structure:
{
  "questions": [
    {
      "question_number": int,
      "question_text": "string with LaTeX",
      "has_image": boolean,
      "question_image_box": [ymin, xmin, ymax, xmax] or null,
      "options": {
        "A": "string text or IMAGE_CONTAINED",
        "B": "string text or IMAGE_CONTAINED",
        "C": "string text or IMAGE_CONTAINED",
        "D": "string text or IMAGE_CONTAINED"
      },
      "option_image_boxes": {
        "A": [ymin, xmin, ymax, xmax] or null,
        "B": [ymin, xmin, ymax, xmax] or null,
        "C": [ymin, xmin, ymax, xmax] or null,
        "D": [ymin, xmin, ymax, xmax] or null
      },
      "correct_answer": null
    }
  ]
}
"""

def crop_and_save(pdf_path, page_num, box, filename_suffix):
    """Helper to cut high-resolution sub-images using pixel boundaries provided by Gemini Studio"""
    if not box or len(box) != 4:
        return None
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        pix = page.get_pixmap(dpi=200)
        img_path = f"temp_render_{uuid.uuid4().hex[:4]}.png"
        pix.save(img_path)
        doc.close()
        
        img = Image.open(img_path)
        w, h = img.size
        ymin, xmin, ymax, xmax = box
        
        # Coordinate map converter
        left = (xmin / 1000) * w
        top = (ymin / 1000) * h
        right = (xmax / 1000) * w
        bottom = (ymax / 1000) * h
        
        cropped = img.crop((left, top, right, bottom))
        final_filename = os.path.join(IMG_OUTPUT_DIR, f"crop_{uuid.uuid4().hex[:6]}_{filename_suffix}.png")
        cropped.save(final_filename, "PNG")
        
        if os.path.exists(img_path):
            os.remove(img_path)
            
        return final_filename
    except Exception as e:
        print(f" -> [CROPPER ERROR]: {e}")
        return None

@app.post("/parse-document/")
async def parse_document(file: UploadFile = File(...)):
    filename = file.filename
    temp_path = f"incoming_{uuid.uuid4().hex[:6]}_{filename}"
    print(f"\n==================== NEW ADVANCED SPATIAL REQUEST: {filename} ====================")
    
    with open(temp_path, "wb") as f:
        f.write(await file.read())
        
    try:
        print("--> Step 1: Uploading document matrix to Gemini Multimodal Cloud...")
        uploaded_file = client.files.upload(file=temp_path)
        
        print("--> Step 2: Running spatial query extraction via Gemini 1.5 Flash...")
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[uploaded_file, STRICT_JSON_PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        # Cloud safe deletion hook
        client.files.delete(name=uploaded_file.name)
        
        page_data = json.loads(response.text)
        extracted_questions = page_data.get("questions", [])
        
        print("--> Step 3: Triggering dynamic image cropper layer using spatial points...")
        final_questions_array = []
        
        for q in extracted_questions:
            # Default fallbacks handled gracefully
            page_index = q.get("extracted_from_page", 2) - 1 
            
            # 1. Question Image Mapping
            q["question_image_path"] = None
            if q.get("question_image_box"):
                saved_path = crop_and_save(temp_path, page_index, q["question_image_box"], f"q_{q['question_number']}")
                q["question_image_path"] = saved_path
                
            # 2. Options Image Mapping
            q["option_image_paths"] = {"A": None, "B": None, "C": None, "D": None}
            opt_boxes = q.get("option_image_boxes", {})
            if opt_boxes:
                for option_key, box in opt_boxes.items():
                    if box:
                        saved_opt_path = crop_and_save(temp_path, page_index, box, f"q_{q['question_number']}_opt_{option_key}")
                        q["option_image_paths"][option_key] = saved_opt_path
            
            # Metadata cleanup before output dispatch
            if "question_image_box" in q: del q["question_image_box"]
            if "option_image_boxes" in q: del q["option_image_boxes"]
            
            final_questions_array.append(q)

        print(f"--> Workflow Complete. Successfully generated metadata for {len(final_questions_array)} elements.")

    except Exception as e:
        print(f"--> ERROR: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)
        print(f"==================== METADATA PIPELINE STREAM END ====================\n")
        
    return JSONResponse(content={
        "file_parsed": filename,
        "total_questions_extracted": len(final_questions_array),
        "questions": final_questions_array
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_with_gemini:app", host="127.0.0.1", port=8000, reload=False)