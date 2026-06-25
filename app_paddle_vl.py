import os
import json
import uuid
import fitz  # PyMuPDF (Still used for precise figure cropping helper)
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

# Official Documentation v3.x New Stable Import
from paddleocr import PaddleOCRVL

app = FastAPI(title="MTG POC - PaddleOCR-VL Official Engine")

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("--> System Status: Initializing Official PaddleOCRVL Engine...")

# 1. INITIALIZE THE NEW OFFICAL PIPELINE (Apple Silicon / CPU optimized)
# Documents orientation aur unwarping ko True kar diya taaki handwritten notes ekdum sahi ho jayein
pipeline = PaddleOCRVL(
    device="cpu", # Apple Silicon plays on CPU device wrapper natively inside paddle core
    use_doc_orientation_classify=True, 
    use_doc_unwarping=True,
    use_layout_detection=True,
    use_formula_recognition=True
)

print("--> System Status: Official PaddleOCRVL Engine Successfully Armed! 🚀")

@app.post("/parse-document/")
async def parse_document(file: UploadFile = File(...)):
    filename = file.filename
    temp_path = f"incoming_{uuid.uuid4().hex[:6]}_{filename}"
    print(f"\n==================== PADDLEOCR-VL OFFICIAL PIPELINE: {filename} ====================")
    
    with open(temp_path, "wb") as f:
        f.write(await file.read())
        
    try:
        # Step 1: Run Official VLM Prediction
        print("--> Step 1: Executing PaddleOCRVL Predictor...")
        output = pipeline.predict(input=temp_path)
        
        # Step 2: Handle PDF Restructuring (Cross-page table merging & multi-level titles)
        print("--> Step 2: Merging pages and restructuring titles layout...")
        pages_res = list(output)
        restructured_output = pipeline.restructure_pages(
            pages_res, 
            merge_tables=True, 
            relevel_titles=True, 
            concatenate_pages=True
        )
        
        # Create isolated session directory for this run
        unique_run_id = uuid.uuid4().hex[:6]
        run_folder_name = f"run_{unique_run_id}_{uuid.uuid4().hex[:4]}"
        current_run_dir = OUTPUT_DIR / run_folder_name
        current_run_dir.mkdir(parents=True, exist_ok=True)
        
        # Print newly created run folder information to console
        print("\n" + "=" * 60)
        print(f"📁 ACTIVE RUN DIRECTORY FOR THIS INFERENCE:")
        print(f"👉 Name: {run_folder_name}")
        print(f"👉 Full Path: {current_run_dir.resolve()}")
        print("=" * 60 + "\n")
        
        # Create imgs subdirectory inside the run folder
        run_imgs_dir = current_run_dir / "imgs"
        run_imgs_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 3: Save official formats to disk logs (As per documentation guidelines)
        for res in restructured_output:
            res.save_to_json(save_path=current_run_dir)
            res.save_to_markdown(save_path=current_run_dir)
            
        # Step 4: Custom Object Extraction for our specific MCQ Frontend
        # Hum manually PyMuPDF bridge se image blocks ko extract karenge taaki dynamic rendering ho sake
        print("--> Step 4: Extracting isolated visual elements via structural positions...")
        doc_bridge = fitz.open(temp_path)
        final_extracted_data = []
        
        # We process pages result array to map visual block coordinates
        for page_idx, page_res in enumerate(pages_res):
            page = doc_bridge[page_idx] if page_idx < len(doc_bridge) else None
            page_elements = []
            
            # Access raw structured layout tokens inside paddle outcome
            # Note: Depending on doc version, page_res attributes contain block arrays
            raw_blocks = getattr(page_res, 'layout_results', []) or getattr(page_res, 'blocks', [])
            
            for element_idx, block in enumerate(raw_blocks):
                # Standard attributes extraction
                block_type = block.get('type', 'text')
                bbox = block.get('bbox', [0, 0, 0, 0]) # [x1, y1, x2, y2]
                
                element_metadata = {
                    "block_id": element_idx,
                    "type": block_type,
                    "coordinates": bbox,
                    "content": block.get('text', None),
                    "cropped_image_path": None
                }
                
                # If PaddleOCRVL detects a dedicated handwritten drawing or figure
                if block_type == 'figure' and page:
                    rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                    try:
                        crop_pix = page.get_pixmap(clip=rect, dpi=200)
                        crop_filename = os.path.join(str(run_imgs_dir), f"official_crop_p{page_idx+1}_b{element_idx}.png")
                        crop_pix.save(crop_filename)
                        element_metadata["cropped_image_path"] = crop_filename
                        element_metadata["content"] = "IMAGE_ASSET_ISOLATED"
                    except Exception as img_err:
                        print(f"   -> Image crop skipping: {img_err}")
                        
                page_elements.append(element_metadata)
                
            final_extracted_data.append({
                "page": page_idx + 1,
                "elements": page_elements
            })
            
        doc_bridge.close()
        print("--> Workflow Complete. Official VLM data compiled successfully.")
        
    except Exception as e:
        print(f"--> ERROR: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(temp_path): 
            os.remove(temp_path)
        print(f"==================== METADATA OFFICIAL VL END ====================\n")
        
    return JSONResponse(content={
        "file_parsed": filename,
        "run_folder": run_folder_name,
        "msg": f"Structured outputs saved in ./output/{run_folder_name}/ directory",
        "structured_layout": final_extracted_data
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_paddle_vl:app", host="127.0.0.1", port=8000, reload=False)