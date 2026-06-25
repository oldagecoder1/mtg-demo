from fastapi import APIRouter, UploadFile, File
from fastapi.responses import HTMLResponse
from app_pcm_master_parser import parse_document, get_dashboard

router = APIRouter(prefix="/digital", tags=["Digital PDF Parser"])

@router.post("/parse-document")
@router.post("/parse-document/")
async def digital_parse_document(file: UploadFile = File(...)):
    return await parse_document(file=file)

@router.get("", include_in_schema=False)
@router.get("/", response_class=HTMLResponse)
async def digital_dashboard():
    return await get_dashboard()
