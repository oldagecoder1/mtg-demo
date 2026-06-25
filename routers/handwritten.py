from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from app_handwritten_formula import recognize, dashboard

router = APIRouter(prefix="/handwritten", tags=["Handwritten Formula Parser"])

@router.post("/recognize")
@router.post("/recognize/")
async def handwritten_recognize(file: UploadFile = File(...), mode: str = Form(default="formula")):
    return await recognize(file=file, mode=mode)

@router.get("", include_in_schema=False)
@router.get("/", response_class=HTMLResponse)
async def handwritten_dashboard():
    return await dashboard()
