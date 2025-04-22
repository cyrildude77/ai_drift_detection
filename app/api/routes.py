from fastapi import APIRouter, UploadFile
from app.services.drift_service import detect_drift

router = APIRouter()

@router.post("/detect")
async def detect(file: UploadFile):
    content = await file.read()
    drift_info = detect_drift(content)
    return {"drift_detected": drift_info}
