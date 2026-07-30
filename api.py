from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse

from models.pipeline import predict as pipeline_predict
from exceptions import AudioProcessingError

ROOT_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Speech Sentiment API",
    version="1.0"
)

@app.get("/")
def home():
    return {"message": "API operationnelle"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filename = Path(file.filename).name
    if not filename.lower().endswith((".wav", ".mp3")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="unsupported format"
        )

    upload_path = UPLOAD_DIR / filename
    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"unable to read upload: {exc}"
        ) from exc

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="empty file"
        )

    try:
        upload_path.write_bytes(content)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"unable to save upload: {exc}"
        ) from exc

    if not upload_path.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="uploaded file is missing after save"
        )

    try:
        result = pipeline_predict(str(upload_path))
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        ) from exc
    except AudioProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"prediction failed: {exc}"
        ) from exc

    return JSONResponse(content=result)
