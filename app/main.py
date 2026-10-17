from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from app.inference import PainterModel
from app.labels import ARTISTS

predictor: PainterModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = PainterModel()
    yield
    predictor = None


app = FastAPI(
    title="Famous paintings classifier",
    description="ResNet-152 that predicts which of 25 painters made an image.",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {
        "status": "ok",
        "device": str(predictor.device),
        "checkpoint": str(predictor.checkpoint),
        "classes": len(ARTISTS),
    }


@app.get("/labels")
def labels():
    return [{"artist_id": index, "artist": name} for index, name in enumerate(ARTISTS)]


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=len(ARTISTS)),
):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty file.")
    try:
        best, top = predictor.predict_bytes(payload, top_k=top_k)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Could not read the image: {error}") from error
    return {"prediction": best, "top": top}
