"""REST API for the mask detector.

    uvicorn app.main:app --reload
    # then open http://localhost:8000/docs to try it

Endpoints:
  GET  /health            -> is the model loaded?
  POST /predict           -> JSON list of detections + summary
  POST /predict/image     -> the image with boxes drawn on it (JPEG)
"""
import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from app.detector import MaskDetector, draw, summarise

WEIGHTS = os.getenv("MODEL_WEIGHTS", "models/best.pt")
MAX_BYTES = 10 * 1024 * 1024

detector: MaskDetector | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the model once at startup, not on every request.
    global detector
    if os.path.exists(WEIGHTS):
        detector = MaskDetector(WEIGHTS, conf=float(os.getenv("CONF_THRESHOLD", "0.35")))
    yield


app = FastAPI(title="Face Mask Detection API", version="1.0", lifespan=lifespan)


async def read_image(file: UploadFile) -> np.ndarray:
    if detector is None:
        raise HTTPException(503, f"Model not loaded. Put trained weights at {WEIGHTS}.")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image too large (max 10 MB).")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Could not read the file as an image.")
    return image


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": detector is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = await read_image(file)
    detections = detector.detect(image)
    return {"summary": summarise(detections), "detections": [d.to_dict() for d in detections]}


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    image = await read_image(file)
    annotated = draw(image, detector.detect(image))
    ok, buf = cv2.imencode(".jpg", annotated)
    return Response(content=buf.tobytes(), media_type="image/jpeg")
