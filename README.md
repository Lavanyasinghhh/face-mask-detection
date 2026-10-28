# Face mask detection with YOLOv8 and FastAPI

[![tests](https://github.com/Lavanyasinghhh/face-mask-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/Lavanyasinghhh/face-mask-detection/actions/workflows/tests.yml)

A real-time object detector that finds faces in an image and classifies each one as **mask**, **no mask** or **mask worn incorrectly**. The model is fine-tuned YOLOv8 and is served through a FastAPI REST API, packaged with Docker.

I worked on face-mask detection during my ML internship at Audax Labs. This repo is my own end-to-end rebuild on a public dataset: data conversion, training, evaluation on a held-out test set, and deployment as an API.

## How it works

1. **Data conversion** (`src/voc_to_yolo.py`)
   - The dataset's Pascal VOC XML boxes (pixel corners) are converted to YOLO format (normalised centre, width and height).
   - Boxes that extend past the image edge are clipped.
   - The data is split 70/15/15 into train, validation and test sets with a fixed seed.
2. **Training** (`notebooks/train_colab.ipynb`)
   - Transfer learning from YOLOv8n pre-trained on COCO.
   - Trained on a free Colab T4 GPU, with early stopping based on validation mAP.
3. **Serving** (`app/`)
   - A FastAPI service loads the model once at startup.
   - It validates uploads (file size and whether the file is a readable image).
   - It returns either JSON detections with a compliance summary, or the annotated image.
4. **Packaging**
   - The Dockerfile uses CPU-only PyTorch to keep the image small.

## API

```bash
uvicorn app.main:app --reload       # interactive docs at http://localhost:8000/docs
```

```bash
curl -F "file=@group_photo.jpg" http://localhost:8000/predict
```

```json
{
  "summary": {"faces": 3, "with_mask": 2, "without_mask": 1, "mask_weared_incorrect": 0, "compliance_rate": 0.667},
  "detections": [
    {"label": "with_mask", "confidence": 0.91, "box": [112, 40, 168, 102]}
  ]
}
```

| Endpoint | Returns |
|---|---|
| `GET /health` | Whether the model is loaded |
| `POST /predict` | JSON detections plus a summary |
| `POST /predict/image` | JPEG with boxes drawn |

## Run it

```bash
pip install -r requirements.txt
python -m unittest                         # tests use a fake model, no weights needed

# with trained weights in models/best.pt:
uvicorn app.main:app --reload
python scripts/webcam.py                   # live webcam demo

# Docker
docker build -t mask-api .
docker run -p 8000:8000 mask-api
```

To train your own weights, open `notebooks/train_colab.ipynb` in Google Colab and follow the steps.

## Project structure

```
src/voc_to_yolo.py          # dataset conversion + train/val/test split
notebooks/train_colab.ipynb # training and evaluation on Colab GPU
app/detector.py             # model wrapper, summary, box drawing
app/main.py                 # FastAPI service
scripts/webcam.py           # live demo
tests/                      # unit + API tests
Dockerfile
```

## Limitations and next steps

- The dataset is small (853 images) and has relatively few examples of incorrectly worn masks. More data or class-weighted sampling would help that class.
- Small faces in crowded scenes are the hardest case. Training at a higher resolution (`imgsz=960`) is worth trying.
- For a real deployment, I'd add request batching and export the model to ONNX for faster CPU inference.

**Tech:** Python, YOLOv8 (Ultralytics), PyTorch, OpenCV, FastAPI, Docker

**Data:** [Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) (Kaggle, 853 images, 3 classes)
