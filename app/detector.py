"""Wraps the trained YOLOv8 model: run detection and draw boxes."""
from dataclasses import asdict, dataclass

import cv2
import numpy as np

COLOURS = {  # BGR
    "with_mask": (60, 180, 75),
    "without_mask": (40, 40, 220),
    "mask_weared_incorrect": (0, 165, 255),
}
LABELS = {"with_mask": "Mask", "without_mask": "No mask", "mask_weared_incorrect": "Incorrect"}


@dataclass
class Detection:
    label: str
    confidence: float
    box: list[int]  # x1, y1, x2, y2 in pixels

    def to_dict(self):
        return asdict(self)


class MaskDetector:
    def __init__(self, weights: str, conf: float = 0.35):
        from ultralytics import YOLO  # imported here so tests don't need it
        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        result = self.model.predict(image_bgr, conf=self.conf, verbose=False)[0]
        names = result.names
        return [
            Detection(label=names[int(c)], confidence=round(float(s), 3),
                      box=[int(v) for v in xyxy])
            for xyxy, s, c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(),
                                  result.boxes.cls.tolist())
        ]


def summarise(detections: list[Detection]) -> dict:
    counts = {k: 0 for k in COLOURS}
    for d in detections:
        counts[d.label] = counts.get(d.label, 0) + 1
    total = len(detections)
    return {"faces": total, **counts,
            "compliance_rate": round(counts["with_mask"] / total, 3) if total else None}


def draw(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = image_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = d.box
        colour = COLOURS.get(d.label, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        text = f"{LABELS.get(d.label, d.label)} {d.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1, th + 6)
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 4, ty), colour, -1)
        cv2.putText(out, text, (x1 + 2, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out
