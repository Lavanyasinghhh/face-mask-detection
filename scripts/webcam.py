"""Live webcam demo:  python scripts/webcam.py  (press q to quit)"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.detector import MaskDetector, draw, summarise  # noqa: E402

detector = MaskDetector("models/best.pt")
cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    dets = detector.detect(frame)
    out = draw(frame, dets)
    s = summarise(dets)
    cv2.putText(out, f"Faces: {s['faces']}  No mask: {s['without_mask']}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Mask detection", out)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
