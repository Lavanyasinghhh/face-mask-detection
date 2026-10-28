"""API tests with a fake model (skipped if FastAPI isn't installed)."""
import importlib.util
import unittest

import cv2
import numpy as np

HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None and importlib.util.find_spec("httpx")


@unittest.skipUnless(HAS_FASTAPI, "fastapi/httpx not installed")
class TestAPI(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        import app.main as main
        from app.detector import Detection

        class FakeDetector:
            def detect(self, image):
                return [Detection("without_mask", 0.9, [5, 5, 40, 40])]

        main.detector = FakeDetector()
        self.client = TestClient(main.app)
        self.png = cv2.imencode(".png", np.zeros((64, 64, 3), np.uint8))[1].tobytes()

    def test_health(self):
        self.assertTrue(self.client.get("/health").json()["model_loaded"])

    def test_predict(self):
        r = self.client.post("/predict", files={"file": ("a.png", self.png, "image/png")})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["summary"]["without_mask"], 1)

    def test_predict_image(self):
        r = self.client.post("/predict/image", files={"file": ("a.png", self.png, "image/png")})
        self.assertEqual(r.headers["content-type"], "image/jpeg")

    def test_rejects_non_image(self):
        r = self.client.post("/predict", files={"file": ("a.txt", b"hello", "text/plain")})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
