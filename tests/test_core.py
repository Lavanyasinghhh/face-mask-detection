"""Run:  python -m unittest
These tests don't need the trained model or the dataset."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.detector import Detection, draw, summarise
from src.voc_to_yolo import convert, voc_to_yolo_lines

XML = """<annotation><size><width>400</width><height>200</height></size>
<object><name>with_mask</name><bndbox><xmin>100</xmin><ymin>50</ymin><xmax>200</xmax><ymax>150</ymax></bndbox></object>
<object><name>without_mask</name><bndbox><xmin>350</xmin><ymin>-5</ymin><xmax>420</xmax><ymax>60</ymax></bndbox></object>
</annotation>"""


class TestConversion(unittest.TestCase):
    def test_box_maths(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.xml"
            p.write_text(XML)
            lines = voc_to_yolo_lines(p)
        self.assertEqual(lines[0], "0 0.375000 0.500000 0.250000 0.500000")
        cls, xc, yc, w, h = lines[1].split()
        self.assertEqual(cls, "1")
        self.assertAlmostEqual(float(xc) + float(w) / 2, 1.0)  # clipped to the right edge
        self.assertAlmostEqual(float(yc) - float(h) / 2, 0.0)  # clipped to the top edge

    def test_split(self):
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d) / "raw", Path(d) / "yolo"
            (src / "images").mkdir(parents=True)
            (src / "annotations").mkdir()
            for i in range(20):
                (src / "annotations" / f"img{i}.xml").write_text(XML)
                (src / "images" / f"img{i}.png").write_bytes(b"x")
            counts = convert(src, out)
            self.assertEqual(counts, {"test": 3, "val": 3, "train": 14})
            self.assertTrue((out / "data.yaml").exists())
            self.assertEqual(len(list((out / "labels" / "train").glob("*.txt"))), 14)


class TestDetectorHelpers(unittest.TestCase):
    dets = [Detection("with_mask", 0.9, [10, 10, 50, 50]),
            Detection("without_mask", 0.8, [60, 10, 100, 50]),
            Detection("with_mask", 0.7, [110, 10, 150, 50])]

    def test_summary(self):
        s = summarise(self.dets)
        self.assertEqual(s["faces"], 3)
        self.assertEqual(s["without_mask"], 1)
        self.assertAlmostEqual(s["compliance_rate"], 0.667)
        self.assertIsNone(summarise([])["compliance_rate"])

    def test_draw_keeps_shape_and_changes_pixels(self):
        img = np.zeros((100, 200, 3), np.uint8)
        out = draw(img, self.dets)
        self.assertEqual(out.shape, img.shape)
        self.assertGreater(out.sum(), 0)
        self.assertEqual(img.sum(), 0)  # original untouched


if __name__ == "__main__":
    unittest.main()
