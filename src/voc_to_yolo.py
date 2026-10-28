"""Convert the Kaggle face-mask dataset (Pascal VOC XML) to YOLO format and
split it into train / val / test.

Dataset: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
  images/maksssksksss0.png ...
  annotations/maksssksksss0.xml ...

YOLO format: one .txt per image, one line per box:
  <class_id> <x_center> <y_center> <width> <height>   (all scaled to 0-1)

Usage:
  python -m src.voc_to_yolo --src data/raw --out data/yolo
"""
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]


def voc_to_yolo_lines(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    w = float(root.find("size/width").text)
    h = float(root.find("size/height").text)
    lines = []
    for obj in root.iter("object"):
        name = obj.find("name").text.strip()
        if name not in CLASSES:
            continue
        box = obj.find("bndbox")
        x1, y1, x2, y2 = (float(box.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        # Clip to the image, since a few annotations go slightly outside it.
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{CLASSES.index(name)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def convert(src: Path, out: Path, val: float = 0.15, test: float = 0.15, seed: int = 42) -> dict:
    xmls = sorted((src / "annotations").glob("*.xml"))
    random.Random(seed).shuffle(xmls)
    n_test, n_val = int(len(xmls) * test), int(len(xmls) * val)
    splits = {"test": xmls[:n_test], "val": xmls[n_test:n_test + n_val],
              "train": xmls[n_test + n_val:]}
    counts = {}
    for split, files in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for xml in files:
            img = next((src / "images").glob(xml.stem + ".*"))
            shutil.copy(img, out / "images" / split / img.name)
            (out / "labels" / split / f"{xml.stem}.txt").write_text("\n".join(voc_to_yolo_lines(xml)))
        counts[split] = len(files)

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    return counts


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("data/yolo"))
    a = p.parse_args()
    print(convert(a.src, a.out))
