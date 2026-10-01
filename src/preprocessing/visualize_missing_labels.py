"""
Visualiza las etiquetas generadas (object + missing) para una imagen
específica, dibujando solo las bounding boxes sobre un lienzo en blanco
(no requiere tener la imagen real descargada).

Uso:
    python visualize_missing_labels.py --image-name train_1005.jpg --split train
"""

import argparse
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd

CSV_COLUMNS = ["image_name", "x1", "y1", "x2", "y2", "class", "image_width", "image_height"]


def plot_labels(image_name: str, split: str, annotations_dir: str, labels_dir: str) -> None:
    csv_path = os.path.join(annotations_dir, f"annotations_{split}.csv")
    df = pd.read_csv(csv_path, header=None, names=CSV_COLUMNS)
    row_sample = df[df["image_name"] == image_name].iloc[0]
    img_width, img_height = row_sample["image_width"], row_sample["image_height"]

    label_path = os.path.join(labels_dir, split, os.path.splitext(image_name)[0] + ".txt")
    with open(label_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)  # invertido porque y crece hacia abajo en imágenes
    ax.set_facecolor("#f0f0f0")

    n_object, n_missing = 0, 0
    for line in lines:
        class_id, xc, yc, w, h = map(float, line.split())
        x1 = (xc - w / 2) * img_width
        y1 = (yc - h / 2) * img_height
        box_w = w * img_width
        box_h = h * img_height

        if int(class_id) == 0:
            color, label = "green", "object"
            n_object += 1
        else:
            color, label = "red", "missing"
            n_missing += 1

        rect = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1.5, edgecolor=color, facecolor="none")
        ax.add_patch(rect)

    ax.set_title(f"{image_name} — {n_object} productos (verde), {n_missing} huecos sintéticos (rojo)")
    plt.savefig("label_preview.png", dpi=120, bbox_inches="tight")
    print(f"Guardado en label_preview.png — {n_object} object, {n_missing} missing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-name", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--annotations-dir", default="data/annotations")
    parser.add_argument("--labels-dir", default="data/yolo_labels_2class")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_labels(args.image_name, args.split, args.annotations_dir, args.labels_dir)