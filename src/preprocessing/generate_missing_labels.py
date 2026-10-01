"""
Genera un dataset YOLO de 2 clases (object=0, missing=1), usando la
heurística geométrica para sintetizar las etiquetas de "missing" a partir
de las anotaciones REALES (no de predicciones de un modelo).

Para cada imagen:
1. Toma las bounding boxes reales de productos (ground truth).
2. Agrupa en filas y detecta huecos entre productos consecutivos
   (misma lógica que gap_detector, aplicada aquí a nivel de anotación).
3. Escribe un .txt YOLO con ambas clases: 0 para cada producto real,
   1 para cada hueco sintetizado.

Uso:
    python generate_missing_labels.py --split train
    python generate_missing_labels.py --split all
"""

import argparse
import logging
import os
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "heuristics"))
from gap_detector import detect_missing_products  # noqa: E402

CSV_COLUMNS = [
    "image_name", "x1", "y1", "x2", "y2", "class", "image_width", "image_height"
]

CLASS_IDS = {"object": 0, "missing": 1}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

KNOWN_CORRUPTED = ["train_4222.jpg", "train_5822.jpg", "train_882.jpg", "train_924.jpg", "test_274.jpg"]


def _to_yolo_line(class_id: int, x1: float, y1: float, x2: float, y2: float,
                   img_width: float, img_height: float) -> str:
    box_width = x2 - x1
    box_height = y2 - y1
    x_center = (x1 + box_width / 2) / img_width
    y_center = (y1 + box_height / 2) / img_height
    width = box_width / img_width
    height = box_height / img_height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def generate_labels_for_image(
    group: pd.DataFrame,
    row_tolerance_ratio: float = 0.6,
    gap_threshold_ratio: float = 1.5,
) -> list[str]:
    """Genera las líneas YOLO (object + missing sintetizado) para una sola imagen."""
    img_width = group.iloc[0]["image_width"]
    img_height = group.iloc[0]["image_height"]

    product_boxes = [
        {"x1": row["x1"], "y1": row["y1"], "x2": row["x2"], "y2": row["y2"]}
        for _, row in group.iterrows()
    ]

    missing_boxes = detect_missing_products(
        product_boxes,
        row_tolerance_ratio=row_tolerance_ratio,
        gap_threshold_ratio=gap_threshold_ratio,
    )

    lines = []
    for box in product_boxes:
        lines.append(_to_yolo_line(CLASS_IDS["object"], box["x1"], box["y1"], box["x2"], box["y2"], img_width, img_height))
    for box in missing_boxes:
        lines.append(_to_yolo_line(CLASS_IDS["missing"], box["x1"], box["y1"], box["x2"], box["y2"], img_width, img_height))

    return lines


def process_split(csv_path: str, output_dir: str, row_tolerance_ratio: float, gap_threshold_ratio: float) -> None:
    logger.info("Cargando anotaciones desde %s", csv_path)
    df = pd.read_csv(csv_path, header=None, names=CSV_COLUMNS)
    df = df[~df["image_name"].isin(KNOWN_CORRUPTED)].reset_index(drop=True)
    logger.info("Cargadas %d filas, %d imágenes únicas", len(df), df["image_name"].nunique())

    os.makedirs(output_dir, exist_ok=True)
    total_missing = 0

    for image_name, group in tqdm(df.groupby("image_name"), desc=f"Generando labels en {output_dir}"):
        lines = generate_labels_for_image(group, row_tolerance_ratio, gap_threshold_ratio)
        total_missing += sum(1 for line in lines if line.startswith(f"{CLASS_IDS['missing']} "))

        label_name = os.path.splitext(image_name)[0] + ".txt"
        with open(os.path.join(output_dir, label_name), "w") as f:
            f.write("\n".join(lines))

    logger.info("Split procesado: %d imágenes, %d huecos sintéticos generados en total",
                df["image_name"].nunique(), total_missing)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera labels YOLO de 2 clases (object + missing sintético).")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--annotations-dir", default="data/annotations")
    parser.add_argument("--output-dir", default="data/yolo_labels_2class")
    parser.add_argument("--row-tolerance-ratio", type=float, default=0.6)
    parser.add_argument("--gap-threshold-ratio", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for split in splits:
        csv_path = os.path.join(args.annotations_dir, f"annotations_{split}.csv")
        output_dir = os.path.join(args.output_dir, split)
        process_split(csv_path, output_dir, args.row_tolerance_ratio, args.gap_threshold_ratio)


if __name__ == "__main__":
    main()