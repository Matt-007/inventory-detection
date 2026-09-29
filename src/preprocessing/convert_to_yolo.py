"""
Convierte las anotaciones del dataset SKU-110K (formato CSV, coordenadas
absolutas x1,y1,x2,y2) al formato de anotación de YOLO (un .txt por imagen,
coordenadas normalizadas class x_center y_center width height).

Uso:
    python convert_to_yolo.py --split train
    python convert_to_yolo.py --split val
    python convert_to_yolo.py --split test
    python convert_to_yolo.py --split all
"""

import argparse
import logging
import os

import pandas as pd
from tqdm import tqdm

CSV_COLUMNS = [
    "image_name", "x1", "y1", "x2", "y2", "class", "image_width", "image_height"
]

# Solo hay una clase en el dataset original ("object" = producto presente).
CLASS_TO_ID = {"object": 0}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_annotations(csv_path: str) -> pd.DataFrame:
    """Carga un CSV de anotaciones SKU-110K (sin encabezado) y le asigna nombres de columna."""
    logger.info("Cargando anotaciones desde %s", csv_path)
    df = pd.read_csv(csv_path, header=None, names=CSV_COLUMNS)
    logger.info("Cargadas %d filas, %d imágenes únicas", len(df), df["image_name"].nunique())
    return df


def convert_bbox_to_yolo(x1: float, y1: float, x2: float, y2: float,
                          img_width: float, img_height: float) -> tuple[float, float, float, float]:
    """
    Convierte una bounding box de coordenadas absolutas (x1, y1, x2, y2)
    al formato YOLO normalizado (x_center, y_center, width, height),
    todos los valores en el rango [0, 1].
    """
    box_width = x2 - x1
    box_height = y2 - y1
    x_center = x1 + box_width / 2
    y_center = y1 + box_height / 2

    return (
        x_center / img_width,
        y_center / img_height,
        box_width / img_width,
        box_height / img_height,
    )


def remove_corrupted_images(df: pd.DataFrame, corrupted_names: list[str]) -> pd.DataFrame:
    """Elimina del dataframe las filas correspondientes a imágenes corruptas o faltantes."""
    before = df["image_name"].nunique()
    df_clean = df[~df["image_name"].isin(corrupted_names)].reset_index(drop=True)
    after = df_clean["image_name"].nunique()
    if before != after:
        logger.info("Removidas %d imágenes corruptas/faltantes de las anotaciones", before - after)
    return df_clean


def write_yolo_labels(df: pd.DataFrame, output_dir: str) -> None:
    """
    Escribe un archivo .txt por cada imagen en output_dir, con una línea
    por bounding box en formato YOLO: class x_center y_center width height
    """
    os.makedirs(output_dir, exist_ok=True)
    grouped = df.groupby("image_name")

    for image_name, group in tqdm(grouped, desc=f"Escribiendo labels en {output_dir}"):
        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(output_dir, label_name)

        lines = []
        for _, row in group.iterrows():
            class_id = CLASS_TO_ID.get(row["class"], 0)
            x_center, y_center, width, height = convert_bbox_to_yolo(
                row["x1"], row["y1"], row["x2"], row["y2"],
                row["image_width"], row["image_height"],
            )
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    logger.info("Se escribieron %d archivos de labels en %s", len(grouped), output_dir)


def process_split(csv_path: str, output_dir: str, corrupted_names: list[str] | None = None) -> None:
    """Procesa un split completo: carga, limpia y convierte a formato YOLO."""
    df = load_annotations(csv_path)
    if corrupted_names:
        df = remove_corrupted_images(df, corrupted_names)
    write_yolo_labels(df, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convierte anotaciones SKU-110K a formato YOLO.")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--annotations-dir", default="data/annotations")
    parser.add_argument("--output-dir", default="data/yolo_labels")
    return parser.parse_args()


# Las 5 imágenes que encontramos corruptas durante el EDA (bytes truncados en el origen S3).
KNOWN_CORRUPTED = ["train_4222.jpg", "train_5822.jpg", "train_882.jpg", "train_924.jpg", "test_274.jpg"]


def main() -> None:
    args = parse_args()
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for split in splits:
        csv_path = os.path.join(args.annotations_dir, f"annotations_{split}.csv")
        output_dir = os.path.join(args.output_dir, split)
        process_split(csv_path, output_dir, corrupted_names=KNOWN_CORRUPTED)


if __name__ == "__main__":
    main()