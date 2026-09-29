"""
Organiza las imágenes descargadas (todas juntas en una carpeta) en la
estructura de carpetas que espera YOLO: images/{train,val,test} y
labels/{train,val,test}, usando el prefijo del nombre de archivo
(train_/val_/test_) para decidir a qué split pertenece cada una.

Uso:
    python organize_yolo_dataset.py \
        --images-dir /content/data/raw/images \
        --labels-dir data/yolo_labels \
        --output-dir /content/data/yolo_dataset
"""

import argparse
import logging
import os
import shutil

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SPLIT_PREFIXES = {"train": "train_", "val": "val_", "test": "test_"}


def organize_split(split: str, images_dir: str, labels_dir: str, output_dir: str, use_symlink: bool = True) -> None:
    """Copia (o enlaza) las imágenes y sus labels correspondientes a la estructura final de un split."""
    split_labels_dir = os.path.join(labels_dir, split)
    if not os.path.isdir(split_labels_dir):
        logger.warning("No se encontró la carpeta de labels para '%s': %s", split, split_labels_dir)
        return

    out_images_dir = os.path.join(output_dir, "images", split)
    out_labels_dir = os.path.join(output_dir, "labels", split)
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)

    label_files = [f for f in os.listdir(split_labels_dir) if f.endswith(".txt")]
    missing_images = 0

    for label_file in tqdm(label_files, desc=f"Organizando split '{split}'"):
        image_name = os.path.splitext(label_file)[0] + ".jpg"
        src_image = os.path.join(images_dir, image_name)
        dst_image = os.path.join(out_images_dir, image_name)
        dst_label = os.path.join(out_labels_dir, label_file)

        if not os.path.exists(src_image):
            missing_images += 1
            continue

        if not os.path.exists(dst_image):
            if use_symlink:
                os.symlink(os.path.abspath(src_image), dst_image)
            else:
                shutil.copy2(src_image, dst_image)

        if not os.path.exists(dst_label):
            shutil.copy2(os.path.join(split_labels_dir, label_file), dst_label)

    if missing_images:
        logger.warning("%d imágenes referenciadas en labels no se encontraron en %s", missing_images, images_dir)
    logger.info("Split '%s': %d imágenes organizadas en %s", split, len(label_files) - missing_images, out_images_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Organiza el dataset en estructura YOLO (images/labels por split).")
    parser.add_argument("--images-dir", required=True, help="Carpeta con todas las imágenes descargadas")
    parser.add_argument("--labels-dir", required=True, help="Carpeta con subcarpetas train/val/test de labels .txt")
    parser.add_argument("--output-dir", required=True, help="Carpeta destino con la estructura final")
    parser.add_argument("--copy", action="store_true", help="Copiar imágenes en vez de usar symlinks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split in SPLIT_PREFIXES:
        organize_split(split, args.images_dir, args.labels_dir, args.output_dir, use_symlink=not args.copy)


if __name__ == "__main__":
    main()