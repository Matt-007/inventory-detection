"""
Tests unitarios para src/preprocessing/convert_to_yolo.py

Uso:
    pytest tests/test_convert_to_yolo.py -v
"""

import os
import sys

import pandas as pd
import pytest

# Permite importar el módulo desde src/preprocessing sin instalar el paquete
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "preprocessing"))

from convert_to_yolo import (
    convert_bbox_to_yolo,
    remove_corrupted_images,
    write_yolo_labels,
    CLASS_TO_ID,
)


class TestConvertBboxToYolo:
    def test_centered_box_in_centered_image(self):
        """Una caja que ocupa el centro exacto de la imagen debe dar x_center=y_center=0.5."""
        x_center, y_center, width, height = convert_bbox_to_yolo(
            x1=25, y1=25, x2=75, y2=75, img_width=100, img_height=100
        )
        assert x_center == pytest.approx(0.5)
        assert y_center == pytest.approx(0.5)
        assert width == pytest.approx(0.5)
        assert height == pytest.approx(0.5)

    def test_box_covering_full_image(self):
        """Una caja que cubre toda la imagen debe dar width=height=1.0."""
        x_center, y_center, width, height = convert_bbox_to_yolo(
            x1=0, y1=0, x2=200, y2=100, img_width=200, img_height=100
        )
        assert x_center == pytest.approx(0.5)
        assert y_center == pytest.approx(0.5)
        assert width == pytest.approx(1.0)
        assert height == pytest.approx(1.0)

    def test_box_in_top_left_corner(self):
        """Una caja pegada a la esquina superior izquierda no debe dar coordenadas negativas."""
        x_center, y_center, width, height = convert_bbox_to_yolo(
            x1=0, y1=0, x2=10, y2=10, img_width=100, img_height=100
        )
        assert x_center > 0
        assert y_center > 0
        assert width == pytest.approx(0.1)
        assert height == pytest.approx(0.1)

    def test_all_values_in_valid_range(self):
        """Todos los valores de salida deben estar entre 0 y 1 para una caja válida dentro de la imagen."""
        result = convert_bbox_to_yolo(x1=50, y1=100, x2=300, y2=400, img_width=1000, img_height=1000)
        for value in result:
            assert 0.0 <= value <= 1.0

    def test_non_square_image(self):
        """Debe manejar correctamente imágenes no cuadradas (ancho != alto)."""
        x_center, y_center, width, height = convert_bbox_to_yolo(
            x1=0, y1=0, x2=100, y2=50, img_width=200, img_height=400
        )
        assert width == pytest.approx(0.5)
        assert height == pytest.approx(0.125)


class TestRemoveCorruptedImages:
    def test_removes_specified_images(self):
        df = pd.DataFrame({
            "image_name": ["a.jpg", "b.jpg", "b.jpg", "c.jpg"],
            "x1": [1, 2, 3, 4],
        })
        result = remove_corrupted_images(df, corrupted_names=["b.jpg"])
        assert "b.jpg" not in result["image_name"].values
        assert set(result["image_name"].unique()) == {"a.jpg", "c.jpg"}

    def test_no_corrupted_images_returns_unchanged(self):
        df = pd.DataFrame({"image_name": ["a.jpg", "b.jpg"], "x1": [1, 2]})
        result = remove_corrupted_images(df, corrupted_names=["z.jpg"])
        assert len(result) == len(df)

    def test_empty_corrupted_list(self):
        df = pd.DataFrame({"image_name": ["a.jpg", "b.jpg"], "x1": [1, 2]})
        result = remove_corrupted_images(df, corrupted_names=[])
        assert len(result) == len(df)

    def test_removes_all_rows_of_corrupted_image_not_just_first(self):
        """Una imagen corrupta puede tener múltiples bounding boxes (múltiples filas)."""
        df = pd.DataFrame({
            "image_name": ["a.jpg", "a.jpg", "a.jpg", "b.jpg"],
            "x1": [1, 2, 3, 4],
        })
        result = remove_corrupted_images(df, corrupted_names=["a.jpg"])
        assert len(result) == 1
        assert result.iloc[0]["image_name"] == "b.jpg"


class TestWriteYoloLabels:
    def test_creates_one_file_per_image(self, tmp_path):
        df = pd.DataFrame({
            "image_name": ["img1.jpg", "img1.jpg", "img2.jpg"],
            "x1": [10, 50, 5], "y1": [10, 50, 5],
            "x2": [30, 70, 25], "y2": [30, 70, 25],
            "class": ["object", "object", "object"],
            "image_width": [100, 100, 100],
            "image_height": [100, 100, 100],
        })
        output_dir = tmp_path / "labels"
        write_yolo_labels(df, str(output_dir))

        assert (output_dir / "img1.txt").exists()
        assert (output_dir / "img2.txt").exists()

    def test_label_file_has_correct_number_of_lines(self, tmp_path):
        df = pd.DataFrame({
            "image_name": ["img1.jpg", "img1.jpg", "img1.jpg"],
            "x1": [10, 50, 20], "y1": [10, 50, 20],
            "x2": [30, 70, 40], "y2": [30, 70, 40],
            "class": ["object", "object", "object"],
            "image_width": [100, 100, 100],
            "image_height": [100, 100, 100],
        })
        output_dir = tmp_path / "labels"
        write_yolo_labels(df, str(output_dir))

        with open(output_dir / "img1.txt") as f:
            lines = f.readlines()
        assert len(lines) == 3

    def test_label_line_format(self, tmp_path):
        """Cada línea debe tener 5 valores: class x_center y_center width height."""
        df = pd.DataFrame({
            "image_name": ["img1.jpg"],
            "x1": [10], "y1": [10], "x2": [30], "y2": [30],
            "class": ["object"],
            "image_width": [100], "image_height": [100],
        })
        output_dir = tmp_path / "labels"
        write_yolo_labels(df, str(output_dir))

        with open(output_dir / "img1.txt") as f:
            line = f.readline().strip()
        parts = line.split(" ")
        assert len(parts) == 5
        assert parts[0] == str(CLASS_TO_ID["object"])
        # Los 4 valores restantes deben ser parseables como float
        for value in parts[1:]:
            float(value)