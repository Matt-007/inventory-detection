"""
Tests unitarios para src/heuristics/gap_detector.py

Uso:
    python -m pytest tests/test_gap_detector.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "heuristics"))

from gap_detector import group_into_rows, detect_gaps_in_row, detect_missing_products


def make_box(x1, y1, x2, y2):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


class TestGroupIntoRows:
    def test_empty_list_returns_empty(self):
        assert group_into_rows([]) == []

    def test_single_detection_returns_one_row(self):
        boxes = [make_box(0, 0, 10, 10)]
        rows = group_into_rows(boxes)
        assert len(rows) == 1
        assert len(rows[0]) == 1

    def test_same_height_boxes_grouped_in_one_row(self):
        """Cajas a la misma altura (mismo y_center) deben quedar en la misma fila."""
        boxes = [
            make_box(0, 100, 50, 200),
            make_box(60, 100, 110, 200),
            make_box(120, 100, 170, 200),
        ]
        rows = group_into_rows(boxes)
        assert len(rows) == 1
        assert len(rows[0]) == 3

    def test_different_heights_create_separate_rows(self):
        """Cajas muy separadas verticalmente deben quedar en filas distintas."""
        boxes = [
            make_box(0, 0, 50, 100),      # fila 1: y_center=50
            make_box(0, 500, 50, 600),    # fila 2: y_center=550, muy lejos
        ]
        rows = group_into_rows(boxes)
        assert len(rows) == 2

    def test_slightly_angled_row_still_grouped(self):
        """Una fila con ligera inclinación (dentro de tolerancia) debe seguir agrupada."""
        boxes = [
            make_box(0, 100, 50, 200),    # y_center=150, height=100
            make_box(60, 110, 110, 210),  # y_center=160, height=100 -> diff=10, tolerance=60
            make_box(120, 120, 170, 220), # y_center=170, height=100 -> diff=10 vs anterior
        ]
        rows = group_into_rows(boxes, row_tolerance_ratio=0.6)
        assert len(rows) == 1

    def test_rows_sorted_left_to_right(self):
        """Dentro de cada fila, las cajas deben quedar ordenadas por posición horizontal."""
        boxes = [
            make_box(200, 100, 250, 200),
            make_box(0, 100, 50, 200),
            make_box(100, 100, 150, 200),
        ]
        rows = group_into_rows(boxes)
        x_centers = [( _b["x1"] + _b["x2"]) / 2 for _b in rows[0]]
        assert x_centers == sorted(x_centers)


class TestDetectGapsInRow:
    def test_empty_row_returns_no_gaps(self):
        assert detect_gaps_in_row([]) == []

    def test_single_box_returns_no_gaps(self):
        row = [make_box(0, 0, 50, 100)]
        assert detect_gaps_in_row(row) == []

    def test_adjacent_boxes_no_gap(self):
        """Cajas pegadas o casi pegadas no deben generar un hueco."""
        row = [make_box(0, 0, 50, 100), make_box(52, 0, 102, 100)]
        gaps = detect_gaps_in_row(row, gap_threshold_ratio=1.5)
        assert len(gaps) == 0

    def test_large_gap_detected(self):
        """Un espacio mucho mayor al ancho promedio debe detectarse como hueco."""
        # Ancho de cada caja: 50. Espacio entre ellas: 200 (4x el ancho) -> debe detectarse
        row = [make_box(0, 0, 50, 100), make_box(250, 0, 300, 100)]
        gaps = detect_gaps_in_row(row, gap_threshold_ratio=1.5)
        assert len(gaps) == 1
        assert gaps[0]["x1"] == 50
        assert gaps[0]["x2"] == 250

    def test_gap_exactly_at_threshold_not_detected(self):
        """Un espacio igual (no mayor) al umbral no debe contar como hueco (comparación estricta >)."""
        # Ancho promedio = 50, threshold_ratio=1.5 -> threshold=75. Espacio = exactamente 75.
        row = [make_box(0, 0, 50, 100), make_box(125, 0, 175, 100)]
        gaps = detect_gaps_in_row(row, gap_threshold_ratio=1.5)
        assert len(gaps) == 0

    def test_multiple_gaps_in_same_row(self):
        row = [
            make_box(0, 0, 50, 100),
            make_box(250, 0, 300, 100),   # gap grande antes de esta
            make_box(310, 0, 360, 100),   # sin gap antes de esta
            make_box(600, 0, 650, 100),   # gap grande antes de esta
        ]
        gaps = detect_gaps_in_row(row, gap_threshold_ratio=1.5)
        assert len(gaps) == 2


class TestDetectMissingProducts:
    def test_no_detections_returns_no_gaps(self):
        assert detect_missing_products([]) == []

    def test_dense_row_no_gaps(self):
        """Productos bien pegados (como en el dataset real) no deben generar falsos huecos."""
        boxes = [make_box(i * 52, 0, i * 52 + 50, 100) for i in range(10)]
        gaps = detect_missing_products(boxes)
        assert len(gaps) == 0

    def test_combines_multiple_rows(self):
        """Debe detectar huecos en varias filas independientemente."""
        row1 = [make_box(0, 0, 50, 100), make_box(250, 0, 300, 100)]
        row2 = [make_box(0, 500, 50, 600), make_box(250, 500, 300, 600)]
        gaps = detect_missing_products(row1 + row2)
        assert len(gaps) == 2