"""
Heurística geométrica para detectar "huecos" (productos faltantes) a partir
de las bounding boxes detectadas por el modelo YOLO de productos.

Lógica:
1. Agrupa las detecciones en "filas" según su posición vertical (y_center),
   con una tolerancia flexible (no exige alineación perfecta).
2. Dentro de cada fila, ordena las detecciones de izquierda a derecha.
3. Mide el espacio horizontal entre cajas consecutivas. Si el espacio es
   mucho mayor al ancho típico de una caja en esa fila, se marca como hueco.

Uso:
    from gap_detector import detect_missing_products
    detections = [
        {"x1": 10, "y1": 100, "x2": 60, "y2": 200},
        {"x1": 65, "y1": 105, "x2": 115, "y2": 205},
        # ... más detecciones, con un espacio grande entre alguna
    ]
    gaps = detect_missing_products(detections)
"""

import logging
from typing import TypedDict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class BBox(TypedDict):
    x1: float
    y1: float
    x2: float
    y2: float


def _y_center(box: BBox) -> float:
    return (box["y1"] + box["y2"]) / 2


def _x_center(box: BBox) -> float:
    return (box["x1"] + box["x2"]) / 2


def _height(box: BBox) -> float:
    return box["y2"] - box["y1"]


def _width(box: BBox) -> float:
    return box["x2"] - box["x1"]


def group_into_rows(detections: list[BBox], row_tolerance_ratio: float = 0.6) -> list[list[BBox]]:
    """
    Agrupa las detecciones en filas según su y_center. Dos detecciones
    quedan en la misma fila si la diferencia entre sus y_center es menor
    a row_tolerance_ratio * altura promedio de ambas cajas.

    No asume alineación horizontal perfecta (funciona razonablemente con
    fotos en ligero ángulo), pero sí asume que las filas del estante son
    aproximadamente horizontales en la imagen.
    """
    if not detections:
        return []

    sorted_by_y = sorted(detections, key=_y_center)
    rows: list[list[BBox]] = [[sorted_by_y[0]]]

    for box in sorted_by_y[1:]:
        last_row = rows[-1]
        reference_box = last_row[-1]
        avg_height = (_height(box) + _height(reference_box)) / 2
        tolerance = avg_height * row_tolerance_ratio

        if abs(_y_center(box) - _y_center(reference_box)) <= tolerance:
            last_row.append(box)
        else:
            rows.append([box])

    for row in rows:
        row.sort(key=_x_center)

    logger.info("Detecciones agrupadas en %d filas", len(rows))
    return rows


def detect_gaps_in_row(row: list[BBox], gap_threshold_ratio: float = 1.5) -> list[BBox]:
    """
    Recorre una fila ya ordenada de izquierda a derecha y devuelve una
    bounding box de tipo "hueco" por cada espacio entre dos cajas
    consecutivas que sea mayor a gap_threshold_ratio veces el ancho
    promedio de las cajas de esa fila.
    """
    if len(row) < 2:
        return []

    avg_width = sum(_width(box) for box in row) / len(row)
    gap_threshold = avg_width * gap_threshold_ratio

    gaps: list[BBox] = []
    for i in range(len(row) - 1):
        current_box = row[i]
        next_box = row[i + 1]
        horizontal_gap = next_box["x1"] - current_box["x2"]

        if horizontal_gap > gap_threshold:
            gap_box: BBox = {
                "x1": current_box["x2"],
                "y1": min(current_box["y1"], next_box["y1"]),
                "x2": next_box["x1"],
                "y2": max(current_box["y2"], next_box["y2"]),
            }
            gaps.append(gap_box)

    return gaps


def detect_missing_products(
    detections: list[BBox],
    row_tolerance_ratio: float = 0.6,
    gap_threshold_ratio: float = 1.5,
) -> list[BBox]:
    """
    Función principal: agrupa las detecciones en filas y devuelve la lista
    de bounding boxes de huecos detectados en todas las filas.
    """
    rows = group_into_rows(detections, row_tolerance_ratio=row_tolerance_ratio)

    all_gaps: list[BBox] = []
    for row in rows:
        gaps = detect_gaps_in_row(row, gap_threshold_ratio=gap_threshold_ratio)
        all_gaps.extend(gaps)

    logger.info("Detectados %d huecos en total", len(all_gaps))
    return all_gaps