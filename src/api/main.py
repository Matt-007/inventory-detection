"""
API de detección de productos e inventario faltante.

Endpoints:
    GET  /                -> UI básica de demo (subir imagen, ver resultado)
    GET  /health           -> chequeo de salud del servicio
    POST /detect           -> recibe una imagen, devuelve productos + huecos detectados en JSON

Ejecutar localmente:
    uvicorn main:app --reload --port 8000
"""

import io
import logging
import os
import sys
from typing import Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "heuristics"))
from gap_detector import detect_missing_products, BBox  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/product_detector_best.pt")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.3"))

app = FastAPI(title="Inventory Detection API", version="1.0.0")

_model = None  # se carga de forma perezosa (lazy) en el primer request o en el startup


def get_model():
    """Carga el modelo YOLO una sola vez (lazy loading) y lo reutiliza en cada request."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No se encontró el modelo en {MODEL_PATH}. "
                "Asegúrate de haber entrenado y copiado best.pt a esa ruta, "
                "o de configurar la variable de entorno MODEL_PATH."
            )
        logger.info("Cargando modelo desde %s", MODEL_PATH)
        _model = YOLO(MODEL_PATH)
    return _model


@app.on_event("startup")
def startup_event() -> None:
    """Intenta precargar el modelo al arrancar. Si falla, el servicio sigue arriba
    pero /detect devolverá un error claro hasta que el modelo esté disponible."""
    try:
        get_model()
        logger.info("Modelo cargado correctamente al iniciar.")
    except FileNotFoundError as e:
        logger.warning("Modelo no disponible al iniciar: %s", e)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": _model is not None, "model_path": MODEL_PATH}


def run_inference(image: Image.Image) -> list[dict[str, Any]]:
    """Corre el modelo YOLO sobre una imagen y devuelve las detecciones como lista de dicts."""
    model = get_model()
    results = model.predict(image, conf=CONFIDENCE_THRESHOLD, verbose=False)

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence": float(box.conf[0]),
        })
    return detections


@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> JSONResponse:
    """Recibe una imagen, devuelve las detecciones de productos y los huecos encontrados."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen enviada.")

    try:
        products = run_inference(image)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    product_boxes: list[BBox] = [
        {"x1": p["x1"], "y1": p["y1"], "x2": p["x2"], "y2": p["y2"]} for p in products
    ]
    missing = detect_missing_products(product_boxes)

    return JSONResponse({
        "image_size": {"width": image.width, "height": image.height},
        "products": products,
        "missing_products": missing,
        "counts": {
            "products_detected": len(products),
            "missing_detected": len(missing),
        },
    })


DEMO_HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Detección de Inventario</title>
    <style>
        body { font-family: sans-serif; max-width: 600px; margin: 40px auto; padding: 0 20px; }
        h1 { font-size: 1.4rem; }
        #result { white-space: pre-wrap; background: #f4f4f4; padding: 12px; border-radius: 6px; margin-top: 16px; }
        button { padding: 8px 16px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>Demo: Detección de productos e inventario faltante</h1>
    <input type="file" id="fileInput" accept="image/*">
    <button onclick="upload()">Analizar imagen</button>
    <div id="result"></div>

    <script>
        async function upload() {
            const fileInput = document.getElementById('fileInput');
            const resultDiv = document.getElementById('result');
            if (!fileInput.files.length) {
                resultDiv.textContent = 'Selecciona una imagen primero.';
                return;
            }
            resultDiv.textContent = 'Procesando...';
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            try {
                const response = await fetch('/detect', { method: 'POST', body: formData });
                const data = await response.json();
                resultDiv.textContent = JSON.stringify(data, null, 2);
            } catch (err) {
                resultDiv.textContent = 'Error: ' + err;
            }
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def demo_ui() -> str:
    return DEMO_HTML