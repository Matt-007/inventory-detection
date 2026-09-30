#!/bin/bash
set -e

MODEL_DIR=$(dirname "$MODEL_PATH")
mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_PATH" ]; then
    echo "Modelo ya presente en $MODEL_PATH, no se descarga de nuevo."
elif [ -n "$MODEL_DOWNLOAD_URL" ]; then
    echo "Descargando modelo desde $MODEL_DOWNLOAD_URL ..."
    curl -L -o "$MODEL_PATH" "$MODEL_DOWNLOAD_URL"
    echo "Modelo descargado en $MODEL_PATH"
else
    echo "ADVERTENCIA: no se encontró el modelo en $MODEL_PATH y MODEL_DOWNLOAD_URL está vacío."
    echo "La API arrancará igual, pero /detect devolverá error 503 hasta que el modelo esté disponible."
fi

exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000