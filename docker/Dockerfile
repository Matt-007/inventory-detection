# Imagen base liviana con Python
FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema necesarias para OpenCV/Pillow/Ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY docker/entrypoint.sh .
RUN chmod +x entrypoint.sh

# Ruta donde el entrypoint dejará el modelo descargado (o donde ya viva si se copió al build)
ENV MODEL_PATH=/app/models/product_detector_best.pt
# URL desde donde descargar el modelo al arrancar, si no está ya presente (configurable en runtime)
ENV MODEL_DOWNLOAD_URL=""

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]