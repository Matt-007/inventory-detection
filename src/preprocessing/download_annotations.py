"""
Descarga solo los CSVs de anotaciones del dataset SKU-110K desde S3.
No descarga las imágenes (esas se descargan aparte, en Colab, cuando
se necesite GPU para entrenar).

Uso:
    python download_annotations.py
"""

import logging
import os

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

BUCKET_NAME = "anyoneai-datasets"
PREFIX = "SKU-110K/SKU110K_fixed/annotations/"
DEST_DIR = "data/annotations"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        config=Config(signature_version="s3v4"),
    )

    os.makedirs(DEST_DIR, exist_ok=True)

    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            filename = os.path.basename(key)
            local_path = os.path.join(DEST_DIR, filename)
            logger.info("Descargando %s -> %s", key, local_path)
            s3_client.download_file(BUCKET_NAME, key, local_path)

    logger.info("Descarga de anotaciones completa.")


if __name__ == "__main__":
    main()