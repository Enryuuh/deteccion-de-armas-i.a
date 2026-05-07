"""
Script 4: Evaluacion del modelo entrenado
==========================================
Calcula mAP@0.5 y mAP@0.5:0.95 sobre el split de test.

Uso:
    python scripts/4_evaluate.py
"""

import logging
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    from ultralytics import YOLO

    weights = Path(cfg["inference"]["weights"])
    if not weights.exists():
        raise FileNotFoundError(f"No se encontraron pesos en {weights}. Entrena primero.")

    data_yaml = Path(cfg["dataset"]["processed_dir"]) / "data.yaml"
    model = YOLO(str(weights))

    logger.info(f"Evaluando {weights} sobre split 'test' de {data_yaml}")
    metrics = model.val(
        data    = str(data_yaml),
        split   = "test",
        imgsz   = cfg["model"]["image_size"],
        device  = cfg["training"].get("device", 0),
        conf    = cfg["inference"]["confidence_threshold"],
        iou     = cfg["inference"]["iou_threshold"],
    )

    logger.info("=== Resultados ===")
    logger.info(f"mAP@0.5      : {metrics.box.map50:.4f}")
    logger.info(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
    logger.info(f"Precision    : {metrics.box.mp:.4f}")
    logger.info(f"Recall       : {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
