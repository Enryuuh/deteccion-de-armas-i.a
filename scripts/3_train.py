"""
Script 3: Entrenamiento YOLOv8
================================
Entrena en GPU (RTX 5050) con FP16 y configuracion equilibrada
para no saturar la tarjeta. Lee toda la config de config.yaml.

Uso:
    python scripts/3_train.py
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

    data_yaml = Path(cfg["dataset"]["processed_dir"]) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"No existe {data_yaml}. Ejecuta primero scripts/2_prepare_dataset.py"
        )

    t = cfg["training"]
    m = cfg["model"]

    logger.info(f"Modelo base: {m['name']}")
    logger.info(f"Dataset: {data_yaml}")
    logger.info(f"Device: {t.get('device', 0)} | Batch: {t['batch']} | Epochs: {t['epochs']}")

    model = YOLO(m["name"])
    model.train(
        data       = str(data_yaml),
        epochs     = t["epochs"],
        imgsz      = m["image_size"],
        batch      = t["batch"],
        workers    = t["workers"],
        device     = t.get("device", 0),
        optimizer  = t.get("optimizer", "AdamW"),
        lr0        = t.get("lr0", 0.001),
        patience   = t.get("patience", 20),
        amp        = t.get("amp", True),
        cache      = t.get("cache", False),
        seed       = t.get("seed", 42),
        project    = t["output_dir"],
        name       = f"{t['project_name']}/{t['run_name']}",
        fliplr      = t.get("fliplr", 0.5),
        flipud      = t.get("flipud", 0.0),
        mosaic      = t.get("mosaic", 1.0),
        mixup       = t.get("mixup", 0.1),
        copy_paste  = t.get("copy_paste", 0.0),
        degrees     = t.get("degrees", 0.0),
        translate   = t.get("translate", 0.1),
        scale       = t.get("scale", 0.5),
        shear       = t.get("shear", 0.0),
        perspective = t.get("perspective", 0.0),
        hsv_h       = t.get("hsv_h", 0.015),
        hsv_s       = t.get("hsv_s", 0.7),
        hsv_v       = t.get("hsv_v", 0.4),
        exist_ok    = True,
    )

    logger.info("Entrenamiento finalizado. Pesos: best.pt en el output_dir.")


if __name__ == "__main__":
    main()
