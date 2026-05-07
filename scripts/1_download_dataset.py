"""
Script 1: Descarga del Dataset
================================
Fuente: Open Images v7 (Google) - CC BY 4.0
Clases: Knife (arma blanca), Firearm/Handgun/Shotgun (armas de fuego)

Uso:
    python scripts/1_download_dataset.py
"""

import logging
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_open_images(cfg: dict):
    try:
        import fiftyone.zoo as foz
    except ImportError:
        logger.error("Instala fiftyone: pip install fiftyone")
        raise

    raw_dir  = Path(cfg["dataset"]["raw_dir"])
    classes  = cfg["dataset"]["open_images_classes"]
    max_samp = cfg["dataset"]["max_samples_per_class"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "validation", "test"]

    for split in splits:
        logger.info(f"==> Descargando split: {split} | clases: {classes}")
        try:
            ds = foz.load_zoo_dataset(
                "open-images-v7",
                split=split,
                label_types=["detections"],
                classes=classes,
                max_samples=max_samp,
                dataset_dir=str(raw_dir / split),
                dataset_name=f"weapons_{split}_{max_samp}",
            )
            logger.info(f"   Split '{split}': {len(ds)} imagenes descargadas")
        except Exception as e:
            logger.warning(f"   No se pudo descargar split '{split}': {e}")

    logger.info("Descarga completada en data/raw/")


def main():
    cfg = load_config()
    logger.info("=== Descarga Open Images v7 - Armas blancas y de fuego ===")
    logger.info(f"Clases objetivo: {cfg['dataset']['open_images_classes']}")
    logger.info(f"Max muestras por clase: {cfg['dataset']['max_samples_per_class']}")
    download_open_images(cfg)


if __name__ == "__main__":
    main()
