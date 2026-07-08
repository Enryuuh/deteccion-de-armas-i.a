"""
Script 12: Fine-tune del modelo v3 con TU dataset propio
==========================================================
Continúa el entrenamiento desde best.pt agregando los frames
que capturaste y etiquetaste con scripts/11.

Espera la siguiente estructura:
    data/finetune_labeled/
        images/train/*.jpg
        labels/train/*.txt    (formato YOLO, mismas 3 clases)
        images/val/*.jpg
        labels/val/*.txt

Uso:
    python scripts/12_finetune.py
"""

import logging
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO          = Path(__file__).resolve().parents[1]
FINETUNE_DIR  = REPO / "data" / "finetune_labeled"


def write_data_yaml() -> Path:
    """Genera data.yaml combinando dataset original + tus frames."""
    out = FINETUNE_DIR / "data.yaml"
    data = {
        "path": str(FINETUNE_DIR.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val":   "images/val",
        "names": {0: "knife", 1: "handgun", 2: "long_gun"},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return out


def main():
    cfg = yaml.safe_load(open(REPO / "config.yaml", encoding="utf-8"))
    weights = REPO / cfg["inference"]["weights"]
    if not weights.exists():
        raise FileNotFoundError(f"No existe {weights}. Entrena primero con scripts/3_train.py")

    if not FINETUNE_DIR.exists() or not (FINETUNE_DIR / "images" / "train").exists():
        raise FileNotFoundError(
            f"No hay dataset etiquetado en {FINETUNE_DIR}.\n"
            "1. Captura frames con scripts/11_capture_finetune_frames.py\n"
            "2. Etiquétalos (Roboflow Annotate o labelImg) con clases knife/handgun/long_gun\n"
            "3. Exporta en formato YOLO a data/finetune_labeled/"
        )

    data_yaml = write_data_yaml()

    from ultralytics import YOLO
    logger.info(f"Pesos base (v3): {weights}")
    logger.info(f"Dataset fine-tune: {data_yaml}")

    model = YOLO(str(weights))
    model.train(
        data       = str(data_yaml),
        epochs     = 50,                # menos epochs porque solo afinamos
        imgsz      = 640,
        batch      = 8,
        workers    = 4,
        device     = 0,
        optimizer  = "AdamW",
        lr0        = 0.0001,            # lr 10x más bajo: NO destruir lo aprendido
        patience   = 15,
        amp        = True,
        cache      = False,
        seed       = 42,
        project    = "models/yolov8_weapons",
        name       = "weapons/yolov8s_v3_finetuned",
        fliplr     = 0.5,
        mosaic     = 0.5,               # menos mosaic en fine-tune
        mixup      = 0.05,
        copy_paste = 0.2,
        degrees    = 15,                # rotación: cuchillos en cualquier ángulo
        hsv_h      = 0.015,
        hsv_s      = 0.7,
        hsv_v      = 0.4,
        exist_ok   = True,
    )

    logger.info("Fine-tune finalizado. best.pt en models/yolov8_weapons/weapons/yolov8s_v3_finetuned/weights/")
    logger.info("Para usarlo, actualiza inference.weights en config.yaml")


if __name__ == "__main__":
    main()
