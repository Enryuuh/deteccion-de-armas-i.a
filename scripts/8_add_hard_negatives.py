"""
Script 8: Agregar Hard Negatives al dataset
============================================
Toma los frames guardados en logs/frames/ (falsos positivos capturados
durante inferencia) y los agrega al split de entrenamiento como
imágenes de fondo (label vacío = ningún objeto).

YOLO aprende de estos ejemplos que esos objetos NO son armas.

Uso:
    python scripts/8_add_hard_negatives.py
"""

import logging
import shutil
import yaml
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(REPO / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def img_ok(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def main():
    cfg = load_config()
    frames_dir = REPO / cfg["alerts"]["frames_dir"]
    processed  = REPO / Path(cfg["dataset"]["processed_dir"])

    img_out = processed / "images" / "train"
    lbl_out = processed / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    frames = sorted([
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])

    if not frames:
        logger.warning(f"No hay frames en {frames_dir}. Ejecuta la cámara primero.")
        return

    logger.info(f"Encontrados {len(frames)} frames de falsos positivos en {frames_dir}")

    # Prefijo único para no colisionar con archivos existentes
    existing = sorted(img_out.glob("hn_*.jpg"))
    start_idx = len(existing) + 1

    added = 0
    for i, src in enumerate(frames, start=start_idx):
        if not img_ok(src):
            logger.warning(f"  Imagen inválida, saltando: {src.name}")
            continue

        dst_img = img_out / f"hn_{i:06d}.jpg"
        dst_lbl = lbl_out / f"hn_{i:06d}.txt"

        shutil.copy2(src, dst_img)
        dst_lbl.write_text("", encoding="utf-8")   # etiqueta vacía = fondo
        added += 1

    logger.info(f"Agregados {added} hard negatives a {img_out}")
    logger.info("Ejecuta scripts/3_train.py para reentrenar con estos negativos.")


if __name__ == "__main__":
    main()
