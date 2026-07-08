"""
Script 9: Descargar negativos extra de Open Images v7 (sin fiftyone)
=====================================================================
Parsea los CSVs de OIv7 ya descargados por fiftyone directamente con
pandas, selecciona imágenes de objetos cotidianos SIN armas, y las
descarga vía HTTP. Evita el crash de MongoDB de fiftyone en Windows.

Uso:
    python scripts/9_download_extra_negatives.py
"""

import logging
import shutil
import yaml
import concurrent.futures
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from io import BytesIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]

# CSVs ya descargados por fiftyone
OI_BASE      = Path.home() / "fiftyone" / "open-images-v7"
VAL_META     = OI_BASE / "validation" / "metadata"
VAL_LABELS   = OI_BASE / "validation" / "labels"

NEGATIVE_DISPLAY_NAMES = {
    "Chair", "Remote control", "Mobile phone", "Bottle",
    "Book", "Pen", "Screwdriver",
    "Bicycle", "Umbrella",
    # ELIMINADO: "Kitchen knife" — confundía al modelo (un knife real ES un weapon)
}

WEAPON_DISPLAY_NAMES = {
    "Knife", "Handgun", "Shotgun", "Rifle", "Gun", "Firearm", "Weapon",
}

MAX_NEGATIVES  = 2000
DOWNLOAD_WORKERS = 8


def load_config() -> dict:
    with open(REPO / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_class_map(classes_csv: Path) -> dict:
    """Devuelve {label_code -> display_name} desde classes.csv de OIv7."""
    df = pd.read_csv(classes_csv, header=None, names=["code", "name"])
    return dict(zip(df["code"], df["name"]))


def select_negative_images(detections_csv: Path, code2name: dict) -> set:
    """
    Lee detections.csv y devuelve IDs de imágenes que:
      - Contienen al menos una clase negativa
      - NO contienen ninguna clase de arma
    """
    neg_codes    = {c for c, n in code2name.items() if n in NEGATIVE_DISPLAY_NAMES}
    weapon_codes = {c for c, n in code2name.items() if n in WEAPON_DISPLAY_NAMES}

    logger.info(f"Leyendo anotaciones: {detections_csv} ({detections_csv.stat().st_size // 1_000_000} MB)...")
    df = pd.read_csv(detections_csv, usecols=["ImageID", "LabelName"])

    has_neg    = set(df[df["LabelName"].isin(neg_codes)]["ImageID"])
    has_weapon = set(df[df["LabelName"].isin(weapon_codes)]["ImageID"])

    clean = has_neg - has_weapon
    logger.info(f"  Con objeto negativo: {len(has_neg)} imgs | Con arma: {len(has_weapon)} imgs | Limpias: {len(clean)} imgs")
    return clean


def load_image_urls(image_ids_csv: Path, valid_ids: set, max_n: int) -> list[tuple]:
    """Devuelve lista de (image_id, url) para los IDs válidos."""
    logger.info(f"Leyendo URLs de imágenes ({image_ids_csv.stat().st_size // 1_000_000} MB)...")
    df = pd.read_csv(image_ids_csv, usecols=["ImageID", "OriginalURL", "Thumbnail300KURL"])
    df = df[df["ImageID"].isin(valid_ids)].head(max_n)
    # Prefiere thumbnail (más rápido), cae a original si no hay
    df["url"] = df["Thumbnail300KURL"].fillna(df["OriginalURL"])
    return list(zip(df["ImageID"], df["url"]))


def download_one(args) -> bool:
    img_id, url, dst_img, dst_lbl = args
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.save(str(dst_img), "JPEG", quality=90)
        dst_lbl.write_text("", encoding="utf-8")
        return True
    except Exception:
        return False


def main():
    cfg        = load_config()
    processed  = REPO / Path(cfg["dataset"]["processed_dir"])
    img_out    = processed / "images" / "train"
    lbl_out    = processed / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    classes_csv    = VAL_META   / "classes.csv"
    detections_csv = VAL_LABELS / "detections.csv"
    image_ids_csv  = VAL_META   / "image_ids.csv"

    for p in [classes_csv, detections_csv, image_ids_csv]:
        if not p.exists():
            raise FileNotFoundError(f"CSV no encontrado: {p}\nEjecuta primero el script 1 para descargar los metadatos.")

    code2name   = load_class_map(classes_csv)
    valid_ids   = select_negative_images(detections_csv, code2name)
    image_urls  = load_image_urls(image_ids_csv, valid_ids, MAX_NEGATIVES)

    if not image_urls:
        logger.warning("No se encontraron imágenes válidas.")
        return

    logger.info(f"Descargando {len(image_urls)} imágenes en paralelo ({DOWNLOAD_WORKERS} workers)...")

    existing = sorted(img_out.glob("neg_*.jpg"))
    start_idx = len(existing) + 1

    tasks = []
    for i, (img_id, url) in enumerate(image_urls, start=start_idx):
        dst_img = img_out / f"neg_{i:06d}.jpg"
        dst_lbl = lbl_out / f"neg_{i:06d}.txt"
        tasks.append((img_id, url, dst_img, dst_lbl))

    added = 0
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
        for j, ok in enumerate(ex.map(download_one, tasks), 1):
            if ok:
                added += 1
            else:
                failed += 1
            if j % 100 == 0:
                logger.info(f"  Progreso: {j}/{len(tasks)} | OK: {added} | Fallidos: {failed}")

    logger.info(f"Agregados {added} negativos extra al dataset de entrenamiento ({failed} fallidos)")
    logger.info("Ejecuta scripts/3_train.py para reentrenar.")


if __name__ == "__main__":
    main()
