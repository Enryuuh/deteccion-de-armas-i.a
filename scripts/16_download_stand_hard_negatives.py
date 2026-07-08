"""
Script 16: Hard negatives específicos para escenario STAND
============================================================
Para que el modelo NO alucine armas en presencia de personas, manos,
teléfonos, llaves, objetos cotidianos. Descarga imágenes SIN armas
de objetos visualmente confundibles y los agrega como background
(label vacío) al training set.

Categorías target (todas SIN armas):
  - Personas (Person) en interiores/exteriores
  - Manos sosteniendo objetos pequeños y oscuros
  - Teléfonos móviles (Mobile phone)
  - Mandos (Remote control) — silueta parecida a pistola
  - Llaves, cartera, monedero
  - Herramientas: destornillador, taladro
  - Cámaras y objetos de mano

Fuentes:
  1. Open Images v7 (cache existente en ~/fiftyone/open-images-v7/)
  2. URLs HTTP directas (LOCATIONS de OIv7)

Salida:
  - imágenes en data/processed/images/train/
  - .txt vacío en data/processed/labels/train/ (background)

Uso:
    python scripts/16_download_stand_hard_negatives.py
"""

import logging
import shutil
from pathlib import Path
import concurrent.futures

import pandas as pd
import requests
from PIL import Image
from io import BytesIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]

OI_BASE     = Path.home() / "fiftyone" / "open-images-v7"
VAL_META    = OI_BASE / "validation" / "metadata"
VAL_LABELS  = OI_BASE / "validation" / "labels"
TRAIN_META  = OI_BASE / "train" / "metadata"
TRAIN_LABELS= OI_BASE / "train" / "labels"

# Clases de OIv7 que son hard negatives PERFECTOS para escenario stand
# (visualmente confundibles con pistolas o presentes en stands)
STAND_NEGATIVE_CLASSES = {
    "Person",            # ~muy importante: gente en el stand
    "Mobile phone",      # objeto frecuente en manos
    "Remote control",    # silueta tipo pistola
    "Telephone",
    "Camera",            # cámaras de fotos
    "Microphone",        # forma tubular
    "Screwdriver",       # herramienta de mano
    "Drill (Tool)",
    "Hair dryer",        # pistola-like
    "Stapler",
    "Tablet computer",
    "Laptop",
    "Calculator",
    "Headphones",
    "Wallet",
    "Watch",
}

# Clases que NO deben estar en la imagen (armas)
WEAPON_CLASSES = {
    "Knife", "Handgun", "Shotgun", "Rifle", "Gun", "Firearm",
    "Weapon", "Pistol", "Submachine gun", "Tank",
}

MAX_NEGATIVES = 2500
DOWNLOAD_WORKERS = 8


def load_class_map(classes_csv):
    df = pd.read_csv(classes_csv, header=None, names=["code", "name"])
    return dict(zip(df["code"], df["name"]))


def select_images_with_neg_classes(detections_csv, code2name, neg_set, weapon_set):
    """Devuelve set de ImageIDs que contienen al menos 1 clase negativa
    y NINGUNA clase de arma."""
    log.info(f"  Leyendo {detections_csv.name} ...")
    df = pd.read_csv(detections_csv, usecols=["ImageID", "LabelName"])
    df["DisplayName"] = df["LabelName"].map(code2name)
    df = df.dropna(subset=["DisplayName"])

    # Imagenes con armas (a excluir)
    weapon_imgs = set(df.loc[df["DisplayName"].isin(weapon_set), "ImageID"].unique())
    # Imagenes con clase negativa target
    neg_imgs = set(df.loc[df["DisplayName"].isin(neg_set), "ImageID"].unique())
    # Negativos limpios
    clean = neg_imgs - weapon_imgs
    log.info(f"  Imgs con clase neg: {len(neg_imgs)} | con arma: {len(weapon_imgs)} | limpios: {len(clean)}")
    return clean


def get_image_urls(image_ids_csv, valid_ids):
    """Lee CSV de image_ids con URLs y filtra por los IDs válidos."""
    df = pd.read_csv(image_ids_csv, usecols=["ImageID", "OriginalURL"])
    df = df[df["ImageID"].isin(valid_ids)]
    return dict(zip(df["ImageID"], df["OriginalURL"]))


def download_one(img_id, url, img_out, lbl_out, prefix):
    dst_img = img_out / f"{prefix}{img_id}.jpg"
    dst_lbl = lbl_out / f"{prefix}{img_id}.txt"
    if dst_img.exists() and dst_lbl.exists():
        return True
    try:
        r = requests.get(url, timeout=15, stream=True)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        if min(img.size) < 200:
            return False
        img.save(dst_img, "JPEG", quality=85)
        dst_lbl.write_text("", encoding="utf-8")  # background = label vacío
        return True
    except Exception:
        return False


def download_negatives_from_split(split_meta, split_labels, prefix, img_out, lbl_out, code2name, target):
    """Descarga negativos de un split (validation o train) de OIv7."""
    detections_csv = split_labels / "detections.csv"
    image_ids_csv = split_meta / "image_ids.csv"
    if not detections_csv.exists() or not image_ids_csv.exists():
        log.warning(f"  faltan CSVs en {split_meta} / {split_labels}")
        return 0

    clean_ids = select_images_with_neg_classes(detections_csv, code2name,
                                                STAND_NEGATIVE_CLASSES, WEAPON_CLASSES)
    if not clean_ids:
        return 0
    urls = get_image_urls(image_ids_csv, clean_ids)
    log.info(f"  {len(urls)} URLs disponibles, descargando max {target}...")

    items = list(urls.items())[:target]
    ok = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
        futs = [ex.submit(download_one, iid, url, img_out, lbl_out, prefix)
                for iid, url in items]
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            if fut.result():
                ok += 1
            if i % 100 == 0:
                log.info(f"  progreso: {i}/{len(items)} (ok={ok})")
    return ok


def main():
    processed = REPO / "data" / "processed"
    img_out = processed / "images" / "train"
    lbl_out = processed / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    if not OI_BASE.exists():
        log.error(f"No existe cache Open Images en {OI_BASE}")
        log.error("Ejecuta scripts/1_download_dataset.py primero (descarga OIv7 via fiftyone)")
        return

    classes_csv = VAL_META / "classes.csv"
    if not classes_csv.exists():
        log.error(f"Falta {classes_csv}")
        return
    code2name = load_class_map(classes_csv)

    total = 0
    # Primero validation (más rápido)
    log.info("=== Split: VALIDATION ===")
    total += download_negatives_from_split(
        VAL_META, VAL_LABELS, "stand_neg_val_",
        img_out, lbl_out, code2name, MAX_NEGATIVES // 2
    )

    # Después train si hay
    if TRAIN_META.exists() and TRAIN_LABELS.exists():
        log.info("=== Split: TRAIN ===")
        remaining = MAX_NEGATIVES - total
        if remaining > 0:
            total += download_negatives_from_split(
                TRAIN_META, TRAIN_LABELS, "stand_neg_train_",
                img_out, lbl_out, code2name, remaining
            )

    log.info("================ RESUMEN ================")
    log.info(f"  Hard negatives stand añadidos: {total}")
    log.info(f"  Carpeta: {img_out}")


if __name__ == "__main__":
    main()
