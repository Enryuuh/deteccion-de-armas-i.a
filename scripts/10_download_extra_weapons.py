"""
Script 10: Descargar más imágenes de armas de OIv7 (sin fiftyone)
==================================================================
Descarga imágenes adicionales de Knife/Handgun/Shotgun/Rifle de
ambos splits (validation + train) parseando los CSVs ya descargados
con pandas en chunks. Genera labels YOLO automáticamente desde las
bounding boxes de OIv7.

Las imágenes se agregan al split train de data/processed/.
Evita el crash de MongoDB de fiftyone en Windows.

Uso:
    python scripts/10_download_extra_weapons.py
"""

import logging
import yaml
import concurrent.futures
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from io import BytesIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO    = Path(__file__).resolve().parents[1]
OI_BASE = Path.home() / "fiftyone" / "open-images-v7"

# Display name OIv7 -> nuestra clase YOLO
WEAPON_TO_CLS = {
    "Knife":   0,    # knife
    "Handgun": 1,    # handgun
    "Shotgun": 2,    # long_gun
    "Rifle":   2,    # long_gun
}

DOWNLOAD_WORKERS = 8
MAX_PER_SPLIT    = 5000   # techo por split (validation tiene ~1500, train mucho más)


def load_class_map(classes_csv: Path) -> dict:
    df = pd.read_csv(classes_csv, header=None, names=["code", "name"])
    return dict(zip(df["code"], df["name"]))


def find_weapon_codes(code2name: dict) -> dict:
    """{label_code -> nuestra_clase_yolo}"""
    return {code: WEAPON_TO_CLS[name] for code, name in code2name.items() if name in WEAPON_TO_CLS}


def collect_annotations_chunked(detections_csv: Path, weapon_codes: dict) -> dict:
    """
    Lee detections.csv en chunks y agrupa anotaciones de armas por imagen.
    Devuelve {ImageID: [(yolo_cls, cx, cy, w, h), ...]}
    """
    size_mb = detections_csv.stat().st_size // 1_000_000
    logger.info(f"Parseando {detections_csv.name} ({size_mb} MB) en chunks...")

    cols = ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax"]
    by_img: dict = {}
    n_rows = 0

    for chunk in pd.read_csv(detections_csv, usecols=cols, chunksize=500_000):
        chunk = chunk[chunk["LabelName"].isin(weapon_codes.keys())]
        n_rows += len(chunk)
        for row in chunk.itertuples(index=False):
            cls = weapon_codes[row.LabelName]
            cx  = (row.XMin + row.XMax) / 2.0
            cy  = (row.YMin + row.YMax) / 2.0
            w   = row.XMax - row.XMin
            h   = row.YMax - row.YMin
            if w <= 0 or h <= 0:
                continue
            by_img.setdefault(row.ImageID, []).append((cls, cx, cy, w, h))

    logger.info(f"  Imágenes con anotaciones de arma: {len(by_img)} ({n_rows} filas)")
    return by_img


def load_image_urls(image_ids_csv: Path, valid_ids: set) -> dict:
    logger.info(f"Leyendo URLs ({image_ids_csv.stat().st_size // 1_000_000} MB)...")
    df = pd.read_csv(image_ids_csv, usecols=["ImageID", "OriginalURL", "Thumbnail300KURL"])
    df = df[df["ImageID"].isin(valid_ids)]
    df["url"] = df["Thumbnail300KURL"].fillna(df["OriginalURL"])
    return dict(zip(df["ImageID"], df["url"]))


def download_one(args) -> bool:
    img_id, url, dst_img, dst_lbl, lines = args
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.save(str(dst_img), "JPEG", quality=90)
        dst_lbl.write_text("\n".join(lines), encoding="utf-8")
        return True
    except Exception:
        return False


def process_split(split: str, img_out: Path, lbl_out: Path, prefix: str, max_n: int) -> int:
    classes_csv    = OI_BASE / split / "metadata" / "classes.csv"
    detections_csv = OI_BASE / split / "labels"   / "detections.csv"
    image_ids_csv  = OI_BASE / split / "metadata" / "image_ids.csv"

    if not all(p.exists() for p in [classes_csv, detections_csv, image_ids_csv]):
        logger.warning(f"  Faltan CSVs para split={split}, saltando")
        return 0

    code2name    = load_class_map(classes_csv)
    weapon_codes = find_weapon_codes(code2name)
    logger.info(f"  Códigos de arma OIv7: {len(weapon_codes)} clases")

    by_img       = collect_annotations_chunked(detections_csv, weapon_codes)
    if not by_img:
        return 0

    # Cap por split
    if len(by_img) > max_n:
        keys = list(by_img.keys())[:max_n]
        by_img = {k: by_img[k] for k in keys}
        logger.info(f"  Cap aplicado: {max_n} imgs")

    urls         = load_image_urls(image_ids_csv, set(by_img.keys()))
    logger.info(f"  URLs disponibles: {len(urls)}")

    existing  = sorted(img_out.glob(f"{prefix}_*.jpg"))
    start_idx = len(existing) + 1

    tasks = []
    for i, (img_id, anns) in enumerate(by_img.items(), start=start_idx):
        if img_id not in urls:
            continue
        url = urls[img_id]
        lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in anns]
        dst_img = img_out / f"{prefix}_{i:06d}.jpg"
        dst_lbl = lbl_out / f"{prefix}_{i:06d}.txt"
        tasks.append((img_id, url, dst_img, dst_lbl, lines))

    logger.info(f"  Descargando {len(tasks)} imágenes en paralelo...")
    added = failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
        for j, ok in enumerate(ex.map(download_one, tasks), 1):
            if ok: added += 1
            else: failed += 1
            if j % 200 == 0:
                logger.info(f"    Progreso {split}: {j}/{len(tasks)} | OK: {added} | Fallidos: {failed}")

    logger.info(f"  Split {split}: agregadas {added} imgs ({failed} fallidos)")
    return added


def main():
    with open(REPO / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    processed = REPO / Path(cfg["dataset"]["processed_dir"])
    img_out   = processed / "images" / "train"
    lbl_out   = processed / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    total = 0
    for split, prefix in [("validation", "wval"), ("train", "wtr")]:
        logger.info(f"=== Procesando split: {split} ===")
        total += process_split(split, img_out, lbl_out, prefix, MAX_PER_SPLIT)

    logger.info(f"TOTAL agregado al dataset: {total} imágenes con armas")
    logger.info("Ejecuta scripts/3_train.py para reentrenar.")


if __name__ == "__main__":
    main()
