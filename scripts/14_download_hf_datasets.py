"""
Script 14: Descarga datasets públicos de HuggingFace para weapon detection
=============================================================================
Datasets:
  1. fcakyon/gun-object-detection   - COCO format (knife/pistol/rifle, descarta grenade)
  2. Simuletic/cctv-knife-detection-dataset - YOLO knife CCTV (114 imgs)

Uso:
    python scripts/14_download_hf_datasets.py
"""

import json
import logging
import shutil
import zipfile
from pathlib import Path

import requests
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO     = Path(__file__).resolve().parents[1]
HF_CACHE = REPO / "data" / "hf_cache"
HF_CACHE.mkdir(parents=True, exist_ok=True)

CLS_IDX = {"knife": 0, "handgun": 1, "long_gun": 2}

# fcakyon: id COCO → nuestra clase ('grenade' se descarta)
FCAKYON_MAP = {
    1: "knife",
    2: "handgun",     # pistol
    3: "long_gun",    # rifle
}


def download_file(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        logger.info(f"  Cache: {dest.name}")
        return
    logger.info(f"  Bajando {dest.name}...")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)


# ─── Dataset 1: fcakyon (COCO format) ───
def integrate_fcakyon(img_out: Path, lbl_out: Path) -> int:
    base_url = "https://huggingface.co/datasets/fcakyon/gun-object-detection/resolve/main/data"
    work = HF_CACHE / "fcakyon_gun"
    work.mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid"]:
        zip_path = work / f"{split}.zip"
        download_file(f"{base_url}/{split}.zip", zip_path)
        ext_dir = work / split
        if not list(ext_dir.glob("*.jpg")):
            logger.info(f"  Descomprimiendo {split}.zip...")
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(ext_dir)

    added = 0
    for split in ["train", "valid"]:
        split_dir = work / split
        coco_path = split_dir / "_annotations.coco.json"
        if not coco_path.exists():
            continue
        with open(coco_path) as f:
            coco = json.load(f)

        # Map image_id → image info
        img_by_id = {im["id"]: im for im in coco["images"]}
        # Group annotations by image
        anns_by_img: dict = {}
        for ann in coco["annotations"]:
            cat = ann["category_id"]
            cls_name = FCAKYON_MAP.get(cat)
            if cls_name is None:  # grenade → skip
                continue
            anns_by_img.setdefault(ann["image_id"], []).append((cls_name, ann["bbox"]))

        for img_id, anns in anns_by_img.items():
            info = img_by_id.get(img_id)
            if not info:
                continue
            src_img = split_dir / info["file_name"]
            if not src_img.exists():
                continue

            iw, ih = info["width"], info["height"]
            lines = []
            for cls_name, bbox in anns:
                x, y, w, h = bbox
                cx = (x + w / 2.0) / iw
                cy = (y + h / 2.0) / ih
                bw = w / iw
                bh = h / ih
                if bw <= 0 or bh <= 0:
                    continue
                cx, cy = max(0, min(1, cx)), max(0, min(1, cy))
                bw, bh = max(0, min(1, bw)), max(0, min(1, bh))
                lines.append(f"{CLS_IDX[cls_name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if not lines:
                continue
            added += 1
            ext = "." + info["file_name"].rsplit(".", 1)[-1].lower()
            dst_img = img_out / f"hf_gun_{added:06d}{ext}"
            dst_lbl = lbl_out / f"hf_gun_{added:06d}.txt"
            shutil.copy2(src_img, dst_img)
            dst_lbl.write_text("\n".join(lines), encoding="utf-8")
    return added


# ─── Dataset 2: Simuletic (YOLO flat, 114 imgs) ───
def integrate_simuletic_knife(img_out: Path, lbl_out: Path) -> int:
    repo = "Simuletic/cctv-knife-detection-dataset"
    base = f"https://huggingface.co/datasets/{repo}/resolve/main"
    api_base = f"https://huggingface.co/api/datasets/{repo}/tree/main"
    work = HF_CACHE / "simuletic_knife"
    work.mkdir(parents=True, exist_ok=True)

    # Listar imágenes en Knife_Dataset/images
    r = requests.get(f"{api_base}/Knife_Dataset/images", timeout=30)
    if r.status_code != 200:
        logger.warning(f"  No se pudo listar archivos: {r.status_code}")
        return 0
    img_files = [f["path"] for f in r.json() if f["type"] == "file"
                 and f["path"].lower().endswith((".png", ".jpg", ".jpeg"))]
    logger.info(f"  {len(img_files)} imágenes encontradas")

    added = 0
    for img_path in img_files:
        stem = img_path.split("/")[-1].rsplit(".", 1)[0]
        ext  = "." + img_path.rsplit(".", 1)[-1].lower()
        lbl_path = f"Knife_Dataset/labels/{stem}.txt"

        try:
            img_data = requests.get(f"{base}/{img_path}", timeout=30).content
            lbl_resp = requests.get(f"{base}/{lbl_path}", timeout=30)
            if lbl_resp.status_code != 200:
                continue
            lbl_text = lbl_resp.text
        except Exception:
            continue

        # Convertir cls_id (todas son knife=0 en este dataset, ya coincide con nuestro 0)
        lines = []
        for line in lbl_text.splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # Forzar cls_id = 0 (knife) por si acaso
            lines.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}")

        if not lines:
            continue

        added += 1
        dst_img = img_out / f"hf_knife_{added:06d}{ext}"
        dst_lbl = lbl_out / f"hf_knife_{added:06d}.txt"
        with open(dst_img, "wb") as f:
            f.write(img_data)
        dst_lbl.write_text("\n".join(lines), encoding="utf-8")

        if added % 25 == 0:
            logger.info(f"    Simuletic: {added}/{len(img_files)}")

    return added


def main():
    processed = REPO / "data" / "processed"
    img_out   = processed / "images" / "train"
    lbl_out   = processed / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    total = 0
    logger.info("=== fcakyon/gun-object-detection (COCO → YOLO) ===")
    try:
        n = integrate_fcakyon(img_out, lbl_out)
        logger.info(f"  Agregadas: {n}")
        total += n
    except Exception as e:
        logger.error(f"  Falló: {e}")

    logger.info("=== Simuletic/cctv-knife-detection-dataset (YOLO) ===")
    try:
        n = integrate_simuletic_knife(img_out, lbl_out)
        logger.info(f"  Agregadas: {n}")
        total += n
    except Exception as e:
        logger.error(f"  Falló: {e}")

    logger.info(f"TOTAL agregado: {total} imágenes públicas")


if __name__ == "__main__":
    main()
