"""
Script 2b: Merge OIv7 + Roboflow datasets en formato YOLO unificado
=====================================================================
Clases finales:
    0 = knife
    1 = handgun
    2 = long_gun  (rifle, shotgun)

Fuentes:
- OIv7 (FiftyOne): Knife, Handgun, Shotgun, Rifle
- RF joseph-nelson/pistols (clase 'pistol' -> handgun)
- RF rifledetection/rifle-detection-p9slr (clase 'Rifle' -> long_gun)

Salida:
    data/processed/
        images/{train,val,test}/*
        labels/{train,val,test}/*.txt
        data.yaml
"""

import logging
import random
import shutil
import yaml
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
PROCESSED = REPO / "data" / "processed"
RF_PISTOLS = REPO / "data" / "rf" / "pistols" / "export"
RF_RIFLE = REPO / "data" / "rf" / "rifle"

CLASSES = ["knife", "handgun", "long_gun"]
CLS_IDX = {n: i for i, n in enumerate(CLASSES)}

# OIv7 -> nuestra clase
OI_TO_CLASS = {
    "Knife":   "knife",
    "Handgun": "handgun",
    "Shotgun": "long_gun",
    "Rifle":   "long_gun",
}

# Roboflow class id (en el data.yaml) -> nuestra clase
RF_PISTOLS_MAP = {0: "handgun"}      # clase 'pistol'
RF_RIFLE_MAP   = {0: "long_gun"}     # clase 'Rifle'

def _load_cfg() -> dict:
    with open(REPO / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)

_CFG = _load_cfg()
_MAX_SAMPLES = _CFG["dataset"]["max_samples_per_class"]

# Cap de imágenes por fuente para mantener balance + tiempo de entreno razonable
CAPS = {
    "rf_pistols":  3000,
    "rf_rifle":    3500,
    "oi_per_split": _MAX_SAMPLES,
}

SPLIT_RATIOS = {"train": 0.85, "val": 0.10, "test": 0.05}

random.seed(42)


def reset_processed():
    if PROCESSED.exists():
        shutil.rmtree(PROCESSED)
    for split in ["train", "val", "test"]:
        (PROCESSED / "images" / split).mkdir(parents=True, exist_ok=True)
        (PROCESSED / "labels" / split).mkdir(parents=True, exist_ok=True)


def img_ok(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def collect_oi():
    """Cosecha (img_path, lines_yolo[]) desde los datasets persistentes de FiftyOne."""
    import fiftyone as fo
    items = []
    for split in ["train", "validation", "test"]:
        ds_name = f"weapons_{split}_{_MAX_SAMPLES}"
        try:
            ds = fo.load_dataset(ds_name)
        except Exception:
            log.warning(f"OIv7: no se encontro dataset '{ds_name}', saltando")
            continue
        for sample in ds:
            src = Path(sample.filepath)
            if not src.exists():
                continue
            lines = []
            if sample.ground_truth and sample.ground_truth.detections:
                for det in sample.ground_truth.detections:
                    cls = OI_TO_CLASS.get(det.label)
                    if cls is None:
                        continue
                    bx, by, bw, bh = det.bounding_box
                    cx = bx + bw / 2.0
                    cy = by + bh / 2.0
                    cx, cy = max(0, min(1, cx)), max(0, min(1, cy))
                    bw, bh = max(0, min(1, bw)), max(0, min(1, bh))
                    if bw <= 0 or bh <= 0:
                        continue
                    lines.append(f"{CLS_IDX[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            if lines:
                items.append((src, lines))
    log.info(f"OIv7: {len(items)} imagenes con anotaciones validas")
    return items


def remap_label_file(label_path: Path, class_map: dict) -> list:
    """Lee un .txt YOLO y devuelve lineas remapeadas con cls_idx final."""
    out = []
    if not label_path.exists():
        return out
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        try:
            old_id = int(parts[0])
        except ValueError:
            continue
        cls = class_map.get(old_id)
        if cls is None:
            continue
        new_id = CLS_IDX[cls]
        out.append(f"{new_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
    return out


def collect_rf_flat(images_dir: Path, labels_dir: Path, class_map: dict, cap: int):
    """Para datasets RF en un solo directorio (export/) o concatenando splits."""
    items = []
    if not images_dir.exists():
        return items
    imgs = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    for img in imgs:
        lbl = labels_dir / (img.stem + ".txt")
        lines = remap_label_file(lbl, class_map)
        if lines:
            items.append((img, lines))
    if cap and len(items) > cap:
        random.shuffle(items)
        items = items[:cap]
    return items


def collect_rf_split(root: Path, class_map: dict, cap: int):
    """Para datasets RF con train/valid/test."""
    items = []
    for sub in ["train", "valid", "test"]:
        items += collect_rf_flat(root / sub / "images", root / sub / "labels", class_map, 0)
    if cap and len(items) > cap:
        random.shuffle(items)
        items = items[:cap]
    return items


def split_and_write(all_items: list):
    random.shuffle(all_items)
    n = len(all_items)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    splits = {
        "train": all_items[:n_train],
        "val":   all_items[n_train:n_train + n_val],
        "test":  all_items[n_train + n_val:],
    }
    counts = {}
    class_counts = {c: {k: 0 for k in CLASSES} for c in splits}
    for split, items in splits.items():
        img_dir = PROCESSED / "images" / split
        lbl_dir = PROCESSED / "labels" / split
        n_ok = 0
        for i, (src, lines) in enumerate(items):
            if not img_ok(src):
                continue
            n_ok += 1
            dst_img = img_dir / f"{n_ok:07d}{src.suffix.lower()}"
            dst_lbl = lbl_dir / f"{n_ok:07d}.txt"
            shutil.copy2(src, dst_img)
            dst_lbl.write_text("\n".join(lines), encoding="utf-8")
            for ln in lines:
                cls_id = int(ln.split()[0])
                class_counts[split][CLASSES[cls_id]] += 1
        counts[split] = n_ok
        log.info(f"  {split}: {n_ok} imgs | clases: {class_counts[split]}")
    return counts


def write_data_yaml():
    out = PROCESSED / "data.yaml"
    data = {
        "path": str(PROCESSED.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "names": {i: n for i, n in enumerate(CLASSES)},
    }
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    log.info(f"data.yaml -> {out}")


def main():
    log.info("=== Merge datasets -> 3 clases (knife/handgun/long_gun) ===")
    reset_processed()

    log.info("[1/3] OIv7 via FiftyOne...")
    oi_items = collect_oi()

    log.info("[2/3] RF pistols (handgun)...")
    pistols_items = collect_rf_flat(
        RF_PISTOLS / "images",
        RF_PISTOLS / "labels",
        RF_PISTOLS_MAP,
        CAPS["rf_pistols"],
    )
    log.info(f"   pistols: {len(pistols_items)} imgs")

    log.info("[3/3] RF rifles (long_gun)...")
    rifle_items = collect_rf_split(RF_RIFLE, RF_RIFLE_MAP, CAPS["rf_rifle"])
    log.info(f"   rifles: {len(rifle_items)} imgs")

    all_items = oi_items + pistols_items + rifle_items
    log.info(f"Total a splitear: {len(all_items)}")

    counts = split_and_write(all_items)
    write_data_yaml()
    log.info(f"Total final: {sum(counts.values())} imgs en data/processed/")


if __name__ == "__main__":
    main()
