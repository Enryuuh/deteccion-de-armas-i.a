"""
Script 2: Preparacion del Dataset -> Formato YOLO
====================================================
Convierte las descargas de FiftyOne (Open Images v7) al formato
YOLO que espera ultralytics:

    data/processed/
    |- data.yaml
    |- images/{train,val,test}/*.jpg
    \- labels/{train,val,test}/*.txt    # cls cx cy w h  (normalizados 0-1)

Uso:
    python scripts/2_prepare_dataset.py
"""

import logging
import shutil
import yaml
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Open Images label -> nuestra clase
OI_TO_LABEL = {
    "Knife":   "knife",
    "Firearm": "firearm",
    "Handgun": "firearm",
    "Shotgun": "firearm",
}


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def class_to_idx(cfg: dict) -> dict:
    """YOLO usa clases 0-indexadas."""
    return {name: idx for idx, name in enumerate(cfg["classes"])}


def convert_split(raw_split: str, out_split: str, processed: Path,
                  cls_idx: dict, max_samples: int) -> dict:
    try:
        import fiftyone as fo
    except ImportError:
        raise ImportError("Instala fiftyone: pip install fiftyone")

    img_dir = processed / "images" / out_split
    lbl_dir = processed / "labels" / out_split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    ds_name = f"weapons_{raw_split}_{max_samples}"
    try:
        dataset = fo.load_dataset(ds_name)
    except Exception:
        logger.warning(f"No se encontro dataset FiftyOne '{ds_name}'. Saltando.")
        return {"images": 0, "annotations": 0, "skipped": 0}

    n_img = 0
    n_ann = 0
    n_skip = 0

    for sample in dataset:
        src = Path(sample.filepath)
        if not src.exists():
            n_skip += 1
            continue

        # Recolectar anotaciones validas
        lines = []
        if sample.ground_truth and sample.ground_truth.detections:
            for det in sample.ground_truth.detections:
                proj = OI_TO_LABEL.get(det.label)
                if proj is None or proj not in cls_idx:
                    continue
                bx, by, bw, bh = det.bounding_box  # ya normalizado [0,1]
                cx = bx + bw / 2.0
                cy = by + bh / 2.0
                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)
                bw = min(max(bw, 0.0), 1.0)
                bh = min(max(bh, 0.0), 1.0)
                if bw <= 0 or bh <= 0:
                    continue
                lines.append(f"{cls_idx[proj]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not lines:
            # YOLO acepta imagenes sin objetos pero no aportan a deteccion
            n_skip += 1
            continue

        # Validar imagen
        try:
            with Image.open(src) as im:
                im.verify()
        except Exception:
            n_skip += 1
            continue

        n_img += 1
        dst_img = img_dir / f"{n_img:06d}{src.suffix.lower()}"
        dst_lbl = lbl_dir / f"{n_img:06d}.txt"
        shutil.copy2(src, dst_img)
        dst_lbl.write_text("\n".join(lines), encoding="utf-8")
        n_ann += len(lines)

    return {"images": n_img, "annotations": n_ann, "skipped": n_skip}


def write_data_yaml(processed: Path, classes: list):
    data = {
        "path": str(processed.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "names": {i: n for i, n in enumerate(classes)},
    }
    out = processed / "data.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    logger.info(f"data.yaml generado en {out}")


def main():
    cfg = load_config()
    processed = Path(cfg["dataset"]["processed_dir"])
    cls_idx = class_to_idx(cfg)
    max_samples = cfg["dataset"]["max_samples_per_class"]

    logger.info("=== Preparacion del Dataset (formato YOLO) ===")
    logger.info(f"Clases: {cls_idx}")

    split_map = {"train": "train", "validation": "val", "test": "test"}

    for raw_split, out_split in split_map.items():
        logger.info(f"--> {raw_split} -> {out_split}")
        stats = convert_split(raw_split, out_split, processed, cls_idx, max_samples)
        logger.info(f"   Imagenes: {stats['images']} | Anotaciones: {stats['annotations']} | Saltadas: {stats['skipped']}")

    write_data_yaml(processed, cfg["classes"])
    logger.info("Dataset listo en data/processed/")


if __name__ == "__main__":
    main()
