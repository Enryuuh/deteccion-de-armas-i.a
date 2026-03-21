"""
Script 4: Evaluación del Modelo — mAP COCO
===========================================
Evalúa el modelo entrenado sobre el conjunto de test.
Genera métricas COCO estándar: mAP@0.5, mAP@0.5:0.95, etc.

Uso:
    python scripts/4_evaluate.py
    python scripts/4_evaluate.py --model-dir models/rtdetr_weapons/best
"""

import sys
import json
import logging
import argparse
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.dataset import WeaponCOCODataset, collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def run_evaluation(model, processor, dataloader, coco_gt, device, conf_thresh: float = 0.0) -> dict:
    model.eval()
    results = []

    for batch in tqdm(dataloader, desc="Evaluando", unit="batch"):
        pixel_values = batch["pixel_values"].to(device)
        image_ids    = batch["image_ids"]
        orig_sizes   = batch["orig_sizes"]

        outputs = model(pixel_values=pixel_values)

        # Post-procesado: convierte logits → bbox en escala original
        target_sizes = torch.tensor(orig_sizes, dtype=torch.float32)
        predictions  = processor.post_process_object_detection(
            outputs,
            threshold=conf_thresh,
            target_sizes=target_sizes,
        )

        for img_id, pred in zip(image_ids, predictions):
            for score, label, box in zip(pred["scores"], pred["labels"], pred["boxes"]):
                x1, y1, x2, y2 = box.tolist()
                results.append({
                    "image_id":    int(img_id),
                    "category_id": int(label),
                    "bbox":        [x1, y1, x2 - x1, y2 - y1],  # COCO [x,y,w,h]
                    "score":       float(score),
                })

    if not results:
        logger.warning("No se generaron predicciones. Revisa el umbral de confianza.")
        return {}

    coco_dt   = coco_gt.loadRes(results)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    metrics = {
        "mAP@0.5:0.95":  float(evaluator.stats[0]),
        "mAP@0.5":        float(evaluator.stats[1]),
        "mAP@0.75":       float(evaluator.stats[2]),
        "mAP_small":      float(evaluator.stats[3]),
        "mAP_medium":     float(evaluator.stats[4]),
        "mAP_large":      float(evaluator.stats[5]),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluación RT-DETR sobre test set")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Ruta al modelo (default: models/rtdetr_weapons/best)")
    parser.add_argument("--conf",      type=float, default=0.01,
                        help="Umbral de confianza para evaluación (default: 0.01)")
    args = parser.parse_args()

    cfg      = load_config()
    proc_dir = Path(cfg["dataset"]["processed_dir"])

    model_dir = Path(args.model_dir) if args.model_dir else Path(cfg["training"]["output_dir"]) / "best"
    if not model_dir.exists():
        logger.error(f"Modelo no encontrado en {model_dir}. Entrena primero con script 3.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"=== Evaluación RT-DETR  |  Dispositivo: {device} ===")
    logger.info(f"    Modelo: {model_dir}")

    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    model     = AutoModelForObjectDetection.from_pretrained(str(model_dir))
    model.to(device)

    test_json  = proc_dir / "annotations_test.json"
    img_test   = proc_dir / "images" / "test"

    if not test_json.exists():
        logger.error("No se encontró annotations_test.json. Ejecuta scripts 1 y 2 primero.")
        sys.exit(1)

    test_dataset = WeaponCOCODataset(test_json, img_test, processor, augment=False, return_meta=True)
    test_loader  = DataLoader(
        test_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["dataloader_num_workers"],
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    coco_gt = COCO(str(test_json))
    metrics = run_evaluation(model, processor, test_loader, coco_gt, device, conf_thresh=args.conf)

    if metrics:
        logger.info("=== Resultados ===")
        for k, v in metrics.items():
            logger.info(f"    {k:20s}: {v:.4f} ({v*100:.2f}%)")

        out_path = Path(cfg["training"]["output_dir"]) / "evaluation_results.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"✅ Métricas guardadas en: {out_path}")


if __name__ == "__main__":
    main()
