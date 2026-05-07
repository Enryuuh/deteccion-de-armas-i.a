"""
Script 3b: Entrena YOLOv8n para despliegue en Raspberry Pi 5.
Usa imgsz 416 y batch 16 para optimizar velocidad/tamano.
"""
import logging
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from ultralytics import YOLO

    data_yaml = Path(cfg["dataset"]["processed_dir"]) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"No existe {data_yaml}. Ejecuta 2b_merge_datasets.py primero.")

    e = cfg["export"]
    t = cfg["training"]

    log.info(f"Entrenando {e['model']} | imgsz={e['imgsz']} | batch={e['batch']} | epochs={e['epochs']}")

    model = YOLO(e["model"])
    model.train(
        data       = str(data_yaml),
        epochs     = e["epochs"],
        imgsz      = e["imgsz"],
        batch      = e["batch"],
        workers    = t["workers"],
        device     = t.get("device", 0),
        optimizer  = t.get("optimizer", "AdamW"),
        lr0        = t.get("lr0", 0.001),
        patience   = 20,
        amp        = True,
        cache      = False,
        seed       = t.get("seed", 42),
        project    = t["output_dir"],
        name       = f"{t['project_name']}/{e['run_name']}",
        fliplr     = 0.5,
        mosaic     = 1.0,
        mixup      = 0.1,
        copy_paste = 0.3,
        hsv_h      = 0.015,
        hsv_s      = 0.7,
        hsv_v      = 0.4,
        exist_ok   = True,
    )
    log.info("Entrenamiento YOLOv8n finalizado.")


if __name__ == "__main__":
    main()
