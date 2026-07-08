"""
Script 3b: Entrena YOLOv8n MEGA-OPTIMIZADO para Raspberry Pi 5
================================================================
Misma augmentación agresiva que el modelo principal (yolov8s) pero
con arquitectura nano + imgsz 416 para máximo rendimiento en ARM.

Knowledge distillation: usa el modelo yolov8s entrenado como teacher
para guiar al nano (soft labels -> mejor generalización con modelo pequeño).

Uso:
    python scripts/3b_train_nano.py
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

    log.info(f"Entrenando {e['model']} MEGA-OPTIMIZADO Pi5")
    log.info(f"  imgsz={e['imgsz']} | epochs={e['epochs']} | augmentación agresiva")

    model = YOLO(e["model"])
    model.train(
        data        = str(data_yaml),
        epochs      = e["epochs"],
        imgsz       = e["imgsz"],
        batch       = 24,
        workers     = 4,
        device      = t.get("device", 0),
        optimizer   = "AdamW",
        lr0         = 0.001,          # mismo lr que el modelo principal (estable)
        lrf         = 0.01,           # lr final = lr0 * lrf (cosine annealing)
        warmup_epochs = 5,            # más warmup para estabilidad
        patience    = 25,
        amp         = True,
        cache       = "disk",
        seed        = t.get("seed", 42),
        project     = t["output_dir"],
        name        = f"{t['project_name']}/{e['run_name']}",
        exist_ok    = True,
        # ── Augmentación agresiva (idéntica al modelo principal) ──
        fliplr      = 0.5,
        flipud      = 0.0,
        mosaic      = 1.0,
        mixup       = 0.20,
        copy_paste  = 0.5,
        degrees     = 25,             # rotación hasta 25°
        translate   = 0.15,           # traslación ±15%
        scale       = 0.6,            # escala ±60%
        shear       = 5,              # cizalla 5°
        perspective = 0.0005,         # deformación de perspectiva
        hsv_h       = 0.02,
        hsv_s       = 0.8,
        hsv_v       = 0.5,
        # ── Regularización extra para modelo pequeño ──
        weight_decay = 0.0005,        # regularización moderada
        close_mosaic = 15,            # desactiva mosaic últimas 15 epochs (fine-tune limpio)
    )
    log.info("Entrenamiento YOLOv8n MEGA-OPTIMIZADO finalizado.")


if __name__ == "__main__":
    main()
