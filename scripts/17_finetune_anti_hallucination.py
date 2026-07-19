"""
Script 17: Fine-tune anti-alucinación + mejora de poses
=========================================================
Parte de best.pt actual y lo afina con:
  - Nuevas imágenes de poses diversas (script 15)
  - Hard negatives de escenario stand (script 16)

Receta clave:
  * lr0 muy bajo (0.0001) → NO destruir lo aprendido.
  * Freeze backbone 10 epochs → solo afina cabeza de detección.
  * cls_loss weight subido (1.0) → penaliza más fuerte falsos positivos.
  * close_mosaic = 10 → fine-tune limpio al final.
  * Augmentación: perspective + degrees moderado para mejor generalización 3D.

ADVERTENCIA: este script ENTRENA. No lo ejecutes hasta que las
descargas de scripts 15 y 16 hayan terminado.

Uso:
    python scripts/17_finetune_anti_hallucination.py [--nano]
        --nano: fine-tunea el modelo nano (best.pt en runs/.../yolov8n_v2)
        sin flag: fine-tunea el modelo small (best.pt en models/.../yolov8s_v2)
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]

WEIGHTS_SMALL = REPO / "models" / "yolov8_weapons" / "weapons" / "yolov8s_v2" / "weights" / "best.pt"
WEIGHTS_NANO  = REPO / "runs" / "detect" / "models" / "yolov8_weapons" / "weapons" / "yolov8n_v2" / "weights" / "best.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nano", action="store_true", help="Fine-tune el nano para Pi5")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs totales")
    parser.add_argument("--imgsz", type=int, default=None, help="Override imgsz")
    parser.add_argument("--name", default=None, help="Nombre del run (override, ej. yolov8s_v5)")
    args = parser.parse_args()

    weights = WEIGHTS_NANO if args.nano else WEIGHTS_SMALL
    if not weights.exists():
        raise FileNotFoundError(f"No existe {weights}")

    data_yaml = REPO / "data" / "processed" / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"No existe {data_yaml}")

    imgsz = args.imgsz or (416 if args.nano else 640)
    # GPU 4060 8GB: batch 16 grande (32 daba OOM a 640px) / 48 nano (416px)
    batch = 48 if args.nano else 16
    run_name = args.name or ("yolov8n_v4_pose_negs" if args.nano else "yolov8s_v4_pose_negs")

    log.info(f"Fine-tune desde: {weights}")
    log.info(f"  imgsz={imgsz} | batch={batch} | epochs={args.epochs}")
    log.info(f"  data={data_yaml}")

    from ultralytics import YOLO
    model = YOLO(str(weights))

    model.train(
        data        = str(data_yaml),
        epochs      = args.epochs,
        imgsz       = imgsz,
        batch       = batch,
        workers     = 16,          # 16 cores: maxima paralelizacion de carga
        device      = 0,
        optimizer   = "AdamW",

        # ── LR muy bajo: no destruir lo aprendido ──
        lr0         = 0.0001,
        lrf         = 0.01,
        warmup_epochs = 3,
        patience    = 15,

        # ── Freeze backbone primeros pasos ──
        # YOLOv8 tiene 23 capas; las primeras 10 son backbone.
        # Freeze=10 congela esas capas → solo cabeza se entrena.
        # (ultralytics lo soporta vía `freeze` arg)
        freeze      = 10,

        # ── Pesos de loss anti-alucinación ──
        # cls = penalización clasificación errónea (subimos)
        # box = localización (estándar)
        # dfl = distribution focal loss
        cls         = 1.0,           # default 0.5 → subido a 1.0 (más penalización a FP)
        box         = 7.5,           # default
        dfl         = 1.5,           # default

        amp         = True,
        # cache=False: el dataset esta en un HDD lento. Leer JPGs pequenos (~3GB,
        # que el SO cachea en RAM) es mas rapido que releer 28GB de .npy del HDD.
        cache       = False,
        seed        = 42,
        project     = "models/yolov8_weapons",
        name        = f"weapons/{run_name}",
        exist_ok    = True,

        # ── Augmentación: foco en 3D / perspective ──
        fliplr      = 0.5,
        flipud      = 0.0,
        mosaic      = 1.0,
        mixup       = 0.15,           # bajado un poco (no mezclar tanto en fine-tune)
        copy_paste  = 0.3,            # bajado
        degrees     = 30,             # subido para más variedad angular
        translate   = 0.15,
        scale       = 0.5,
        shear       = 5,
        perspective = 0.001,          # subido 2x → más vistas tipo POV/frontal
        hsv_h       = 0.02,
        hsv_s       = 0.7,
        hsv_v       = 0.4,

        weight_decay = 0.0005,
        close_mosaic = 10,            # apaga mosaic últimas 10 epochs (afinado limpio)
    )
    log.info("Fine-tune completado.")


if __name__ == "__main__":
    main()
