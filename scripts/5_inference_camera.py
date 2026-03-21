"""
Script 5: Inferencia en Tiempo Real con Cámara
===============================================
Detecta cuchillos y armas de fuego usando el modelo entrenado.
Optimizado para RTX 4060 — ~30-60 FPS con FP16.

Uso:
    python scripts/5_inference_camera.py
    python scripts/5_inference_camera.py --model-dir models/rtdetr_weapons/best
    python scripts/5_inference_camera.py --source video.mp4   (archivo de video)
    python scripts/5_inference_camera.py --source 0           (cámara USB)
"""

import sys
import time
import logging
import argparse
import yaml
from pathlib import Path
from collections import deque

import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.visualization import draw_detections, draw_hud
from utils.alerts import AlertSystem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def open_source(source: str) -> cv2.VideoCapture:
    """Abre cámara o archivo de video."""
    try:
        cam_idx = int(source)
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)  # DirectShow para Windows
    except ValueError:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente: {source}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reducir latencia
    return cap


@torch.no_grad()
def infer_frame(model, processor, frame_rgb, device, conf_thresh: float, use_fp16: bool):
    """Corre RT-DETR sobre un frame y retorna bboxes, scores y labels."""
    pil_img  = Image.fromarray(frame_rgb)
    inputs   = processor(images=pil_img, return_tensors="pt")
    pv       = inputs["pixel_values"].to(device)

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
        outputs = model(pixel_values=pv)

    h, w = frame_rgb.shape[:2]
    target_sizes = torch.tensor([[h, w]], dtype=torch.float32)
    preds = processor.post_process_object_detection(
        outputs, threshold=conf_thresh, target_sizes=target_sizes
    )[0]

    boxes  = preds["boxes"].cpu().numpy()    # [x1,y1,x2,y2]
    scores = preds["scores"].cpu().numpy()
    labels = preds["labels"].cpu().numpy()

    return boxes, scores, labels


def main():
    parser = argparse.ArgumentParser(description="Inferencia en tiempo real con RT-DETR")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--source",    type=str, default=None,
                        help="Índice de cámara (ej. 0) o ruta a video")
    parser.add_argument("--conf",      type=float, default=None)
    parser.add_argument("--no-alert",  action="store_true")
    args = parser.parse_args()

    cfg  = load_config()
    icfg = cfg["inference"]

    model_dir   = Path(args.model_dir)  if args.model_dir else Path(cfg["training"]["output_dir"]) / "best"
    source      = args.source           if args.source     else str(icfg["camera_index"])
    conf_thresh = args.conf             if args.conf       else icfg["confidence_threshold"]

    if not model_dir.exists():
        logger.error(f"Modelo no encontrado en {model_dir}. Entrena con script 3 primero.")
        sys.exit(1)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda"
    logger.info(f"=== Inferencia RT-DETR  |  {device}  |  FP16: {use_fp16} ===")

    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    model     = AutoModelForObjectDetection.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    if use_fp16:
        model.half()

    alerts = AlertSystem(cfg, disabled=args.no_alert)
    cap    = open_source(source)

    fps_history = deque(maxlen=30)
    frame_idx   = 0
    logger.info("=== Iniciando detección — presiona 'q' para salir ===")

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            logger.info("Fin del stream.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Inferencia
        boxes, scores, labels_ids = infer_frame(
            model, processor, frame_rgb, device, conf_thresh, use_fp16
        )

        # Mapa id → nombre
        id2label = model.config.id2label

        # Calcular FPS
        elapsed = time.perf_counter() - t0
        fps_history.append(1.0 / max(elapsed, 1e-6))
        fps = np.mean(fps_history)

        # Dibujar resultados
        weapon_detected = len(boxes) > 0
        frame_display = draw_detections(frame, boxes, scores, labels_ids, id2label)
        frame_display = draw_hud(frame_display, fps, weapon_detected, conf_thresh)

        # Alertas
        if weapon_detected:
            class_names = [id2label.get(int(lid), "weapon") for lid in labels_ids]
            alerts.trigger(class_names, frame, frame_idx)

        cv2.imshow("Detección de Armas — RT-DETR | 'q' para salir", frame_display)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    alerts.close()
    logger.info("=== Sesión finalizada ===")


if __name__ == "__main__":
    main()
