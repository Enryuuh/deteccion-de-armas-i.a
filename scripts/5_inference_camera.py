"""
Script 5: Inferencia en tiempo real desde camara
==================================================
Modo laptop: usa los pesos .pt directamente con ultralytics.
Para Raspberry Pi 5 usar scripts/7_inference_pi.py (ONNX Runtime).

Uso:
    python scripts/5_inference_camera.py
"""

import logging
import sys
import time
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.visualization import draw_detections, draw_hud
from utils.alerts import AlertSystem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    from ultralytics import YOLO

    weights = Path(cfg["inference"]["weights"])
    if not weights.exists():
        raise FileNotFoundError(f"Pesos no encontrados: {weights}. Entrena primero.")

    model = YOLO(str(weights))
    id2label = {i: n for i, n in enumerate(cfg["classes"])}
    conf_th  = cfg["inference"]["confidence_threshold"]
    iou_th   = cfg["inference"]["iou_threshold"]
    imgsz    = cfg["inference"]["imgsz"]
    cam_idx  = cfg["inference"]["camera_index"]

    alerts = AlertSystem(cfg)

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la camara index={cam_idx}")

    logger.info("Camara abierta. ESC o q para salir.")
    frame_idx = 0
    fps_avg = 0.0
    t_prev = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            results = model.predict(
                source = frame,
                imgsz  = imgsz,
                conf   = conf_th,
                iou    = iou_th,
                verbose = False,
            )[0]

            if results.boxes is not None and len(results.boxes) > 0:
                boxes_xyxy = results.boxes.xyxy.cpu().numpy()
                scores     = results.boxes.conf.cpu().numpy()
                cls_ids    = results.boxes.cls.cpu().numpy().astype(int)
            else:
                boxes_xyxy = []
                scores     = []
                cls_ids    = []

            weapon_detected = len(cls_ids) > 0
            frame = draw_detections(frame, boxes_xyxy, scores, cls_ids, id2label)

            t_now = time.time()
            inst_fps = 1.0 / max(t_now - t_prev, 1e-6)
            fps_avg = 0.9 * fps_avg + 0.1 * inst_fps if fps_avg > 0 else inst_fps
            t_prev = t_now

            frame = draw_hud(frame, fps_avg, weapon_detected, conf_th)

            if weapon_detected:
                names = [id2label.get(int(c), "weapon") for c in cls_ids]
                alerts.trigger(names, frame, frame_idx)

            cv2.imshow("Deteccion de Armas", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        alerts.close()
        logger.info("Camara cerrada.")


if __name__ == "__main__":
    main()
