"""
Script 5: Inferencia en tiempo real desde camara
==================================================
Modo laptop: usa los pesos .pt directamente con ultralytics.
Para Raspberry Pi 5 usar scripts/7_inference_pi.py (ONNX Runtime).

Features:
  - ByteTrack persistent tracking
  - Filtros plausibilidad + temporal
  - Dashboard web en http://0.0.0.0:8080 (monitoreo desde celular)
  - Alertas multi-canal (sonido + log + JSON + video clips + Telegram)

Uso:
    python scripts/5_inference_camera.py
    python scripts/5_inference_camera.py --no-dashboard
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.visualization import draw_detections, draw_hud, draw_faces
from utils.alerts import AlertSystem
from utils.temporal_filter import TemporalDetectionFilter
from utils.plausibility_filter import PlausibilityFilter
from utils.face_recognition import FaceRecognizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Deteccion de armas - camara")
    parser.add_argument("--no-dashboard", action="store_true", help="Desactivar dashboard web")
    parser.add_argument("--weights", default=None, help="Ruta al modelo .pt (override config)")
    parser.add_argument("--cam", type=int, default=None, help="Indice de camara (override config)")
    args = parser.parse_args()

    cfg = load_config()
    from ultralytics import YOLO

    weights = Path(args.weights) if args.weights else Path(cfg["inference"]["weights"])
    if not weights.exists():
        raise FileNotFoundError(f"Pesos no encontrados: {weights}. Entrena primero.")

    model = YOLO(str(weights))
    logger.info("Modelo cargado: %s", weights.name)
    id2label = {i: n for i, n in enumerate(cfg["classes"])}
    conf_th  = cfg["inference"]["confidence_threshold"]
    iou_th   = cfg["inference"]["iou_threshold"]
    imgsz    = cfg["inference"]["imgsz"]
    cam_idx  = args.cam if args.cam is not None else cfg["inference"]["camera_index"]

    alerts = AlertSystem(cfg)

    # ByteTrack ya hace voting+smoothing, asi que el filtro temporal lo relajo
    tf = TemporalDetectionFilter(window=3, min_hits=2, smooth_alpha=0.40)
    # Filtro de plausibilidad: umbral por clase + geometria de caja
    pf = PlausibilityFilter(cfg, id2label)

    # Reconocimiento facial (identificacion de personas)
    fcfg = cfg.get("face_recognition", {}) or {}
    face_only_on_weapon = bool(fcfg.get("only_on_weapon", False))
    face_every = max(1, int(fcfg.get("recognize_every", 5)))
    try:
        face_rec = FaceRecognizer(cfg)
    except Exception as e:
        logger.warning(f"Reconocimiento facial desactivado: {e}")
        face_rec = None
    if face_rec and face_rec.enabled:
        logger.info("Reconocimiento facial ACTIVO (%d personas matriculadas)",
                    len(face_rec.db_names))
    last_faces: list = []

    # Dashboard web
    dashboard_fn = None
    dash_cfg = cfg.get("dashboard", {})
    if dash_cfg.get("enabled", False) and not args.no_dashboard:
        try:
            from utils.web_dashboard import start_dashboard, update_dashboard
            start_dashboard(
                port=dash_cfg.get("port", 8080),
                fps_limit=dash_cfg.get("fps_limit", 15),
            )
            dashboard_fn = update_dashboard
        except Exception as e:
            logger.warning(f"Dashboard no disponible: {e}")

    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la camara index={cam_idx}")

    # Pedir resolucion HD (la camara cae a la mas cercana si no la soporta)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Camara abierta a {real_w}x{real_h}. ESC o q para salir.")
    if dashboard_fn:
        logger.info(f"Dashboard: http://localhost:{dash_cfg.get('port', 8080)}")

    cv2.namedWindow("Deteccion de Armas", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Deteccion de Armas", real_w, real_h)
    frame_idx = 0
    fps_avg = 0.0
    t_prev = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            # Ring buffer para video clips (ANTES de dibujar)
            alerts.push_frame(frame)

            # ByteTrack: tracker integrado de YOLOv8
            results = model.track(
                source  = frame,
                imgsz   = imgsz,
                conf    = conf_th,
                iou     = iou_th,
                tracker = "bytetrack.yaml",
                persist = True,
                verbose = False,
            )[0]

            track_ids: list = []
            if results.boxes is not None and len(results.boxes) > 0:
                raw_boxes   = results.boxes.xyxy.cpu().numpy()
                raw_scores  = results.boxes.conf.cpu().numpy()
                raw_cls_ids = results.boxes.cls.cpu().numpy().astype(int)
                if results.boxes.id is not None:
                    track_ids = results.boxes.id.cpu().numpy().astype(int).tolist()
                else:
                    track_ids = [-1] * len(raw_boxes)
            else:
                raw_boxes = raw_scores = raw_cls_ids = []

            h_frame, w_frame = frame.shape[:2]

            # 1. Filtro de plausibilidad
            if len(raw_boxes) > 0:
                np_boxes  = np.array(raw_boxes)
                np_scores = np.array(raw_scores)
                np_cls    = np.array(raw_cls_ids)
                pf_boxes, pf_scores, pf_cls = pf.filter(np_boxes, np_scores, np_cls, h_frame, w_frame)

                kept_ids = []
                for i, (box, score, cid) in enumerate(zip(np_boxes, np_scores, np_cls)):
                    label = id2label.get(int(cid), "")
                    th    = pf.class_thresholds.get(label, 0.50)
                    if float(score) < th:
                        continue
                    if not pf._geometry_ok(box, label, h_frame, w_frame):
                        continue
                    kept_ids.append(track_ids[i] if i < len(track_ids) else -1)
            else:
                pf_boxes = pf_scores = pf_cls = np.array([])
                kept_ids = []

            # 2. Filtro temporal
            boxes_xyxy, scores, cls_ids = tf.update(pf_boxes, pf_scores, pf_cls)
            display_ids = kept_ids[:len(boxes_xyxy)] if kept_ids else [-1] * len(boxes_xyxy)

            weapon_detected = len(cls_ids) > 0

            # ── Reconocimiento facial (sobre el frame limpio, antes de dibujar armas) ──
            persons = None
            if face_rec and face_rec.enabled:
                run_face = (
                    (weapon_detected if face_only_on_weapon else True)
                    and (frame_idx % face_every == 0 or weapon_detected)
                )
                if run_face:
                    last_faces = face_rec.recognize(frame)
                if last_faces:
                    frame = draw_faces(frame, last_faces)
                    known = [f.name for f in last_faces if f.is_known]
                    if weapon_detected:
                        persons = known if known else ["Desconocido"]

            frame = draw_detections(frame, boxes_xyxy, scores, cls_ids, id2label, display_ids)

            t_now = time.time()
            inst_fps = 1.0 / max(t_now - t_prev, 1e-6)
            fps_avg = 0.9 * fps_avg + 0.1 * inst_fps if fps_avg > 0 else inst_fps
            t_prev = t_now

            names = [id2label.get(int(c), "weapon") for c in cls_ids] if weapon_detected else None
            frame = draw_hud(frame, fps_avg, weapon_detected, conf_th, names)

            if weapon_detected:
                alerts.trigger(names, frame, frame_idx, persons)

            # Dashboard web
            if dashboard_fn:
                dashboard_fn(frame, fps_avg, weapon_detected, names)

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
