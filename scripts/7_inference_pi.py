"""
Script 7: Inferencia en Raspberry Pi 5 (ONNX Runtime)
========================================================
Pensado para correr en la Pi5 SIN PyTorch ni ultralytics.
Solo necesita: onnxruntime, opencv-python, numpy, pyyaml, pygame.

Uso (en la Pi):
    python scripts/7_inference_pi.py --weights models/export/best.onnx
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.visualization import draw_detections, draw_hud
from utils.alerts import AlertSystem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def letterbox(img: np.ndarray, new_size: int = 416, color=(114, 114, 114)):
    """Resize manteniendo aspect ratio + padding (formato YOLO)."""
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top  = (new_size - nh) // 2
    left = (new_size - nw) // 2
    out = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, r, left, top


def nms(boxes: np.ndarray, scores: np.ndarray, iou_th: float) -> list:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_th]
    return keep


def postprocess(output: np.ndarray, ratio: float, pad_x: int, pad_y: int,
                conf_th: float, iou_th: float, num_classes: int):
    """
    output shape esperada de YOLOv8 ONNX: [1, 4+num_classes, N]
    Devuelve boxes_xyxy (sobre frame original), scores, class_ids.
    """
    pred = output[0]                         # (4+nc, N)
    pred = pred.transpose(1, 0)              # (N, 4+nc)
    boxes_cxcywh = pred[:, :4]
    cls_scores = pred[:, 4:4 + num_classes]
    cls_ids = cls_scores.argmax(axis=1)
    confs = cls_scores.max(axis=1)

    mask = confs >= conf_th
    boxes_cxcywh = boxes_cxcywh[mask]
    confs = confs[mask]
    cls_ids = cls_ids[mask]
    if len(boxes_cxcywh) == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

    cx, cy, bw, bh = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # Quitar padding y desescalar
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes /= ratio

    keep = nms(boxes, confs, iou_th)
    return boxes[keep], confs[keep], cls_ids[keep]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None,
                        help="Ruta al .onnx. Por defecto: models/export/best.onnx")
    parser.add_argument("--source", default=None,
                        help="Indice de camara (int) o ruta de video. Por defecto config.")
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        logger.error("Instala onnxruntime: pip install onnxruntime")
        raise

    cfg = load_config()
    weights = Path(args.weights or (Path(cfg["export"]["output_dir"]) / "best.onnx"))
    if not weights.exists():
        raise FileNotFoundError(f"No existe {weights}. Ejecuta scripts/6_export.py")

    classes  = cfg["classes"]
    id2label = {i: n for i, n in enumerate(classes)}
    conf_th  = cfg["inference"]["confidence_threshold"]
    iou_th   = cfg["inference"]["iou_threshold"]
    imgsz    = cfg["export"]["imgsz"]

    if args.source is None:
        src = cfg["inference"]["camera_index"]
    else:
        src = int(args.source) if args.source.isdigit() else args.source

    logger.info(f"Modelo: {weights}")
    logger.info(f"Providers: {ort.get_available_providers()}")
    sess = ort.InferenceSession(str(weights), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    alerts = AlertSystem(cfg)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente: {src}")

    logger.info("Inferencia iniciada. ESC o q para salir.")
    frame_idx = 0
    fps_avg = 0.0
    t_prev = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            img, ratio, pad_x, pad_y = letterbox(frame, imgsz)
            blob = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
            blob = np.ascontiguousarray(blob)[None, ...]

            outputs = sess.run(None, {in_name: blob})
            boxes, scores, cls_ids = postprocess(
                outputs[0], ratio, pad_x, pad_y, conf_th, iou_th, len(classes)
            )

            weapon_detected = len(cls_ids) > 0
            frame = draw_detections(frame, boxes, scores, cls_ids, id2label)

            t_now = time.time()
            inst_fps = 1.0 / max(t_now - t_prev, 1e-6)
            fps_avg = 0.9 * fps_avg + 0.1 * inst_fps if fps_avg > 0 else inst_fps
            t_prev = t_now

            frame = draw_hud(frame, fps_avg, weapon_detected, conf_th)

            if weapon_detected:
                names = [id2label.get(int(c), "weapon") for c in cls_ids]
                alerts.trigger(names, frame, frame_idx)

            cv2.imshow("Deteccion de Armas (Pi)", frame)
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
