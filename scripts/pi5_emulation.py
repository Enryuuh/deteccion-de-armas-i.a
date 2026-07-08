"""
Emulacion Pi5: corre el modelo con PyTorch CPU + 4 threads.
"""
import sys
import time
from pathlib import Path

import torch

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.visualization import draw_detections, draw_hud
from utils.plausibility_filter import PlausibilityFilter
from utils.temporal_filter import TemporalDetectionFilter

from ultralytics import YOLO


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    weights = Path(cfg["inference"]["weights"])
    model = YOLO(str(weights))
    # Usar GPU pero imgsz=416 para simular resolución de la Pi

    id2label = {i: n for i, n in enumerate(cfg["classes"])}
    conf_th = cfg["inference"]["confidence_threshold"]
    iou_th = cfg["inference"]["iou_threshold"]

    tf = TemporalDetectionFilter(window=3, min_hits=2, smooth_alpha=0.40)
    pf = PlausibilityFilter(cfg, id2label)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"Camera: {int(cap.get(3))}x{int(cap.get(4))}", flush=True)

    cv2.namedWindow("Pi5 Emulation (CPU-only)", cv2.WINDOW_NORMAL)
    fps_avg = 0.0
    t_prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(
            frame, imgsz=416, conf=conf_th, iou=iou_th,
            verbose=False
        )[0]

        if results.boxes is not None and len(results.boxes) > 0:
            raw_boxes = results.boxes.xyxy.cpu().numpy()
            raw_scores = results.boxes.conf.cpu().numpy()
            raw_cls = results.boxes.cls.cpu().numpy().astype(int)
        else:
            raw_boxes = raw_scores = raw_cls = np.array([])

        h, w = frame.shape[:2]
        if len(raw_boxes) > 0:
            pf_b, pf_s, pf_c = pf.filter(raw_boxes, raw_scores, raw_cls, h, w)
        else:
            pf_b = pf_s = pf_c = np.array([])

        boxes, scores, cls_ids = tf.update(pf_b, pf_s, pf_c)
        weapon = len(cls_ids) > 0
        frame = draw_detections(frame, boxes, scores, cls_ids, id2label)

        t_now = time.time()
        inst_fps = 1.0 / max(t_now - t_prev, 1e-6)
        fps_avg = 0.9 * fps_avg + 0.1 * inst_fps if fps_avg > 0 else inst_fps
        t_prev = t_now

        names = [id2label.get(int(c), "weapon") for c in cls_ids] if weapon else None
        frame = draw_hud(frame, fps_avg, weapon, conf_th, names)

        cv2.imshow("Pi5 Emulation (CPU-only)", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done", flush=True)


if __name__ == "__main__":
    main()
