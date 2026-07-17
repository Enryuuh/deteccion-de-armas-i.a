"""
Script 7: Inferencia MEGA-OPTIMIZADA en Raspberry Pi 5 (ONNX Runtime)
========================================================================
Pensado para correr en la Pi5 SIN PyTorch ni ultralytics.
Solo necesita: onnxruntime, opencv-python-headless, numpy, pyyaml, pygame.

Optimizaciones Pi5:
  - ONNX INT8 (~3 MB) con onnxruntime CPU
  - imgsz 416 (4x menos pixels que 640)
  - Skip-frame: inferencia cada N frames, interpolación entre medias
  - Filtros plausibilidad + temporal para eliminar falsos positivos
  - Threading: captura de cámara en hilo separado
  - Pre-allocated buffers (zero-copy donde posible)

Uso (en la Pi):
    python scripts/7_inference_pi.py
    python scripts/7_inference_pi.py --weights models/export/yolov8n_v2_int8.onnx
    python scripts/7_inference_pi.py --skip 2 --imgsz 320   # ultra-fast mode
"""

import argparse
import logging
import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np
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


# ─── Threaded camera capture (evita bloqueo por I/O de cámara) ───

class CameraThread:
    """Captura frames en hilo separado para no bloquear inferencia."""
    def __init__(self, src, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # mínimo buffer -> menos latencia
        self.ok = False
        self.frame = None
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            with self.lock:
                self.ok = ok
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ok, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.stopped = True
        self.thread.join(timeout=2)
        self.cap.release()

    @property
    def isOpened(self):
        return self.cap.isOpened()


# ─── Pre-processing optimizado ───

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


def preprocess(img: np.ndarray, imgsz: int):
    """Letterbox + BGR->RGB + HWC->NCHW + float32 normalizado."""
    lb, ratio, pad_x, pad_y = letterbox(img, imgsz)
    blob = lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.ascontiguousarray(blob)[None, ...]
    return blob, ratio, pad_x, pad_y


# ─── Post-processing ───

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
    output shape: [1, 4+num_classes, N] (YOLOv8 ONNX format)
    Devuelve boxes_xyxy (coords originales), scores, class_ids.
    """
    pred = output[0].transpose(1, 0)       # (N, 4+nc)
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
    parser = argparse.ArgumentParser(description="Inferencia optimizada Pi 5")
    parser.add_argument("--weights", default=None,
                        help="Ruta al .onnx (default: models/export/yolov8n_v2_int8.onnx)")
    parser.add_argument("--source", default=None,
                        help="Indice de camara o ruta de video")
    parser.add_argument("--skip", type=int, default=1,
                        help="Inferir cada N frames (1=todos, 2=cada 2, etc.)")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Tamaño de entrada (default: config.yaml export.imgsz)")
    parser.add_argument("--width", type=int, default=640,
                        help="Ancho de captura de cámara")
    parser.add_argument("--height", type=int, default=480,
                        help="Alto de captura de cámara")
    parser.add_argument("--no-display", action="store_true",
                        help="Headless mode (sin ventana, solo alertas)")
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        logger.error("Instala onnxruntime: pip install onnxruntime")
        raise

    cfg = load_config()

    # Pesos: prioridad INT8 > FP32 > argumento
    export_dir = Path(cfg["export"]["output_dir"])
    if args.weights:
        weights = Path(args.weights)
    elif (export_dir / "yolov8n_v4_int8.onnx").exists():
        weights = export_dir / "yolov8n_v4_int8.onnx"
    elif (export_dir / "yolov8n_v4_fp32.onnx").exists():
        weights = export_dir / "yolov8n_v4_fp32.onnx"
    elif (export_dir / "yolov8n_v2_int8.onnx").exists():
        weights = export_dir / "yolov8n_v2_int8.onnx"
    else:
        weights = export_dir / "best.onnx"

    if not weights.exists():
        raise FileNotFoundError(f"No existe {weights}. Ejecuta scripts/6b_export_int8.py")

    classes  = cfg["classes"]
    id2label = {i: n for i, n in enumerate(classes)}
    conf_th  = cfg["inference"]["confidence_threshold"]
    iou_th   = cfg["inference"]["iou_threshold"]
    imgsz    = args.imgsz or cfg["export"]["imgsz"]

    if args.source is None:
        src = cfg["inference"]["camera_index"]
    else:
        src = int(args.source) if args.source.isdigit() else args.source

    # ── Sesión ONNX optimizada ──
    logger.info(f"Modelo: {weights} ({weights.stat().st_size / 1024:.0f} KB)")
    logger.info(f"Providers: {ort.get_available_providers()}")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = 4       # Pi5 tiene 4 cores Cortex-A76
    sess_opts.inter_op_num_threads = 1
    sess_opts.enable_mem_pattern = True
    sess_opts.enable_cpu_mem_arena = True

    sess = ort.InferenceSession(str(weights), sess_opts, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    # ── Filtros (mismos que laptop) ──
    tf = TemporalDetectionFilter(window=3, min_hits=2, smooth_alpha=0.40)
    pf = PlausibilityFilter(cfg, id2label)
    alerts = AlertSystem(cfg)

    # Reconocimiento facial (en la Pi: solo al detectar arma, para ahorrar CPU)
    try:
        face_rec = FaceRecognizer(cfg)
    except Exception as e:
        logger.warning(f"Reconocimiento facial desactivado: {e}")
        face_rec = None
    if face_rec and face_rec.enabled:
        logger.info("Reconocimiento facial ACTIVO (%d personas matriculadas)",
                    len(face_rec.db_names))

    # ── Cámara con threading ──
    logger.info(f"Abriendo cámara {src} a {args.width}x{args.height}...")
    cam = CameraThread(src, args.width, args.height)
    if not cam.isOpened:
        raise RuntimeError(f"No se pudo abrir la cámara: {src}")

    skip_n = max(1, args.skip)
    logger.info(f"imgsz={imgsz} | skip={skip_n} | display={'OFF' if args.no_display else 'ON'}")
    logger.info("Inferencia iniciada. ESC o q para salir.")

    if not args.no_display:
        cv2.namedWindow("Deteccion de Armas (Pi5)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Deteccion de Armas (Pi5)", args.width, args.height)

    frame_idx = 0
    fps_avg = 0.0
    t_prev = time.time()

    # Cache de última detección (para skip-frame)
    last_boxes = np.empty((0, 4))
    last_scores = np.empty((0,))
    last_cls_ids = np.empty((0,), dtype=int)

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            frame_idx += 1

            # ── Skip-frame: solo inferir cada N frames ──
            if frame_idx % skip_n == 0:
                blob, ratio, pad_x, pad_y = preprocess(frame, imgsz)
                outputs = sess.run(None, {in_name: blob})
                raw_boxes, raw_scores, raw_cls_ids = postprocess(
                    outputs[0], ratio, pad_x, pad_y, conf_th, iou_th, len(classes)
                )

                # Filtro de plausibilidad
                h_frame, w_frame = frame.shape[:2]
                if len(raw_boxes) > 0:
                    pf_boxes, pf_scores, pf_cls = pf.filter(
                        raw_boxes, raw_scores, raw_cls_ids, h_frame, w_frame
                    )
                else:
                    pf_boxes = pf_scores = pf_cls = np.array([])

                # Filtro temporal
                last_boxes, last_scores, last_cls_ids = tf.update(pf_boxes, pf_scores, pf_cls)

            # ── Dibujar (usa última detección para frames skipped) ──
            weapon_detected = len(last_cls_ids) > 0

            # ── Reconocimiento facial: solo cuando hay arma (ahorra CPU en la Pi) ──
            persons = None
            if face_rec and face_rec.enabled and weapon_detected:
                faces = face_rec.recognize(frame)
                if faces:
                    frame = draw_faces(frame, faces)
                    known = [f.name for f in faces if f.is_known]
                    persons = known if known else ["Desconocido"]

            frame = draw_detections(frame, last_boxes, last_scores, last_cls_ids, id2label)

            t_now = time.time()
            inst_fps = 1.0 / max(t_now - t_prev, 1e-6)
            fps_avg = 0.9 * fps_avg + 0.1 * inst_fps if fps_avg > 0 else inst_fps
            t_prev = t_now

            names = [id2label.get(int(c), "weapon") for c in last_cls_ids] if weapon_detected else None
            frame = draw_hud(frame, fps_avg, weapon_detected, conf_th, names)

            if weapon_detected:
                alerts.trigger(names, frame, frame_idx, persons)

            if not args.no_display:
                cv2.imshow("Deteccion de Armas (Pi5)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
            else:
                # Headless: salir con Ctrl+C
                pass

    except KeyboardInterrupt:
        logger.info("Interrumpido por usuario.")
    finally:
        cam.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        alerts.close()
        logger.info("Camara cerrada.")


if __name__ == "__main__":
    main()
