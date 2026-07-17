"""
utils/visualization.py
=======================
HUD minimalista + bounding boxes con etiquetas. Optimizado para correr
a 30+ FPS sin copias innecesarias del frame.
"""

import cv2
import numpy as np

# Paleta por clase (BGR)
CLASS_COLORS = {
    "knife":    (50, 180, 255),   # naranja
    "handgun":  (60,  80, 255),   # rojo claro
    "long_gun": (40,  40, 220),   # rojo oscuro
    "firearm":  (60,  80, 255),
}
DEFAULT_COLOR = (0, 200, 255)
ALERT_COLOR   = (45, 45, 230)
SAFE_COLOR    = (90, 200, 110)
TEXT_COLOR    = (245, 245, 245)
TEXT_DIM      = (170, 170, 170)
HUD_BG        = (25, 25, 30)

FONT      = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX


def _put_text(img, text, org, scale=0.55, color=TEXT_COLOR, thick=1, font=FONT):
    """Texto con sombra sutil para legibilidad sobre cualquier fondo."""
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y),         font, scale, color,    thick,     cv2.LINE_AA)


def _alpha_rect(img, p1, p2, color, alpha):
    """Rectángulo semi-transparente in-place sin copia completa del frame."""
    x1, y1 = p1
    x2, y2 = p2
    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


def draw_detections(frame, boxes, scores, label_ids, id2label, track_ids=None):
    """Dibuja cajas con esquinas decorativas, etiquetas y track ID si existe."""
    if len(boxes) == 0:
        return frame

    if track_ids is None:
        track_ids = [-1] * len(boxes)

    for box, score, lid, tid in zip(boxes, scores, label_ids, track_ids):
        class_name = id2label.get(int(lid), "weapon")
        color      = CLASS_COLORS.get(class_name, DEFAULT_COLOR)
        x1, y1, x2, y2 = map(int, box)

        # Caja principal (línea fina)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Esquinas decorativas (gruesas)
        corner = max(8, min(18, (x2 - x1) // 5))
        for cx, cy, dx, dy in [
            (x1, y1,  1,  1),
            (x2, y1, -1,  1),
            (x1, y2,  1, -1),
            (x2, y2, -1, -1),
        ]:
            cv2.line(frame, (cx, cy), (cx + dx * corner, cy), color, 4, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy * corner), color, 4, cv2.LINE_AA)

        # Etiqueta con fondo (incluye track ID si existe)
        if int(tid) > 0:
            label = f"#{int(tid)} {class_name}  {score:.0%}"
        else:
            label = f"{class_name}  {score:.0%}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
        pad = 6
        ly1 = max(y1 - th - pad * 2 - 2, 0)
        lx2 = x1 + tw + pad * 2
        cv2.rectangle(frame, (x1, ly1), (lx2, ly1 + th + pad * 2), color, -1, cv2.LINE_AA)
        cv2.putText(frame, label, (x1 + pad, ly1 + th + pad), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


FACE_KNOWN_COLOR   = (90, 200, 110)   # verde  (persona identificada)
FACE_UNKNOWN_COLOR = (0, 170, 255)    # naranja (desconocido)


def draw_faces(frame, faces):
    """
    Dibuja las caras reconocidas con su nombre.
    `faces` es una lista de objetos con .name, .score, .bbox y .is_known
    (FaceResult de utils.face_recognition).
    """
    if not faces:
        return frame

    for f in faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        color = FACE_KNOWN_COLOR if f.is_known else FACE_UNKNOWN_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        label = f"{f.name}  {f.score:.0%}" if f.is_known else f.name
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        pad = 5
        ly2 = min(y2 + th + pad * 2 + 2, frame.shape[0] - 1)
        cv2.rectangle(frame, (x1, y2), (x1 + tw + pad * 2, ly2), color, -1, cv2.LINE_AA)
        cv2.putText(frame, label, (x1 + pad, ly2 - pad), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_hud(frame, fps, weapon_detected, conf_thresh, class_names=None):
    """
    HUD compacto en esquina superior izquierda + indicador de alerta arriba derecha.
    """
    h, w = frame.shape[:2]

    # ── Borde rojo de alerta cuando hay arma (mucho menos invasivo que tinte de pantalla)
    if weapon_detected:
        thick = 6
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), ALERT_COLOR, thick, cv2.LINE_AA)

    # ── Panel HUD esquina superior izquierda
    pw, ph = 200, 78
    _alpha_rect(frame, (0, 0), (pw, ph), HUD_BG, 0.72)
    cv2.rectangle(frame, (0, 0), (pw, ph), (60, 60, 70), 1, cv2.LINE_AA)

    # FPS y umbral
    _put_text(frame, f"FPS  {fps:5.1f}",          (12, 22), 0.5, TEXT_DIM, 1, FONT_SMALL)
    _put_text(frame, f"Umbral  {conf_thresh:.0%}", (12, 42), 0.5, TEXT_DIM, 1, FONT_SMALL)

    # Estado: verde / rojo
    status_color = ALERT_COLOR if weapon_detected else SAFE_COLOR
    cv2.circle(frame, (20, 64), 5, status_color, -1, cv2.LINE_AA)
    status_txt = "ALERTA" if weapon_detected else "SEGURO"
    _put_text(frame, status_txt, (32, 69), 0.55, status_color, 1, FONT)

    # ── Banner compacto top-right cuando hay alerta
    if weapon_detected and class_names:
        # Contar por clase
        counts = {}
        for n in class_names:
            counts[n] = counts.get(n, 0) + 1
        summary = "  ".join(f"{c} x{n}" if n > 1 else c for c, n in counts.items())
        text = f"⚠  {summary.upper()}"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 1)
        pad = 10
        bx1 = w - tw - pad * 2 - 12
        bx2 = w - 12
        by1 = 12
        by2 = by1 + th + pad * 2
        _alpha_rect(frame, (bx1, by1), (bx2, by2), ALERT_COLOR, 0.85)
        _put_text(frame, text, (bx1 + pad, by1 + th + pad), 0.6, (255, 255, 255), 1, FONT)

    return frame
