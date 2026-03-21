"""
utils/visualization.py
=======================
Funciones para dibujar bounding boxes, etiquetas y HUD
sobre los frames de la cámara.
"""

import cv2
import numpy as np

# Paleta de colores por clase (BGR)
CLASS_COLORS = {
    "knife":    (0, 60, 255),    # rojo
    "firearm":  (0, 0, 220),     # rojo más oscuro
}
DEFAULT_COLOR = (0, 200, 255)

ALERT_COLOR   = (0, 0, 255)      # rojo para overlay de alerta
SAFE_COLOR    = (0, 200, 80)     # verde cuando no hay arma
HUD_BG_COLOR  = (20, 20, 20)

FONT          = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE    = 0.65
LINE_THICK    = 2


def draw_detections(
    frame: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    label_ids: np.ndarray,
    id2label: dict,
) -> np.ndarray:
    """
    Dibuja bounding boxes y etiquetas sobre el frame.
    boxes: array [N, 4] en formato [x1, y1, x2, y2]
    """
    output = frame.copy()

    for box, score, lid in zip(boxes, scores, label_ids):
        class_name = id2label.get(int(lid), "weapon")
        color      = CLASS_COLORS.get(class_name, DEFAULT_COLOR)

        x1, y1, x2, y2 = map(int, box)

        # Rectángulo principal
        cv2.rectangle(output, (x1, y1), (x2, y2), color, LINE_THICK)

        # Esquinas decorativas
        corner_len = max(10, min(20, (x2 - x1) // 4))
        for cx, cy, dx, dy in [
            (x1, y1,  1,  1),
            (x2, y1, -1,  1),
            (x1, y2,  1, -1),
            (x2, y2, -1, -1),
        ]:
            cv2.line(output, (cx, cy), (cx + dx * corner_len, cy), color, 3)
            cv2.line(output, (cx, cy), (cx, cy + dy * corner_len), color, 3)

        # Etiqueta con fondo
        label_text = f"{class_name}  {score:.0%}"
        (tw, th), baseline = cv2.getTextSize(label_text, FONT, FONT_SCALE, 1)
        tag_y = max(y1 - 6, th + 6)
        cv2.rectangle(output, (x1, tag_y - th - baseline - 4), (x1 + tw + 6, tag_y + 2), color, -1)
        cv2.putText(output, label_text, (x1 + 3, tag_y - baseline), FONT, FONT_SCALE, (255, 255, 255), 1)

    return output


def draw_hud(
    frame: np.ndarray,
    fps: float,
    weapon_detected: bool,
    conf_thresh: float,
) -> np.ndarray:
    """
    Dibuja el HUD informativo en la esquina superior izquierda
    y un overlay rojo semi-transparente si hay arma detectada.
    """
    output = frame.copy()
    h, w   = output.shape[:2]

    # Overlay rojo al detectar arma
    if weapon_detected:
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), ALERT_COLOR, -1)
        cv2.addWeighted(overlay, 0.12, output, 0.88, 0, output)

        # Banner de alerta
        alert_text = "⚠  ARMA DETECTADA"
        (tw, th), _ = cv2.getTextSize(alert_text, FONT, 1.1, 2)
        ax = (w - tw) // 2
        ay = 55
        cv2.rectangle(output, (ax - 12, ay - th - 10), (ax + tw + 12, ay + 8), ALERT_COLOR, -1)
        cv2.putText(output, alert_text, (ax, ay), FONT, 1.1, (255, 255, 255), 2)

    # HUD background
    hud_h = 90
    cv2.rectangle(output, (0, 0), (240, hud_h), HUD_BG_COLOR, -1)
    cv2.addWeighted(output[:hud_h, :240], 0.0, frame[:hud_h, :240], 0.0, 0, output[:hud_h, :240])
    cv2.rectangle(output, (0, 0), (240, hud_h), (50, 50, 50), 1)

    status_text  = "ARMA" if weapon_detected else "SEGURO"
    status_color = ALERT_COLOR if weapon_detected else SAFE_COLOR

    cv2.putText(output, f"FPS:    {fps:5.1f}",         (10, 22), FONT, 0.55, (200, 200, 200), 1)
    cv2.putText(output, f"Umbral: {conf_thresh:.0%}",  (10, 44), FONT, 0.55, (200, 200, 200), 1)
    cv2.putText(output, f"Estado: {status_text}",      (10, 70), FONT, 0.65, status_color, 1)

    return output
