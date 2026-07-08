"""
Script 11: Captura frames de la cámara para fine-tuning
==========================================================
Graba frames de tu cámara cada N segundos (o pulsando SPACE)
para que después los etiquetes manualmente y se usen para
fine-tunear el modelo en TU escena específica.

Salida: data/finetune_raw/cap_YYYYMMDD_HHMMSS.jpg

Controles en la ventana:
    SPACE  → captura inmediata
    a      → toggle captura automática (cada AUTO_INTERVAL_SEC)
    +/-    → ajusta intervalo automático
    q/ESC  → salir

Uso:
    python scripts/11_capture_finetune_frames.py
"""

import logging
import time
from datetime import datetime
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO        = Path(__file__).resolve().parents[1]
OUT_DIR     = REPO / "data" / "finetune_raw"
CAM_INDEX   = 0
AUTO_INTERVAL_SEC = 2.0


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara index={CAM_INDEX}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    cv2.namedWindow("Captura para Fine-tuning", cv2.WINDOW_NORMAL)

    auto = False
    interval = AUTO_INTERVAL_SEC
    last_auto = 0.0
    captured = len(list(OUT_DIR.glob("cap_*.jpg")))

    logger.info(f"Salida: {OUT_DIR}")
    logger.info(f"Frames previos: {captured}. Controles: SPACE=captura, a=auto, +/-=intervalo, q=salir")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Captura automática
            now = time.time()
            do_capture = False
            if auto and (now - last_auto) >= interval:
                do_capture = True
                last_auto = now

            display = frame.copy()
            h, w = display.shape[:2]

            # Overlay informativo
            cv2.rectangle(display, (0, 0), (340, 70), (0, 0, 0), -1)
            cv2.putText(display, f"Capturados: {captured}",                (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            modo_txt = f"AUTO ({interval:.1f}s)" if auto else "MANUAL"
            modo_color = (0, 255, 0) if auto else (0, 200, 255)
            cv2.putText(display, f"Modo: {modo_txt}",                       (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, modo_color, 1)

            # Indicador de captura (flash blanco)
            if do_capture:
                cv2.rectangle(display, (0, 0), (w-1, h-1), (255, 255, 255), 8)

            cv2.imshow("Captura para Fine-tuning", display)

            if do_capture:
                fname = OUT_DIR / f"cap_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                captured += 1
                logger.info(f"  Guardado #{captured}: {fname.name}")

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord(" "):
                fname = OUT_DIR / f"cap_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                captured += 1
                logger.info(f"  Guardado #{captured}: {fname.name}")
            elif key == ord("a"):
                auto = not auto
                logger.info(f"  Modo automático: {'ON' if auto else 'OFF'}")
            elif key == ord("+") or key == ord("="):
                interval = max(0.5, interval - 0.5)
            elif key == ord("-") or key == ord("_"):
                interval = min(10.0, interval + 0.5)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info(f"Total capturado en esta sesión: ver {OUT_DIR}")


if __name__ == "__main__":
    main()
