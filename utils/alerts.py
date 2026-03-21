"""
utils/alerts.py
===============
Sistema de alertas cuando se detecta un arma:
  - Sonido (pygame)
  - Log de detecciones a archivo
  - Guardado de frames de evidencia
"""

import logging
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2

logger = logging.getLogger(__name__)


class AlertSystem:
    """
    Gestiona todas las alertas del sistema de detección.
    Respeta un cooldown para no saturar alertas repetidas.
    """

    def __init__(self, cfg: dict, disabled: bool = False):
        self.disabled = disabled
        if disabled:
            return

        acfg = cfg.get("alerts", {})
        icfg = cfg.get("inference", {})

        self.sound_enabled  = acfg.get("sound_enabled", True)
        self.log_enabled    = acfg.get("log_detections", True)
        self.save_frames    = acfg.get("save_frames", True)
        self.cooldown       = icfg.get("alert_cooldown_seconds", 5)
        self.last_alert_t   = 0.0

        # Directorios de salida
        self.log_path   = Path(acfg.get("log_file",   "logs/detections.log"))
        self.frames_dir = Path(acfg.get("frames_dir", "logs/frames"))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        # Sonido
        self._sound = None
        if self.sound_enabled:
            self._init_sound(acfg.get("sound_file", "assets/alert.wav"))

    def _init_sound(self, sound_file: str):
        try:
            import pygame
            pygame.mixer.init()
            fp = Path(sound_file)
            if fp.exists():
                self._sound = pygame.mixer.Sound(str(fp))
                logger.info(f"Sonido cargado: {fp}")
            else:
                # Generar beep sintético si no hay archivo
                sample_rate  = 44100
                duration     = 0.4
                freq         = 880
                t            = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
                wave         = (32767 * 0.7 * np.sin(2 * np.pi * freq * t)).astype(np.int16)
                stereo_wave  = np.column_stack([wave, wave])
                self._sound  = pygame.sndarray.make_sound(stereo_wave)
                logger.info("Sonido sintético generado (no se encontró assets/alert.wav)")
        except Exception as e:
            logger.warning(f"No se pudo inicializar sonido: {e}")
            self._sound = None

    def _should_fire(self) -> bool:
        return (time.time() - self.last_alert_t) >= self.cooldown

    def trigger(self, class_names: list, frame: np.ndarray, frame_idx: int):
        """Dispara la alerta si no está en cooldown."""
        if self.disabled or not self._should_fire():
            return

        self.last_alert_t = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        classes_str = ", ".join(set(class_names))

        # Sonido
        if self._sound:
            try:
                self._sound.play()
            except Exception:
                pass

        # Log
        if self.log_enabled:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp}]  Frame #{frame_idx:06d}  |  Detectado: {classes_str}\n")
            except Exception as e:
                logger.warning(f"Error al escribir log: {e}")

        # Guardar frame
        if self.save_frames and frame is not None:
            try:
                fname = self.frames_dir / f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            except Exception as e:
                logger.warning(f"Error al guardar frame: {e}")

        logger.warning(f"🚨 ALERTA  [{timestamp}]  Detectado: {classes_str}")

    def close(self):
        if self.disabled:
            return
        try:
            import pygame
            pygame.mixer.quit()
        except Exception:
            pass
