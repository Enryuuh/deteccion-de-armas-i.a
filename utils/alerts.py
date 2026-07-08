"""
utils/alerts.py
===============
Sistema de alertas multi-canal:
  - Sonido (pygame beep)
  - Log texto plano
  - Log JSON estructurado (para análisis posterior)
  - Guardado de frames JPG
  - Video clips (ring buffer pre + post detección)
  - Telegram push (foto + texto)
"""

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2

logger = logging.getLogger(__name__)


class VideoClipRecorder:
    """Ring buffer de frames + grabación post-detección en hilo separado."""

    def __init__(self, pre_seconds: float = 5, post_seconds: float = 10,
                 output_dir: str = "logs/clips", fps: float = 15):
        self.pre_frames = int(pre_seconds * fps)
        self.post_frames = int(post_seconds * fps)
        self.fps = fps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.buffer: deque = deque(maxlen=self.pre_frames)
        self.recording = False
        self.record_frames: list = []
        self.frames_remaining = 0
        self._lock = threading.Lock()

    def push_frame(self, frame: np.ndarray):
        """Llamar cada frame. Mantiene ring buffer y graba si está activo."""
        with self._lock:
            if self.recording:
                self.record_frames.append(frame.copy())
                self.frames_remaining -= 1
                if self.frames_remaining <= 0:
                    self._save_clip()
                    self.recording = False
            else:
                self.buffer.append(frame.copy())

    def start_recording(self, class_names: list):
        """Inicia grabación: vuelca ring buffer + graba post_frames más."""
        with self._lock:
            if self.recording:
                # Ya grabando, extender
                self.frames_remaining = self.post_frames
                return
            self.recording = True
            self.record_frames = list(self.buffer)
            self.buffer.clear()
            self.frames_remaining = self.post_frames
            self._trigger_classes = class_names

    def _save_clip(self):
        """Guarda el clip en un hilo para no bloquear inferencia."""
        frames = self.record_frames
        classes = getattr(self, "_trigger_classes", ["unknown"])
        self.record_frames = []

        if not frames:
            return

        def _write():
            try:
                h, w = frames[0].shape[:2]
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                cls_str = "_".join(set(classes))
                path = self.output_dir / f"clip_{ts}_{cls_str}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(path), fourcc, self.fps, (w, h))
                for f in frames:
                    writer.write(f)
                writer.release()
                logger.info(f"Video clip guardado: {path} ({len(frames)} frames)")
            except Exception as e:
                logger.warning(f"Error guardando clip: {e}")

        threading.Thread(target=_write, daemon=True).start()


class TelegramNotifier:
    """Envía foto + texto por Telegram cuando detecta arma."""

    def __init__(self, bot_token: str, chat_id: str, cooldown: float = 30):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.cooldown = cooldown
        self.last_sent = 0.0
        self._url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"

    def send(self, frame: np.ndarray, class_names: list):
        """Envía notificación si no está en cooldown. No bloquea."""
        now = time.time()
        if (now - self.last_sent) < self.cooldown:
            return
        self.last_sent = now

        def _send():
            try:
                import requests
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                ts = datetime.now().strftime("%H:%M:%S")
                classes_str = ", ".join(set(class_names))
                caption = f"⚠️ ARMA DETECTADA\n{classes_str}\n🕐 {ts}"
                resp = requests.post(
                    self._url,
                    data={"chat_id": self.chat_id, "caption": caption},
                    files={"photo": ("alert.jpg", buf.tobytes(), "image/jpeg")},
                    timeout=10,
                )
                if resp.ok:
                    logger.info(f"Telegram enviado a {self.chat_id}")
                else:
                    logger.warning(f"Telegram error: {resp.status_code} {resp.text[:100]}")
            except Exception as e:
                logger.warning(f"Telegram falló: {e}")

        threading.Thread(target=_send, daemon=True).start()


class AlertSystem:
    """
    Gestiona todas las alertas del sistema de detección.
    Multi-canal: sonido + log + JSON + frames + video clips + Telegram.
    """

    def __init__(self, cfg: dict, disabled: bool = False):
        self.disabled = disabled
        if disabled:
            return

        acfg = cfg.get("alerts", {})
        icfg = cfg.get("inference", {})

        self.sound_enabled  = acfg.get("sound_enabled", True)
        self.log_enabled    = acfg.get("log_detections", True)
        self.json_enabled   = acfg.get("log_json", False)
        self.save_frames    = acfg.get("save_frames", True)
        self.cooldown       = icfg.get("alert_cooldown_seconds", 5)
        self.last_alert_t   = 0.0

        # Directorios de salida
        self.log_path      = Path(acfg.get("log_file", "logs/detections.log"))
        self.json_log_path = Path(acfg.get("log_json_file", "logs/detections.jsonl"))
        self.frames_dir    = Path(acfg.get("frames_dir", "logs/frames"))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        # Sonido
        self._sound = None
        if self.sound_enabled:
            self._init_sound(acfg.get("sound_file", "assets/alert.wav"))

        # Video clips
        self.clip_recorder = None
        if acfg.get("video_clip_enabled", False):
            self.clip_recorder = VideoClipRecorder(
                pre_seconds=acfg.get("video_clip_pre_seconds", 5),
                post_seconds=acfg.get("video_clip_post_seconds", 10),
                output_dir=acfg.get("video_clip_dir", "logs/clips"),
            )
            logger.info("Video clip recorder activado")

        # Telegram
        self.telegram = None
        if acfg.get("telegram_enabled", False):
            token = acfg.get("telegram_bot_token", "")
            chat_id = acfg.get("telegram_chat_id", "")
            if token and chat_id:
                self.telegram = TelegramNotifier(
                    bot_token=token,
                    chat_id=chat_id,
                    cooldown=acfg.get("telegram_cooldown_seconds", 30),
                )
                logger.info(f"Telegram notifier activado -> chat_id={chat_id}")

    def _init_sound(self, sound_file: str):
        try:
            import pygame
            pygame.mixer.init()
            fp = Path(sound_file)
            if fp.exists():
                self._sound = pygame.mixer.Sound(str(fp))
                logger.info(f"Sonido cargado: {fp}")
            else:
                sample_rate  = 44100
                duration     = 0.4
                freq         = 880
                t            = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
                wave         = (32767 * 0.7 * np.sin(2 * np.pi * freq * t)).astype(np.int16)
                stereo_wave  = np.column_stack([wave, wave])
                self._sound  = pygame.sndarray.make_sound(stereo_wave)
                logger.info("Sonido sintetico generado")
        except Exception as e:
            logger.warning(f"No se pudo inicializar sonido: {e}")
            self._sound = None

    def _should_fire(self) -> bool:
        return (time.time() - self.last_alert_t) >= self.cooldown

    def push_frame(self, frame: np.ndarray):
        """Llamar CADA frame para el ring buffer de video clips."""
        if self.clip_recorder:
            self.clip_recorder.push_frame(frame)

    def trigger(self, class_names: list, frame: np.ndarray, frame_idx: int):
        """Dispara alertas multi-canal si no está en cooldown."""
        if self.disabled or not self._should_fire():
            # Aun en cooldown, si estamos grabando clip, extender
            if self.clip_recorder and self.clip_recorder.recording:
                self.clip_recorder.frames_remaining = self.clip_recorder.post_frames
            return

        self.last_alert_t = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        classes_str = ", ".join(set(class_names))

        # 1. Sonido
        if self._sound:
            try:
                self._sound.play()
            except Exception:
                pass

        # 2. Log texto
        if self.log_enabled:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp}]  Frame #{frame_idx:06d}  |  Detectado: {classes_str}\n")
            except Exception as e:
                logger.warning(f"Error log texto: {e}")

        # 3. Log JSON estructurado
        if self.json_enabled:
            try:
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "frame_idx": frame_idx,
                    "classes": list(set(class_names)),
                    "counts": {n: class_names.count(n) for n in set(class_names)},
                }
                with open(self.json_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.warning(f"Error log JSON: {e}")

        # 4. Guardar frame JPG
        if self.save_frames and frame is not None:
            try:
                fname = self.frames_dir / f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            except Exception as e:
                logger.warning(f"Error guardando frame: {e}")

        # 5. Video clip
        if self.clip_recorder:
            self.clip_recorder.start_recording(class_names)

        # 6. Telegram
        if self.telegram and frame is not None:
            self.telegram.send(frame, class_names)

        logger.warning(f"ALERTA  [{timestamp}]  Detectado: {classes_str}")

    def close(self):
        if self.disabled:
            return
        try:
            import pygame
            pygame.mixer.quit()
        except Exception:
            pass
