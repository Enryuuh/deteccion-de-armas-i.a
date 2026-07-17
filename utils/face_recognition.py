"""
Reconocimiento facial (identificacion de personas)
===================================================
Detecta caras e IDENTIFICA de quien son, comparando contra una base de
personas previamente matriculadas (enrolled). Se apoya en InsightFace
(RetinaFace para detectar + ArcFace para el embedding de 512-D), todo en
ONNX -> funciona con GPU (laptop/RTX) y con CPU (Raspberry Pi 5).

Como funciona:
  1. matricular (enroll): con 3-5 fotos por persona se calcula su embedding
     promedio y se guarda en face_db.npz junto a su nombre.
  2. reconocer: a cada cara detectada se le calcula el embedding y se compara
     (similitud coseno) contra la base. Si supera el umbral -> nombre;
     si no -> "Desconocido".

No se entrena nada: el modelo viene pre-entrenado. Matricular = guardar
vectores, cuestion de segundos.

Uso tipico:
    from utils.face_recognition import FaceRecognizer
    fr = FaceRecognizer(cfg)
    faces = fr.recognize(frame)   # -> [FaceResult(name, score, bbox), ...]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FaceResult:
    """Una cara detectada en un frame."""
    name: str            # nombre matriculado o "Desconocido"
    score: float         # similitud coseno con la mejor coincidencia (0-1)
    bbox: tuple          # (x1, y1, x2, y2) en pixeles
    is_known: bool       # True si supero el umbral


class FaceRecognizer:
    """Detector + identificador de caras basado en InsightFace."""

    def __init__(self, cfg: dict):
        fcfg = (cfg or {}).get("face_recognition", {}) or {}
        self.enabled: bool = bool(fcfg.get("enabled", False))
        self.threshold: float = float(fcfg.get("similarity_threshold", 0.35))
        self.det_size: int = int(fcfg.get("det_size", 640))
        self.model_pack: str = fcfg.get("model_pack", "buffalo_l")
        self.db_path: str = fcfg.get("db_file", "models/faces/face_db.npz")
        self.use_gpu: bool = bool(fcfg.get("use_gpu", True))

        self.app = None
        self.db_names: list[str] = []
        self.db_embeds: Optional[np.ndarray] = None  # (N, 512) normalizados

        if self.enabled:
            self._load_model()
            self.load_db(self.db_path)

    # ---------------------------------------------------------------- modelo
    def _load_model(self):
        """Carga InsightFace. Elige GPU si esta disponible, si no CPU."""
        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ImportError(
                "Falta insightface. Instala:  pip install insightface onnxruntime\n"
                "(en laptop con GPU:  pip install onnxruntime-gpu)"
            ) from e

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.use_gpu else ["CPUExecutionProvider"]
        )
        self.app = FaceAnalysis(name=self.model_pack, providers=providers)
        # ctx_id = 0 -> primera GPU;  -1 -> CPU
        ctx_id = 0 if self.use_gpu else -1
        self.app.prepare(ctx_id=ctx_id, det_size=(self.det_size, self.det_size))
        logger.info("InsightFace '%s' listo (providers=%s)", self.model_pack, providers)

    # ---------------------------------------------------------- base de datos
    def load_db(self, path: str) -> int:
        """Carga la base de caras matriculadas. Devuelve nº de personas."""
        p = Path(path)
        if not p.exists():
            logger.warning("Base de caras no encontrada (%s). Todos seran 'Desconocido' "
                           "hasta que matricules con scripts/20_enroll_faces.py", path)
            self.db_names, self.db_embeds = [], None
            return 0
        data = np.load(p, allow_pickle=True)
        self.db_names = list(data["names"])
        self.db_embeds = data["embeds"].astype(np.float32)
        # asegurar normalizados (por si acaso)
        self.db_embeds = self._normalize(self.db_embeds)
        logger.info("Base de caras: %d personas (%s)", len(self.db_names), ", ".join(self.db_names))
        return len(self.db_names)

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        v = np.atleast_2d(v).astype(np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return v / norms

    def embed(self, face_img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Embedding 512-D (normalizado) de la cara MAS GRANDE del recorte, o None."""
        if self.app is None:
            self._load_model()
        faces = self.app.get(face_img_bgr)
        if not faces:
            return None
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        return self._normalize(faces[0].normed_embedding)[0]

    # ------------------------------------------------------------- reconocer
    def recognize(self, frame_bgr: np.ndarray) -> list[FaceResult]:
        """Detecta TODAS las caras del frame y las identifica."""
        if not self.enabled or self.app is None:
            return []
        results: list[FaceResult] = []
        for f in self.app.get(frame_bgr):
            emb = self._normalize(f.normed_embedding)[0]
            name, score, known = self._match(emb)
            x1, y1, x2, y2 = f.bbox.astype(int)
            results.append(FaceResult(name, score, (int(x1), int(y1), int(x2), int(y2)), known))
        return results

    def _match(self, emb: np.ndarray) -> tuple[str, float, bool]:
        """Compara un embedding contra la base -> (nombre, similitud, conocido)."""
        if self.db_embeds is None or len(self.db_names) == 0:
            return "Desconocido", 0.0, False
        sims = self.db_embeds @ emb          # coseno (vectores normalizados)
        idx = int(np.argmax(sims))
        best = float(sims[idx])
        if best >= self.threshold:
            return self.db_names[idx], best, True
        return "Desconocido", best, False
