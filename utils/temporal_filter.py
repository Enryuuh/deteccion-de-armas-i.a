"""
utils/temporal_filter.py
========================
Filtro temporal para reducir falsos positivos y flickering.

Lógica:
  - Guarda las últimas `window` detecciones por frame.
  - Una clase se "confirma" solo si apareció en >= `min_hits` de esos frames.
  - Las coordenadas del bounding box se suavizan con EMA para eliminar saltos.
"""

from collections import deque
import numpy as np


class TemporalDetectionFilter:

    def __init__(self, window: int = 7, min_hits: int = 4, smooth_alpha: float = 0.20):
        """
        window      : cuántos frames hacia atrás considerar
        min_hits    : mínimo de frames con detección para confirmar
        smooth_alpha: peso del frame actual en EMA (0=inmóvil, 1=sin suavizado)
                      0.20 → 80% estado anterior, muy estable
        """
        self.window       = window
        self.min_hits     = min_hits
        self.smooth_alpha = smooth_alpha

        # Historial: deque de dicts {cls_id: (box_xyxy, score)}
        self.history: deque = deque(maxlen=window)
        # Estado suavizado: {cls_id: {'box': np.array, 'score': float}}
        self.tracked: dict  = {}

    def update(
        self,
        boxes:    np.ndarray,
        scores:   np.ndarray,
        cls_ids:  np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Recibe detecciones crudas del modelo y devuelve las filtradas + suavizadas.
        Siempre devuelve arrays (posiblemente vacíos).
        """
        # -- 1. Mejor detección por clase en este frame --
        frame: dict[int, tuple] = {}
        for box, score, cid in zip(boxes, scores, cls_ids):
            cid = int(cid)
            if cid not in frame or float(score) > frame[cid][1]:
                frame[cid] = (np.asarray(box, dtype=float), float(score))

        self.history.append(frame)

        # -- 2. Contar presencia de cada clase en la ventana --
        class_hits: dict[int, int] = {}
        for past_frame in self.history:
            for cid in past_frame:
                class_hits[cid] = class_hits.get(cid, 0) + 1

        confirmed = {cid for cid, hits in class_hits.items() if hits >= self.min_hits}

        # -- 3. Actualizar estado suavizado para clases confirmadas --
        for cid in confirmed:
            if cid in frame:
                raw_box, raw_score = frame[cid]
                if cid in self.tracked:
                    a = self.smooth_alpha
                    smooth_box   = a * raw_box                   + (1 - a) * self.tracked[cid]["box"]
                    smooth_score = a * raw_score                  + (1 - a) * self.tracked[cid]["score"]
                else:
                    smooth_box   = raw_box.copy()
                    smooth_score = raw_score
                self.tracked[cid] = {"box": smooth_box, "score": smooth_score}

        # -- 4. Eliminar clases que ya no están confirmadas --
        for cid in list(self.tracked.keys()):
            if cid not in confirmed:
                del self.tracked[cid]

        # -- 5. Construir arrays de salida --
        if not self.tracked:
            empty = np.array([])
            return empty, empty, empty

        out_boxes   = np.array([s["box"]   for s in self.tracked.values()])
        out_scores  = np.array([s["score"] for s in self.tracked.values()])
        out_cls_ids = np.array(list(self.tracked.keys()), dtype=int)
        return out_boxes, out_scores, out_cls_ids

    def reset(self):
        self.history.clear()
        self.tracked.clear()
