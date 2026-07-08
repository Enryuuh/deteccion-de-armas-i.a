"""
utils/plausibility_filter.py
=============================
Descarta detecciones geométricamente imposibles para armas:
  - Caja demasiado pequeña o demasiado grande respecto al frame
  - Aspect ratio fuera de rango para la clase
  - Confianza por debajo del umbral específico de la clase
"""

import numpy as np


# Geometría esperada por clase  [min_ratio, max_ratio, min_ar, max_ar]
# ar = max(w,h) / min(w,h)
_CLASS_GEOMETRY = {
    # (min_area_ratio, max_area_ratio, min_aspect_ratio, max_aspect_ratio)
    "knife":    (0.002, 0.45, 1.1, 18.0),  # AR mínimo bajado: cuchillo apuntando al frente se ve cuadrado
    "handgun":  (0.003, 0.50, 1.0,  6.0),
    "long_gun": (0.005, 0.60, 2.0, 22.0),
}
_DEFAULT_GEOMETRY = (0.002, 0.55, 1.0, 22.0)


class PlausibilityFilter:
    """
    Aplica dos filtros independientes:
      1. Umbral de confianza por clase (class_thresholds en config)
      2. Geometría de la caja (área relativa + aspect ratio por clase)
    """

    def __init__(self, cfg: dict, id2label: dict):
        icfg = cfg.get("inference", {})
        pcfg = cfg.get("plausibility", {})

        self.id2label = id2label

        # Umbrales por clase (default = confidence_threshold global)
        default_th = icfg.get("confidence_threshold", 0.50)
        raw_th = icfg.get("class_thresholds", {})
        self.class_thresholds = {
            label: raw_th.get(label, default_th)
            for label in id2label.values()
        }

        # Geometría global (fallback si no hay por clase en config)
        self.min_area  = pcfg.get("min_area_ratio", 0.002)
        self.max_area  = pcfg.get("max_area_ratio", 0.55)

    def _geometry_ok(self, box: np.ndarray, label: str, frame_h: int, frame_w: int) -> bool:
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            return False

        area_ratio = (w * h) / (frame_w * frame_h)
        min_area, max_area, min_ar, max_ar = _CLASS_GEOMETRY.get(label, _DEFAULT_GEOMETRY)

        if not (min_area <= area_ratio <= max_area):
            return False

        ar = max(w, h) / max(min(w, h), 1e-6)
        if not (min_ar <= ar <= max_ar):
            return False

        return True

    def filter(
        self,
        boxes:   np.ndarray,
        scores:  np.ndarray,
        cls_ids: np.ndarray,
        frame_h: int,
        frame_w: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Devuelve solo las detecciones que pasan ambos filtros."""
        if len(boxes) == 0:
            empty = np.array([])
            return empty, empty, empty

        keep = []
        for i, (box, score, cid) in enumerate(zip(boxes, scores, cls_ids)):
            label = self.id2label.get(int(cid), "")
            th = self.class_thresholds.get(label, 0.50)
            if float(score) < th:
                continue
            if not self._geometry_ok(box, label, frame_h, frame_w):
                continue
            keep.append(i)

        if not keep:
            empty = np.array([])
            return empty, empty, empty

        return boxes[keep], scores[keep], cls_ids[keep]
