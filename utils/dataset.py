"""
utils/dataset.py
================
Dataset PyTorch compatible con COCO JSON para RT-DETR.
Incluye aumentaciones de datos para entrenamiento.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import random

logger = logging.getLogger(__name__)


class WeaponCOCODataset(Dataset):
    """
    Dataset COCO para detección de armas.
    Soporta augmentación en entrenamiento y retorno de metadatos para evaluación.
    """

    def __init__(
        self,
        annotation_json: Path,
        images_dir: Path,
        processor,
        augment: bool = False,
        return_meta: bool = False,
    ):
        self.images_dir  = Path(images_dir)
        self.processor   = processor
        self.augment     = augment
        self.return_meta = return_meta

        with open(annotation_json, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

        # Indexar anotaciones por image_id
        self.ann_index: dict = {}
        for ann in coco.get("annotations", []):
            img_id = ann["image_id"]
            self.ann_index.setdefault(img_id, []).append(ann)

        # Filtrar imágenes que tengan al menos una anotación
        self.images = [
            img for img in coco["images"]
            if img["id"] in self.ann_index and (self.images_dir / img["file_name"]).exists()
        ]

        if len(self.images) == 0:
            logger.warning(f"Dataset vacío en {images_dir}. Verifica que los scripts 1 y 2 corrieron.")
        else:
            logger.info(f"Dataset cargado: {len(self.images)} imágenes con anotaciones.")

    def __len__(self) -> int:
        return len(self.images)

    def _augment(self, image: Image.Image, boxes: list) -> tuple:
        """Aumentaciones simples que no rompen las bbox."""

        # Flip horizontal con prob 0.5
        if random.random() > 0.5:
            image = TF.hflip(image)
            w = image.width
            boxes = [[w - b[0] - b[2], b[1], b[2], b[3]] for b in boxes]

        # Ajuste de color
        image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.7, 1.3))
        image = TF.adjust_contrast(image,   contrast_factor=random.uniform(0.8, 1.2))
        image = TF.adjust_saturation(image, saturation_factor=random.uniform(0.8, 1.2))

        return image, boxes

    def __getitem__(self, idx: int) -> dict:
        img_info = self.images[idx]
        img_path = self.images_dir / img_info["file_name"]

        image = Image.open(img_path).convert("RGB")
        anns  = self.ann_index.get(img_info["id"], [])

        # COCO bbox → [x, y, width, height]
        boxes  = [ann["bbox"] for ann in anns]
        labels = [ann["category_id"] for ann in anns]

        if self.augment and boxes:
            image, boxes = self._augment(image, boxes)

        # Convertir bbox a [x_min, y_min, x_max, y_max] para el processor
        boxes_xyxy = []
        for b in boxes:
            x, y, w, h = b
            boxes_xyxy.append([x, y, x + w, y + h])

        encoding = self.processor(
            images=image,
            annotations=[{
                "image_id": img_info["id"],
                "annotations": [
                    {"bbox": b_orig, "category_id": lbl, "area": b_orig[2]*b_orig[3], "iscrowd": 0}
                    for b_orig, lbl in zip(boxes, labels)
                ],
            }] if boxes else None,
            return_tensors="pt",
        )

        item = {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels":       encoding["labels"][0] if "labels" in encoding else {},
        }

        if self.return_meta:
            item["image_id"]  = img_info["id"]
            item["orig_sizes"] = (img_info["height"], img_info["width"])

        return item


def collate_fn(batch: list) -> dict:
    """Agrupa muestras en un batch, paddeando labels de longitud variable."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])

    # Labels pueden tener distinto número de objetos → lista de dicts
    labels = [item["labels"] for item in batch]

    result: dict = {"pixel_values": pixel_values, "labels": labels}

    if "image_id" in batch[0]:
        result["image_ids"]  = [item["image_id"]   for item in batch]
        result["orig_sizes"] = [item["orig_sizes"]  for item in batch]

    return result
