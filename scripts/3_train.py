"""
Script 3: Entrenamiento RT-DETR — RTX 4060 (8 GB VRAM)
=========================================================
Fine-tuning de RT-DETR-R50 sobre el dataset de armas.
Usa FP16 automático para maximizar rendimiento en RTX 4060.

Requisitos: CUDA 12.x instalado, pip install -r requirements.txt
            Haber ejecutado scripts 1 y 2 primero.

Uso:
    python scripts/3_train.py

Monitoreo:
    tensorboard --logdir models/rtdetr_weapons/logs
"""

import os
import sys
import json
import logging
import yaml
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from torch.utils.tensorboard import SummaryWriter

# Fix para importar utils desde cualquier directorio
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.dataset import WeaponCOCODataset, collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_label_maps(cfg: dict) -> tuple[dict, dict]:
    """Genera id2label y label2id para el modelo."""
    id2label = {0: "background"}
    label2id = {"background": 0}
    for idx, name in enumerate(cfg["classes"]):
        cat_id = idx + 1
        id2label[cat_id] = name
        label2id[name]   = cat_id
    return id2label, label2id


def train_one_epoch(
    model, dataloader, optimizer, scheduler, scaler, device, epoch, writer
) -> float:
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")

    for step, batch in enumerate(pbar):
        pixel_values = batch["pixel_values"].to(device)
        labels       = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        optimizer.zero_grad()
        with autocast():
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss    = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_val = loss.item()
        total_loss += loss_val
        pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        global_step = epoch * len(dataloader) + step
        writer.add_scalar("Loss/train_step", loss_val, global_step)

    avg_loss = total_loss / len(dataloader)
    writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, device, epoch, writer) -> float:
    model.eval()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc=f"  Val Epoch {epoch}", unit="batch"):
        pixel_values = batch["pixel_values"].to(device)
        labels       = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        with autocast():
            outputs = model(pixel_values=pixel_values, labels=labels)

        total_loss += outputs.loss.item()

    avg_loss = total_loss / len(dataloader)
    writer.add_scalar("Loss/val_epoch", avg_loss, epoch)
    return avg_loss


def main():
    cfg = load_config()
    tcfg = cfg["training"]

    torch.manual_seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"=== Entrenamiento RT-DETR  |  Dispositivo: {device} ===")

    if device.type == "cuda":
        logger.info(f"    GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"    VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        logger.info(f"    FP16 (mixed precision): {'activado' if tcfg['fp16'] else 'desactivado'}")

    # Rutas
    proc_dir    = Path(cfg["dataset"]["processed_dir"])
    output_dir  = Path(tcfg["output_dir"])
    logs_dir    = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Label maps
    id2label, label2id = build_label_maps(cfg)
    num_labels = len(id2label)  # incluye background
    logger.info(f"    Clases: {id2label}")

    # Procesador e imagen
    model_name = cfg["model"]["name"]
    logger.info(f"    Cargando procesador desde: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    processor.size = {"width": cfg["model"]["image_size"], "height": cfg["model"]["image_size"]}

    # Datasets
    train_json = proc_dir / "annotations_train.json"
    val_json   = proc_dir / "annotations_val.json"
    img_train  = proc_dir / "images" / "train"
    img_val    = proc_dir / "images" / "val"

    if not train_json.exists():
        logger.error("No se encontró annotations_train.json. Ejecuta scripts 1 y 2 primero.")
        sys.exit(1)

    train_dataset = WeaponCOCODataset(train_json, img_train, processor, augment=True)
    val_dataset   = WeaponCOCODataset(val_json,   img_val,   processor, augment=False)
    logger.info(f"    Train: {len(train_dataset)} muestras | Val: {len(val_dataset)} muestras")

    train_loader = DataLoader(
        train_dataset,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=tcfg["dataloader_num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=tcfg["batch_size"],
        shuffle=False,
        num_workers=tcfg["dataloader_num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Modelo
    logger.info(f"    Cargando modelo RT-DETR desde: {model_name}")
    model = AutoModelForObjectDetection.from_pretrained(
        model_name,
        id2label=id2label,
        label2id=label2id,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # Optimizador y scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
    )
    total_steps = len(train_loader) * tcfg["num_epochs"]
    scheduler = OneCycleLR(
        optimizer,
        max_lr=tcfg["learning_rate"],
        total_steps=total_steps,
        pct_start=tcfg["warmup_steps"] / total_steps,
    )
    scaler  = GradScaler(enabled=tcfg["fp16"] and device.type == "cuda")
    writer  = SummaryWriter(log_dir=str(logs_dir))

    best_val_loss = float("inf")
    logger.info(f"=== Iniciando entrenamiento: {tcfg['num_epochs']} epochs ===")

    for epoch in range(1, tcfg["num_epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch, writer)
        val_loss   = validate(model, val_loader, device, epoch, writer)

        logger.info(f"Epoch {epoch:03d}/{tcfg['num_epochs']}  |  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")

        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = output_dir / "best"
            model.save_pretrained(str(best_dir))
            processor.save_pretrained(str(best_dir))
            logger.info(f"   ✅ Mejor modelo guardado (val_loss={val_loss:.4f})")

        # Checkpoint periódico
        if epoch % 10 == 0:
            ckpt_dir = output_dir / f"checkpoint_epoch_{epoch}"
            model.save_pretrained(str(ckpt_dir))
            processor.save_pretrained(str(ckpt_dir))

    writer.close()
    logger.info(f"=== Entrenamiento completado ===")
    logger.info(f"   Mejor val_loss: {best_val_loss:.4f}")
    logger.info(f"   Modelo guardado en: {output_dir / 'best'}")
    logger.info(f"   TensorBoard: tensorboard --logdir {logs_dir}")


if __name__ == "__main__":
    main()
