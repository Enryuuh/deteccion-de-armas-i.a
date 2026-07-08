"""
Script 19: Re-integra PranomVignesh completo + reentrena ambos modelos
========================================================================
Hace en cadena, sin intervención:
  1. Elimina las 688 imgs antiguas `hfprv_*` del training set.
  2. Re-integra el dataset PranomVignesh completo desde el cache (ahora ~8.9K).
  3. Entrena nano (yolov8n_v4_pose_negs).
  4. Entrena small (yolov8s_v4_pose_negs).
  5. Cada uno se valida + exporta + reporta automáticamente (script 18).

Uso:
    python scripts/19_reintegrate_and_retrain.py
"""

import logging
import subprocess
import sys
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
CLS_IDX = {"knife": 0, "handgun": 1, "long_gun": 2}


def reintegrate_pranom():
    img_out = REPO / "data" / "processed" / "images" / "train"
    lbl_out = REPO / "data" / "processed" / "labels" / "train"

    # 1) Borrar las 688 antiguas
    log.info("Borrando archivos hfprv_* antiguos...")
    n_del_img, n_del_lbl = 0, 0
    for f in img_out.glob("hfprv_*"):
        f.unlink()
        n_del_img += 1
    for f in lbl_out.glob("hfprv_*"):
        f.unlink()
        n_del_lbl += 1
    log.info(f"  borrados {n_del_img} imgs + {n_del_lbl} labels")

    # 2) Re-integrar desde cache completo
    cache = REPO / "data" / "hf_pose" / "PranomVignesh_HandGuns"
    data_yaml = cache / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"No existe {data_yaml}")
    with open(data_yaml, encoding="utf-8") as f:
        ds_cfg = yaml.safe_load(f)
    names = ds_cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names.keys())]
    log.info(f"  clases ds: {names} -> map {{'gun': 'handgun'}}")

    total_imgs, total_pos = 0, 0
    for split in ["train", "valid", "test"]:
        images_dir = cache / split / "images"
        labels_dir = cache / split / "labels"
        if not images_dir.exists():
            continue
        added, with_obj = 0, 0
        for img in images_dir.iterdir():
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl = labels_dir / (img.stem + ".txt")
            if not lbl.exists():
                continue
            mapped = []
            for line in lbl.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    old_id = int(parts[0])
                except ValueError:
                    continue
                if old_id >= len(names):
                    continue
                old_name = names[old_id]
                if old_name != "gun":
                    continue
                new_id = CLS_IDX["handgun"]
                mapped.append(f"{new_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
            if not mapped:
                continue
            added += 1
            with_obj += 1
            dst_img = img_out / f"hfprv_handguns_{split}_{added:06d}{img.suffix.lower()}"
            dst_lbl = lbl_out / f"hfprv_handguns_{split}_{added:06d}.txt"
            import shutil
            shutil.copy2(img, dst_img)
            dst_lbl.write_text("\n".join(mapped), encoding="utf-8")
        log.info(f"  {split}: {added} imgs integradas ({with_obj} con armas)")
        total_imgs += added
        total_pos  += with_obj
    log.info(f"PranomVignesh TOTAL re-integrado: {total_imgs} ({total_pos} con armas)")
    return total_imgs


def edit_finetune_runname(new_suffix="v4_pose_negs"):
    """Actualiza el run_name del script 17 para no sobreescribir v3."""
    s17 = REPO / "scripts" / "17_finetune_anti_hallucination.py"
    text = s17.read_text(encoding="utf-8")
    text = text.replace("yolov8n_v3_pose_negs", f"yolov8n_{new_suffix}")
    text = text.replace("yolov8s_v3_pose_negs", f"yolov8s_{new_suffix}")
    s17.write_text(text, encoding="utf-8")
    log.info(f"Script 17 actualizado para usar suffix {new_suffix}")


def edit_post_runname(new_suffix="v4_pose_negs"):
    s18 = REPO / "scripts" / "18_post_finetune.py"
    text = s18.read_text(encoding="utf-8")
    text = text.replace('"yolov8n_v3_pose_negs"', f'"yolov8n_{new_suffix}"')
    text = text.replace('"yolov8s_v3_pose_negs"', f'"yolov8s_{new_suffix}"')
    s18.write_text(text, encoding="utf-8")
    log.info(f"Script 18 actualizado para usar suffix {new_suffix}")


def run(cmd):
    log.info(f"$ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(REPO))
    if r.returncode != 0:
        log.error(f"  FAILED rc={r.returncode}")
        return False
    return True


def main():
    total = reintegrate_pranom()
    log.info(f"================ DATASET ACTUALIZADO ================")
    img_out = REPO / "data" / "processed" / "images" / "train"
    lbl_out = REPO / "data" / "processed" / "labels" / "train"
    n_img = len(list(img_out.iterdir()))
    n_lbl = len(list(lbl_out.iterdir()))
    log.info(f"  Imgs totales: {n_img} | Labels: {n_lbl}")

    # cambiar runs a v4
    edit_finetune_runname("v4_pose_negs")
    edit_post_runname("v4_pose_negs")

    # NANO
    log.info("================ ENTRENANDO NANO v4 ================")
    if not run([sys.executable, "scripts/17_finetune_anti_hallucination.py", "--nano"]):
        return
    log.info("================ POST NANO v4 ================")
    run([sys.executable, "scripts/18_post_finetune.py", "--nano"])

    # SMALL
    log.info("================ ENTRENANDO SMALL v4 ================")
    if not run([sys.executable, "scripts/17_finetune_anti_hallucination.py"]):
        return
    log.info("================ POST SMALL v4 ================")
    run([sys.executable, "scripts/18_post_finetune.py"])

    log.info("================ PIPELINE COMPLETO ================")
    log.info("Reporte: docs/finetune_results.md")


if __name__ == "__main__":
    main()
