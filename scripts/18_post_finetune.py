"""
Script 18: Post-fine-tune — valida, exporta y genera reporte
=============================================================
Para cada modelo (nano y/o small) recién entrenado:
  1. Compara métricas vs versión original (yolov8s_v2 / yolov8n_v2)
  2. Exporta a ONNX + NCNN (listo para Pi5)
  3. Escribe docs/finetune_results.md con tabla comparativa

Uso:
    python scripts/18_post_finetune.py --nano       # solo nano
    python scripts/18_post_finetune.py --small      # solo small
    python scripts/18_post_finetune.py              # ambos si existen
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]

ORIG = {
    "nano":  REPO / "runs" / "detect" / "models" / "yolov8_weapons" / "weapons" / "yolov8n_v2" / "weights" / "best.pt",
    "small": REPO / "models" / "yolov8_weapons" / "weapons" / "yolov8s_v2" / "weights" / "best.pt",
}
def _find_new(run_name: str) -> Path:
    for base in [REPO / "runs" / "detect" / "models" / "yolov8_weapons" / "weapons",
                 REPO / "models" / "yolov8_weapons" / "weapons"]:
        p = base / run_name / "weights" / "best.pt"
        if p.exists():
            return p
    return REPO / "runs" / "detect" / "models" / "yolov8_weapons" / "weapons" / run_name / "weights" / "best.pt"

NEW = {
    "nano":  _find_new("yolov8n_v4_pose_negs"),
    "small": _find_new("yolov8s_v4_pose_negs"),
}
IMGSZ = {"nano": 416, "small": 640}

DATA_YAML = REPO / "data" / "processed" / "data.yaml"


def val_model(path: Path, imgsz: int):
    from ultralytics import YOLO
    m = YOLO(str(path))
    r = m.val(data=str(DATA_YAML), imgsz=imgsz, batch=16, verbose=False, plots=False)
    return {
        "params_M": sum(p.numel() for p in m.model.parameters()) / 1e6,
        "mAP50":    float(r.box.map50),
        "mAP50_95": float(r.box.map),
        "P":        float(r.box.mp),
        "R":        float(r.box.mr),
    }


def export_model(path: Path, imgsz: int):
    from ultralytics import YOLO
    m = YOLO(str(path))
    log.info(f"  → exportando ONNX")
    m.export(format="onnx", imgsz=imgsz, simplify=True)
    log.info(f"  → exportando NCNN")
    m.export(format="ncnn", imgsz=imgsz)


def process(tag: str):
    new_path = NEW[tag]
    orig_path = ORIG[tag]
    if not new_path.exists():
        log.warning(f"[{tag}] no existe {new_path} — ¿terminó el entrenamiento?")
        return None
    imgsz = IMGSZ[tag]

    log.info(f"========== {tag.upper()} ==========")
    log.info(f"[1/3] Val original: {orig_path}")
    orig = val_model(orig_path, imgsz)
    log.info(f"  mAP50={orig['mAP50']:.4f}  mAP50-95={orig['mAP50_95']:.4f}  P={orig['P']:.4f}  R={orig['R']:.4f}")

    log.info(f"[2/3] Val nuevo: {new_path}")
    new = val_model(new_path, imgsz)
    log.info(f"  mAP50={new['mAP50']:.4f}  mAP50-95={new['mAP50_95']:.4f}  P={new['P']:.4f}  R={new['R']:.4f}")

    log.info(f"[3/3] Export {tag} a ONNX + NCNN")
    export_model(new_path, imgsz)

    return {"tag": tag, "orig": orig, "new": new, "imgsz": imgsz,
            "new_path": str(new_path), "orig_path": str(orig_path)}


def write_report(results):
    out = REPO / "docs" / "finetune_results.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Resultados Fine-tune Anti-Alucinación + Poses",
        "",
        "Comparativa entre los modelos **originales** (yolov8X_v2) y los **fine-tuned** (yolov8X_v3_pose_negs).",
        "",
        "Validación sobre `data/processed/data.yaml` (865 imgs val).",
        "",
        "| Modelo | Param (M) | imgsz | mAP@50 | mAP@50-95 | Precision | Recall |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r is None: continue
        o, n = r["orig"], r["new"]
        lines.append(f"| {r['tag']} ORIGINAL | {o['params_M']:.2f} | {r['imgsz']} | {o['mAP50']:.4f} | {o['mAP50_95']:.4f} | {o['P']:.4f} | {o['R']:.4f} |")
        lines.append(f"| {r['tag']} **NUEVO** | {n['params_M']:.2f} | {r['imgsz']} | **{n['mAP50']:.4f}** | **{n['mAP50_95']:.4f}** | **{n['P']:.4f}** | **{n['R']:.4f}** |")

    lines += ["", "## Deltas", "", "| Modelo | Δ mAP@50 | Δ mAP@50-95 | Δ P | Δ R |", "|---|---|---|---|---|"]
    for r in results:
        if r is None: continue
        o, n = r["orig"], r["new"]
        d50  = (n["mAP50"]    - o["mAP50"])    * 100
        d595 = (n["mAP50_95"] - o["mAP50_95"]) * 100
        dP   = (n["P"]        - o["P"])        * 100
        dR   = (n["R"]        - o["R"])        * 100
        def f(x): return f"{x:+.2f}pp"
        lines.append(f"| {r['tag']} | {f(d50)} | {f(d595)} | {f(dP)} | {f(dR)} |")

    lines += [
        "",
        "## Archivos generados",
        "",
    ]
    for r in results:
        if r is None: continue
        tag = r["tag"]
        weights_dir = Path(r["new_path"]).parent
        lines += [
            f"### {tag}",
            f"- Pesos: `{r['new_path']}`",
            f"- ONNX: `{weights_dir / 'best.onnx'}`",
            f"- NCNN: `{weights_dir / 'best_ncnn_model/'}` (param + bin + metadata.yaml)",
            "",
        ]

    lines += [
        "## Notas",
        "- El fine-tune partió de los pesos `v2` y se hizo con `lr0=0.0001`, `freeze=10`, `cls=1.0`, `close_mosaic=10`.",
        "- Se añadieron al training set: 5,341 imgs Subh775, 688 PranomVignesh, 230 Simuletic, 605 móviles, 846 negativos OpenImages (= +7,710 imgs).",
        "- Para usar en Pi5: copiar la carpeta `best_ncnn_model/` y correr `yolo predict model=best_ncnn_model imgsz=416 source=0`.",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Reporte: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nano",  action="store_true")
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()

    tags = []
    if args.nano:  tags.append("nano")
    if args.small: tags.append("small")
    if not tags:
        if NEW["nano"].exists():  tags.append("nano")
        if NEW["small"].exists(): tags.append("small")

    if not tags:
        log.error("No hay modelos nuevos para procesar.")
        return

    results = [process(t) for t in tags]
    write_report(results)


if __name__ == "__main__":
    main()
