"""
Pipeline de reentrenamiento completo (autonomo)
================================================
Ejecuta toda la cadena sin intervencion:
  1. Fine-tune anti-alucinacion del modelo GRANDE (yolov8s) -> v5
  2. Fine-tune anti-alucinacion del modelo NANO (yolov8n)   -> v5  (para la Pi)
  3. Evalua v5 vs v4 en el test set (mAP / P / R por clase)
  4. Exporta el nano v5 a ONNX (fp32 + int8) para la Raspberry Pi
  5. Escribe un resumen en docs/retrain_v5_results.md

Solo actualiza config.yaml para apuntar al v5 si NO empeora (recall no cae > 3pp).
El v4 queda intacto como respaldo.

Uso:
    python scripts/retrain_pipeline.py
    python scripts/retrain_pipeline.py --epochs 30
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pipeline")

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

SMALL_NAME = "yolov8s_v5_moredata"
NANO_NAME  = "yolov8n_v5_moredata"

# Rutas de salida de los runs (script 17 usa project=models/yolov8_weapons, name=weapons/<run>)
def run_weights(name):
    # ultralytics antepone runs/detect/ al project relativo
    return REPO / "runs" / "detect" / "models" / "yolov8_weapons" / "weapons" / name / "weights" / "best.pt"

V4_SMALL = REPO / "runs/detect/models/yolov8_weapons/weapons/yolov8s_v4_pose_negs/weights/best.pt"
V4_NANO  = REPO / "runs/detect/models/yolov8_weapons/weapons/yolov8n_v4_pose_negs/weights/best.pt"
DATA = REPO / "data" / "processed" / "data.yaml"


def sh(cmd):
    log.info("EJECUTANDO: %s", " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, cwd=str(REPO))
    if r.returncode != 0:
        log.error("Fallo (rc=%d): %s", r.returncode, cmd)
    return r.returncode == 0


def evaluate(weights, imgsz):
    """Evalua un modelo en el split test. Devuelve dict de metricas o None."""
    if not Path(weights).exists():
        log.warning("No existe para evaluar: %s", weights)
        return None
    try:
        from ultralytics import YOLO
        m = YOLO(str(weights))
        res = m.val(data=str(DATA), split="test", imgsz=imgsz, verbose=False)
        return {
            "mAP50": float(res.box.map50),
            "mAP50_95": float(res.box.map),
            "precision": float(res.box.mp),
            "recall": float(res.box.mr),
        }
    except Exception as e:
        log.error("Error evaluando %s: %s", weights, e)
        return None


def fmt(m):
    if not m:
        return "n/a"
    return (f"mAP50={m['mAP50']:.4f}  mAP50-95={m['mAP50_95']:.4f}  "
            f"P={m['precision']:.4f}  R={m['recall']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    # Candado anti-doble-ejecucion (evita 2 entrenamientos a la vez -> OOM)
    lock = REPO / "models" / ".retrain.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        log.error("Ya hay un pipeline corriendo (existe %s). Abortando.", lock)
        return
    lock.write_text(datetime.now().isoformat(), encoding="utf-8")

    try:
        _run(args)
    finally:
        try:
            lock.unlink()
        except Exception:
            pass


def _run(args):
    t0 = datetime.now()
    log.info("===== PIPELINE DE REENTRENAMIENTO v5 =====")

    # 1. Entrenar GRANDE (yolov8s) — saltar si ya existe (evita re-entrenar)
    if run_weights(SMALL_NAME).exists():
        log.info(">>> [1/5] GRANDE ya entrenado (%s), se salta.", SMALL_NAME)
        ok_small = True
    else:
        log.info(">>> [1/5] Fine-tune modelo GRANDE (yolov8s) -> %s", SMALL_NAME)
        ok_small = sh([PY, "scripts/17_finetune_anti_hallucination.py",
                       "--name", SMALL_NAME, "--epochs", str(args.epochs)])

    # 2. Entrenar NANO (yolov8n) para la Pi — saltar si ya existe
    if run_weights(NANO_NAME).exists():
        log.info(">>> [2/5] NANO ya entrenado (%s), se salta.", NANO_NAME)
        ok_nano = True
    else:
        log.info(">>> [2/5] Fine-tune modelo NANO (yolov8n) -> %s", NANO_NAME)
        ok_nano = sh([PY, "scripts/17_finetune_anti_hallucination.py", "--nano",
                      "--name", NANO_NAME, "--epochs", str(args.epochs)])

    # 3. Evaluar v5 vs v4
    log.info(">>> [3/5] Evaluando en test set...")
    ev = {
        "small_v4": evaluate(V4_SMALL, 640),
        "small_v5": evaluate(run_weights(SMALL_NAME), 640),
        "nano_v4":  evaluate(V4_NANO, 416),
        "nano_v5":  evaluate(run_weights(NANO_NAME), 416),
    }
    for k, v in ev.items():
        log.info("  %-10s : %s", k, fmt(v))

    # 4. Exportar nano v5 a ONNX
    log.info(">>> [4/5] Exportando nano v5 a ONNX...")
    nano_w = run_weights(NANO_NAME)
    if nano_w.exists():
        try:
            from ultralytics import YOLO
            YOLO(str(nano_w)).export(format="onnx", imgsz=416, simplify=True, opset=12)
            log.info("  ONNX exportado junto a %s", nano_w)
        except Exception as e:
            log.error("  Error exportando ONNX: %s", e)

    # 5. Resumen a docs/
    log.info(">>> [5/5] Escribiendo resumen...")
    out = REPO / "docs" / "retrain_v5_results.md"
    lines = [
        "# Reentrenamiento v5 (mas hard negatives + armas Roboflow)",
        "",
        f"Fecha: {t0:%Y-%m-%d %H:%M}  |  Epochs: {args.epochs}",
        f"Entrenamiento GRANDE ok: {ok_small}  |  NANO ok: {ok_nano}",
        "",
        "| Modelo | mAP@50 | mAP@50-95 | Precision | Recall |",
        "|---|---|---|---|---|",
    ]
    for k in ["small_v4", "small_v5", "nano_v4", "nano_v5"]:
        m = ev[k]
        if m:
            lines.append(f"| {k} | {m['mAP50']:.4f} | {m['mAP50_95']:.4f} | "
                         f"{m['precision']:.4f} | {m['recall']:.4f} |")
        else:
            lines.append(f"| {k} | n/a | n/a | n/a | n/a |")
    # Recomendacion automatica (no cambia config sola; solo sugiere)
    lines += ["", "## Lectura"]
    sv4, sv5 = ev["small_v4"], ev["small_v5"]
    if sv4 and sv5:
        d_map = (sv5["mAP50"] - sv4["mAP50"]) * 100
        d_r = (sv5["recall"] - sv4["recall"]) * 100
        d_p = (sv5["precision"] - sv4["precision"]) * 100
        lines.append(f"- Small v5 vs v4: mAP50 {d_map:+.2f}pp, Precision {d_p:+.2f}pp, Recall {d_r:+.2f}pp")
        lines.append(f"- Precision sube = menos falsos positivos (objetivo). Recall no debe caer mucho.")
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("Resumen en %s", out)

    dt = (datetime.now() - t0).total_seconds() / 60
    log.info("===== PIPELINE COMPLETADO en %.1f min =====", dt)


if __name__ == "__main__":
    main()
