"""
Script 6: Exportar modelo para Raspberry Pi 5
================================================
Exporta los pesos entrenados a formatos livianos para correr en
Raspberry Pi 5 sin PyTorch:
  - ONNX (recomendado, corre con onnxruntime)
  - NCNN (opcional, mas rapido en ARM)

Uso:
    python scripts/6_export.py
"""

import logging
import shutil
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    from ultralytics import YOLO

    weights = Path(cfg["inference"]["weights"])
    if not weights.exists():
        raise FileNotFoundError(f"Pesos no encontrados: {weights}. Entrena primero.")

    out_dir = Path(cfg["export"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    imgsz   = cfg["export"]["imgsz"]
    half    = cfg["export"].get("half", False)
    simplif = cfg["export"].get("simplify", True)
    opset   = cfg["export"].get("opset", 12)

    for fmt in cfg["export"]["formats"]:
        logger.info(f"==> Exportando a {fmt} (imgsz={imgsz}, half={half})")
        try:
            exported = model.export(
                format   = fmt,
                imgsz    = imgsz,
                half     = half,
                simplify = simplif if fmt == "onnx" else False,
                opset    = opset if fmt == "onnx" else None,
            )
            src = Path(exported)
            if src.is_file():
                dst = out_dir / src.name
                shutil.copy2(src, dst)
                logger.info(f"   Copiado: {dst}")
            elif src.is_dir():
                dst = out_dir / src.name
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                logger.info(f"   Copiado dir: {dst}")
        except Exception as e:
            logger.error(f"   Fallo export {fmt}: {e}")

    logger.info(f"Export completado en {out_dir}")
    logger.info("Copia el archivo a la Raspberry Pi:")
    logger.info(f"   scp {out_dir}/best.onnx pi@<ip>:~/weapons/")


if __name__ == "__main__":
    main()
