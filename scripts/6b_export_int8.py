"""
Script 6b: Export YOLOv8n a ONNX FP32 + cuantizacion INT8 calibrada.

Salidas en models/export/:
    yolov8n_v2_fp32.onnx       ~6 MB
    yolov8n_v2_int8.onnx       ~3 MB (cuantizado, listo para Pi)
"""
import logging
import shutil
import random
import yaml
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
CALIB_N = 100  # imgs para calibrar


def load_cfg():
    with open(REPO / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def export_fp32(weights_pt: Path, imgsz: int, opset: int) -> Path:
    from ultralytics import YOLO
    model = YOLO(str(weights_pt))
    log.info(f"Exportando {weights_pt.name} -> ONNX FP32 (imgsz={imgsz})")
    out = model.export(
        format="onnx",
        imgsz=imgsz,
        half=False,
        simplify=True,
        opset=opset,
        dynamic=False,
    )
    return Path(out)


class CalibReader:
    """Calibration data reader para onnxruntime.quantization."""
    def __init__(self, image_paths, input_name, imgsz):
        self.image_paths = image_paths
        self.input_name = input_name
        self.imgsz = imgsz
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.image_paths):
            return None
        p = self.image_paths[self.idx]
        self.idx += 1
        try:
            with Image.open(p) as im:
                im = im.convert("RGB").resize((self.imgsz, self.imgsz), Image.BILINEAR)
                arr = np.asarray(im, dtype=np.float32) / 255.0
                arr = arr.transpose(2, 0, 1)[None, ...]  # NCHW
            log.info(f"  calib {self.idx}/{len(self.image_paths)}: {p.name}")
            return {self.input_name: arr.astype(np.float32)}
        except Exception as e:
            log.warning(f"  saltada {p.name}: {e}")
            return self.get_next()


def quantize_int8(fp32_path: Path, out_path: Path, calib_paths, imgsz: int):
    from onnxruntime.quantization import quantize_static, QuantType, CalibrationMethod
    import onnxruntime as ort

    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    log.info(f"Input ONNX: {input_name}")

    reader = CalibReader(calib_paths, input_name, imgsz)
    log.info(f"Cuantizando -> INT8 con {len(calib_paths)} imgs de calibracion...")
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(out_path),
        calibration_data_reader=reader,
        quant_format=2,  # QDQ
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )


def main():
    cfg = load_cfg()
    out_dir = REPO / cfg["export"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pesos YOLOv8n entrenado
    nano_run = cfg["export"]["run_name"]
    nano_weights = REPO / "runs" / "detect" / "models" / "yolov8_weapons" / "weapons" / nano_run / "weights" / "best.pt"
    if not nano_weights.exists():
        # Fallback al path del config si lo movimos
        alt = REPO / "models" / "yolov8_weapons" / "weapons" / nano_run / "weights" / "best.pt"
        if alt.exists():
            nano_weights = alt
        else:
            raise FileNotFoundError(f"No encontrado: {nano_weights}")
    log.info(f"Pesos: {nano_weights}")

    imgsz = cfg["export"]["imgsz"]
    opset = cfg["export"].get("opset", 12)

    # 1. ONNX FP32
    fp32_temp = export_fp32(nano_weights, imgsz, opset)
    fp32_dst = out_dir / "yolov8n_v2_fp32.onnx"
    shutil.copy2(fp32_temp, fp32_dst)
    log.info(f"FP32 -> {fp32_dst} ({fp32_dst.stat().st_size/1024:.1f} KB)")

    # 2. Calibration set: 100 imgs random del train
    train_imgs_dir = REPO / "data" / "processed" / "images" / "train"
    all_imgs = list(train_imgs_dir.glob("*.jpg")) + list(train_imgs_dir.glob("*.png"))
    random.seed(42)
    calib = random.sample(all_imgs, min(CALIB_N, len(all_imgs)))
    log.info(f"Set de calibracion: {len(calib)} imgs de {train_imgs_dir}")

    # 3. INT8
    int8_dst = out_dir / "yolov8n_v2_int8.onnx"
    quantize_int8(fp32_dst, int8_dst, calib, imgsz)
    log.info(f"INT8 -> {int8_dst} ({int8_dst.stat().st_size/1024:.1f} KB)")

    # 4. Resumen
    fp32_kb = fp32_dst.stat().st_size / 1024
    int8_kb = int8_dst.stat().st_size / 1024
    log.info("=" * 50)
    log.info(f"FP32: {fp32_kb:>8.1f} KB ({fp32_kb/1024:.2f} MB)")
    log.info(f"INT8: {int8_kb:>8.1f} KB ({int8_kb/1024:.2f} MB)  -> {(1-int8_kb/fp32_kb)*100:.0f}% mas chico")
    log.info("=" * 50)
    log.info(f"Listo para copiar a la Pi:")
    log.info(f"   scp {int8_dst} pi@<ip>:~/weapons/best.onnx")


if __name__ == "__main__":
    main()
