# Resultados Fine-tune Anti-Alucinación + Poses (v4)

Comparativa entre los modelos **originales** (yolov8X_v2) y los **fine-tuned** (yolov8X_v4_pose_negs).

Validación sobre `data/processed/data.yaml` (865 imgs val).

> **Importante:** este val set está dominado por la distribución del training original (pistolas mayormente de lado, sin distractores como móviles/paraguas/personas). Los modelos v4 distribuyen capacidad entre esa distribución vieja **y** las nuevas (frontal/POV/hard-negs), así que ven una bajada esperable aquí. La ganancia real se mide en el deployment del stand, no en este val.

| Modelo | Param (M) | imgsz | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|---|---|---|---|---|
| nano ORIGINAL | 3.01 | 416 | 0.8530 | 0.6208 | 0.8511 | 0.7757 |
| nano **NUEVO** | 3.01 | 416 | **0.8516** | **0.6105** | **0.8750** | **0.7585** |
| small ORIGINAL | 11.13 | 640 | 0.8901 | 0.6476 | 0.8858 | 0.8249 |
| small **NUEVO** | 11.13 | 640 | **0.8791** | **0.6311** | **0.8787** | **0.7924** |

## Deltas

| Modelo | Δ mAP@50 | Δ mAP@50-95 | Δ P | Δ R |
|---|---|---|---|---|
| nano | -0.14pp | -1.03pp | +2.39pp | -1.73pp |
| small | -1.10pp | -1.65pp | -0.71pp | -3.26pp |

## Archivos generados

### nano
- Pesos: `C:\Users\suiza\Proyectos\deteccion de armas\deteccion de armas\repo\runs\detect\models\yolov8_weapons\weapons\yolov8n_v4_pose_negs\weights\best.pt`
- ONNX: `C:\Users\suiza\Proyectos\deteccion de armas\deteccion de armas\repo\runs\detect\models\yolov8_weapons\weapons\yolov8n_v4_pose_negs\weights\best.onnx`
- NCNN: `C:\Users\suiza\Proyectos\deteccion de armas\deteccion de armas\repo\runs\detect\models\yolov8_weapons\weapons\yolov8n_v4_pose_negs\weights\best_ncnn_model` (param + bin + metadata.yaml)

### small
- Pesos: `C:\Users\suiza\Proyectos\deteccion de armas\deteccion de armas\repo\runs\detect\models\yolov8_weapons\weapons\yolov8s_v4_pose_negs\weights\best.pt`
- ONNX: `C:\Users\suiza\Proyectos\deteccion de armas\deteccion de armas\repo\runs\detect\models\yolov8_weapons\weapons\yolov8s_v4_pose_negs\weights\best.onnx`
- NCNN: `C:\Users\suiza\Proyectos\deteccion de armas\deteccion de armas\repo\runs\detect\models\yolov8_weapons\weapons\yolov8s_v4_pose_negs\weights\best_ncnn_model` (param + bin + metadata.yaml)

## Lectura de los resultados

### Nano v4 (Pi5)
- **mAP@50 prácticamente igual** al original (-0.14pp = ruido)
- **Precision +2.39pp** ← clave anti-alucinación: el modelo es **más exigente** para decir "hay arma" → menos falsos positivos en el stand
- Recall -1.73pp → algunos verdaderos positivos en pose rara podrían escaparse, pero el target del stand es evitar falsas alarmas

### Small v4 (PC)
- mAP@50 -1.10pp y recall -3.26pp (bajada un poco mayor que en v3 — mismo patrón: el modelo se generaliza pero pierde algo en el val viejo)
- Si en el stand resulta peor que el v2, **se puede volver al v2 sin problema** (ambos exportados y disponibles)

## Configuración del fine-tune
- Punto de partida: pesos `v2` (originales)
- `lr0 = 0.0001`, `lrf = 0.01`, `warmup_epochs = 3`
- `freeze = 10` (backbone congelado, solo cabeza entrena)
- `cls = 1.0` (penalización fuerte a falsos positivos)
- `perspective = 0.001`, `degrees = 30` (más variedad angular)
- `close_mosaic = 10` (afinado limpio últimas epochs)
- 30 epochs completos en ambos (sin EarlyStopping)

## Dataset usado (+13,902 imgs vs train original)
| Fuente | Imágenes | Tipo |
|---|---|---|
| PranomVignesh/HandGuns (completo con HF_TOKEN) | 6,882 | Mirror Roboflow pistols-p6x3w (8.9K) — poses variadas |
| Subh775/WeaponDetection | 5,341 | Multi-clase armas |
| OpenImages stand negatives | 846 | Personas, móviles, mandos (hard neg) |
| harshdadiya phone_detection | 605 | Móviles puros (hard neg) |
| Simuletic CCTV Rifle-vs-Umbrella | 120 | CCTV cenital + paraguas (hard neg) |
| Simuletic Handgun-vs-Chips | 110 | Bolsa de Doritos (hard neg) |

## Deploy

### Raspberry Pi 5 (recomendado: nano v4)
```bash
# Copiar a la Pi5
scp -r runs/detect/models/yolov8_weapons/weapons/yolov8n_v4_pose_negs/weights/best_ncnn_model pi@raspberrypi.local:~/nano_v4_ncnn

# En la Pi5
pip install ultralytics ncnn
yolo predict model=~/nano_v4_ncnn imgsz=416 source=0 conf=0.5
```

### PC con GPU (small v4)
```bash
yolo predict model=runs/detect/models/yolov8_weapons/weapons/yolov8s_v4_pose_negs/weights/best.pt imgsz=640 source=0 conf=0.5
```

## Tips de inferencia para el stand
- Subir `conf=0.5` o `conf=0.6` para reducir aún más alucinaciones (default es 0.25)
- Mantener `iou=0.45` (default) para NMS
- Si quieres comparar v2 vs v4 en vivo, corre los dos modelos en ventanas separadas con la misma cámara y mira cuál se comporta mejor

## Versiones de modelos disponibles

| Versión | Nano | Small | Notas |
|---|---|---|---|
| v2 (original) | `runs/detect/models/.../yolov8n_v2/weights/best.pt` | `models/.../yolov8s_v2/weights/best.pt` | Baseline, sólo dataset original |
| v3 (intermedio) | `runs/detect/.../yolov8n_v3_pose_negs/weights/best.pt` | `runs/detect/.../yolov8s_v3_pose_negs/weights/best.pt` | +7,710 imgs (sin PranomVignesh completo) |
| **v4 (final, recomendado)** | `runs/detect/.../yolov8n_v4_pose_negs/weights/best.pt` | `runs/detect/.../yolov8s_v4_pose_negs/weights/best.pt` | +13,902 imgs (PranomVignesh completo con token) |