# Deteccion de Armas - Sistema de Vision Artificial

Sistema de deteccion en tiempo real de **armas blancas (cuchillos)** y **armas de fuego (pistolas, escopetas)** usando **YOLOv8**, optimizado para entrenar en una laptop con **RTX 5050** y desplegar en **Raspberry Pi 5**.

> Licencias: AGPL-3.0 (ultralytics) / Apache 2.0 / BSD / CC BY 4.0.

---

## Inicio rapido (clonar y ejecutar)

**El modelo ya entrenado viene incluido en el repo.** No hace falta descargar el
dataset ni reentrenar: clonas, instalas dependencias y corre.

```bash
# 1. Clonar
git clone https://github.com/Enryuuh/deteccion-de-armas-i.a.git
cd deteccion-de-armas-i.a

# 2. Crear entorno virtual (Python 3.10+)
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS

# 3. Instalar PyTorch
#    -- Con GPU NVIDIA (CUDA 12.x):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#    -- Solo CPU (sin tarjeta grafica):
# pip install torch torchvision

# 4. Instalar el resto de dependencias
pip install -r requirements.txt

# 5. Ejecutar la deteccion con la camara
python scripts/5_inference_camera.py
```

> `fiftyone` (en requirements.txt) solo se usa para **descargar** el dataset.
> Si solo vas a ejecutar la deteccion, puedes omitirlo si te da problemas al instalar.

### Pesos incluidos en el repo

| Archivo | Uso | Tamano |
|---|---|---|
| `runs/detect/models/yolov8_weapons/weapons/yolov8s_v4_pose_negs/weights/best.pt` | **Laptop** (modelo por defecto en `config.yaml`) | 22 MB |
| `runs/detect/models/yolov8_weapons/weapons/yolov8n_v4_pose_negs/weights/best.pt` | Raspberry Pi (PyTorch) | 6 MB |
| `models/export/yolov8n_v4_fp32.onnx` | Raspberry Pi (ONNX Runtime) | 12 MB |
| `models/export/yolov8n_v4_int8.onnx` | Raspberry Pi (ONNX cuantizado) | 3.3 MB |

El resto de secciones (descarga de dataset, entrenamiento, evaluacion) solo son
necesarias si quieres **reentrenar** el modelo desde cero.

---

## Arquitectura

| Componente | Tecnologia |
|---|---|
| Modelo (entrenamiento) | YOLOv8s (~11M params) |
| Modelo (Raspberry Pi) | YOLOv8n exportado a ONNX/NCNN (~6 MB) |
| Dataset | Open Images v7 (clases: Knife, Firearm, Handgun, Shotgun) |
| Framework entrenamiento | PyTorch + Ultralytics |
| Inferencia laptop | Ultralytics |
| Inferencia Pi5 | ONNX Runtime (CPU) |
| Vision | OpenCV |
| Alertas | pygame + log + frames de evidencia |

---

## Estructura del proyecto

```
deteccion-de-armas-i.a/
├── config.yaml                    # configuracion central
├── requirements.txt               # deps de la laptop (entrenamiento)
├── scripts/
│   ├── 1_download_dataset.py      # Open Images v7 via FiftyOne
│   ├── 2_prepare_dataset.py       # convierte a formato YOLO
│   ├── 3_train.py                 # entrena YOLOv8 en RTX 5050
│   ├── 4_evaluate.py              # mAP en test
│   ├── 5_inference_camera.py      # inferencia laptop (.pt)
│   ├── 6_export.py                # exporta ONNX/NCNN para Pi5
│   └── 7_inference_pi.py          # inferencia Raspberry Pi 5 (ONNX)
├── utils/
│   ├── visualization.py           # bboxes + HUD
│   └── alerts.py                  # sonido / log / frames
├── data/                          # dataset (no en git)
├── models/                        # pesos (no en git)
└── logs/                          # detecciones (no en git)
```

---

## Instalacion (laptop de entrenamiento)

### 1. Requisitos
- Python 3.10+
- NVIDIA RTX 5050 (8 GB VRAM) con CUDA 12.x
- ~20 GB libres de disco

### 2. Instalar PyTorch con CUDA 12.x
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Instalar el resto
```bash
pip install -r requirements.txt
```

---

## Flujo completo

```bash
# 1. Descargar dataset (~5-15 GB)
python scripts/1_download_dataset.py

# 2. Convertir a formato YOLO
python scripts/2_prepare_dataset.py

# 3. Entrenar (RTX 5050, ~30-90 min para 80 epochs)
python scripts/3_train.py

# 4. Evaluar
python scripts/4_evaluate.py

# 5. Probar en camara de la laptop
python scripts/5_inference_camera.py

# 6. Exportar para Raspberry Pi 5
python scripts/6_export.py
# -> genera models/export/best.onnx
```

---

## Despliegue en Raspberry Pi 5

En la Pi solo hace falta un set minimo de dependencias:

```bash
# En la Pi
pip install opencv-python onnxruntime numpy pyyaml pygame
```

Copiar el modelo y los archivos esenciales:

```bash
# Desde la laptop
scp models/export/best.onnx pi@<ip>:~/weapons/
scp config.yaml             pi@<ip>:~/weapons/
scp -r utils                pi@<ip>:~/weapons/
scp scripts/7_inference_pi.py pi@<ip>:~/weapons/
```

Ejecutar en la Pi:

```bash
python 7_inference_pi.py --weights best.onnx
```

**No se necesita PyTorch ni ultralytics en la Pi.**

---

## Reconocimiento facial (identificacion de personas)

Ademas de detectar armas, el sistema puede **identificar de quien** son las
caras presentes, comparando contra personas previamente matriculadas. Usa
InsightFace (embeddings ArcFace, ONNX) — **no se entrena**, se matricula.

```bash
# 1. Matricular personas (registrar quien es quien)
python scripts/20_enroll_faces.py --capture "Tu Nombre"        # con webcam
#   o desde carpetas faces_db/<nombre>/*.jpg:
python scripts/20_enroll_faces.py --from-folder faces_db
python scripts/20_enroll_faces.py --list                       # ver matriculados

# 2. Activar en config.yaml:  face_recognition.enabled: true
# 3. Correr la deteccion normal: las caras conocidas apareceran con su nombre,
#    y las alertas registraran QUIEN portaba el arma.
python scripts/5_inference_camera.py
```

- **Laptop/GPU:** `model_pack: buffalo_l`, `use_gpu: true`.
- **Raspberry Pi 5:** `model_pack: buffalo_sc`, `use_gpu: false`, `only_on_weapon: true`
  (reconoce solo al detectar un arma, para no bajar los FPS).
- Los datos biometricos (`faces_db/`, `models/faces/`) **no se suben a git**.
  Matricula solo a personas que dieron su consentimiento.

---

## Clases detectadas

| ID | Clase | Descripcion |
|---|---|---|
| 0 | `knife` | Armas blancas (cuchillos) |
| 1 | `firearm` | Armas de fuego (pistolas, escopetas) |

---

## Rendimiento esperado

| Plataforma | Modelo | Resolucion | FPS |
|---|---|---|---|
| RTX 5050 Laptop | YOLOv8s `.pt` | 640 | 80-120 |
| Iris Xe (laptop sin CUDA) | YOLOv8s OpenVINO | 640 | 15-25 |
| Raspberry Pi 5 | YOLOv8n ONNX | 416 | 8-15 |

mAP@0.5 esperado tras fine-tuning: ~0.70-0.85.

---

## Licencias y atribucion

- **YOLOv8 / Ultralytics**: AGPL-3.0 (uso academico/no comercial). Para uso comercial requiere licencia.
- **Open Images v7**: Google LLC - CC BY 4.0 (requiere atribucion).
- **PyTorch**: Meta - BSD 3-Clause.
- **OpenCV**: OpenCV team - Apache 2.0.
- **ONNX Runtime**: Microsoft - MIT.
