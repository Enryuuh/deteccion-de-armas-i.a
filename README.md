# 🔫 Detección de Armas — Sistema de Visión Artificial

Sistema de detección en tiempo real de **cuchillos y armas de fuego** usando **RT-DETR** (Real-Time Detection Transformer) y una cámara web estándar.

> **Licencias**: Apache 2.0 / BSD / CC BY 4.0 — 100% libre para uso académico, comercial e institucional.

---

## 🏗️ Tecnología

| Componente | Tecnología | Licencia |
|---|---|---|
| Modelo | RT-DETR-R50 (Hugging Face) | Apache 2.0 |
| Dataset | Open Images v7 (Google) | CC BY 4.0 |
| Framework | PyTorch 2.x + CUDA | BSD |
| Visión | OpenCV | Apache 2.0 |
| Métricas | pycocotools | Apache 2.0 |

**GPU recomendada:** NVIDIA RTX 4060+ (8 GB VRAM) con CUDA 12.x

---

## 📁 Estructura del Proyecto

```
deteccion de armas i.a/
├── config.yaml                    ← Configuración central
├── requirements.txt               ← Dependencias
├── scripts/
│   ├── 1_download_dataset.py      ← Descarga Open Images v7
│   ├── 2_prepare_dataset.py       ← Convierte a formato COCO
│   ├── 3_train.py                 ← Entrena RT-DETR en tu GPU
│   ├── 4_evaluate.py              ← Mide mAP en test set
│   └── 5_inference_camera.py      ← Detección en tiempo real
├── utils/
│   ├── dataset.py                 ← Dataset COCO personalizado
│   ├── visualization.py           ← Bounding boxes y UI
│   └── alerts.py                  ← Sistema de alertas
├── data/                          ← Dataset descargado (no en git)
├── models/                        ← Pesos entrenados (no en git)
└── logs/                          ← Logs y frames detectados
```

---

## 🚀 Instalación y Uso

### 1. Requisitos del Sistema

- Python 3.10+
- NVIDIA RTX 4060 (8 GB VRAM)
- CUDA 12.x + cuDNN
- ~20 GB espacio en disco

### 2. Instalar Dependencias

```bash
pip install -r requirements.txt
```

> Para CUDA: instalar PyTorch desde https://pytorch.org/get-started/locally/
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

### 3. Descargar Dataset

```bash
python scripts/1_download_dataset.py
```

### 4. Preparar Dataset

```bash
python scripts/2_prepare_dataset.py
```

### 5. Entrenar Modelo

```bash
python scripts/3_train.py
```

> El entrenamiento con RTX 4060 toma ~2-4 horas con 2000 imágenes por clase.
> Progreso visible en TensorBoard: `tensorboard --logdir models/`

### 6. Evaluar Modelo

```bash
python scripts/4_evaluate.py
```

### 7. Inferencia en Tiempo Real

```bash
python scripts/5_inference_camera.py
```

---

## 🎯 Clases Detectadas

| ID | Clase | Descripción |
|---|---|---|
| 1 | `knife` | Cuchillos |
| 2 | `firearm` | Armas de fuego (pistolas, escopetas) |

---

## 📊 Rendimiento Esperado (RTX 4060)

| Métrica | Valor esperado |
|---|---|
| Velocidad inferencia | ~30-60 FPS |
| mAP@0.5 (fine-tuned) | ~75-85% |
| VRAM usada | ~4-6 GB |
| Precisión FP16 | ✅ Activada |

---

## 📄 Licencias y Atribución

- **RT-DETR**: Baidu Inc. — Apache License 2.0
- **Open Images v7**: Google LLC — CC BY 4.0 (requiere atribución)
- **PyTorch**: Meta Platforms — BSD 3-Clause
- **OpenCV**: OpenCV team — Apache 2.0

---

## ✍️ Autor

Proyecto desarrollado para presentación ante entidades públicas de financiación I+D.
