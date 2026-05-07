================================================================================
  DETECCION DE ARMAS CON YOLOv8 - GUIA DE USO COMPLETA
================================================================================

Sistema de deteccion en tiempo real de armas blancas (cuchillos) y armas de
fuego (pistolas, rifles, escopetas) usando YOLOv8.

  - Entrenamiento: laptop con GPU NVIDIA (CUDA)
  - Despliegue:    Raspberry Pi 5 (ONNX Runtime CPU, modelo INT8 cuantizado)

Resultados v2:
  - mAP@0.5 (test):    0.73
  - mAP@0.5 handgun:   0.80
  - Modelo Pi (INT8):  3.25 MB

================================================================================
  TABLA DE CONTENIDOS
================================================================================

  A) ARRANCAR EN UNA PC NUEVA DESDE CERO
  B) FLUJO COMPLETO PASO A PASO
  C) USO RAPIDO (modelo ya entrenado)
  D) DESPLIEGUE EN RASPBERRY PI 5
  E) ESTRUCTURA DEL REPOSITORIO
  F) TROUBLESHOOTING COMUN
  G) VARIABLES DE ENTORNO Y CONFIG IMPORTANTE


================================================================================
  A) ARRANCAR EN UNA PC NUEVA DESDE CERO
================================================================================

REQUISITOS:
  - Windows 10/11 o Linux con Python 3.10 u 3.11 (NO 3.12+)
  - GPU NVIDIA (recomendado): drivers >= 550 con soporte CUDA 12.x
  - 32 GB RAM (16 GB minimo)
  - 30 GB libres en disco
  - Cuenta gratuita en Roboflow (https://app.roboflow.com) con API key

NOTA IMPORTANTE: si tu GPU es serie RTX 50xx (Blackwell, sm_120), NECESITAS
PyTorch compilado contra CUDA 12.8 o superior. Las versiones cu121/cu124 no
funcionan con sm_120 y caen al CPU sin avisar.

------------------------------------------------------------
  PASO 1: Clonar el repositorio
------------------------------------------------------------

  git clone https://github.com/Enryuuh/deteccion-de-armas-i.a.git
  cd deteccion-de-armas-i.a

------------------------------------------------------------
  PASO 2: Instalar Python 3.11 (si no lo tienes)
------------------------------------------------------------

  Windows:
    winget install -e --id Python.Python.3.11

  Linux:
    sudo apt install python3.11 python3.11-venv python3-pip

------------------------------------------------------------
  PASO 3: Crear virtualenv e instalar PyTorch con CUDA correcta
------------------------------------------------------------

  Windows (PowerShell):
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

  Linux/Mac:
    python3.11 -m venv .venv
    source .venv/bin/activate

  Actualizar pip:
    python -m pip install --upgrade pip

  Instalar PyTorch (elegir la opcion que corresponda):

    GPU NVIDIA RTX 50xx (Blackwell, sm_120) - OBLIGATORIO cu128:
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

    GPU NVIDIA RTX 30xx/40xx:
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    Sin GPU (CPU only):
      pip install torch torchvision

------------------------------------------------------------
  PASO 4: Instalar el resto de dependencias
------------------------------------------------------------

  pip install -r requirements.txt

  Para el export INT8 (paso 9):
    pip install reportlab roboflow

------------------------------------------------------------
  PASO 5: Verificar que PyTorch ve la GPU
------------------------------------------------------------

  python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

  Salida esperada con GPU:
    CUDA: True
    Device: NVIDIA GeForce RTX 5050 Laptop GPU

  Si dice "CUDA: False" pero tienes GPU, REVISA la version de PyTorch instalada.

------------------------------------------------------------
  PASO 6: Configurar API key de Roboflow
------------------------------------------------------------

  1) Ir a https://app.roboflow.com/login y crear cuenta.
  2) Settings -> Workspace -> Roboflow API -> copiar Private API Key.
  3) Editar scripts/2b_merge_datasets.py o exportar como variable de entorno:

  Windows:
    $env:ROBOFLOW_API_KEY="tu-api-key-aqui"

  Linux:
    export ROBOFLOW_API_KEY="tu-api-key-aqui"


================================================================================
  B) FLUJO COMPLETO PASO A PASO
================================================================================

Tiempo total estimado: 4-6 horas (dominado por descarga + entrenamiento).

------------------------------------------------------------
  PASO 7: Descargar Open Images v7 (FiftyOne)
------------------------------------------------------------

  python scripts/1_download_dataset.py

  Tiempo: 5-15 min segun la conexion.
  Output: ~/fiftyone/open-images-v7/ + datasets persistentes en FiftyOne.

------------------------------------------------------------
  PASO 8: Descargar datasets de Roboflow Universe
------------------------------------------------------------

  Editar scripts/2b_merge_datasets.py si quieres cambiar la API key
  hardcodeada (cerca del top), o usa la variable de entorno.

  Descargar pistols + rifles desde Roboflow:

  python -c "
  from roboflow import Roboflow
  import os
  os.makedirs('data/rf', exist_ok=True)
  os.chdir('data/rf')
  rf = Roboflow(api_key=os.environ.get('ROBOFLOW_API_KEY'))
  rf.workspace('joseph-nelson').project('pistols').version(1).download('yolov8', location='pistols')
  rf.workspace('rifledetection').project('rifle-detection-p9slr').version(1).download('yolov8', location='rifle')
  "

  Tiempo: ~3 min. Total descargado: ~1.8 GB.

------------------------------------------------------------
  PASO 9: Mergear datasets en formato YOLO unificado
------------------------------------------------------------

  python scripts/2b_merge_datasets.py

  Salida: data/processed/ con images/ + labels/ + data.yaml.
  Total: 8.656 imagenes con 3 clases (knife, handgun, long_gun).

------------------------------------------------------------
  PASO 10: Entrenar YOLOv8s (modelo de referencia)
------------------------------------------------------------

  Recomendado: usar el wrapper para que active env vars de proteccion VRAM:

  Windows:
    .\run_train.ps1

  Linux:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES=0 \
    python scripts/3_train.py

  Tiempo en RTX 5050: ~3 horas (100 epochs).
  Output: runs/detect/models/yolov8_weapons/weapons/yolov8s_v2/weights/best.pt

------------------------------------------------------------
  PASO 11: Entrenar YOLOv8n (para Raspberry Pi)
------------------------------------------------------------

  python scripts/3b_train_nano.py

  Tiempo: ~50 min con cache RAM.
  Output: runs/detect/models/yolov8_weapons/weapons/yolov8n_v2/weights/best.pt

------------------------------------------------------------
  PASO 12: Evaluar el modelo en test
------------------------------------------------------------

  python scripts/4_evaluate.py

  Imprime mAP@0.5, mAP@0.5:0.95, Precision, Recall por clase.

------------------------------------------------------------
  PASO 13: Probar inferencia en camara (laptop)
------------------------------------------------------------

  python scripts/5_inference_camera.py

  Abre ventana en vivo con bbox y FPS. Esc para salir.

------------------------------------------------------------
  PASO 14: Exportar a ONNX INT8 para la Pi
------------------------------------------------------------

  python scripts/6b_export_int8.py

  Output:
    models/export/yolov8n_v2_fp32.onnx   ~12 MB
    models/export/yolov8n_v2_int8.onnx   ~3.25 MB  <-- el de la Pi


================================================================================
  C) USO RAPIDO (modelo ya entrenado)
================================================================================

Si solo quieres USAR el modelo (sin reentrenar):

  1) Clonar el repo (paso 1).
  2) Crear venv e instalar dependencias (pasos 2-4).
  3) Descargar los pesos pre-entrenados desde Releases del repositorio
     (cuando esten disponibles) o entrenarlos con los pasos 7-11.
  4) Correr inferencia en camara:

     python scripts/5_inference_camera.py

  Si la confidence es muy baja para tu prueba, editar config.yaml:

     inference:
       confidence_threshold: 0.30   # bajar de 0.50


================================================================================
  D) DESPLIEGUE EN RASPBERRY PI 5
================================================================================

Ver seccion 12 del PDF (docs/Documento_Tecnico_Deteccion_Armas.pdf) para
el detalle completo con SSH desde VS Code, systemd, etc.

Resumen ultra-rapido:

  # En la Pi (una sola vez):
  sudo apt update && sudo apt install -y python3-pip python3-venv libatlas-base-dev
  python3 -m venv ~/weapons-env
  source ~/weapons-env/bin/activate
  pip install opencv-python onnxruntime numpy pyyaml pygame
  mkdir -p ~/weapons/logs/frames

  # Desde la laptop:
  scp models/export/yolov8n_v2_int8.onnx pi@<ip>:~/weapons/best.onnx
  scp config.yaml pi@<ip>:~/weapons/
  scp -r utils pi@<ip>:~/weapons/
  scp scripts/7_inference_pi.py pi@<ip>:~/weapons/

  # En la Pi:
  cd ~/weapons
  python 7_inference_pi.py --weights best.onnx


================================================================================
  E) ESTRUCTURA DEL REPOSITORIO
================================================================================

  deteccion-de-armas-i.a/
  |- README.txt                    (este archivo)
  |- requirements.txt              (dependencias Python)
  |- config.yaml                   (configuracion central: clases, paths, hiperparametros)
  |- run_train.ps1                 (wrapper PowerShell con env vars de seguridad VRAM)
  |- .gitignore
  |
  |- scripts/
  |  |- 1_download_dataset.py      (descarga Open Images v7 via FiftyOne)
  |  |- 2_prepare_dataset.py       (legacy v1, conversion OIv7 -> YOLO)
  |  |- 2b_merge_datasets.py       (v2: merge OIv7 + Roboflow, 3 clases)
  |  |- 3_train.py                 (entrena YOLOv8s)
  |  |- 3b_train_nano.py           (entrena YOLOv8n para Pi)
  |  |- 4_evaluate.py              (mAP en test)
  |  |- 5_inference_camera.py      (inferencia laptop con webcam)
  |  |- 6_export.py                (export ONNX/NCNN sin cuantizar)
  |  |- 6b_export_int8.py          (export ONNX FP32 + INT8 calibrado)
  |  \- 7_inference_pi.py          (inferencia Raspberry Pi 5 con ONNX Runtime)
  |
  |- utils/
  |  |- visualization.py           (dibujo de bboxes y HUD)
  |  \- alerts.py                  (sonido + log + frames de evidencia)
  |
  |- docs/
  |  |- Documento_Tecnico_Deteccion_Armas.pdf
  |  \- generate_doc.py            (genera el PDF anterior)
  |
  |- data/                         (NO en git - generado)
  |  |- raw/                       (descargas originales)
  |  |- rf/                        (Roboflow datasets)
  |  \- processed/                 (formato YOLO + data.yaml)
  |
  |- models/                       (NO en git - generado)
  |  |- yolov8_weapons/
  |  |  \- weapons/
  |  |     |- yolov8s_v2/weights/best.pt    (modelo grande, laptop)
  |  |     \- yolov8n_v2/weights/best.pt    (modelo nano, base de Pi)
  |  \- export/
  |     |- yolov8n_v2_fp32.onnx
  |     \- yolov8n_v2_int8.onnx             (3.25 MB, listo para Pi)
  |
  \- runs/                         (NO en git - generado por Ultralytics)
     \- detect/
        |- models/yolov8_weapons/...
        \- val/, val-2/, ...


================================================================================
  F) TROUBLESHOOTING COMUN
================================================================================

PROBLEMA: torch.cuda.is_available() = False con GPU NVIDIA presente

  Causa probable: PyTorch instalado contra CUDA equivocada.
  Solucion:
    pip uninstall torch torchvision -y
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

  Para RTX 50xx (Blackwell) DEBES usar cu128 o superior.

----------------------------------------

PROBLEMA: CUDA out of memory durante el entrenamiento

  Causas: batch demasiado alto, fragmentacion, otra app usando VRAM.
  Solucion:
    1) En config.yaml bajar batch (12 -> 8 -> 6).
    2) Setear: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    3) Cerrar Chrome/Edge y otros consumidores de VRAM.
    4) En config.yaml bajar workers (8 -> 4).

----------------------------------------

PROBLEMA: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED

  Tipico de Blackwell con batch grandes en yolov8n a 416px.
  Solucion: bajar batch (32 -> 24 -> 16).

----------------------------------------

PROBLEMA: Dataset no descarga (FiftyOne timeout)

  Solucion: reintentar. Si persiste, descargar manualmente desde:
    https://storage.googleapis.com/openimages/web/download.html

----------------------------------------

PROBLEMA: Roboflow API key rechazada

  Verificar que copiaste la "Private API Key" (no la Publishable).
  Esta en: https://app.roboflow.com/?workspace=... -> Settings -> Workspace -> Roboflow API.

----------------------------------------

PROBLEMA: Inferencia con webcam abre ventana negra

  Solucion: probar otros indices de camara editando config.yaml:
    inference:
      camera_index: 0  # probar 1, 2

  En Linux: verificar permisos con ls /dev/video*

----------------------------------------

PROBLEMA: Pantalla parpadea negro durante el entrenamiento (TDR de Windows)

  Esto es normal con GPU al 86%+ sostenido. El computo CUDA NO se interrumpe.
  Si te molesta, agregar al registro:
    HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\TdrDelay = 60 (DWORD)
  Reiniciar.

----------------------------------------

PROBLEMA: La pistola 3D impresa no la detecta

  Causas: color/material fuera de distribucion, mucha distancia.
  Soluciones (en orden):
    1) Imprimir en NEGRO MATE (no rosa, blanco, ni brillante).
    2) Acercar a 50 cm de la camara.
    3) Bajar threshold en config.yaml: confidence_threshold: 0.30
    4) Mostrar de costado (no frontal).


================================================================================
  G) VARIABLES DE ENTORNO Y CONFIG IMPORTANTE
================================================================================

config.yaml - secciones criticas:

  classes:
    - "knife"
    - "handgun"
    - "long_gun"

  training:
    batch: 8        # bajar si OOM
    workers: 4
    epochs: 100
    patience: 30

  inference:
    weights: "models/yolov8_weapons/weapons/yolov8s_v2/weights/best.pt"
    confidence_threshold: 0.50    # bajar a 0.30 para pruebas
    iou_threshold: 0.45
    camera_index: 0

  export:
    model: "yolov8n.pt"
    run_name: "yolov8n_v2"
    imgsz: 416

Variables de entorno recomendadas durante entrenamiento:

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # evita OOM al final
  CUDA_VISIBLE_DEVICES=0                              # blinda la GPU NVIDIA
  KMP_DUPLICATE_LIB_OK=TRUE                           # workaround OpenMP en Win

================================================================================
  CONTACTO Y RECURSOS
================================================================================

  - Repositorio:  https://github.com/Enryuuh/deteccion-de-armas-i.a
  - Documento tecnico completo: docs/Documento_Tecnico_Deteccion_Armas.pdf
  - YOLOv8 docs:  https://docs.ultralytics.com/
  - Roboflow:     https://universe.roboflow.com/

================================================================================
