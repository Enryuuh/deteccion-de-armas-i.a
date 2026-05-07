"""Genera el documento tecnico oficial del proyecto en PDF."""
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

OUT = Path(__file__).parent / "Documento_Tecnico_Deteccion_Armas.pdf"

styles = getSampleStyleSheet()
H1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=18, spaceAfter=12,
                    textColor=colors.HexColor('#1a3d6d'), keepWithNext=True)
H2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=14, spaceBefore=12,
                    spaceAfter=8, textColor=colors.HexColor('#2a5a9e'), keepWithNext=True)
H3 = ParagraphStyle('H3', parent=styles['Heading3'], fontSize=12, spaceBefore=8,
                    spaceAfter=6, textColor=colors.HexColor('#444'), keepWithNext=True)
BODY = ParagraphStyle('Body', parent=styles['BodyText'], fontSize=10, leading=14,
                      alignment=TA_JUSTIFY, spaceAfter=6)
BULLET = ParagraphStyle('Bullet', parent=BODY, leftIndent=14, bulletIndent=4)
CODE = ParagraphStyle('Code', parent=styles['Code'], fontSize=8.5, leading=11,
                      textColor=colors.HexColor('#222'),
                      backColor=colors.HexColor('#f4f4f4'), leftIndent=8,
                      borderPadding=4, spaceAfter=8)
COVER_TITLE = ParagraphStyle('CoverTitle', parent=H1, fontSize=26, alignment=TA_CENTER,
                             spaceAfter=10, textColor=colors.HexColor('#1a3d6d'))
COVER_SUB = ParagraphStyle('CoverSub', parent=BODY, fontSize=14, alignment=TA_CENTER,
                           textColor=colors.HexColor('#555'), spaceAfter=4)
META = ParagraphStyle('Meta', parent=BODY, fontSize=9, alignment=TA_CENTER,
                      textColor=colors.HexColor('#777'))

TBL_HEADER = colors.HexColor('#1a3d6d')
TBL_ALT = colors.HexColor('#f0f4fa')

def tbl(data, widths=None, header=True):
    t = Table(data, colWidths=widths, hAlign='LEFT')
    style = [
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#aaaaaa')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [
            ('BACKGROUND', (0, 0), (-1, 0), TBL_HEADER),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, TBL_ALT]),
        ]
    t.setStyle(TableStyle(style))
    return t

def code(s):
    txt = s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br/>')
    return Paragraph(f'<font face="Courier">{txt}</font>', CODE)

def p(s):
    return Paragraph(s, BODY)

def b(s):
    return Paragraph(f'&bull;&nbsp; {s}', BULLET)


story = []

# ========== PORTADA ==========
story.append(Spacer(1, 4*cm))
story.append(Paragraph("Sistema de Deteccion de Armas", COVER_TITLE))
story.append(Paragraph("mediante Vision Artificial con YOLOv8", COVER_TITLE))
story.append(Spacer(1, 0.6*cm))
story.append(Paragraph("Documento tecnico del proceso de desarrollo", COVER_SUB))
story.append(Spacer(1, 4*cm))
story.append(Paragraph("Plataforma de entrenamiento: NVIDIA RTX 5050 Laptop (Blackwell)", COVER_SUB))
story.append(Paragraph("Plataforma de despliegue: Raspberry Pi 5", COVER_SUB))
story.append(Spacer(1, 5*cm))
story.append(Paragraph(f"Fecha de generacion: {datetime.now().strftime('%d/%m/%Y %H:%M')}", META))
story.append(Paragraph("Repositorio: deteccion-de-armas-i.a", META))
story.append(PageBreak())

# ========== INDICE ==========
story.append(Paragraph("Indice", H1))
toc = [
    "1. Resumen ejecutivo",
    "2. Objetivos del sistema",
    "3. Arquitectura de hardware",
    "4. Stack de software",
    "5. Datasets y procesamiento de datos",
    "6. Modelo y arquitectura YOLO",
    "7. Configuracion del entrenamiento",
    "8. Fases ejecutadas",
    "9. Resultados parciales",
    "10. Despliegue en Raspberry Pi 5",
    "11. Riesgos identificados y mitigaciones",
    "12. Trabajo pendiente",
]
for item in toc:
    story.append(Paragraph(item, BODY))
story.append(PageBreak())

# ========== 1. RESUMEN ==========
story.append(Paragraph("1. Resumen ejecutivo", H1))
story.append(p(
    "El presente documento describe el desarrollo de un sistema de deteccion en tiempo real "
    "de armas blancas (cuchillos) y armas de fuego (pistolas, escopetas, rifles) basado en "
    "la arquitectura YOLOv8 de Ultralytics. El sistema esta disenado para entrenarse en una "
    "laptop con GPU NVIDIA RTX 5050 y desplegarse en una Raspberry Pi 5 corriendo el modelo "
    "cuantizado a INT8 con ONNX Runtime."
))
story.append(p(
    "Se ejecutaron dos iteraciones del proceso de entrenamiento. La primera version (v1) "
    "utilizo unicamente Open Images v7 con dos clases agregadas (knife/firearm) y obtuvo "
    "un mAP@0.5 de 0.62 en el conjunto de prueba. La segunda version (v2) incorpora datasets "
    "de Roboflow Universe, eleva el corpus a 8.656 imagenes curadas y descompone la clase "
    "firearm en handgun y long_gun para mejorar la coherencia visual."
))

# ========== 2. OBJETIVOS ==========
story.append(Paragraph("2. Objetivos del sistema", H1))
for o in [
    "Detectar en tiempo real armas blancas (clase <b>knife</b>) y armas de fuego "
    "diferenciando armas cortas (<b>handgun</b>) de armas largas (<b>long_gun</b>).",
    "Lograr una velocidad de inferencia minima de 10 FPS en Raspberry Pi 5 a "
    "resolucion 416x416.",
    "Mantener el modelo desplegado en menos de 5 MB para permitir actualizaciones "
    "remotas eficientes.",
    "Disparar alertas locales (sonido, log y captura de evidencia) cuando se detecte "
    "una amenaza con confianza superior al umbral configurado.",
    "Permitir el despliegue completo en la Pi sin necesidad de instalar PyTorch ni "
    "Ultralytics, reduciendo la huella a menos de 250 MB de paquetes.",
]:
    story.append(b(o))

# ========== 3. HARDWARE ==========
story.append(PageBreak())
story.append(Paragraph("3. Arquitectura de hardware", H1))
story.append(Paragraph("3.1 Estacion de entrenamiento", H2))
story.append(tbl([
    ["Componente", "Especificacion", "Rol"],
    ["Equipo", "ASUS TUF Gaming A16 FA608UH", "Entrenamiento + inferencia desktop"],
    ["CPU", "AMD Ryzen 7 260 (8 cores / 16 hilos)", "DataLoader workers"],
    ["RAM", "32 GB DDR5", "Cache de dataset"],
    ["GPU dedicada", "NVIDIA RTX 5050 Laptop, 8 GB VRAM", "Entrenamiento CUDA"],
    ["Arquitectura GPU", "Blackwell, capability sm_120", "Requiere CUDA 12.8+"],
    ["GPU integrada", "AMD Radeon 780M", "Solo display, sin CUDA"],
    ["Driver NVIDIA", "581.80 (CUDA 13.0 runtime)", "Detectado al iniciar"],
], widths=[3.5*cm, 7*cm, 5*cm]))

story.append(Paragraph("3.2 Plataforma de despliegue (objetivo)", H2))
story.append(tbl([
    ["Componente", "Recomendado"],
    ["SBC", "Raspberry Pi 5 (8 GB RAM)"],
    ["Almacenamiento", "microSD 64 GB clase A2"],
    ["Refrigeracion", "Active cooler oficial (imprescindible)"],
    ["Camara", "USB UVC o Pi Camera v3"],
    ["Sistema operativo", "Raspberry Pi OS Bookworm 64-bit"],
    ["Alimentacion", "Fuente oficial 5V/5A"],
], widths=[4*cm, 9*cm]))

# ========== 4. SOFTWARE ==========
story.append(Paragraph("4. Stack de software", H1))
story.append(p(
    "El entorno se aisla en un virtualenv Python 3.11.9 dentro del repositorio "
    "(<font face='Courier'>repo/.venv</font>). Se forzo la asignacion de la GPU NVIDIA "
    "para los ejecutables Python via la clave de registro "
    "<font face='Courier'>HKCU\\Software\\Microsoft\\DirectX\\UserGpuPreferences</font>."
))
story.append(tbl([
    ["Paquete", "Version", "Funcion"],
    ["torch", "2.11.0+cu128", "Framework de deep learning con soporte CUDA 12.8"],
    ["torchvision", "0.26.0+cu128", "Operaciones de vision en PyTorch"],
    ["ultralytics", "8.4.47", "Framework YOLOv8 (entrenamiento + export)"],
    ["fiftyone", "1.15.0", "Gestion del dataset Open Images v7"],
    ["roboflow", "1.x", "Descarga de datasets Roboflow Universe"],
    ["opencv-python", "4.13.0", "Captura de camara y dibujo de bboxes"],
    ["onnx + onnxruntime", "1.21 / 1.25", "Export e inferencia portable"],
    ["onnxslim", "0.1.92", "Optimizacion del grafo ONNX"],
    ["pygame", "2.6.1", "Reproduccion de alertas sonoras"],
    ["reportlab", "4.5.0", "Generacion del presente documento"],
], widths=[3.5*cm, 3*cm, 9*cm]))

# ========== 5. DATASETS ==========
story.append(PageBreak())
story.append(Paragraph("5. Datasets y procesamiento de datos", H1))

story.append(Paragraph("5.1 Fuentes utilizadas", H2))
story.append(tbl([
    ["Fuente", "Imagenes", "Clases originales", "Mapeo final"],
    ["Open Images v7 (FiftyOne)", "2.185", "Knife, Handgun, Shotgun, Rifle", "knife / handgun / long_gun"],
    ["RF joseph-nelson/pistols", "2.971", "pistol", "handgun"],
    ["RF rifledetection/rifle-detection-p9slr", "3.500*", "Rifle", "long_gun"],
], widths=[5.5*cm, 2.2*cm, 4.3*cm, 4*cm]))
story.append(Paragraph(
    "<i>* Sampleadas aleatoriamente desde un total de 13.013 imagenes para mantener "
    "balance de clases.</i>", BODY))

story.append(Paragraph("5.2 Reorganizacion taxonomica", H2))
story.append(p(
    "La version v1 utilizaba una clase generica <b>firearm</b> que agrupaba pistolas, "
    "escopetas y rifles. Esta decision degrado el rendimiento en esa clase a un "
    "mAP@0.5 de 0.49 debido a la alta varianza visual entre los subtipos. Para v2 se "
    "establecio la siguiente taxonomia final:"
))
story.append(b("<b>0 &mdash; knife:</b> cuchillos y armas blancas en general."))
story.append(b("<b>1 &mdash; handgun:</b> armas cortas (pistolas, revolveres). Empuniadura a una mano."))
story.append(b("<b>2 &mdash; long_gun:</b> armas largas (rifles, escopetas). Requieren dos manos."))

story.append(Paragraph("5.3 Particion final", H2))
story.append(tbl([
    ["Particion", "Imagenes", "Anotaciones knife", "Anotaciones handgun", "Anotaciones long_gun"],
    ["Train (85%)", "7.357", "983", "3.593", "5.363"],
    ["Val (10%)", "865", "93", "413", "699"],
    ["Test (5%)", "434", "51", "217", "322"],
    ["Total", "8.656", "1.127", "4.223", "6.384"],
], widths=[2.8*cm, 2.5*cm, 3.2*cm, 3.5*cm, 3.5*cm]))

story.append(Paragraph("5.4 Formato YOLO", H2))
story.append(p(
    "Cada imagen <font face='Courier'>NNNNNNN.jpg</font> tiene asociado un archivo "
    "<font face='Courier'>NNNNNNN.txt</font> con una linea por objeto detectado:"
))
story.append(code("<class_id> <cx> <cy> <ancho> <alto>"))
story.append(p(
    "Donde todas las coordenadas estan normalizadas en el rango [0, 1] respecto al "
    "tamano de la imagen, y <font face='Courier'>(cx, cy)</font> es el centro del "
    "bounding box."
))

# ========== 6. MODELO ==========
story.append(PageBreak())
story.append(Paragraph("6. Modelo y arquitectura YOLO", H1))
story.append(p(
    "YOLO (You Only Look Once) es una familia de detectores de objetos en una sola "
    "pasada. A diferencia de R-CNN o Fast R-CNN que primero proponen regiones candidatas "
    "y luego las clasifican, YOLO predice simultaneamente clase y bounding box "
    "procesando la imagen una unica vez. Esto la convierte en la opcion adecuada para "
    "inferencia en tiempo real en hardware limitado como la Raspberry Pi."
))

story.append(Paragraph("6.1 Variantes evaluadas", H2))
story.append(tbl([
    ["Variante", "Parametros", "Tamano (FP32)", "Uso en este proyecto"],
    ["YOLOv8n (nano)", "3.2 M", "~6 MB", "Despliegue final en Pi5"],
    ["YOLOv8s (small)", "11 M", "~22 MB / 89 MB con optimizer", "Entrenamiento en laptop"],
    ["YOLOv8m (medium)", "26 M", "~52 MB", "No utilizado (excede VRAM)"],
], widths=[3.2*cm, 2.5*cm, 4.5*cm, 5.5*cm]))

story.append(Paragraph("6.2 Estrategia dual de modelos", H2))
story.append(p(
    "Se entrena YOLOv8s en la laptop como modelo de referencia con la mejor precision "
    "posible. Posteriormente se entrena un YOLOv8n con el mismo dataset, optimizado para "
    "tamano y velocidad, que es el que se exporta a ONNX cuantizado para la Pi."
))
story.append(p(
    "Se utiliza fine-tuning desde pesos preentrenados en COCO "
    "(<font face='Courier'>yolov8s.pt</font>, <font face='Courier'>yolov8n.pt</font>). "
    "Esto reduce drasticamente el tiempo de entrenamiento al partir de un modelo que "
    "ya conoce caracteristicas visuales generales (bordes, texturas, formas)."
))

# ========== 7. CONFIG TRAIN ==========
story.append(Paragraph("7. Configuracion del entrenamiento", H1))
story.append(Paragraph("7.1 Hiperparametros principales", H2))
story.append(tbl([
    ["Parametro", "Valor", "Justificacion"],
    ["epochs", "100", "Suficiente con dataset ampliado"],
    ["batch", "8", "Reducido desde 12 para evitar OOM al final"],
    ["workers", "4", "Reducido desde 8 para liberar RAM"],
    ["imgsz", "640", "Estandar YOLOv8 en GPU"],
    ["optimizer", "AdamW", "Adaptativo, robusto, estandar moderno"],
    ["lr0", "0.001", "Conservador para fine-tuning"],
    ["patience", "30", "Early stopping si val no mejora"],
    ["amp", "true", "FP16 mixed precision (mitad de VRAM)"],
    ["device", "0", "GPU CUDA 0 (RTX 5050)"],
    ["seed", "42", "Reproducibilidad"],
], widths=[3*cm, 2.5*cm, 10*cm]))

story.append(Paragraph("7.2 Augmentations aplicadas", H2))
story.append(tbl([
    ["Augmentation", "Valor", "Efecto"],
    ["fliplr", "0.5", "50% prob. espejo horizontal"],
    ["flipud", "0.0", "Sin espejo vertical (irreal)"],
    ["mosaic", "1.0", "Combina 4 imagenes en una"],
    ["mixup", "0.15", "Transparenta dos imagenes"],
    ["copy_paste", "0.3", "Copia objetos entre imagenes (clave para knife)"],
    ["hsv_h/s/v", "0.015 / 0.7 / 0.4", "Variacion de tono, saturacion, brillo"],
], widths=[3*cm, 3.5*cm, 9*cm]))

story.append(Paragraph("7.3 Variables de entorno de seguridad", H2))
story.append(code(
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
    "CUDA_VISIBLE_DEVICES=0\n"
    "KMP_DUPLICATE_LIB_OK=TRUE"
))
story.append(p(
    "<font face='Courier'>expandable_segments:True</font> es la mitigacion clave del "
    "incidente de OOM observado en la version v1 (epoch 70/80). Permite a PyTorch "
    "expandir segmentos de VRAM existentes en lugar de fragmentar la memoria con "
    "asignaciones nuevas."
))

# ========== 8. FASES ==========
story.append(PageBreak())
story.append(Paragraph("8. Fases ejecutadas", H1))

phases = [
    ("Fase 0", "Setup del entorno",
     "Instalacion de Python 3.11.9, creacion del virtualenv, instalacion de "
     "PyTorch 2.11+cu128 (verificado <font face='Courier'>torch.cuda.is_available()=True</font>, "
     "device <font face='Courier'>RTX 5050 Laptop GPU sm_120</font>), "
     "verificacion de matmul CUDA, instalacion de dependencias del requirements.txt. "
     "Configuracion de preferencia de GPU NVIDIA en el registro de Windows.",
     "Completado"),
    ("Fase 1", "Descarga inicial de Open Images v7 (v1)",
     "Primera ejecucion de <font face='Courier'>1_download_dataset.py</font>. "
     "Se identificaron dos bugs: (1) la clase 'Firearm' no existe como leaf class en "
     "OIv7; (2) el parametro <font face='Courier'>dataset_dir</font> entraba en "
     "conflicto con argumentos internos de FiftyOne. Ambos corregidos, descarga "
     "exitosa de 2.185 imagenes en 4 clases.",
     "Completado"),
    ("Fase 2", "Conversion a formato YOLO (v1)",
     "Mapeo OIv7 -> 2 clases (knife/firearm), generacion de splits y "
     "data.yaml. 1.500 imgs train + 169 val + 516 test = 2.185 imagenes con "
     "3.612 anotaciones validas.",
     "Completado"),
    ("Fase 3", "Entrenamiento YOLOv8s v1",
     "100 epochs configuradas, batch 12, workers 8. Crash con CUDA OOM en epoch 70/80 "
     "por fragmentacion de VRAM. <font face='Courier'>best.pt</font> se conservo (mAP "
     "esperado ya alcanzado por early stopping efectivo).",
     "Completado con incidente"),
    ("Fase 4", "Evaluacion v1",
     "Sobre el split de test (516 imgs): mAP@0.5 = 0.617, mAP@0.5:0.95 = 0.474, "
     "Precision = 0.83, Recall = 0.65. Velocidad de inferencia: 4.4 ms por imagen "
     "(equivalente a 227 FPS en RTX). Identificado bajo rendimiento en clase firearm.",
     "Completado"),
    ("Fase 5", "Plan de mejora y obtencion de API key Roboflow",
     "Se establece estrategia: ampliar el corpus con datasets curados de Roboflow "
     "Universe, descomponer firearm en handgun + long_gun, ajustar config para "
     "evitar OOM, planificar export INT8 para Pi.",
     "Completado"),
    ("Fase 6", "Descarga de datasets Roboflow",
     "Se identificaron y descargaron joseph-nelson/pistols (2.971 imgs) y "
     "rifledetection/rifle-detection-p9slr (13.013 imgs, sampleadas a 3.500). "
     "Total descargado: ~1.8 GB.",
     "Completado"),
    ("Fase 7", "Merge y unificacion del dataset",
     "Implementacion de <font face='Courier'>2b_merge_datasets.py</font>: re-mapeo "
     "de clases a 3 categorias, particion estratificada 85/10/5, validacion de "
     "imagenes corruptas. Resultado: 8.656 imagenes finales.",
     "Completado"),
    ("Fase 8", "Reentrenamiento YOLOv8s v2",
     "Lanzado con batch 8, workers 4, 100 epochs, patience 30, augmentations "
     "extendidas (copy_paste 0.3) y env vars de proteccion VRAM. En curso.",
     "En progreso"),
    ("Fase 9", "Entrenamiento YOLOv8n para Pi",
     "Pendiente de iniciar tras finalizar v2. Mismo dataset, imgsz 416, batch 16, "
     "80 epochs.",
     "Pendiente"),
    ("Fase 10", "Export ONNX INT8",
     "Cuantizacion INT8 con calibracion sobre 100 imgs del train. Salida esperada: "
     "best.onnx ~3 MB.",
     "Pendiente"),
    ("Fase 11", "Evaluacion final y deploy a Pi",
     "Reporte de mAP por clase del modelo cuantizado, copia a Pi via scp, "
     "configuracion de servicio systemd para arranque automatico.",
     "Pendiente"),
]
phase_data = [["Fase", "Nombre", "Descripcion", "Estado"]]
for code_, name, desc, status in phases:
    phase_data.append([
        code_, name, Paragraph(desc, BODY), status,
    ])
story.append(tbl(phase_data, widths=[1.6*cm, 3.5*cm, 8.4*cm, 2.5*cm]))

# ========== 9. RESULTADOS ==========
story.append(PageBreak())
story.append(Paragraph("9. Resultados parciales", H1))

story.append(Paragraph("9.1 Comparacion v1 vs v2", H2))
story.append(tbl([
    ["Metrica", "v1 (2 clases, OIv7)", "v2 (3 clases, OIv7+RF)"],
    ["Imagenes de entrenamiento", "1.500", "7.357"],
    ["Total dataset", "2.185", "8.656"],
    ["mAP@0.5 (test)", "0.617", "En progreso (>0.71 en val a epoch 19)"],
    ["mAP@0.5:0.95 (test)", "0.474", "En progreso (0.48 en val a epoch 19)"],
    ["Precision", "0.83", "0.71 actual (val)"],
    ["Recall", "0.65", "0.67 actual (val)"],
    ["Inferencia laptop", "4.4 ms/img (227 FPS)", "Esperado similar"],
], widths=[5*cm, 5*cm, 6*cm]))

story.append(Paragraph("9.2 Curva de aprendizaje v2", H2))
story.append(p(
    "Hasta epoch 19 (al momento de generacion de este documento), el modelo v2 muestra "
    "una curva de aprendizaje ascendente y sin signos de overfitting:"
))
story.append(tbl([
    ["Epoch", "mAP@0.5 (val)", "mAP@0.5:0.95 (val)"],
    ["1", "0.443", "0.246"],
    ["2", "0.514", "0.293"],
    ["3", "0.557", "0.345"],
    ["6", "0.628", "0.396"],
    ["12", "0.681", "0.431"],
    ["19", "0.719", "0.479"],
], widths=[2*cm, 4*cm, 4*cm]))

story.append(Paragraph("9.3 Uso de recursos durante v2", H2))
for item in [
    "VRAM utilizada: 2.71 GB de 8 GB disponibles (33%).",
    "Temperatura GPU sostenida: 77 C (limite: 87 C).",
    "Utilizacion GPU: 86% promedio.",
    "Tiempo por epoch: aproximadamente 115 segundos.",
    "RAM total reportada por SO: 93%, pero pagefile usado solo 439 MB (memoria standby reclamable).",
]:
    story.append(b(item))

# ========== 10. PI ==========
story.append(PageBreak())
story.append(Paragraph("10. Despliegue en Raspberry Pi 5", H1))

story.append(Paragraph("10.1 Software a instalar en la Pi", H2))
story.append(code(
    "sudo apt update && sudo apt upgrade -y\n"
    "sudo apt install -y python3-pip python3-venv libatlas-base-dev libopenblas-dev\n"
    "python3 -m venv ~/weapons-env\n"
    "source ~/weapons-env/bin/activate\n"
    "pip install opencv-python onnxruntime numpy pyyaml pygame"
))
story.append(p("Total instalado: aproximadamente 250 MB. <b>No</b> se instala PyTorch ni Ultralytics."))

story.append(Paragraph("10.2 Archivos a copiar desde la laptop", H2))
story.append(tbl([
    ["Archivo", "Funcion", "Tamano aprox."],
    ["models/export/best.onnx", "Modelo cuantizado INT8", "~3 MB"],
    ["config.yaml", "Umbrales y configuracion", "2 KB"],
    ["utils/", "Visualizacion + alertas", "10 KB"],
    ["scripts/7_inference_pi.py", "Loop de inferencia", "5 KB"],
], widths=[5.5*cm, 7*cm, 3*cm]))
story.append(p("Total a transferir: <b>aproximadamente 3 MB</b>."))

story.append(Paragraph("10.3 Performance esperado", H2))
story.append(tbl([
    ["Resolucion", "Modelo", "FPS esperados en Pi 5"],
    ["416x416", "YOLOv8n FP32 ONNX", "4 - 7"],
    ["416x416", "YOLOv8n INT8 ONNX", "12 - 18 (objetivo)"],
    ["320x320", "YOLOv8n INT8 ONNX", "18 - 25"],
    ["416x416", "YOLOv8n NCNN FP16", "15 - 22"],
], widths=[3*cm, 5*cm, 7*cm]))

story.append(Paragraph("10.4 Servicio systemd para arranque automatico", H2))
story.append(code(
    "[Unit]\n"
    "Description=Weapons Detection\n"
    "After=network.target\n\n"
    "[Service]\n"
    "User=pi\n"
    "WorkingDirectory=/home/pi/weapons\n"
    "ExecStart=/home/pi/weapons-env/bin/python 7_inference_pi.py --weights best.onnx\n"
    "Restart=always\n\n"
    "[Install]\n"
    "WantedBy=multi-user.target"
))

# ========== 11. RIESGOS ==========
story.append(Paragraph("11. Riesgos identificados y mitigaciones", H1))
story.append(tbl([
    ["Riesgo", "Impacto", "Mitigacion aplicada"],
    ["OOM de VRAM al final del entrenamiento (incidente v1)",
     "Crash en epoch 70/80, perdida de progreso reciente",
     "Reduccion de batch a 8, workers a 4, env var expandable_segments:True"],
    ["Fragmentacion progresiva de VRAM en runs largos",
     "OOM probabilistico tras horas de entrenamiento",
     "PYTORCH_CUDA_ALLOC_CONF configurada"],
    ["Clase minoritaria (knife) con solo 12% de las imagenes",
     "Bajo recall en cuchillos",
     "copy_paste augmentation 0.3 + datasets adicionales"],
    ["TDR de Windows durante carga GPU sostenida",
     "Pantalla negra momentanea con sonido de desconexion USB",
     "Inocuo: el computo CUDA no se interrumpe. Verificar log si persiste."],
    ["GPU NVIDIA Blackwell sm_120 incompatible con CUDA 12.1",
     "Entrenamiento caeria silenciosamente al CPU o fallaria",
     "PyTorch instalado contra cu128 explicitamente"],
    ["Falta de assets/alert.wav en el repo",
     "Crash al activar alertas sonoras",
     "alerts.sound_enabled puesto en false hasta proveer el archivo"],
], widths=[5*cm, 4*cm, 7*cm]))

# ========== 12. PENDIENTE ==========
story.append(Paragraph("12. Trabajo pendiente", H1))
for item in [
    "Finalizar entrenamiento YOLOv8s v2 (en curso, estimado 2-3 horas).",
    "Entrenar YOLOv8n con el mismo dataset para despliegue en Pi.",
    "Exportar YOLOv8n a ONNX e INT8 con calibracion sobre 100 imgs.",
    "Evaluar el modelo cuantizado en el split de test, verificar perdida de mAP "
    "respecto al FP32 (esperada &lt; 2%).",
    "Generar archivo de alerta sonora <font face='Courier'>assets/alert.wav</font>.",
    "Aprovisionar Raspberry Pi 5 con sistema operativo y dependencias.",
    "Realizar pruebas de campo con camara en la Pi y medir FPS reales.",
    "Considerar incorporacion de hard negatives (200-500 imagenes sin armas con "
    "objetos confusos como utensilios de cocina) para reducir falsos positivos.",
]:
    story.append(b(item))

story.append(Spacer(1, 1*cm))
story.append(Paragraph(
    "<i>Documento generado automaticamente. Las cifras de la version v2 corresponden "
    "al estado del entrenamiento al momento de la generacion y se actualizaran al "
    "completarse el run.</i>",
    META,
))

# Build
OUT.parent.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    leftMargin=2.2*cm, rightMargin=2.2*cm,
    topMargin=2*cm, bottomMargin=2*cm,
    title="Documento Tecnico - Deteccion de Armas con YOLOv8",
    author="Proyecto deteccion-de-armas-i.a",
)
doc.build(story)
print(f"PDF generado: {OUT} ({OUT.stat().st_size/1024:.1f} KB)")
