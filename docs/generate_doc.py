"""Genera el documento tecnico final del proyecto en PDF."""
from pathlib import Path
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

OUT = Path(__file__).parent / "Documento_Tecnico_Deteccion_Armas.pdf"

styles = getSampleStyleSheet()
H1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=18, spaceAfter=12,
                    textColor=colors.HexColor('#1a3d6d'), keepWithNext=True)
H2 = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=14, spaceBefore=12,
                    spaceAfter=8, textColor=colors.HexColor('#2a5a9e'), keepWithNext=True)
H3 = ParagraphStyle('H3', parent=styles['Heading3'], fontSize=11.5, spaceBefore=8,
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
story.append(Paragraph("Documento tecnico final del proceso completo", COVER_SUB))
story.append(Spacer(1, 4*cm))
story.append(Paragraph("Plataforma de entrenamiento: NVIDIA RTX 5050 Laptop (Blackwell)", COVER_SUB))
story.append(Paragraph("Plataforma de despliegue: Raspberry Pi 5", COVER_SUB))
story.append(Spacer(1, 5*cm))
story.append(Paragraph(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}", META))
story.append(Paragraph("Repositorio: github.com/Enryuuh/deteccion-de-armas-i.a", META))
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
    "8. Fases ejecutadas (cronologia completa)",
    "9. Resultados finales",
    "10. Cuantizacion INT8 para Raspberry Pi",
    "11. Despliegue en Raspberry Pi 5",
    "12. Guia: SSH a la Pi desde Visual Studio Code",
    "13. Riesgos identificados y mitigaciones",
    "14. Reproducibilidad y trabajo futuro",
]
for item in toc:
    story.append(Paragraph(item, BODY))
story.append(PageBreak())

# ========== 1. RESUMEN ==========
story.append(Paragraph("1. Resumen ejecutivo", H1))
story.append(p(
    "El presente documento describe el desarrollo completo de un sistema de deteccion en "
    "tiempo real de armas blancas (cuchillos) y armas de fuego (pistolas, escopetas, rifles) "
    "basado en la arquitectura YOLOv8 de Ultralytics. El sistema fue entrenado en una laptop "
    "ASUS TUF Gaming A16 con GPU NVIDIA RTX 5050 (Blackwell, sm_120) y exportado a ONNX "
    "cuantizado INT8 para despliegue en Raspberry Pi 5."
))
story.append(p(
    "Se ejecutaron <b>dos iteraciones</b> del proceso de entrenamiento. La version v1 utilizo "
    "unicamente Open Images v7 con dos clases agregadas (knife/firearm) y obtuvo un mAP@0.5 "
    "de 0.62 en test. La version v2 incorporo datasets de Roboflow Universe, elevo el corpus "
    "a 8.656 imagenes curadas y descompuso la clase firearm en <b>handgun</b> y <b>long_gun</b>, "
    "alcanzando un <b>mAP@0.5 de 0.73 en test</b> con la clase handgun pasando de 0.49 a 0.80."
))
story.append(p(
    "Adicionalmente se entreno un <b>YOLOv8n</b> con el mismo dataset para despliegue en la Pi, "
    "obteniendo un mAP@0.5 de 0.83 en val y un tamano final cuantizado a INT8 de "
    "<b>3.25 MB</b> (72% menor que el FP32). El modelo final esta listo para inferir en tiempo "
    "real en la Raspberry Pi 5."
))

# ========== 2. OBJETIVOS ==========
story.append(Paragraph("2. Objetivos del sistema", H1))
for o in [
    "Detectar en tiempo real armas blancas (clase <b>knife</b>) y armas de fuego "
    "diferenciando armas cortas (<b>handgun</b>) de armas largas (<b>long_gun</b>: rifles, escopetas).",
    "Lograr una velocidad de inferencia minima de 10 FPS en Raspberry Pi 5 a "
    "resolucion 416x416.",
    "Mantener el modelo desplegado en menos de 5 MB para permitir actualizaciones "
    "remotas eficientes (objetivo cumplido: 3.25 MB).",
    "Disparar alertas locales (sonido, log y captura de evidencia) cuando se detecte "
    "una amenaza con confianza superior al umbral configurado.",
    "Permitir el despliegue completo en la Pi sin necesidad de instalar PyTorch ni "
    "Ultralytics, reduciendo la huella a menos de 250 MB de paquetes Python.",
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
    ["Driver NVIDIA", "581.80 (CUDA 13.0 runtime)", "Verificado al iniciar"],
], widths=[3.5*cm, 7*cm, 5*cm]))

story.append(Paragraph("3.2 Plataforma de despliegue", H2))
story.append(tbl([
    ["Componente", "Recomendado"],
    ["SBC", "Raspberry Pi 5 (8 GB RAM)"],
    ["Almacenamiento", "microSD 64 GB clase A2"],
    ["Refrigeracion", "Active cooler oficial (imprescindible bajo carga)"],
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
    ["onnx + onnxruntime", "1.21 / 1.25", "Export e inferencia portable + cuantizacion"],
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

story.append(Paragraph("5.2 Reorganizacion taxonomica (decision clave del v2)", H2))
story.append(p(
    "La version v1 utilizaba una clase generica <b>firearm</b> que agrupaba pistolas, "
    "escopetas y rifles. Esta decision degrado el rendimiento en esa clase a un "
    "mAP@0.5 de 0.49 debido a la alta varianza visual entre los subtipos. Para v2 se "
    "establecio la siguiente taxonomia final:"
))
story.append(b("<b>0 &mdash; knife:</b> cuchillos y armas blancas en general."))
story.append(b("<b>1 &mdash; handgun:</b> armas cortas (pistolas, revolveres). Empuniadura a una mano."))
story.append(b("<b>2 &mdash; long_gun:</b> armas largas (rifles, escopetas). Requieren dos manos."))

story.append(Paragraph("5.3 Particion final del dataset v2", H2))
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
    "pasada. A diferencia de R-CNN que primero propone regiones y luego las clasifica, "
    "YOLO predice simultaneamente clase y bounding box procesando la imagen una unica "
    "vez, lo que la convierte en la opcion adecuada para inferencia en tiempo real en "
    "hardware limitado como la Raspberry Pi."
))

story.append(Paragraph("6.1 Variantes evaluadas", H2))
story.append(tbl([
    ["Variante", "Parametros", "Tamano FP32", "Uso en este proyecto"],
    ["YOLOv8n (nano)", "3.2 M", "~6 MB / 12 MB ONNX", "Despliegue final en Pi5"],
    ["YOLOv8s (small)", "11 M", "~22 MB", "Entrenamiento en laptop (referencia)"],
    ["YOLOv8m (medium)", "26 M", "~52 MB", "No utilizado"],
], widths=[3.2*cm, 2.5*cm, 4.5*cm, 5.5*cm]))

story.append(Paragraph("6.2 Estrategia dual de modelos", H2))
story.append(p(
    "Se entrenaron <b>dos modelos en paralelo</b> sobre el mismo dataset:"
))
story.append(b(
    "<b>YOLOv8s</b> (small) como modelo de referencia con la mejor precision posible. "
    "Permite validar la calidad del dataset y servir como benchmark."
))
story.append(b(
    "<b>YOLOv8n</b> (nano) optimizado para tamano y velocidad. Es el modelo que se "
    "exporta a ONNX cuantizado INT8 para correr en la Pi."
))
story.append(p(
    "El nano alcanzo un mAP@0.5 muy similar al small (0.83 vs 0.84) pero con "
    "<b>3.6x menos peso</b> y <b>4.4x mas velocidad</b>, validando la decision."
))

# ========== 7. CONFIG TRAIN ==========
story.append(Paragraph("7. Configuracion del entrenamiento", H1))
story.append(Paragraph("7.1 Hiperparametros YOLOv8s v2", H2))
story.append(tbl([
    ["Parametro", "Valor", "Justificacion"],
    ["epochs", "100", "Suficiente con dataset ampliado"],
    ["batch", "8", "Reducido desde 12 (v1 sufrio OOM en epoch 70)"],
    ["workers", "4", "Reducido desde 8 para liberar RAM"],
    ["imgsz", "640", "Estandar YOLOv8 en GPU"],
    ["optimizer", "AdamW", "Adaptativo, robusto, estandar moderno"],
    ["lr0", "0.001", "Conservador para fine-tuning"],
    ["patience", "30", "Early stopping si val no mejora"],
    ["amp", "true", "FP16 mixed precision (mitad de VRAM)"],
    ["device", "0", "GPU CUDA 0 (RTX 5050)"],
], widths=[3*cm, 2.5*cm, 10*cm]))

story.append(Paragraph("7.2 Hiperparametros YOLOv8n", H2))
story.append(tbl([
    ["Parametro", "Valor", "Justificacion"],
    ["model", "yolov8n.pt", "Pesos preentrenados en COCO"],
    ["epochs", "80", "Modelo mas pequeno converge antes"],
    ["batch", "24", "32 dispara cuDNN error en Blackwell, 24 estable"],
    ["workers", "6", "Aprovecha mejor las 16 hilos del CPU"],
    ["imgsz", "416", "Resolucion reducida para FPS en Pi"],
    ["cache", "ram", "8.6k imgs en RAM (~3 GB) -> sin lectura disco"],
], widths=[3*cm, 2.5*cm, 10*cm]))

story.append(Paragraph("7.3 Augmentations aplicadas", H2))
story.append(tbl([
    ["Augmentation", "Valor", "Efecto"],
    ["fliplr", "0.5", "50% prob. espejo horizontal"],
    ["flipud", "0.0", "Sin espejo vertical (irreal)"],
    ["mosaic", "1.0", "Combina 4 imagenes en una"],
    ["mixup", "0.15 (v8s) / 0.1 (v8n)", "Transparenta dos imagenes"],
    ["copy_paste", "0.3", "Copia objetos entre imagenes (clave para knife)"],
    ["hsv_h/s/v", "0.015 / 0.7 / 0.4", "Variacion de tono, saturacion, brillo"],
], widths=[3*cm, 3.5*cm, 9*cm]))

story.append(Paragraph("7.4 Variables de entorno de seguridad", H2))
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
story.append(Paragraph("8. Fases ejecutadas (cronologia completa)", H1))

phases = [
    ("Fase 0", "Setup del entorno",
     "Instalacion de Python 3.11.9, virtualenv, PyTorch 2.11+cu128 (verificado "
     "torch.cuda.is_available()=True, RTX 5050 sm_120), instalacion del requirements.txt. "
     "Configuracion de preferencia de GPU NVIDIA en el registro de Windows.",
     "Completado"),
    ("Fase 1", "Descarga inicial de Open Images v7 (v1)",
     "Primera ejecucion de 1_download_dataset.py. Se identificaron dos bugs: (1) la clase "
     "'Firearm' no existe como leaf class en OIv7; (2) el parametro dataset_dir entraba en "
     "conflicto con argumentos internos de FiftyOne. Ambos corregidos, descarga exitosa de "
     "2.185 imagenes en 4 clases.",
     "Completado"),
    ("Fase 2", "Conversion a formato YOLO (v1)",
     "Mapeo OIv7 -> 2 clases (knife/firearm), generacion de splits y data.yaml. "
     "1.500 train + 169 val + 516 test = 2.185 imagenes con 3.612 anotaciones validas.",
     "Completado"),
    ("Fase 3", "Entrenamiento YOLOv8s v1",
     "100 epochs configuradas, batch 12, workers 8. Crash con CUDA OOM en epoch 70/80 por "
     "fragmentacion de VRAM. best.pt se conservo.",
     "Completado con incidente"),
    ("Fase 4", "Evaluacion v1",
     "Sobre el split de test (516 imgs): mAP@0.5 = 0.617, mAP@0.5:0.95 = 0.474, "
     "Precision = 0.83, Recall = 0.65. Velocidad: 4.4 ms/img (227 FPS en RTX). "
     "Identificado bajo rendimiento en clase firearm (0.49).",
     "Completado"),
    ("Fase 5", "Plan de mejora y obtencion de API key Roboflow",
     "Estrategia: ampliar corpus con datasets curados de RF Universe, descomponer firearm "
     "en handgun + long_gun, ajustar config para evitar OOM, planificar export INT8.",
     "Completado"),
    ("Fase 6", "Descarga de datasets Roboflow",
     "joseph-nelson/pistols (2.971 imgs) y rifledetection/rifle-detection-p9slr (13.013 "
     "imgs, sampleadas a 3.500). Total descargado: ~1.8 GB.",
     "Completado"),
    ("Fase 7", "Merge y unificacion del dataset",
     "Implementacion de 2b_merge_datasets.py: re-mapeo a 3 categorias, particion estratificada "
     "85/10/5, validacion de imagenes corruptas. Resultado: 8.656 imagenes finales.",
     "Completado"),
    ("Fase 8", "Reentrenamiento YOLOv8s v2",
     "100 epochs en 2h 55min. Sin OOM. mAP@0.5 val final: 0.838. "
     "Best epoch ~80 con metrica estable.",
     "Completado"),
    ("Fase 9", "Evaluacion v2 en test",
     "mAP@0.5 = 0.729 (+11pp vs v1). Por clase: knife 0.72, handgun 0.80, long_gun 0.66.",
     "Completado"),
    ("Fase 10", "Entrenamiento YOLOv8n para Pi",
     "80 epochs en 52 min con cache RAM. mAP@0.5 val: 0.833 (-0.5pp vs YOLOv8s). "
     "Modelo final: 6.2 MB.",
     "Completado"),
    ("Fase 11", "Export ONNX + cuantizacion INT8",
     "ONNX FP32: 11.6 MB. ONNX INT8 calibrado con 100 imgs: 3.25 MB (-72%). "
     "Listo para copiar a la Pi.",
     "Completado"),
    ("Fase 12", "Publicacion en GitHub",
     "Commit con todos los cambios: scripts modificados, scripts nuevos (2b, 3b, 6b), "
     "configuracion v2, documentacion. Push exitoso a master.",
     "Completado"),
]
phase_data = [["Fase", "Nombre", "Descripcion", "Estado"]]
for code_, name, desc, status in phases:
    phase_data.append([
        code_, name, Paragraph(desc, BODY), status,
    ])
story.append(tbl(phase_data, widths=[1.6*cm, 3.5*cm, 8.4*cm, 2.5*cm]))

# ========== 9. RESULTADOS ==========
story.append(PageBreak())
story.append(Paragraph("9. Resultados finales", H1))

story.append(Paragraph("9.1 Comparativa global v1 vs v2", H2))
story.append(tbl([
    ["Metrica", "v1 (OIv7, 2 clases)", "v2 (OIv7+RF, 3 clases)", "Delta"],
    ["Imagenes totales", "2.185", "8.656", "+296%"],
    ["mAP@0.5 (test)", "0.617", "0.729", "+11.2 pp"],
    ["mAP@0.5:0.95 (test)", "0.474", "0.589", "+11.5 pp"],
    ["Precision", "0.83", "0.87", "+4 pp"],
    ["Recall", "0.65", "0.74", "+9 pp"],
], widths=[3.8*cm, 4*cm, 4.5*cm, 2.7*cm]))

story.append(Paragraph("9.2 Resultados v2 por clase (split test)", H2))
story.append(tbl([
    ["Clase", "Instancias", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"],
    ["all", "589", "0.868", "0.745", "0.729", "0.589"],
    ["knife", "51", "0.760", "0.745", "0.724", "0.655"],
    ["handgun", "217", "0.951", "0.806", "0.802", "0.649"],
    ["long_gun", "321", "0.894", "0.682", "0.660", "0.464"],
], widths=[2.5*cm, 2*cm, 2.2*cm, 2.2*cm, 2.5*cm, 3*cm]))
story.append(p(
    "<b>Hallazgo clave:</b> la clase handgun paso de 0.49 (v1, agregada como firearm) a "
    "0.80 (v2, separada). La descomposicion taxonomica fue la mejora individual mas grande."
))

story.append(Paragraph("9.3 YOLOv8n vs YOLOv8s (val)", H2))
story.append(tbl([
    ["Metrica", "YOLOv8s (small)", "YOLOv8n (nano)", "Diferencia"],
    ["mAP@0.5 global", "0.838", "0.833", "-0.5%"],
    ["mAP@0.5 knife", "0.858", "0.845", "-1.3%"],
    ["mAP@0.5 handgun", "0.908", "0.883", "-2.5%"],
    ["mAP@0.5 long_gun", "0.748", "0.772", "+2.4%"],
    ["Tamano (.pt)", "22.5 MB", "6.2 MB", "-72%"],
    ["Inferencia", "3.1 ms/img", "0.7 ms/img", "-77%"],
    ["FPS en RTX", "~322", "~1428", "+343%"],
], widths=[3.5*cm, 3.5*cm, 3.5*cm, 3*cm]))

story.append(Paragraph("9.4 Curva de aprendizaje YOLOv8s v2", H2))
story.append(tbl([
    ["Epoch", "mAP@0.5", "mAP@0.5:0.95"],
    ["1", "0.443", "0.246"],
    ["12", "0.681", "0.431"],
    ["26", "0.780", "0.549"],
    ["47", "0.820", "0.598"],
    ["71", "0.844", "0.625"],
    ["100 (final)", "0.838", "0.639"],
], widths=[2.5*cm, 4*cm, 4*cm]))

# ========== 10. CUANTIZACION ==========
story.append(PageBreak())
story.append(Paragraph("10. Cuantizacion INT8 para Raspberry Pi", H1))
story.append(p(
    "La cuantizacion convierte los pesos del modelo de 32 bits flotantes (FP32) a 8 bits "
    "enteros (INT8). Esto reduce el tamano del archivo en ~75% y acelera la inferencia "
    "en hardware ARM en aproximadamente 4x, con una perdida tipica de mAP menor al 2%."
))

story.append(Paragraph("10.1 Proceso aplicado", H2))
story.append(p(
    "Se utilizo <font face='Courier'>onnxruntime.quantization.quantize_static()</font> con "
    "calibracion estatica:"
))
story.append(b("Modelo de entrada: yolov8n_v2_fp32.onnx (11.6 MB)"))
story.append(b("Set de calibracion: 100 imagenes seleccionadas aleatoriamente del split de train"))
story.append(b("Formato: QDQ (Quantize-DeQuantize), per-channel"))
story.append(b("Pesos: QInt8 | Activaciones: QUInt8"))
story.append(b("Metodo de calibracion: MinMax"))

story.append(Paragraph("10.2 Resultado", H2))
story.append(tbl([
    ["Modelo", "Formato", "Tamano", "Reduccion"],
    ["yolov8n_v2_fp32.onnx", "FP32", "11.61 MB", "baseline"],
    ["yolov8n_v2_int8.onnx", "INT8 estatico QDQ", "3.25 MB", "-72%"],
], widths=[5*cm, 4.5*cm, 3*cm, 3*cm]))

# ========== 11. PI ==========
story.append(Paragraph("11. Despliegue en Raspberry Pi 5", H1))

story.append(Paragraph("11.1 Software a instalar en la Pi", H2))
story.append(code(
    "sudo apt update && sudo apt upgrade -y\n"
    "sudo apt install -y python3-pip python3-venv libatlas-base-dev libopenblas-dev\n"
    "python3 -m venv ~/weapons-env\n"
    "source ~/weapons-env/bin/activate\n"
    "pip install opencv-python onnxruntime numpy pyyaml pygame"
))
story.append(p("Total instalado: ~250 MB. <b>No</b> se instala PyTorch ni Ultralytics."))

story.append(Paragraph("11.2 Archivos a copiar desde la laptop", H2))
story.append(tbl([
    ["Archivo", "Funcion", "Tamano"],
    ["models/export/yolov8n_v2_int8.onnx", "Modelo cuantizado INT8", "3.25 MB"],
    ["config.yaml", "Umbrales y configuracion", "2 KB"],
    ["utils/", "Visualizacion + alertas", "10 KB"],
    ["scripts/7_inference_pi.py", "Loop de inferencia", "5 KB"],
], widths=[6.5*cm, 6*cm, 3*cm]))

story.append(Paragraph("11.3 Comando scp para transferir", H2))
story.append(code(
    "scp models/export/yolov8n_v2_int8.onnx pi@<ip-pi>:~/weapons/best.onnx\n"
    "scp config.yaml                        pi@<ip-pi>:~/weapons/\n"
    "scp -r utils                           pi@<ip-pi>:~/weapons/\n"
    "scp scripts/7_inference_pi.py          pi@<ip-pi>:~/weapons/"
))

story.append(Paragraph("11.4 Performance esperado", H2))
story.append(tbl([
    ["Resolucion", "Modelo", "FPS esperados en Pi 5"],
    ["416x416", "YOLOv8n FP32 ONNX", "4 - 7"],
    ["<b>416x416</b>", "<b>YOLOv8n INT8 ONNX</b>", "<b>12 - 18 (objetivo)</b>"],
    ["320x320", "YOLOv8n INT8 ONNX", "18 - 25"],
    ["416x416", "YOLOv8n NCNN FP16", "15 - 22"],
], widths=[3*cm, 5*cm, 7*cm]))

# ========== 12. SSH desde VS Code ==========
story.append(PageBreak())
story.append(Paragraph("12. Guia: SSH a la Pi desde Visual Studio Code", H1))
story.append(p(
    "Esta seccion documenta el flujo completo para administrar y desarrollar en la "
    "Raspberry Pi 5 desde Visual Studio Code en la laptop, usando la extension Remote-SSH."
))

story.append(Paragraph("12.1 Setup inicial de la Pi", H2))
story.append(b("Flashear microSD con <b>Raspberry Pi Imager</b> (Pi OS Lite 64-bit)."))
story.append(b("En el menu avanzado del Imager (icono de engranaje), configurar:"))
story.append(b("&nbsp;&nbsp;&nbsp;&nbsp;- Hostname: <font face='Courier'>raspberrypi</font>"))
story.append(b("&nbsp;&nbsp;&nbsp;&nbsp;- Habilitar SSH (con autenticacion por password o llave)"))
story.append(b("&nbsp;&nbsp;&nbsp;&nbsp;- Usuario y password (ej. <font face='Courier'>pi</font> / tu password)"))
story.append(b("&nbsp;&nbsp;&nbsp;&nbsp;- Configurar WiFi (SSID + password)"))
story.append(b("Insertar la SD en la Pi, conectar fuente y esperar 60-90 segundos al primer boot."))

story.append(Paragraph("12.2 Encontrar la IP de la Pi", H2))
story.append(p("Desde la laptop, opcion mas rapida (ping mDNS):"))
story.append(code("ping raspberrypi.local"))
story.append(p("Si no responde, escanear la red:"))
story.append(code(
    "# Windows (PowerShell)\n"
    "arp -a | findstr 'b8-27-eb\\|dc-a6-32\\|e4-5f-01\\|2c-cf-67'\n\n"
    "# Linux/Mac\n"
    "nmap -sn 192.168.1.0/24"
))

story.append(Paragraph("12.3 Configurar SSH key (recomendado)", H2))
story.append(p("Desde la laptop, generar par de llaves si no existe:"))
story.append(code("ssh-keygen -t ed25519 -C 'pi-deploy'"))
story.append(p("Copiar la llave publica a la Pi:"))
story.append(code(
    "# Windows PowerShell\n"
    "type $env:USERPROFILE\\.ssh\\id_ed25519.pub | ssh pi@<ip> 'cat &gt;&gt; ~/.ssh/authorized_keys'\n\n"
    "# Linux/Mac\n"
    "ssh-copy-id pi@<ip>"
))
story.append(p("A partir de aqui no se pide password al conectar."))

story.append(Paragraph("12.4 Editar ~/.ssh/config en la laptop", H2))
story.append(p(
    "Esto permite conectarse simplemente con <font face='Courier'>ssh pi5</font>:"
))
story.append(code(
    "Host pi5\n"
    "    HostName 192.168.1.42       # reemplazar por la IP real\n"
    "    User pi\n"
    "    IdentityFile ~/.ssh/id_ed25519\n"
    "    ServerAliveInterval 60"
))

story.append(Paragraph("12.5 Instalar la extension Remote-SSH en VS Code", H2))
story.append(b("Abrir Visual Studio Code en la laptop."))
story.append(b("Ir a Extensions (Ctrl+Shift+X)."))
story.append(b("Buscar e instalar: <b>Remote - SSH</b> (publisher: ms-vscode-remote)."))
story.append(b("Opcionalmente instalar tambien <b>Remote - SSH: Editing Configuration Files</b>."))

story.append(Paragraph("12.6 Conectarse a la Pi desde VS Code", H2))
story.append(b("Abrir Command Palette: <font face='Courier'>Ctrl+Shift+P</font>."))
story.append(b("Escribir: <b>Remote-SSH: Connect to Host...</b>"))
story.append(b("Seleccionar <font face='Courier'>pi5</font> (o el alias configurado)."))
story.append(b("Se abre una nueva ventana de VS Code conectada a la Pi."))
story.append(b("La primera vez descarga el VS Code Server en la Pi (~80 MB, una sola vez)."))
story.append(b("Esquina inferior izquierda mostrara: <font face='Courier'>SSH: pi5</font> (verde)."))

story.append(Paragraph("12.7 Estructura de carpetas en la Pi", H2))
story.append(code(
    "/home/pi/\n"
    "  weapons-env/                  # virtualenv\n"
    "  weapons/\n"
    "    best.onnx                   # modelo INT8 (3.25 MB)\n"
    "    config.yaml\n"
    "    utils/\n"
    "      visualization.py\n"
    "      alerts.py\n"
    "    7_inference_pi.py\n"
    "    logs/\n"
    "      detections.log\n"
    "      frames/                   # capturas de evidencia"
))

story.append(Paragraph("12.8 Subir archivos: dos opciones", H2))
story.append(p("<b>Opcion A &mdash; arrastrar y soltar en VS Code:</b>"))
story.append(b("Una vez conectado, abrir el explorador de archivos (Ctrl+Shift+E)."))
story.append(b("Navegar a <font face='Courier'>/home/pi/weapons</font>."))
story.append(b("Arrastrar archivos desde el explorador del SO directamente a la ventana."))
story.append(p("<b>Opcion B &mdash; via terminal con scp:</b>"))
story.append(code(
    "scp models/export/yolov8n_v2_int8.onnx pi5:~/weapons/best.onnx\n"
    "scp config.yaml pi5:~/weapons/\n"
    "scp -r utils pi5:~/weapons/\n"
    "scp scripts/7_inference_pi.py pi5:~/weapons/"
))

story.append(Paragraph("12.9 Ejecutar inferencia desde la terminal de VS Code", H2))
story.append(b("En VS Code conectado a la Pi, abrir terminal: <font face='Courier'>Ctrl+\\`</font>."))
story.append(b("Activar el venv:"))
story.append(code("source ~/weapons-env/bin/activate"))
story.append(b("Ejecutar inferencia:"))
story.append(code("cd ~/weapons && python 7_inference_pi.py --weights best.onnx"))
story.append(b("Para detener: <font face='Courier'>Ctrl+C</font>."))

story.append(Paragraph("12.10 Debugging remoto con breakpoints", H2))
story.append(p(
    "En la pestana <b>Run and Debug</b> (Ctrl+Shift+D), crear "
    "<font face='Courier'>.vscode/launch.json</font> en la Pi con:"
))
story.append(code(
    '{\n'
    '  "version": "0.2.0",\n'
    '  "configurations": [\n'
    '    {\n'
    '      "name": "Inferencia Pi",\n'
    '      "type": "debugpy",\n'
    '      "request": "launch",\n'
    '      "program": "${workspaceFolder}/7_inference_pi.py",\n'
    '      "args": ["--weights", "best.onnx"],\n'
    '      "console": "integratedTerminal",\n'
    '      "python": "/home/pi/weapons-env/bin/python"\n'
    '    }\n'
    '  ]\n'
    '}'
))
story.append(p("Ahora F5 lanza el script con breakpoints funcionales."))

story.append(Paragraph("12.11 Servicio systemd para arranque automatico", H2))
story.append(p(
    "Para que la deteccion arranque sola cada vez que se enchufa la Pi, crear "
    "<font face='Courier'>/etc/systemd/system/weapons.service</font>:"
))
story.append(code(
    "[Unit]\n"
    "Description=Weapons Detection\n"
    "After=network.target\n\n"
    "[Service]\n"
    "User=pi\n"
    "WorkingDirectory=/home/pi/weapons\n"
    "ExecStart=/home/pi/weapons-env/bin/python 7_inference_pi.py --weights best.onnx\n"
    "Restart=always\n"
    "RestartSec=5\n\n"
    "[Install]\n"
    "WantedBy=multi-user.target"
))
story.append(p("Activar:"))
story.append(code(
    "sudo systemctl daemon-reload\n"
    "sudo systemctl enable weapons\n"
    "sudo systemctl start weapons\n"
    "sudo systemctl status weapons   # verificar"
))

story.append(Paragraph("12.12 Comandos utiles desde la terminal", H2))
story.append(tbl([
    ["Comando", "Funcion"],
    ["journalctl -u weapons -f", "Ver logs en vivo del servicio"],
    ["sudo systemctl restart weapons", "Reiniciar deteccion"],
    ["htop", "Monitor de CPU y RAM"],
    ["vcgencmd measure_temp", "Temperatura del SoC"],
    ["v4l2-ctl --list-devices", "Listar camaras detectadas"],
    ["uptime", "Tiempo encendido + carga"],
    ["df -h /", "Espacio en disco"],
], widths=[6*cm, 9*cm]))

story.append(Paragraph("12.13 Troubleshooting comun", H2))
story.append(tbl([
    ["Problema", "Solucion"],
    ["Camara no detectada (cv2.VideoCapture(0))",
     "Probar otros indices (1, 2). Verificar permisos: sudo usermod -aG video pi"],
    ["FPS muy bajos (<5)", "Verificar throttling: vcgencmd get_throttled. Si distinto de 0x0, problema termico/voltaje"],
    ["No se reproduce sonido", "Verificar audio: aplay -l. Configurar default device en ~/.asoundrc"],
    ["VS Code Remote-SSH no conecta", "Borrar ~/.vscode-server en la Pi y reintentar. Verificar version de SSH (>=8.0)"],
    ["systemctl status reporta failed", "Ver journalctl -u weapons -n 50 para detalle"],
    ["Poca memoria al inferir", "Cerrar X11/desktop si se uso Pi OS Desktop (preferir Lite)"],
], widths=[6*cm, 9*cm]))

# ========== 13. RIESGOS ==========
story.append(PageBreak())
story.append(Paragraph("13. Riesgos identificados y mitigaciones", H1))
story.append(tbl([
    ["Riesgo", "Impacto", "Mitigacion aplicada"],
    ["OOM de VRAM al final del entrenamiento (incidente v1)",
     "Crash en epoch 70/80",
     "batch 12->8, workers 8->4, expandable_segments:True"],
    ["Fragmentacion progresiva de VRAM en runs largos",
     "OOM probabilistico",
     "PYTORCH_CUDA_ALLOC_CONF configurada"],
    ["Clase minoritaria knife (12% del dataset)",
     "Bajo recall en cuchillos",
     "copy_paste augmentation 0.3"],
    ["TDR de Windows durante carga GPU sostenida",
     "Pantalla negra momentanea",
     "Inocuo: el computo CUDA no se interrumpe"],
    ["GPU NVIDIA Blackwell sm_120 incompatible con CUDA &lt;12.8",
     "Entrenamiento caeria al CPU",
     "PyTorch instalado contra cu128 explicitamente"],
    ["batch 32 dispara cuDNN error en yolov8n",
     "Crash en epoch 1",
     "Reducido a batch 24"],
    ["Falta de assets/alert.wav en el repo",
     "Crash al activar alertas",
     "alerts.sound_enabled=false hasta proveer el archivo"],
    ["Pistola 3D impresa fuera de distribucion (color, textura)",
     "Bajo confidence en pruebas",
     "Imprimir en negro mate; bajar threshold a 0.30 si necesario"],
], widths=[5*cm, 4*cm, 7*cm]))

# ========== 14. REPRODUCIBILIDAD ==========
story.append(Paragraph("14. Reproducibilidad y trabajo futuro", H1))

story.append(Paragraph("14.1 Para reproducir todo desde cero", H2))
story.append(p("Ver el archivo <font face='Courier'>README.txt</font> en la raiz del repositorio."))

story.append(Paragraph("14.2 Trabajo futuro sugerido", H2))
for item in [
    "Incorporar 200-500 hard negatives (utensilios de cocina, controles, taladros) para "
    "reducir falsos positivos.",
    "Ampliar dataset con mas ejemplos de armas en contextos reales (manos, ropa, "
    "interiores, exteriores).",
    "Probar variantes YOLOv8 mas grandes (m, l) en la laptop para tener un modelo "
    "&quot;teacher&quot; para destilacion al nano.",
    "Generar archivo de alerta sonora <font face='Courier'>assets/alert.wav</font>.",
    "Implementar deteccion de tracking entre frames (ByteTrack/BoTSORT) para reducir "
    "alertas duplicadas y mejorar la confiabilidad temporal.",
    "Integrar con un sistema de notificacion remota (Telegram, email) para alertas en "
    "ubicaciones desatendidas.",
    "Optimizar el modelo NCNN para ARM y comparar FPS reales contra ONNX Runtime.",
    "Considerar deteccion en stream RTSP (camaras IP) ademas de USB/Pi Camera.",
]:
    story.append(b(item))

story.append(Spacer(1, 1*cm))
story.append(Paragraph(
    "<i>Documento generado automaticamente por docs/generate_doc.py.</i>", META,
))

# ============== BUILD ==============
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
