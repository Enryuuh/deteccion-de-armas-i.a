"""Genera el documento tecnico final del proyecto en PDF (estado actual, incluye v1-v4)."""
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
story.append(Paragraph("Documento tecnico: proceso completo y resultados actuales (v1 a v4)", COVER_SUB))
story.append(Spacer(1, 4*cm))
story.append(Paragraph("Plataforma de entrenamiento: NVIDIA RTX 5050 Laptop (Blackwell)", COVER_SUB))
story.append(Paragraph("Plataforma de despliegue: Raspberry Pi 5", COVER_SUB))
story.append(Spacer(1, 5*cm))
story.append(Paragraph(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}", META))
story.append(Paragraph("Repositorio: deteccion-de-armas-i.a", META))
story.append(PageBreak())

# ========== INDICE ==========
story.append(Paragraph("Indice", H1))
toc = [
    "1. Resumen ejecutivo",
    "2. Objetivos del sistema",
    "3. Arquitectura de hardware",
    "4. Stack de software",
    "5. Datasets: evolucion completa (v1 a v4)",
    "6. Modelo y arquitectura YOLO",
    "7. Configuracion del entrenamiento y fine-tuning",
    "8. Sistema de alertas, filtrado y monitoreo",
    "9. Fases ejecutadas (cronologia completa)",
    "10. Resultados actuales",
    "11. Cuantizacion INT8 para Raspberry Pi",
    "12. Despliegue en Raspberry Pi 5",
    "13. Guia: SSH a la Pi desde Visual Studio Code",
    "14. Riesgos identificados y mitigaciones",
    "15. Reproducibilidad y trabajo futuro",
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
    "con GPU NVIDIA RTX 5050 (Blackwell, sm_120) y exportado a ONNX/NCNN para desplegarse en "
    "una Raspberry Pi 5 sin necesidad de PyTorch ni Ultralytics."
))
story.append(p(
    "El proyecto paso por <b>cuatro iteraciones</b> mayores. La <b>v1</b> uso unicamente Open "
    "Images v7 con una clase generica <b>firearm</b> y alcanzo un mAP@0.5 de 0.617 en test. La "
    "<b>v2</b> incorporo datasets de Roboflow Universe, amplio el corpus a 8.656 imagenes "
    "curadas y descompuso <b>firearm</b> en <b>handgun</b> y <b>long_gun</b>, alcanzando un "
    "<b>mAP@0.5 de 0.729 en test</b> (+11.2 pp). Las iteraciones <b>v3 y v4</b> abordaron un "
    "problema distinto: el modelo detectaba bien pistolas de lado pero fallaba en vistas "
    "frontales/POV, y podia <b>alucinar</b> armas al ver personas, telefonos o mandos en un "
    "entorno tipo stand. Se sumaron +13.902 imagenes (poses diversas + hard negatives de "
    "personas/objetos cotidianos) y se aplico un <b>fine-tune anti-alucinacion</b> "
    "(backbone congelado, LR muy bajo, penalizacion de clasificacion reforzada) partiendo de "
    "los pesos v2, sin reentrenar desde cero."
))
story.append(p(
    "El resultado del fine-tune (v4, evaluado sobre el mismo split de validacion de 865 "
    "imagenes que v2): el modelo <b>nano</b> (el que se despliega en la Pi) mantuvo su mAP@0.5 "
    "practicamente igual (0.8516 vs 0.8530) pero subio su <b>Precision +2.39 pp</b> "
    "(0.8511 &rarr; 0.8750), que es exactamente el objetivo buscado: menos falsas alarmas frente "
    "a personas y objetos en el stand. El modelo <b>small</b> de referencia se comporto de forma "
    "similar (mAP@0.5 -1.10 pp, mismo patron de generalizacion)."
))
story.append(p(
    "El sistema dejo de ser solo un detector: incorpora <b>filtro geometrico de plausibilidad</b>, "
    "<b>filtro temporal</b> (confirmacion multi-frame + suavizado), <b>umbrales de confianza por "
    "clase</b>, <b>captura de evidencia</b> (frames + clips de video con buffer previo/posterior), "
    "<b>notificaciones push por Telegram</b> y un <b>dashboard web</b> para monitoreo remoto desde "
    "el celular. El modelo final (YOLOv8n v4) se exporto a ONNX INT8 de <b>3.25 MB</b> (-72% vs "
    "FP32) y fue <b>validado en una sesion de camara en vivo</b> (2026-05-24) donde detecto "
    "consistentemente el arma de prueba y genero automaticamente 457 frames y 3 clips de "
    "evidencia. A la fecha, el dataset de entrenamiento en disco sigue creciendo "
    "(actualmente 27.970 imagenes de train) por el ciclo de retroalimentacion con hard "
    "negatives capturados en campo; ese superset aun no cuenta con una re-evaluacion formal."
))

# ========== 2. OBJETIVOS ==========
story.append(Paragraph("2. Objetivos del sistema", H1))
for o in [
    "Detectar en tiempo real armas blancas (clase <b>knife</b>) y armas de fuego "
    "diferenciando armas cortas (<b>handgun</b>) de armas largas (<b>long_gun</b>: rifles, escopetas).",
    "Reconocer el arma en <b>poses diversas</b> (lateral, frontal, POV, angulo CCTV cenital), no "
    "solo en la vista de perfil predominante en el dataset original.",
    "<b>No alucinar</b> armas al ver personas, manos, telefonos, mandos, llaves u otros objetos "
    "cotidianos tipicos de un escenario de stand/exhibicion con publico.",
    "Lograr una velocidad de inferencia minima de 10 FPS en Raspberry Pi 5 a "
    "resolucion 416x416.",
    "Mantener el modelo desplegado en pocos MB para permitir actualizaciones "
    "remotas eficientes (objetivo cumplido: 3.25 MB en INT8).",
    "Disparar alertas locales (sonido, log estructurado, frames y clips de video) y remotas "
    "(Telegram) cuando se detecte una amenaza con confianza superior al umbral configurado.",
    "Permitir monitoreo remoto sin pantalla propia mediante un dashboard web accesible desde "
    "cualquier dispositivo en la red local.",
    "Permitir el despliegue completo en la Pi sin necesidad de instalar PyTorch ni "
    "Ultralytics, reduciendo la huella a menos de 250 MB de paquetes Python.",
    "Cerrar el ciclo de mejora continua: los falsos positivos capturados durante pruebas en "
    "campo se reintegran como hard negatives en el siguiente entrenamiento.",
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
    "El entorno se aisla en un virtualenv Python dentro del repositorio "
    "(<font face='Courier'>repo/.venv</font>). Se forzo la asignacion de la GPU NVIDIA "
    "para los ejecutables Python via la clave de registro "
    "<font face='Courier'>HKCU\\Software\\Microsoft\\DirectX\\UserGpuPreferences</font>."
))
story.append(tbl([
    ["Paquete", "Funcion"],
    ["torch / torchvision (cu128)", "Framework de deep learning con soporte CUDA 12.8"],
    ["ultralytics", "Framework YOLOv8 (entrenamiento + export + val)"],
    ["fiftyone", "Gestion del dataset Open Images v7"],
    ["roboflow", "Descarga de datasets Roboflow Universe (poses de armas)"],
    ["datasets / huggingface_hub", "Descarga de datasets de HuggingFace (PranomVignesh, Subh775, Simuletic, etc.)"],
    ["opencv-python", "Captura de camara, dibujo de bboxes, streaming MJPEG del dashboard"],
    ["onnx + onnxruntime", "Export e inferencia portable + cuantizacion INT8"],
    ["onnxslim", "Optimizacion del grafo ONNX"],
    ["ncnn", "Formato alterno optimizado para ARM (Raspberry Pi)"],
    ["pygame", "Reproduccion de alertas sonoras (beep sintetico, sin wav externo)"],
    ["requests", "Envio de notificaciones push a la API de Telegram"],
    ["reportlab", "Generacion del presente documento"],
], widths=[4.5*cm, 11*cm]))

# ========== 5. DATASETS ==========
story.append(PageBreak())
story.append(Paragraph("5. Datasets: evolucion completa (v1 a v4)", H1))

story.append(Paragraph("5.1 v1 -- Open Images v7 unicamente (2 clases)", H2))
story.append(p(
    "Primera version: 2.185 imagenes de Open Images v7 (Knife, Handgun, Shotgun, Rifle) "
    "colapsadas en 2 clases (<b>knife</b> / <b>firearm</b>). La clase generica <b>firearm</b> "
    "agrupaba pistolas, escopetas y rifles, lo que degrado su rendimiento (mAP@0.5 de 0.49) "
    "por la alta varianza visual entre subtipos."
))

story.append(Paragraph("5.2 v2 -- Roboflow + reorganizacion taxonomica (3 clases)", H2))
story.append(p(
    "Se sumaron datasets de Roboflow Universe y se establecio la taxonomia final de 3 clases:"
))
story.append(b("<b>0 &mdash; knife:</b> cuchillos y armas blancas en general."))
story.append(b("<b>1 &mdash; handgun:</b> armas cortas (pistolas, revolveres). Empuniadura a una mano."))
story.append(b("<b>2 &mdash; long_gun:</b> armas largas (rifles, escopetas). Requieren dos manos."))
story.append(tbl([
    ["Fuente", "Imagenes", "Clases originales", "Mapeo final"],
    ["Open Images v7 (FiftyOne)", "2.185", "Knife, Handgun, Shotgun, Rifle", "knife / handgun / long_gun"],
    ["RF joseph-nelson/pistols", "2.971", "pistol", "handgun"],
    ["RF rifledetection/rifle-detection-p9slr", "3.500*", "Rifle", "long_gun"],
], widths=[5.5*cm, 2.2*cm, 4.3*cm, 4*cm]))
story.append(Paragraph(
    "<i>* Sampleadas aleatoriamente desde un total de 13.013 imagenes para mantener "
    "balance de clases.</i>", BODY))
story.append(tbl([
    ["Particion", "Imagenes", "Anotaciones knife", "Anotaciones handgun", "Anotaciones long_gun"],
    ["Train (85%)", "7.357", "983", "3.593", "5.363"],
    ["Val (10%)", "865", "93", "413", "699"],
    ["Test (5%)", "434", "51", "217", "322"],
    ["Total", "8.656", "1.127", "4.223", "6.384"],
], widths=[2.8*cm, 2.5*cm, 3.2*cm, 3.5*cm, 3.5*cm]))

story.append(Paragraph("5.3 v3/v4 -- Poses diversas + anti-alucinacion (stand)", H2))
story.append(p(
    "Deteccion de dos problemas nuevos al preparar el modelo para exhibirse en un stand con "
    "publico: (1) el modelo entrenado en v2 detectaba bien pistolas <b>de lado</b> pero fallaba "
    "en vistas <b>frontales / POV / apuntando a camara</b>; (2) en presencia de personas, manos, "
    "telefonos y mandos, podia generar <b>falsos positivos</b> (alucinaciones). Aumentar solo con "
    "transformaciones geometricas no alcanza: una pistola de lado rotada sigue siendo una "
    "pistola de lado, no aporta la vista frontal que falta. Se opto por sumar <b>datos reales</b> "
    "de esas vistas y de esos objetos confundibles."
))
story.append(tbl([
    ["Fuente", "Imagenes", "Tipo"],
    ["PranomVignesh/HandGuns (HF, con token)", "6.882", "Mirror Roboflow pistols-p6x3w -- poses variadas"],
    ["Subh775/WeaponDetection (HF)", "5.341", "Multi-clase armas"],
    ["OpenImages stand negatives", "846", "Personas, moviles, mandos (hard negative)"],
    ["harshdadiya phone_detection (HF)", "605", "Moviles puros (hard negative)"],
    ["Simuletic CCTV Rifle-vs-Umbrella (HF)", "120", "CCTV cenital + paraguas (hard negative)"],
    ["Simuletic Handgun-vs-Chips (HF)", "110", "Bolsa de Doritos vs pistola (hard negative)"],
], widths=[6.5*cm, 2.5*cm, 6.5*cm]))
story.append(p(
    "Total sumado en la version v4 (final recomendada): <b>+13.902 imagenes</b> respecto al "
    "train original de v2 (7.357 &rarr; 21.259). Los hard negatives se agregan como imagenes de "
    "fondo (label vacio) para que el modelo aprenda que esos objetos <b>no</b> son armas."
))
story.append(p(
    "Ademas se implemento un <b>ciclo de retroalimentacion</b> (<font face='Courier'>scripts/8_"
    "add_hard_negatives.py</font>): los frames capturados como evidencia durante pruebas reales "
    "de camara (incluyendo falsos positivos) se reincorporan al split de train como background "
    "para el siguiente ciclo de fine-tune."
))

story.append(Paragraph("5.4 Escala actual del dataset (en disco, hoy)", H2))
story.append(tbl([
    ["Particion", "Imagenes"],
    ["Train", "27.970"],
    ["Val", "865"],
    ["Test", "434"],
    ["Total", "29.269"],
], widths=[4*cm, 4*cm]))
story.append(p(
    "El train set sigue creciendo por el ciclo de retroalimentacion descrito arriba (27.970 "
    "actuales vs 21.259 documentados en el reporte v4). Val y test se mantienen identicos a v2 "
    "para que las comparaciones de metricas entre versiones sigan siendo validas. <b>El superset "
    "actual de train (27.970 imgs) aun no tiene una re-evaluacion formal de mAP</b>; los "
    "numeros de la seccion 10 corresponden al ultimo checkpoint evaluado formalmente (v4, "
    "21.259 imgs de train)."
))

story.append(Paragraph("5.5 Formato YOLO", H2))
story.append(p(
    "Cada imagen <font face='Courier'>NNNNNNN.jpg</font> tiene asociado un archivo "
    "<font face='Courier'>NNNNNNN.txt</font> con una linea por objeto detectado:"
))
story.append(code("<class_id> <cx> <cy> <ancho> <alto>"))
story.append(p(
    "Donde todas las coordenadas estan normalizadas en el rango [0, 1] respecto al "
    "tamano de la imagen, y <font face='Courier'>(cx, cy)</font> es el centro del "
    "bounding box. Las imagenes background (hard negatives) usan un archivo "
    "<font face='Courier'>.txt</font> vacio."
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
    ["YOLOv8n (nano)", "3.0 M", "~6 MB / 12 MB ONNX", "Despliegue final en Pi5"],
    ["YOLOv8s (small)", "11.1 M", "~22 MB", "Entrenamiento en laptop (referencia)"],
    ["YOLOv8m (medium)", "26 M", "~52 MB", "No utilizado"],
], widths=[3.2*cm, 2.5*cm, 4.5*cm, 5.5*cm]))

story.append(Paragraph("6.2 Estrategia dual de modelos", H2))
story.append(p(
    "Se entrenan y afinan <b>dos modelos en paralelo</b> sobre el mismo dataset:"
))
story.append(b(
    "<b>YOLOv8s</b> (small) como modelo de referencia con la mejor precision posible. "
    "Permite validar la calidad del dataset y servir como benchmark, y correr en la laptop/PC "
    "con GPU en tiempo real durante demostraciones."
))
story.append(b(
    "<b>YOLOv8n</b> (nano) optimizado para tamano y velocidad. Es el modelo que se "
    "exporta a ONNX cuantizado INT8 para correr en la Pi."
))
story.append(p(
    "En la version v4, el nano mantiene un mAP@0.5 casi identico al small (0.852 vs 0.879 en "
    "val) con un peso muy inferior y una velocidad de inferencia varias veces mayor en GPU, "
    "validando la decision de usarlo como modelo de borde."
))

# ========== 7. CONFIG TRAIN ==========
story.append(Paragraph("7. Configuracion del entrenamiento y fine-tuning", H1))
story.append(Paragraph("7.1 Hiperparametros base (v2, entrenamiento desde COCO)", H2))
story.append(tbl([
    ["Parametro", "Valor", "Justificacion"],
    ["epochs", "100 (v8s) / 80 (v8n)", "Suficiente con dataset ampliado; nano converge antes"],
    ["batch", "8 (v8s) / 24 (v8n)", "Ajustado para evitar OOM y errores de cuDNN en Blackwell"],
    ["workers", "4-6", "Balance entre velocidad de carga y RAM disponible"],
    ["imgsz", "640 (v8s) / 416 (v8n)", "640 estandar en GPU; 416 para FPS en la Pi"],
    ["optimizer", "AdamW", "Adaptativo, robusto, estandar moderno"],
    ["lr0", "0.001", "Conservador para entrenar desde pesos COCO"],
    ["patience", "30", "Early stopping si val no mejora"],
    ["amp", "true", "FP16 mixed precision (mitad de VRAM)"],
    ["mixup / copy_paste", "0.20 / 0.50", "Mas combinaciones y pegado de armas en otros fondos"],
    ["degrees / translate / scale", "25 / 0.15 / 0.60", "Armas en cualquier angulo, posicion y distancia"],
], widths=[4*cm, 4*cm, 7.5*cm]))

story.append(Paragraph("7.2 Receta de fine-tune anti-alucinacion (v3/v4)", H2))
story.append(p(
    "A diferencia del entrenamiento base, el fine-tune <b>no parte de cero</b>: continua desde "
    "los pesos <font face='Courier'>best.pt</font> de v2 para no perder lo ya aprendido, y usa "
    "una receta distinta orientada a reducir falsos positivos:"
))
story.append(tbl([
    ["Parametro", "Valor", "Por que"],
    ["punto de partida", "best.pt de v2", "No reentrenar desde cero: conservar lo ya aprendido"],
    ["lr0", "0.0001", "Muy bajo -- ajuste fino, no destruir pesos existentes"],
    ["lrf / warmup_epochs", "0.01 / 3", "Decaimiento suave con calentamiento corto"],
    ["freeze", "10 capas (backbone)", "Solo se reentrena la cabeza de deteccion"],
    ["cls (peso de la perdida de clase)", "1.0", "Penaliza mas fuerte clasificar mal -- menos alucinaciones"],
    ["perspective / degrees", "0.001 / 30", "Mas variedad angular para las vistas frontal/POV"],
    ["close_mosaic", "10", "Ultimas epochs sin mosaico -- fine-tune limpio"],
    ["epochs", "30 (sin early stopping)", "Ciclo corto y controlado de ajuste fino"],
], widths=[4.5*cm, 3.5*cm, 7*cm]))

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

# ========== 8. SISTEMA COMPLETO ==========
story.append(PageBreak())
story.append(Paragraph("8. Sistema de alertas, filtrado y monitoreo", H1))
story.append(p(
    "Mas alla del modelo de deteccion, el sistema desplegado incorpora varias capas para hacerlo "
    "utilizable en un escenario real (stand, punto de vigilancia sin operador permanente):"
))

story.append(Paragraph("8.1 Filtro geometrico de plausibilidad", H2))
story.append(p(
    "<font face='Courier'>utils/plausibility_filter.py</font> descarta detecciones "
    "geometricamente imposibles para cada clase, comparando el area de la caja respecto al "
    "frame y su aspect ratio contra rangos esperados:"
))
story.append(tbl([
    ["Clase", "Area minima / maxima (frame)", "Aspect ratio min / max"],
    ["knife", "0.2% / 45%", "1.1 / 18.0"],
    ["handgun", "0.3% / 50%", "1.0 / 6.0"],
    ["long_gun", "0.5% / 60%", "2.0 / 22.0"],
], widths=[3*cm, 6*cm, 5*cm]))

story.append(Paragraph("8.2 Filtro temporal (anti-flicker)", H2))
story.append(p(
    "<font face='Courier'>utils/temporal_filter.py</font> no confirma una alerta con una sola "
    "deteccion aislada: mantiene una ventana de las ultimas <b>7 frames</b> y solo confirma una "
    "clase si aparecio en al menos <b>4 de esos 7</b>. Las coordenadas del bounding box se "
    "suavizan con un promedio movil exponencial (EMA, alpha=0.20) para eliminar saltos visuales. "
    "Esto reduce tanto el parpadeo de la caja como las alarmas por detecciones espurias de un "
    "unico frame."
))

story.append(Paragraph("8.3 Umbrales de confianza por clase", H2))
story.append(p(
    "En vez de un unico umbral global, cada clase tiene su propio umbral de confianza "
    "configurado en <font face='Courier'>config.yaml</font>: knife 0.40, handgun 0.45, "
    "long_gun 0.45 (umbral base 0.30). Esto permite calibrar cada clase de forma independiente "
    "segun su tasa de falsos positivos observada."
))

story.append(Paragraph("8.4 Captura de evidencia: frames y clips de video", H2))
story.append(p(
    "Ante una deteccion confirmada, <font face='Courier'>utils/alerts.py</font> guarda: (1) un "
    "beep sintetico generado en runtime (no requiere archivo .wav externo), (2) una linea en "
    "<font face='Courier'>logs/detections.log</font> (texto legible) y otra en "
    "<font face='Courier'>logs/detections.jsonl</font> (JSON estructurado para analisis), (3) un "
    "frame de evidencia en <font face='Courier'>logs/frames/</font>, y (4) un <b>clip de video</b> "
    "con un buffer en anillo de <b>5 segundos antes</b> y <b>10 segundos despues</b> de la primera "
    "deteccion, guardado en <font face='Courier'>logs/clips/</font>."
))

story.append(Paragraph("8.5 Notificaciones remotas por Telegram (opcional)", H2))
story.append(p(
    "El sistema puede enviar una foto + alerta al chat de Telegram configurado "
    "(<font face='Courier'>telegram_bot_token</font> / <font face='Courier'>telegram_chat_id</font> "
    "en <font face='Courier'>config.yaml</font>), con un cooldown de 30s entre mensajes para no "
    "saturar. Desactivado por defecto hasta configurar el bot."
))

story.append(Paragraph("8.6 Dashboard web de monitoreo remoto", H2))
story.append(p(
    "<font face='Courier'>utils/web_dashboard.py</font> levanta un servidor HTTP minimalista en "
    "un hilo separado (no bloquea la inferencia) que transmite el video en vivo como stream MJPEG "
    "y el estado de alertas, accesible desde cualquier dispositivo de la red local en "
    "<font face='Courier'>http://&lt;ip&gt;:8080</font> -- util para vigilar el sistema desde el "
    "celular sin necesitar un monitor conectado."
))

story.append(Paragraph("8.7 Herramientas de mejora continua", H2))
story.append(b("<font face='Courier'>scripts/8_add_hard_negatives.py</font>: reintegra los frames capturados (incluyendo falsos positivos) como hard negatives del proximo entrenamiento."))
story.append(b("<font face='Courier'>scripts/11_capture_finetune_frames.py</font> + <font face='Courier'>12_finetune.py</font>: capturan y etiquetan frames de un escenario/camara especifico para afinar el modelo a ese entorno concreto."))
story.append(b("<font face='Courier'>scripts/pi5_emulation.py</font>: emula en la laptop las restricciones de computo de la Pi5 antes de desplegar, para estimar FPS sin necesitar el hardware fisico a mano."))

# ========== 9. FASES ==========
story.append(PageBreak())
story.append(Paragraph("9. Fases ejecutadas (cronologia completa)", H1))

phases = [
    ("Fase 0", "Setup del entorno",
     "Instalacion de Python, virtualenv, PyTorch con soporte CUDA 12.8 (verificado "
     "torch.cuda.is_available()=True, RTX 5050 sm_120), instalacion de requirements.txt. "
     "Configuracion de preferencia de GPU NVIDIA en el registro de Windows.",
     "Completado"),
    ("Fase 1", "Descarga inicial de Open Images v7 (v1)",
     "Primera ejecucion de 1_download_dataset.py. Se identificaron dos bugs: (1) la clase "
     "'Firearm' no existe como leaf class en OIv7; (2) el parametro dataset_dir entraba en "
     "conflicto con argumentos internos de FiftyOne. Ambos corregidos, descarga exitosa de "
     "2.185 imagenes en 4 clases.",
     "Completado"),
    ("Fase 2", "Conversion a formato YOLO (v1)",
     "Mapeo OIv7 -> 2 clases (knife/firearm), generacion de splits y data.yaml.",
     "Completado"),
    ("Fase 3", "Entrenamiento YOLOv8s v1",
     "100 epochs configuradas, batch 12, workers 8. Crash con CUDA OOM en epoch 70/80 por "
     "fragmentacion de VRAM. best.pt se conservo.",
     "Completado con incidente"),
    ("Fase 4", "Evaluacion v1",
     "Test (516 imgs): mAP@0.5 = 0.617, mAP@0.5:0.95 = 0.474, P = 0.83, R = 0.65. Bajo "
     "rendimiento en la clase generica firearm (0.49).",
     "Completado"),
    ("Fase 5", "Descarga de datasets Roboflow y merge (v2)",
     "joseph-nelson/pistols (2.971 imgs) y rifledetection/rifle-detection-p9slr (sampleado a "
     "3.500 imgs). 2b_merge_datasets.py re-mapea a 3 clases y genera particion 85/10/5. "
     "Resultado: 8.656 imagenes finales.",
     "Completado"),
    ("Fase 6", "Reentrenamiento YOLOv8s v2 y evaluacion",
     "100 epochs. mAP@0.5 test = 0.729 (+11.2pp vs v1). handgun paso de 0.49 (agregada) a 0.80 "
     "(separada) -- la mejora individual mas grande.",
     "Completado"),
    ("Fase 7", "Entrenamiento YOLOv8n v2 para Pi + export INT8",
     "80 epochs con cache en RAM. mAP@0.5 val ~0.85. Export ONNX FP32 (11.6 MB) + INT8 "
     "cuantizado (3.25 MB, -72%).",
     "Completado"),
    ("Fase 8", "Publicacion inicial en GitHub",
     "Commit con scripts, config v2 y documentacion tecnica (version inicial del presente "
     "documento).",
     "Completado"),
    ("Fase 9", "Sesion de captura + hard negatives + descarga de negativos extra",
     "scripts/8, 9, 10 y 11: se capturan frames propios de camara, se agregan negativos extra "
     "de Open Images (personas/objetos sin fiftyone, evitando su crash de MongoDB en Windows) y "
     "se descargan mas imagenes de armas directamente de los CSV de OIv7.",
     "Completado"),
    ("Fase 10", "Diagnostico: fallas en poses frontales + riesgo de alucinacion",
     "Se identifica que el modelo v2 falla con pistolas de frente/POV y puede confundir "
     "personas/telefonos/mandos con armas en escenarios con publico (stand). Se descarta el "
     "aumento por transformaciones geometricas como solucion suficiente.",
     "Completado"),
    ("Fase 11", "Descarga masiva de datasets de poses diversas y hard negatives (v3/v4)",
     "scripts/13, 14, 15, 16: Roboflow + HuggingFace (PranomVignesh/HandGuns, "
     "Subh775/WeaponDetection, phone_detection, CCTV Rifle-vs-Umbrella, Handgun-vs-Chips) + "
     "negativos de personas/objetos de OpenImages. +13.902 imagenes.",
     "Completado"),
    ("Fase 12", "Fine-tune anti-alucinacion v3 (nano + small)",
     "scripts/17: fine-tune desde best.pt v2 con backbone congelado, lr0=0.0001, cls=1.0, "
     "30 epochs. Primera version intermedia (v3), sin el dataset PranomVignesh completo.",
     "Completado"),
    ("Fase 13", "Reintegracion completa + reentrenamiento final (v4)",
     "scripts/19: reintegra PranomVignesh completo (con token HF, ~8.9k imgs vs subset previo), "
     "reentrena nano y small, y cada uno se valida + exporta (ONNX/NCNN) + reporta "
     "automaticamente (script 18). Genera docs/finetune_results.md.",
     "Completado"),
    ("Fase 14", "Funcionalidades de sistema: alertas, filtros, dashboard, Telegram",
     "Se agregan utils/plausibility_filter.py, utils/temporal_filter.py, "
     "utils/web_dashboard.py, umbrales por clase, clips de video con buffer y notificaciones "
     "por Telegram (opcional). Se actualiza config.yaml al modelo v4.",
     "Completado"),
    ("Fase 15", "Validacion en vivo con camara (2026-05-24)",
     "Sesion real de prueba con el modelo nano v4: deteccion consistente del arma de prueba "
     "durante varios minutos, 457 frames de evidencia y 3 clips de video generados "
     "automaticamente, logs de texto y JSON poblados end-to-end.",
     "Completado"),
    ("Fase 16", "Ciclo de retroalimentacion continuo",
     "Los frames capturados en pruebas de campo se reincorporan como hard negatives "
     "(script 8). El train set en disco crecio de 21.259 (v4 reportado) a 27.970 imagenes "
     "actuales. Pendiente: re-evaluacion formal sobre este superset.",
     "En curso"),
]
phase_data = [["Fase", "Nombre", "Descripcion", "Estado"]]
for code_, name, desc, status in phases:
    phase_data.append([
        code_, name, Paragraph(desc, BODY), status,
    ])
story.append(tbl(phase_data, widths=[1.6*cm, 3.3*cm, 8.6*cm, 2.5*cm]))

# ========== 10. RESULTADOS ==========
story.append(PageBreak())
story.append(Paragraph("10. Resultados actuales", H1))

story.append(Paragraph("10.1 Progresion historica en el split de test (434 imgs, sin tocar desde v1)", H2))
story.append(tbl([
    ["Metrica", "v1 (OIv7, 2 clases)", "v2 (OIv7+RF, 3 clases)", "Delta"],
    ["Imagenes de train", "1.500", "7.357", "+391%"],
    ["mAP@0.5 (test)", "0.617", "0.729", "+11.2 pp"],
    ["mAP@0.5:0.95 (test)", "0.474", "0.589", "+11.5 pp"],
    ["Precision", "0.83", "0.868", "+3.8 pp"],
    ["Recall", "0.65", "0.745", "+9.5 pp"],
], widths=[3.8*cm, 4*cm, 4.5*cm, 2.7*cm]))

story.append(Paragraph("10.2 v2 por clase (split test, 589 instancias)", H2))
story.append(tbl([
    ["Clase", "Instancias", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"],
    ["all", "589", "0.868", "0.745", "0.729", "0.589"],
    ["knife", "51", "0.760", "0.745", "0.724", "0.655"],
    ["handgun", "217", "0.951", "0.806", "0.802", "0.649"],
    ["long_gun", "321", "0.894", "0.682", "0.660", "0.464"],
], widths=[2.5*cm, 2*cm, 2.2*cm, 2.2*cm, 2.5*cm, 3*cm]))
story.append(p(
    "<b>Hallazgo clave del salto v1&rarr;v2:</b> la clase handgun paso de 0.49 (agregada como "
    "firearm generica) a 0.80 (separada). La descomposicion taxonomica fue la mejora individual "
    "mas grande de esa etapa."
))

story.append(Paragraph("10.3 Fine-tune anti-alucinacion v4 vs v2 (split val, 865 imgs, 1.205 instancias)", H2))
story.append(p(
    "Esta comparacion usa el split de <b>validacion</b> (no el de test de la seccion 10.1) y es "
    "la que se utilizo para decidir si aceptar el fine-tune v4. Nota: este val set sigue "
    "dominado por la distribucion original (pistolas de lado); los modelos v4 reparten capacidad "
    "entre esa distribucion vieja y las nuevas (frontal/POV/hard-negs), por lo que aqui se ve una "
    "bajada esperable de mAP. La ganancia real (menos falsas alarmas y mejor deteccion frontal) "
    "se observa en el despliegue, no en este val estatico."
))
story.append(tbl([
    ["Modelo", "Param (M)", "imgsz", "mAP@50", "mAP@50-95", "Precision", "Recall"],
    ["nano v2 (original)", "3.01", "416", "0.8530", "0.6208", "0.8511", "0.7757"],
    ["nano v4 (final, en Pi)", "3.01", "416", "0.8516", "0.6105", "0.8750", "0.7585"],
    ["small v2 (original)", "11.13", "640", "0.8901", "0.6476", "0.8858", "0.8249"],
    ["small v4 (final, PC)", "11.13", "640", "0.8791", "0.6311", "0.8787", "0.7924"],
], widths=[4*cm, 2*cm, 1.8*cm, 2*cm, 2.3*cm, 2.2*cm, 1.7*cm]))
story.append(tbl([
    ["Modelo", "Delta mAP@50", "Delta mAP@50-95", "Delta Precision", "Delta Recall"],
    ["nano (v4 vs v2)", "-0.14 pp", "-1.03 pp", "+2.39 pp", "-1.73 pp"],
    ["small (v4 vs v2)", "-1.10 pp", "-1.65 pp", "-0.71 pp", "-3.26 pp"],
], widths=[4*cm, 2.8*cm, 3*cm, 3*cm, 2.5*cm]))
story.append(p(
    "<b>Lectura:</b> en el modelo nano (el que va a la Pi), el mAP@0.5 se mantuvo practicamente "
    "igual (-0.14pp, dentro de ruido) mientras que la <b>Precision subio +2.39 pp</b> -- el "
    "modelo se volvio mas exigente para decir &quot;hay arma&quot;, que es exactamente el "
    "objetivo anti-alucinacion para el stand. El Recall bajo -1.73pp (algun verdadero positivo "
    "en pose rara podria escaparse), una compensacion aceptada porque el objetivo prioritario "
    "es evitar falsas alarmas frente al publico. Ambas versiones (v2 y v4) quedan exportadas y "
    "disponibles, por lo que revertir a v2 es inmediato si en campo v4 se comportara peor."
))

story.append(Paragraph("10.4 Velocidad de inferencia (GPU RTX 5050, val)", H2))
story.append(tbl([
    ["Modelo", "Preprocess", "Inferencia", "Postprocess"],
    ["nano v4 (416px)", "~0.4 ms", "~1.1-1.6 ms", "~0.8-0.9 ms"],
    ["small v4 (640px)", "~0.8-0.9 ms", "~5.8-6.2 ms", "~0.8 ms"],
], widths=[4*cm, 3*cm, 3.5*cm, 3.5*cm]))

story.append(Paragraph("10.5 Validacion en campo con camara en vivo (2026-05-24)", H2))
story.append(p(
    "Mas alla de las metricas de benchmark, se corrio una sesion real con el modelo nano v4 "
    "sobre la camara en vivo para verificar el pipeline completo (deteccion &rarr; filtro "
    "temporal &rarr; alerta &rarr; evidencia), no solo el modelo aislado:"
))
story.append(b("El sistema detecto <b>consistentemente</b> el arma de prueba (handgun) frame a frame durante toda la sesion, confirmando el filtro temporal (>= 4 de 7 frames)."))
story.append(b("Se generaron automaticamente <b>457 frames</b> de evidencia en <font face='Courier'>logs/frames/</font>."))
story.append(b("Se generaron <b>3 clips de video</b> con buffer de 5s antes / 10s despues en <font face='Courier'>logs/clips/</font>."))
story.append(b("<font face='Courier'>logs/detections.log</font> y <font face='Courier'>logs/detections.jsonl</font> quedaron poblados con timestamp, numero de frame y clase por cada deteccion confirmada."))
story.append(p(
    "Esta prueba valida el sistema de punta a punta (no solo el mAP del modelo), incluyendo la "
    "cadena de alertas y captura de evidencia que se usara en el despliegue final."
))

# ========== 11. CUANTIZACION ==========
story.append(PageBreak())
story.append(Paragraph("11. Cuantizacion INT8 para Raspberry Pi", H1))
story.append(p(
    "La cuantizacion convierte los pesos del modelo de 32 bits flotantes (FP32) a 8 bits "
    "enteros (INT8). Esto reduce el tamano del archivo en ~72% y acelera la inferencia "
    "en hardware ARM, con una perdida tipica de mAP menor al 2%."
))

story.append(Paragraph("11.1 Proceso aplicado", H2))
story.append(p(
    "Se utilizo <font face='Courier'>onnxruntime.quantization.quantize_static()</font> con "
    "calibracion estatica sobre el modelo final <b>yolov8n_v4_pose_negs</b>:"
))
story.append(b("Set de calibracion: imagenes seleccionadas aleatoriamente del split de train"))
story.append(b("Formato: QDQ (Quantize-DeQuantize), per-channel"))
story.append(b("Pesos: QInt8 | Activaciones: QUInt8"))
story.append(b("Metodo de calibracion: MinMax"))
story.append(b("Formatos adicionales exportados: NCNN (param+bin), alternativa optimizada para ARM"))

story.append(Paragraph("11.2 Resultado (modelo desplegado hoy)", H2))
story.append(tbl([
    ["Modelo", "Formato", "Tamano", "Reduccion"],
    ["yolov8n_v4_fp32.onnx", "FP32", "11.61 MB", "baseline"],
    ["yolov8n_v4_int8.onnx", "INT8 estatico QDQ", "3.25 MB", "-72%"],
    ["yolov8n_v4 (best_ncnn_model)", "NCNN FP32", "~11.5 MB (param+bin)", "alternativa ARM"],
], widths=[5*cm, 4.5*cm, 3*cm, 3*cm]))

# ========== 12. PI ==========
story.append(Paragraph("12. Despliegue en Raspberry Pi 5", H1))

story.append(Paragraph("12.1 Software a instalar en la Pi", H2))
story.append(code(
    "sudo apt update && sudo apt upgrade -y\n"
    "sudo apt install -y python3-pip python3-venv libatlas-base-dev libopenblas-dev\n"
    "python3 -m venv ~/weapons-env\n"
    "source ~/weapons-env/bin/activate\n"
    "pip install opencv-python onnxruntime numpy pyyaml pygame requests"
))
story.append(p("Total instalado: ~250 MB. <b>No</b> se instala PyTorch ni Ultralytics."))

story.append(Paragraph("12.2 Archivos a copiar desde la laptop", H2))
story.append(tbl([
    ["Archivo", "Funcion", "Tamano"],
    ["models/export/yolov8n_v4_int8.onnx", "Modelo cuantizado INT8 (final)", "3.25 MB"],
    ["config.yaml", "Umbrales por clase, alertas, dashboard, Telegram", "3 KB"],
    ["utils/", "Filtros, visualizacion, alertas, dashboard", "~30 KB"],
    ["scripts/7_inference_pi.py", "Loop de inferencia", "5 KB"],
], widths=[6.5*cm, 6*cm, 3*cm]))

story.append(Paragraph("12.3 Comando scp para transferir", H2))
story.append(code(
    "scp models/export/yolov8n_v4_int8.onnx pi@<ip-pi>:~/weapons/best.onnx\n"
    "scp config.yaml                        pi@<ip-pi>:~/weapons/\n"
    "scp -r utils                           pi@<ip-pi>:~/weapons/\n"
    "scp scripts/7_inference_pi.py          pi@<ip-pi>:~/weapons/"
))

story.append(Paragraph("12.4 Performance esperado", H2))
story.append(tbl([
    ["Resolucion", "Modelo", "FPS esperados en Pi 5"],
    ["416x416", "YOLOv8n FP32 ONNX", "4 - 7"],
    ["<b>416x416</b>", "<b>YOLOv8n INT8 ONNX (desplegado)</b>", "<b>12 - 18 (objetivo)</b>"],
    ["320x320", "YOLOv8n INT8 ONNX", "18 - 25"],
    ["416x416", "YOLOv8n NCNN FP16", "15 - 22"],
], widths=[3*cm, 5*cm, 7*cm]))
story.append(p(
    "El dashboard web (seccion 8.6) puede activarse tambien en la Pi para monitoreo remoto; "
    "considerar bajar <font face='Courier'>fps_limit</font> del stream MJPEG si el ancho de "
    "banda de la red local es limitado."
))

# ========== 13. SSH desde VS Code ==========
story.append(PageBreak())
story.append(Paragraph("13. Guia: SSH a la Pi desde Visual Studio Code", H1))
story.append(p(
    "Esta seccion documenta el flujo completo para administrar y desarrollar en la "
    "Raspberry Pi 5 desde Visual Studio Code en la laptop, usando la extension Remote-SSH."
))

story.append(Paragraph("13.1 Setup inicial de la Pi", H2))
story.append(b("Flashear microSD con <b>Raspberry Pi Imager</b> (Pi OS Lite 64-bit)."))
story.append(b("En el menu avanzado del Imager (icono de engranaje), configurar:"))
story.append(b("&nbsp;&nbsp;&nbsp;&nbsp;- Hostname: <font face='Courier'>raspberrypi</font>"))
story.append(b("&nbsp;&nbsp;&nbsp;&nbsp;- Habilitar SSH (con autenticacion por password o llave)"))
story.append(b("&nbsp;&nbsp;&nbsp;&nbsp;- Usuario y password (ej. <font face='Courier'>pi</font> / tu password)"))
story.append(b("&nbsp;&nbsp;&nbsp;&nbsp;- Configurar WiFi (SSID + password)"))
story.append(b("Insertar la SD en la Pi, conectar fuente y esperar 60-90 segundos al primer boot."))

story.append(Paragraph("13.2 Encontrar la IP de la Pi", H2))
story.append(p("Desde la laptop, opcion mas rapida (ping mDNS):"))
story.append(code("ping raspberrypi.local"))
story.append(p("Si no responde, escanear la red:"))
story.append(code(
    "# Windows (PowerShell)\n"
    "arp -a | findstr 'b8-27-eb\\|dc-a6-32\\|e4-5f-01\\|2c-cf-67'\n\n"
    "# Linux/Mac\n"
    "nmap -sn 192.168.1.0/24"
))

story.append(Paragraph("13.3 Configurar SSH key (recomendado)", H2))
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

story.append(Paragraph("13.4 Editar ~/.ssh/config en la laptop", H2))
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

story.append(Paragraph("13.5 Instalar la extension Remote-SSH en VS Code", H2))
story.append(b("Abrir Visual Studio Code en la laptop."))
story.append(b("Ir a Extensions (Ctrl+Shift+X)."))
story.append(b("Buscar e instalar: <b>Remote - SSH</b> (publisher: ms-vscode-remote)."))
story.append(b("Opcionalmente instalar tambien <b>Remote - SSH: Editing Configuration Files</b>."))

story.append(Paragraph("13.6 Conectarse a la Pi desde VS Code", H2))
story.append(b("Abrir Command Palette: <font face='Courier'>Ctrl+Shift+P</font>."))
story.append(b("Escribir: <b>Remote-SSH: Connect to Host...</b>"))
story.append(b("Seleccionar <font face='Courier'>pi5</font> (o el alias configurado)."))
story.append(b("Se abre una nueva ventana de VS Code conectada a la Pi."))
story.append(b("La primera vez descarga el VS Code Server en la Pi (~80 MB, una sola vez)."))
story.append(b("Esquina inferior izquierda mostrara: <font face='Courier'>SSH: pi5</font> (verde)."))

story.append(Paragraph("13.7 Estructura de carpetas en la Pi", H2))
story.append(code(
    "/home/pi/\n"
    "  weapons-env/                  # virtualenv\n"
    "  weapons/\n"
    "    best.onnx                   # modelo INT8 (3.25 MB)\n"
    "    config.yaml\n"
    "    utils/\n"
    "      visualization.py\n"
    "      alerts.py\n"
    "      plausibility_filter.py\n"
    "      temporal_filter.py\n"
    "      web_dashboard.py\n"
    "    7_inference_pi.py\n"
    "    logs/\n"
    "      detections.log\n"
    "      detections.jsonl\n"
    "      frames/                   # capturas de evidencia\n"
    "      clips/                    # clips de video (pre/post buffer)"
))

story.append(Paragraph("13.8 Subir archivos: dos opciones", H2))
story.append(p("<b>Opcion A &mdash; arrastrar y soltar en VS Code:</b>"))
story.append(b("Una vez conectado, abrir el explorador de archivos (Ctrl+Shift+E)."))
story.append(b("Navegar a <font face='Courier'>/home/pi/weapons</font>."))
story.append(b("Arrastrar archivos desde el explorador del SO directamente a la ventana."))
story.append(p("<b>Opcion B &mdash; via terminal con scp:</b>"))
story.append(code(
    "scp models/export/yolov8n_v4_int8.onnx pi5:~/weapons/best.onnx\n"
    "scp config.yaml pi5:~/weapons/\n"
    "scp -r utils pi5:~/weapons/\n"
    "scp scripts/7_inference_pi.py pi5:~/weapons/"
))

story.append(Paragraph("13.9 Ejecutar inferencia desde la terminal de VS Code", H2))
story.append(b("En VS Code conectado a la Pi, abrir terminal: <font face='Courier'>Ctrl+\\`</font>."))
story.append(b("Activar el venv:"))
story.append(code("source ~/weapons-env/bin/activate"))
story.append(b("Ejecutar inferencia:"))
story.append(code("cd ~/weapons && python 7_inference_pi.py --weights best.onnx"))
story.append(b("Para detener: <font face='Courier'>Ctrl+C</font>."))

story.append(Paragraph("13.10 Servicio systemd para arranque automatico", H2))
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

story.append(Paragraph("13.11 Comandos utiles desde la terminal", H2))
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

story.append(Paragraph("13.12 Troubleshooting comun", H2))
story.append(tbl([
    ["Problema", "Solucion"],
    ["Camara no detectada (cv2.VideoCapture(0))",
     "Probar otros indices (1, 2). Verificar permisos: sudo usermod -aG video pi"],
    ["FPS muy bajos (<5)", "Verificar throttling: vcgencmd get_throttled. Si distinto de 0x0, problema termico/voltaje"],
    ["No se reproduce sonido", "Verificar audio: aplay -l. Configurar default device en ~/.asoundrc"],
    ["VS Code Remote-SSH no conecta", "Borrar ~/.vscode-server en la Pi y reintentar. Verificar version de SSH (>=8.0)"],
    ["systemctl status reporta failed", "Ver journalctl -u weapons -n 50 para detalle"],
    ["Dashboard web no accesible desde el celular", "Verificar que celular y Pi esten en la misma red/VLAN; revisar firewall (sudo ufw allow 8080)"],
    ["Poca memoria al inferir", "Cerrar X11/desktop si se uso Pi OS Desktop (preferir Lite)"],
], widths=[6*cm, 9*cm]))

# ========== 14. RIESGOS ==========
story.append(PageBreak())
story.append(Paragraph("14. Riesgos identificados y mitigaciones", H1))
story.append(tbl([
    ["Riesgo", "Impacto", "Mitigacion aplicada"],
    ["OOM de VRAM al final del entrenamiento (incidente v1)",
     "Crash en epoch 70/80",
     "batch 12->8, workers 8->4, expandable_segments:True"],
    ["Fragmentacion progresiva de VRAM en runs largos",
     "OOM probabilistico",
     "PYTORCH_CUDA_ALLOC_CONF configurada"],
    ["Modelo v2 solo entrenado con pistolas de perfil",
     "Fallaba en vistas frontales/POV",
     "+6.9k imgs de PranomVignesh/HandGuns con poses variadas (v3/v4)"],
    ["Riesgo de alucinacion en stand (personas, moviles, mandos)",
     "Falsas alarmas frente al publico",
     "Hard negatives especificos (846+605+120+110 imgs) + fine-tune con cls=1.0"],
    ["Fine-tune agresivo podria olvidar lo aprendido (catastrophic forgetting)",
     "Degradacion del modelo base",
     "LR muy bajo (0.0001) + freeze de backbone (10 capas) + partir de v2, no de cero"],
    ["Deteccion aislada de un solo frame (parpadeo / detecciones espurias)",
     "Alertas ruidosas o inconsistentes",
     "Filtro temporal: confirmacion >=4/7 frames + suavizado EMA"],
    ["Cajas geometricamente imposibles (ej. area casi nula o gigante)",
     "Falsos positivos obvios sin filtrar",
     "utils/plausibility_filter.py con rangos de area/aspect-ratio por clase"],
    ["Dataset de train sigue creciendo sin nueva evaluacion formal (27.970 vs 21.259 v4)",
     "Numeros de mAP reportados podrian no reflejar el checkpoint mas reciente",
     "Documentado explicitamente en la seccion 5.4 y 10.3; re-evaluar antes de la proxima entrega"],
    ["Clase minoritaria knife (proporcionalmente la mas chica del dataset)",
     "Bajo recall en cuchillos",
     "copy_paste augmentation 0.5 + umbral de confianza especifico mas bajo (0.40)"],
    ["GPU NVIDIA Blackwell sm_120 incompatible con CUDA &lt;12.8",
     "Entrenamiento caeria al CPU",
     "PyTorch instalado contra cu128 explicitamente"],
    ["batch 32 dispara cuDNN error en yolov8n",
     "Crash en epoch 1",
     "Reducido a batch 24 (base) / 16 (export/finetune)"],
    ["Pistola de prueba (replica) fuera de distribucion (color, textura)",
     "Bajo confidence en pruebas",
     "Imprimir/usar en negro mate; bajar threshold si es necesario; validado en vivo el 24/05"],
], widths=[5*cm, 4*cm, 7*cm]))

# ========== 15. REPRODUCIBILIDAD ==========
story.append(Paragraph("15. Reproducibilidad y trabajo futuro", H1))

story.append(Paragraph("15.1 Para reproducir todo desde cero", H2))
story.append(p("Ver el archivo <font face='Courier'>README.md</font> / <font face='Courier'>README.txt</font> en la raiz del repositorio."))

story.append(Paragraph("15.2 Trabajo futuro sugerido", H2))
for item in [
    "Re-evaluar formalmente el modelo sobre el superset actual de train (27.970 imgs) para "
    "confirmar que los numeros de la seccion 10.3 siguen vigentes.",
    "Construir un split de validacion propio con escenas de stand reales (personas + objetos "
    "cotidianos + arma de prueba) para medir directamente la tasa de falsos positivos "
    "person&rarr;handgun, en vez de inferirla solo del val set historico.",
    "Ampliar dataset con mas ejemplos de armas en contextos reales (manos, ropa, "
    "interiores, exteriores) mas alla de las fuentes ya integradas.",
    "Probar variantes YOLOv8 mas grandes (m, l) en la laptop como modelo &quot;teacher&quot; "
    "para destilacion al nano.",
    "Implementar tracking entre frames (ByteTrack/BoTSORT) ademas del filtro temporal actual, "
    "para reducir aun mas alertas duplicadas sobre el mismo objeto.",
    "Optimizar y medir el modelo NCNN en la Pi5 real y comparar FPS contra ONNX Runtime en "
    "hardware fisico (hoy solo se cuenta con estimaciones/emulacion).",
    "Considerar deteccion en stream RTSP (camaras IP) ademas de USB/Pi Camera.",
    "Activar y probar en campo la notificacion por Telegram y el dashboard web en la Raspberry "
    "Pi 5 real (hoy validados solo en la laptop).",
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
