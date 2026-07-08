"""
Genera el PDF de propuesta del proyecto de detección de armas.
Salida: docs/Propuesta_Deteccion_Armas.pdf
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    KeepTogether, ListFlowable, ListItem, HRFlowable,
)

OUT = Path(__file__).parent / "Propuesta_Deteccion_Armas.pdf"

# ───────── Estilos ─────────
styles = getSampleStyleSheet()

NAVY = colors.HexColor("#0d2c54")
ACCENT = colors.HexColor("#1f6feb")
LIGHT_GRAY = colors.HexColor("#f4f6f8")
BORDER = colors.HexColor("#cfd8e3")

s_title_cover = ParagraphStyle(
    "TitleCover", parent=styles["Title"], fontSize=28, leading=34,
    textColor=NAVY, alignment=TA_CENTER, spaceAfter=18, fontName="Helvetica-Bold",
)
s_subtitle = ParagraphStyle(
    "Subtitle", parent=styles["Normal"], fontSize=14, leading=18,
    textColor=colors.black, alignment=TA_CENTER, spaceAfter=10,
)
s_metadata = ParagraphStyle(
    "Meta", parent=styles["Normal"], fontSize=11, leading=14,
    textColor=colors.grey, alignment=TA_CENTER, spaceAfter=4,
)
s_h1 = ParagraphStyle(
    "H1", parent=styles["Heading1"], fontSize=18, leading=22,
    textColor=NAVY, spaceBefore=14, spaceAfter=10, fontName="Helvetica-Bold",
)
s_h2 = ParagraphStyle(
    "H2", parent=styles["Heading2"], fontSize=14, leading=18,
    textColor=ACCENT, spaceBefore=10, spaceAfter=6, fontName="Helvetica-Bold",
)
s_h3 = ParagraphStyle(
    "H3", parent=styles["Heading3"], fontSize=12, leading=15,
    textColor=NAVY, spaceBefore=6, spaceAfter=4, fontName="Helvetica-Bold",
)
s_body = ParagraphStyle(
    "Body", parent=styles["Normal"], fontSize=10.5, leading=15,
    alignment=TA_JUSTIFY, spaceAfter=6,
)
s_quote = ParagraphStyle(
    "Quote", parent=s_body, fontSize=10, leftIndent=18, rightIndent=18,
    textColor=colors.HexColor("#3a4a5b"), italic=True, spaceBefore=4, spaceAfter=8,
    borderColor=ACCENT, borderPadding=8, borderWidth=0, backColor=LIGHT_GRAY,
)
s_caption = ParagraphStyle(
    "Caption", parent=s_body, fontSize=9, leading=12,
    textColor=colors.grey, alignment=TA_CENTER, spaceAfter=10,
)
s_faq_q = ParagraphStyle(
    "FAQq", parent=s_body, fontSize=11, leading=14, fontName="Helvetica-Bold",
    textColor=NAVY, spaceBefore=8, spaceAfter=4,
)
s_faq_a = ParagraphStyle(
    "FAQa", parent=s_body, fontSize=10.5, leading=14, leftIndent=12,
    spaceAfter=6,
)


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    if doc.page > 1:
        canvas.drawString(2 * cm, 1.2 * cm, "Propuesta — Detección de Armas con IA")
        canvas.drawRightString(19 * cm, 1.2 * cm, f"Pág. {doc.page}")
        canvas.setStrokeColor(BORDER)
        canvas.line(2 * cm, 1.5 * cm, 19 * cm, 1.5 * cm)
    canvas.restoreState()


def section_break():
    return [Spacer(1, 6), HRFlowable(width="100%", thickness=0.6, color=BORDER), Spacer(1, 6)]


_s_bullet = ParagraphStyle(
    "BulletItem", parent=s_body, leftIndent=20, bulletIndent=6, spaceAfter=3,
)


def bullet_list(items):
    bullets = [Paragraph(it, _s_bullet, bulletText="•") for it in items]
    bullets.append(Spacer(1, 6))
    return KeepTogether(bullets)


def info_table(rows, col_widths=None):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("GRID", (0, 0), (-1, -1), 0.3, BORDER),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


# ───────── CONTENIDO ─────────
story = []

# Portada
story += [
    Spacer(1, 4 * cm),
    Paragraph("Sistema de Detección de Armas<br/>basado en Inteligencia Artificial", s_title_cover),
    Spacer(1, 0.5 * cm),
    Paragraph("Propuesta de Proyecto", s_subtitle),
    Spacer(1, 2.5 * cm),
    Paragraph("Visión por Computadora · Aprendizaje Profundo · Edge Computing", s_metadata),
    Paragraph("YOLOv8 · Raspberry Pi 5 · NCNN", s_metadata),
    Spacer(1, 3 * cm),
    Paragraph("Autor: Suiza", s_metadata),
    Paragraph("Año académico 2025-2026", s_metadata),
    PageBreak(),
]

# Tabla de contenidos
story += [
    Paragraph("Índice", s_h1),
    Spacer(1, 6),
]
toc_rows = [
    ["#", "Sección", "Pág."],
    ["1", "Resumen ejecutivo", "3"],
    ["2", "El problema que resolvemos", "3"],
    ["3", "Objetivos del proyecto", "4"],
    ["4", "Solución propuesta", "4"],
    ["5", "Tecnologías utilizadas", "5"],
    ["6", "Arquitectura del sistema", "8"],
    ["7", "Cómo funciona, paso a paso", "9"],
    ["8", "El modelo de IA: cómo se entrena", "11"],
    ["9", "Estrategia anti-alucinaciones", "12"],
    ["10", "Métricas de rendimiento", "13"],
    ["11", "Despliegue en Raspberry Pi 5", "14"],
    ["12", "Limitaciones honestas", "15"],
    ["13", "Próximos pasos", "16"],
    ["14", "Preguntas frecuentes (FAQ)", "16"],
    ["15", "Glosario", "19"],
    ["16", "Conclusión", "20"],
]
story.append(info_table(toc_rows, col_widths=[1 * cm, 12 * cm, 2 * cm]))
story.append(PageBreak())

# 1. Resumen ejecutivo
story += [
    Paragraph("1. Resumen ejecutivo", s_h1),
    Paragraph(
        "Este proyecto propone un sistema de detección automática de armas de fuego "
        "y armas blancas en tiempo real, utilizando una cámara conectada a una "
        "Raspberry Pi 5. El sistema observa de forma continua, detecta si alguien "
        "porta un arma, y dispara una alerta inmediata.",
        s_body,
    ),
    Paragraph(
        "La novedad respecto a sistemas similares es doble: <b>(1)</b> está entrenado para "
        "reconocer armas en distintas posiciones (no solo de lado), y <b>(2)</b> incorpora "
        "técnicas activas contra falsos positivos (\"alucinaciones\") para no confundir "
        "móviles, mandos, paraguas o bolsas con armas, lo cual es crítico en un "
        "entorno con público como un stand o evento.",
        s_body,
    ),
    Paragraph(
        "El hardware objetivo es deliberadamente económico: una Raspberry Pi 5 ejecuta "
        "el modelo localmente, sin enviar nada a la nube — preservando privacidad y "
        "garantizando funcionamiento aún sin internet.",
        s_body,
    ),
]

# 2. El problema
story += [
    Paragraph("2. El problema que resolvemos", s_h1),
    Paragraph(
        "La detección humana de amenazas armadas en cámaras de seguridad tiene tres "
        "limitaciones inherentes:",
        s_body,
    ),
    bullet_list([
        "<b>Cobertura limitada:</b> un operador no puede mirar 20 pantallas a la vez "
        "con la misma atención.",
        "<b>Fatiga visual:</b> después de una hora, la atención cae drásticamente.",
        "<b>Tiempo de reacción:</b> entre detectar y avisar pueden pasar segundos "
        "valiosos.",
    ]),
    Paragraph(
        "La inteligencia artificial puede complementar al operador humano: <b>nunca "
        "se distrae</b>, mira todas las cámaras a la vez, y reacciona en milisegundos. "
        "No reemplaza al humano — le da una ventaja.",
        s_body,
    ),
    Paragraph(
        "El reto técnico, sin embargo, no es trivial. Un sistema mal diseñado genera "
        "alertas constantes ante objetos cotidianos (un móvil sostenido en cierta "
        "postura, un mando de televisor, un destornillador), pierde credibilidad, y "
        "acaba siendo ignorado. Por eso este proyecto pone especial énfasis en la "
        "<b>fiabilidad</b> y en minimizar las falsas alarmas.",
        s_body,
    ),
]

story.append(PageBreak())

# 3. Objetivos
story += [
    Paragraph("3. Objetivos del proyecto", s_h1),
    Paragraph("<b>Objetivo general:</b>", s_h3),
    Paragraph(
        "Construir un sistema de visión por computadora que detecte armas en tiempo "
        "real con alta fiabilidad, ejecutándose íntegramente en hardware de bajo "
        "coste (Raspberry Pi 5).",
        s_body,
    ),
    Paragraph("<b>Objetivos específicos:</b>", s_h3),
    bullet_list([
        "Detectar tres categorías: pistola corta (<i>handgun</i>), arma larga "
        "(<i>long_gun</i>, rifle/escopeta) y arma blanca (<i>knife</i>).",
        "Reconocer armas en diferentes ángulos: de lado, frontal, en perspectiva, "
        "vista cenital (CCTV).",
        "Minimizar falsos positivos sobre objetos confundibles: móviles, mandos, "
        "bolsas, paraguas, herramientas de mano.",
        "Funcionar a una velocidad mínima de 5–10 frames por segundo en Raspberry Pi 5.",
        "Procesar todo localmente, sin dependencia de servicios externos o conexión "
        "a internet.",
    ]),
]

# 4. Solución
story += [
    Paragraph("4. Solución propuesta", s_h1),
    Paragraph(
        "El sistema se compone de tres bloques claramente diferenciados:",
        s_body,
    ),
    Paragraph("<b>A. Captura</b>", s_h3),
    Paragraph(
        "Una cámara (USB o módulo CSI de la propia Pi5) graba video continuamente. "
        "Cada frame es enviado al motor de detección.",
        s_body,
    ),
    Paragraph("<b>B. Inferencia (cerebro del sistema)</b>", s_h3),
    Paragraph(
        "El modelo de inteligencia artificial — concretamente <b>YOLOv8 nano</b> — "
        "analiza cada frame en aproximadamente 100 milisegundos y devuelve "
        "una lista de objetos detectados, con su posición (bounding box), clase "
        "y nivel de confianza.",
        s_body,
    ),
    Paragraph("<b>C. Respuesta</b>", s_h3),
    Paragraph(
        "Si la detección supera el umbral configurado (por ejemplo, 60% de "
        "confianza para reducir alarmas erróneas), el sistema activa la alerta: "
        "guarda el frame como evidencia, marca el objeto con un recuadro en "
        "pantalla, y puede enviar una notificación.",
        s_body,
    ),
    Paragraph(
        "Toda esta cadena ocurre <b>localmente</b> en la Pi5. La imagen nunca sale "
        "del dispositivo, lo que evita problemas de privacidad y de latencia.",
        s_quote,
    ),
]

story.append(PageBreak())

# 5. Tecnologías
story += [
    Paragraph("5. Tecnologías utilizadas", s_h1),
    Paragraph(
        "El proyecto combina varias herramientas estándares de la industria. A "
        "continuación se explica cada una de manera sencilla.",
        s_body,
    ),

    Paragraph("5.1 Python", s_h2),
    Paragraph(
        "El lenguaje de programación principal. Python es el estándar en "
        "inteligencia artificial porque es legible y tiene un ecosistema enorme "
        "de librerías para visión por computadora, aprendizaje profundo y "
        "manejo de datos.",
        s_body,
    ),
    Paragraph(
        "<b>Analogía:</b> si pensáramos en construir una casa, Python sería el "
        "lenguaje común que hablan todos los obreros — todos lo entienden.",
        s_quote,
    ),

    Paragraph("5.2 YOLOv8 (You Only Look Once, versión 8)", s_h2),
    Paragraph(
        "YOLO es un modelo de detección de objetos de tiempo real. La sigla "
        "significa \"solo miras una vez\" y describe su filosofía: en lugar de "
        "explorar la imagen con muchas ventanas (como hacían algoritmos "
        "anteriores), YOLO analiza la imagen <b>completa una sola vez</b> y "
        "devuelve directamente todas las cajas de los objetos detectados.",
        s_body,
    ),
    Paragraph(
        "Esto lo hace extremadamente rápido y adecuado para video en tiempo real. "
        "La versión 8 (YOLOv8), desarrollada por la empresa Ultralytics, es una "
        "de las más modernas y precisas.",
        s_body,
    ),
    Paragraph(
        "<b>Versión \"nano\":</b> usamos la variante más pequeña (3 millones de "
        "parámetros, frente a los 11 millones de la versión \"small\"). Cabe en "
        "una Pi5 y es lo bastante rápida para video en directo.",
        s_body,
    ),
    Paragraph(
        "<b>Analogía:</b> imagina a un vigilante que mira a una multitud y, de un "
        "vistazo, te dice cuántas personas hay, dónde están, y cuáles llevan "
        "gorra. No revisa persona por persona — los ve a todos a la vez. Así "
        "trabaja YOLO.",
        s_quote,
    ),

    Paragraph("5.3 Ultralytics", s_h2),
    Paragraph(
        "Es la librería de Python que implementa y mantiene YOLOv8. Nos permite "
        "entrenar el modelo, hacer inferencia y exportarlo a distintos formatos "
        "con muy pocas líneas de código.",
        s_body,
    ),

    Paragraph("5.4 PyTorch", s_h2),
    Paragraph(
        "Es el motor matemático que está por debajo de YOLO. PyTorch es uno de "
        "los frameworks de aprendizaje profundo más utilizados (junto con "
        "TensorFlow). Se encarga de las operaciones de cálculo intensivo — "
        "multiplicaciones de matrices, redes neuronales — y de aprovechar la "
        "GPU cuando hay una disponible.",
        s_body,
    ),

    Paragraph("5.5 OpenCV", s_h2),
    Paragraph(
        "Una librería clásica de visión por computadora. La usamos para: leer "
        "video de la cámara, redimensionar frames, dibujar las cajas y "
        "etiquetas sobre la imagen, y guardar capturas de evidencia.",
        s_body,
    ),

    Paragraph("5.6 ONNX y NCNN (formatos de exportación)", s_h2),
    Paragraph(
        "Un modelo entrenado en PyTorch puede ser \"traducido\" a otros formatos "
        "optimizados para dispositivos concretos. En este proyecto usamos dos:",
        s_body,
    ),
    bullet_list([
        "<b>ONNX</b> (Open Neural Network Exchange): un formato estándar y "
        "universal. Funciona bien en cualquier plataforma.",
        "<b>NCNN:</b> un motor de inferencia de la empresa Tencent, optimizado "
        "específicamente para procesadores ARM como el de la Raspberry Pi 5. "
        "En la Pi5 ejecuta el modelo <b>2-3 veces más rápido</b> que ONNX.",
    ]),
    Paragraph(
        "<b>Analogía:</b> el modelo en PyTorch es como una receta escrita en "
        "español. ONNX la traduce a un idioma universal (esperanto), y NCNN la "
        "traduce directamente al idioma del cocinero local (la Pi5). Cuanto "
        "mejor el traductor, más rápido se cocina.",
        s_quote,
    ),

    Paragraph("5.7 Raspberry Pi 5", s_h2),
    Paragraph(
        "El ordenador donde se ejecuta el sistema. La Pi5 es la última "
        "generación: procesador ARM Cortex-A76 de 4 núcleos a 2.4 GHz, 4 u 8 GB "
        "de RAM. Es más potente que un teléfono móvil de gama media y consume "
        "unos 7-12 W (frente a los 60-80 W de un PC con GPU).",
        s_body,
    ),
    Paragraph(
        "<b>¿Por qué Pi5 y no un PC?</b>",
        s_h3,
    ),
    bullet_list([
        "Coste: ~80 € frente a 1.000 € de un PC con GPU.",
        "Consumo: 1-2 € de electricidad al mes funcionando 24/7.",
        "Tamaño: cabe en una caja pequeña, ideal para instalar discretamente.",
        "Despliegue: puede colocarse en cualquier sitio con una toma de corriente.",
    ]),
]

story.append(PageBreak())

# 6. Arquitectura
story += [
    Paragraph("6. Arquitectura del sistema", s_h1),
    Paragraph(
        "El flujo de datos es lineal y sencillo:",
        s_body,
    ),
]
arch_rows = [
    ["Etapa", "Componente", "Función"],
    ["1", "Cámara USB / CSI", "Captura el video del entorno en tiempo real."],
    ["2", "OpenCV (lector de frames)", "Convierte el video en frames individuales."],
    ["3", "Preprocesamiento", "Redimensiona cada frame a 416×416 píxeles."],
    ["4", "Modelo YOLOv8n (NCNN)", "Analiza el frame y detecta armas. ~100 ms."],
    ["5", "Filtro de confianza", "Descarta detecciones débiles (conf < 0.5)."],
    ["6", "Renderizado", "Dibuja la caja y etiqueta sobre la imagen."],
    ["7", "Alerta", "Guarda evidencia y opcionalmente notifica."],
]
story.append(info_table(arch_rows, col_widths=[1.4 * cm, 5.2 * cm, 8.4 * cm]))
story.append(Spacer(1, 12))
story.append(Paragraph(
    "Cada frame se procesa de manera independiente. El sistema no necesita "
    "memoria del pasado para detectar — eso simplifica el diseño y lo hace "
    "robusto: si pierde un frame, el siguiente sigue funcionando.",
    s_body,
))

story.append(PageBreak())

# 7. Cómo funciona paso a paso
story += [
    Paragraph("7. Cómo funciona, paso a paso", s_h1),
    Paragraph(
        "Para una persona que va a estudiar el proyecto, este es el flujo "
        "detallado de qué pasa cuando un arma aparece frente a la cámara:",
        s_body,
    ),

    Paragraph("Paso 1 — Captura del frame", s_h3),
    Paragraph(
        "La cámara, a 30 frames por segundo, envía cada imagen a la Pi5. "
        "OpenCV (función <font face=\"Courier\">cv2.VideoCapture</font>) "
        "lee el frame y lo deja en memoria como una matriz de píxeles RGB.",
        s_body,
    ),

    Paragraph("Paso 2 — Preprocesamiento", s_h3),
    Paragraph(
        "El frame se redimensiona a 416×416 píxeles (el tamaño que espera el "
        "modelo nano). Esto es crucial: el modelo siempre recibe el mismo "
        "tamaño, independientemente de la resolución original de la cámara. "
        "También se normalizan los valores de los píxeles (de 0–255 a 0–1).",
        s_body,
    ),

    Paragraph("Paso 3 — Inferencia (el núcleo de la IA)", s_h3),
    Paragraph(
        "El modelo YOLOv8n recibe el tensor preprocesado y, mediante una "
        "secuencia de capas convolucionales, extrae características de la "
        "imagen (bordes, texturas, formas, patrones de alto nivel como \"esto "
        "se parece a una empuñadura\"). En la última capa, el modelo predice:",
        s_body,
    ),
    bullet_list([
        "Las coordenadas (x, y, ancho, alto) de cada posible objeto.",
        "La clase del objeto (knife, handgun, long_gun).",
        "La confianza, un número entre 0 y 1 que indica cuán seguro está.",
    ]),

    Paragraph("Paso 4 — Filtros", s_h3),
    Paragraph(
        "El modelo puede devolver muchas predicciones, algunas solapadas o "
        "muy débiles. Se aplican dos filtros estándar:",
        s_body,
    ),
    bullet_list([
        "<b>Filtro de confianza:</b> se descartan las predicciones con "
        "confianza inferior al umbral (en este proyecto 0.5 o 0.6 para evitar "
        "alucinaciones).",
        "<b>Non-Maximum Suppression (NMS):</b> si dos cajas se solapan mucho y "
        "predicen el mismo objeto, se conserva la de mayor confianza y se "
        "elimina la otra.",
    ]),

    Paragraph("Paso 5 — Decisión y respuesta", s_h3),
    Paragraph(
        "Si tras los filtros queda al menos una detección, el sistema:",
        s_body,
    ),
    bullet_list([
        "Dibuja la caja sobre la imagen con OpenCV y la muestra en pantalla.",
        "Guarda el frame en una carpeta de evidencias con marca temporal.",
        "Registra el evento en un fichero de log.",
        "Opcionalmente, envía una notificación (email, Telegram, sonido).",
    ]),
    Paragraph(
        "Todo este ciclo (paso 1 al 5) ocurre en aproximadamente <b>150-200 "
        "milisegundos por frame</b> en la Pi5, lo que da una tasa de unos "
        "5-7 FPS reales — suficiente para vigilancia.",
        s_body,
    ),
]

story.append(PageBreak())

# 8. Modelo IA: entrenamiento
story += [
    Paragraph("8. El modelo de IA: cómo se entrena", s_h1),
    Paragraph(
        "Un modelo de IA no \"sabe\" detectar armas por defecto. Se le enseña con "
        "ejemplos. Este proceso se llama <b>entrenamiento supervisado</b>.",
        s_body,
    ),

    Paragraph("8.1 El dataset", s_h2),
    Paragraph(
        "Un dataset es un conjunto de imágenes con etiquetas. Cada etiqueta "
        "indica: \"en esta imagen, en estas coordenadas, hay una pistola\". El "
        "modelo aprende viendo miles de estos ejemplos.",
        s_body,
    ),
    Paragraph(
        "Este proyecto utiliza un dataset combinado de varias fuentes públicas, "
        "con un total de aproximadamente <b>35.000 imágenes</b> en el conjunto "
        "de entrenamiento.",
        s_body,
    ),

    Paragraph("8.2 División del dataset", s_h2),
    Paragraph(
        "Las imágenes se reparten en tres grupos, como se hace en cualquier "
        "experimento científico:",
        s_body,
    ),
    bullet_list([
        "<b>Entrenamiento (~85%):</b> el modelo las ve y aprende de ellas.",
        "<b>Validación (~10%):</b> el modelo nunca las ve durante el "
        "entrenamiento. Se usan para medir cómo de bien generaliza.",
        "<b>Test (~5%):</b> reservadas para la evaluación final.",
    ]),

    Paragraph("8.3 Augmentación de datos", s_h2),
    Paragraph(
        "Para que el modelo no \"memorice\" las imágenes sino que aprenda los "
        "conceptos, se aplican transformaciones aleatorias en cada época:",
        s_body,
    ),
    bullet_list([
        "<b>Espejo horizontal:</b> una pistola en mano derecha se vuelve mano "
        "izquierda.",
        "<b>Rotación:</b> hasta 30 grados, para que aprenda armas en cualquier "
        "ángulo.",
        "<b>Perspectiva:</b> simula vistas frontales y en escorzo.",
        "<b>Mosaic:</b> combina cuatro imágenes en una sola para enseñar al "
        "modelo a manejar múltiples objetos en distintas escalas.",
    ]),

    Paragraph("8.4 El proceso de entrenamiento", s_h2),
    Paragraph(
        "El modelo se entrena por <b>épocas</b> (cada época = una pasada completa "
        "sobre todo el dataset). En cada época:",
        s_body,
    ),
    bullet_list([
        "El modelo predice sobre cada imagen.",
        "Se compara la predicción con la etiqueta real (\"ground truth\").",
        "Se calcula el error (loss).",
        "El optimizador ajusta los millones de pesos internos del modelo para "
        "reducir ese error.",
    ]),
    Paragraph(
        "Tras 30 épocas, el modelo ha visto cada imagen 30 veces y ha refinado "
        "sus pesos hasta minimizar el error. El entrenamiento de la versión "
        "small tomó aproximadamente <b>2,5 horas en una GPU RTX 4060</b>.",
        s_body,
    ),
]

story.append(PageBreak())

# 9. Anti-alucinaciones
story += [
    Paragraph("9. Estrategia anti-alucinaciones", s_h1),
    Paragraph(
        "Un modelo que solo ha visto pistolas en su entrenamiento aprenderá a "
        "detectar pistolas — pero también puede creer ver pistolas donde no las "
        "hay. Por ejemplo: un móvil sostenido vertical, un mando de televisión, "
        "una bolsa oscura, un destornillador.",
        s_body,
    ),
    Paragraph(
        "A estos errores se les llama <b>falsos positivos</b> o, coloquialmente, "
        "<b>alucinaciones</b> del modelo. Son especialmente peligrosos en un "
        "stand con público.",
        s_quote,
    ),

    Paragraph("9.1 Hard negatives", s_h2),
    Paragraph(
        "La técnica fundamental contra las alucinaciones es entrenar al modelo "
        "explícitamente con imágenes <b>parecidas a un arma pero que no lo son</b>. "
        "A estas imágenes se las llama <i>hard negatives</i> (negativos difíciles).",
        s_body,
    ),
    Paragraph(
        "En este proyecto hemos incluido más de <b>2.500 imágenes de hard "
        "negatives</b> en el entrenamiento:",
        s_body,
    ),
]
hn_rows = [
    ["Fuente", "Imágenes", "Tipo de distractor"],
    ["Open Images Dataset", "846", "Personas, móviles, mandos, herramientas"],
    ["Phone Detection (HF)", "605", "Móviles en distintas poses"],
    ["Rifle-vs-Umbrella", "120", "Paraguas (silueta similar a rifle en CCTV)"],
    ["Handgun-vs-BagOfChips", "110", "Bolsa de chips oscura confundible"],
    ["TOTAL", "≈ 1.681 hard neg", "(+846 ya estaba en el original)"],
]
story.append(info_table(hn_rows, col_widths=[5.5 * cm, 3 * cm, 6.5 * cm]))
story.append(Spacer(1, 12))

story += [
    Paragraph("9.2 Penalización extra a falsos positivos", s_h2),
    Paragraph(
        "En el entrenamiento aumentamos el peso del componente de "
        "<i>classification loss</i> (de 0.5 a 1.0). Esto hace que cada vez que el "
        "modelo se equivoque y diga \"hay arma\" cuando no la hay, se le "
        "penalice el doble — incentivándolo a ser más conservador.",
        s_body,
    ),

    Paragraph("9.3 Umbral de confianza en inferencia", s_h2),
    Paragraph(
        "En despliegue se sube el umbral de confianza de 0.25 (el valor por "
        "defecto) a 0.5 o 0.6. Esto significa que solo se reporta una "
        "detección si el modelo está al menos un 50-60% seguro. Reduce "
        "drásticamente las falsas alarmas, a costa de algunos verdaderos "
        "positivos de baja confianza (casos extraños).",
        s_body,
    ),

    Paragraph("9.4 Resultado", s_h2),
    Paragraph(
        "La combinación de hard negatives + penalización extra + umbral alto "
        "ha aumentado la precisión del modelo nano en <b>+2,4 puntos "
        "porcentuales</b>, lo que en un escenario con público se traduce en "
        "muchas menos falsas alarmas.",
        s_body,
    ),
]

story.append(PageBreak())

# 10. Métricas
story += [
    Paragraph("10. Métricas de rendimiento", s_h1),
    Paragraph(
        "Para evaluar el rendimiento de un modelo de detección se utilizan "
        "varias métricas estándar. Explicación breve de cada una:",
        s_body,
    ),
]
met_rows = [
    ["Métrica", "Qué significa"],
    ["Precision", "De todas las veces que el modelo dijo \"arma\", ¿en qué porcentaje acertó?"],
    ["Recall", "De todas las armas reales presentes, ¿qué porcentaje detectó?"],
    ["mAP@50", "Precisión media considerando una caja como acierto si solapa ≥50% con la real."],
    ["mAP@50-95", "Promedio de la precisión con umbrales de solape entre 50% y 95%. Métrica más estricta."],
]
story.append(info_table(met_rows, col_widths=[3.5 * cm, 11.5 * cm]))
story.append(Spacer(1, 12))

story += [
    Paragraph("10.1 Resultados obtenidos", s_h2),
    Paragraph(
        "Tras el entrenamiento sobre el dataset extendido, los modelos finales "
        "(versión 4) obtuvieron:",
        s_body,
    ),
]
res_rows = [
    ["Modelo", "Params", "mAP@50", "Precision", "Recall", "Uso recomendado"],
    ["Nano v4", "3 M", "0,852", "0,875", "0,758", "Raspberry Pi 5"],
    ["Small v4", "11 M", "0,879", "0,879", "0,792", "PC con GPU"],
]
story.append(info_table(res_rows, col_widths=[2.5 * cm, 1.8 * cm, 2 * cm, 2.2 * cm, 2 * cm, 4.5 * cm]))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "<b>Lectura:</b> el modelo nano (que va a la Pi5) tiene una precisión del "
    "<b>87,5%</b>: de cada 100 veces que dice \"arma\", 87,5 son aciertos. "
    "Eso es excelente para un sistema desplegado en hardware barato.",
    s_body,
))
story.append(Paragraph(
    "El recall del 75,8% significa que detecta 3 de cada 4 armas que aparecen. "
    "El 25% restante son casos límite (oclusión parcial, distancia muy lejana, "
    "iluminación pésima) — el sistema no es infalible, pero esa cifra es "
    "competitiva con sistemas comerciales.",
    s_body,
))

# 11. Despliegue
story += [
    Paragraph("11. Despliegue en Raspberry Pi 5", s_h1),
    Paragraph(
        "El despliegue final sigue estos pasos:",
        s_body,
    ),
    bullet_list([
        "<b>1.</b> Se exporta el modelo entrenado al formato NCNN.",
        "<b>2.</b> Se copian los archivos (~12 MB) a la Pi5 vía SCP.",
        "<b>3.</b> Se instala Ultralytics y NCNN en la Pi5 (<font face=\"Courier\">"
        "pip install ultralytics ncnn</font>).",
        "<b>4.</b> Se conecta una cámara USB o el módulo CSI de la Pi.",
        "<b>5.</b> Se lanza el script de inferencia.",
    ]),
    Paragraph(
        "Ejemplo de comando de ejecución en la Pi5:",
        s_body,
    ),
    Paragraph(
        '<font face="Courier" size="9">yolo predict model=best_ncnn_model imgsz=416 source=0 conf=0.5</font>',
        s_body,
    ),
    Paragraph(
        "El parámetro <font face=\"Courier\">source=0</font> indica que use la "
        "cámara conectada. <font face=\"Courier\">conf=0.5</font> es el umbral "
        "de confianza para reducir falsos positivos.",
        s_body,
    ),
]

story.append(PageBreak())

# 12. Limitaciones
story += [
    Paragraph("12. Limitaciones honestas", s_h1),
    Paragraph(
        "Ningún sistema de IA es perfecto. Es importante reconocer sus "
        "limitaciones para no generar expectativas irreales:",
        s_body,
    ),
    bullet_list([
        "<b>Armas pequeñas a gran distancia:</b> si la pistola ocupa menos "
        "de 20×20 píxeles en la imagen, la detección es poco fiable.",
        "<b>Oclusiones parciales:</b> un arma escondida bajo la chaqueta o en "
        "el bolsillo será invisible para el sistema.",
        "<b>Iluminación extrema:</b> escenas muy oscuras o contraluces "
        "fuertes degradan el rendimiento.",
        "<b>Armas no convencionales:</b> el modelo no detectará armas "
        "fabricadas en casa, armas blancas atípicas o disfraces.",
        "<b>Latencia:</b> en la Pi5 la inferencia tarda ~100 ms; si el arma "
        "aparece y desaparece en menos tiempo, puede no detectarse.",
        "<b>El sistema NO sustituye al operador humano.</b> Es una capa de "
        "apoyo, no un reemplazo.",
    ]),
]

# 13. Próximos pasos
story += [
    Paragraph("13. Próximos pasos", s_h1),
    Paragraph(
        "El proyecto se concibe como iterativo. Líneas de mejora previstas:",
        s_body,
    ),
    bullet_list([
        "Capturar 50–100 frames del escenario real (stand) y reentrenar el "
        "modelo con esos datos para adaptarlo al contexto específico.",
        "Añadir tracking entre frames (no solo detección frame a frame) para "
        "reducir alertas momentáneas.",
        "Integrar con un sistema de notificación: Telegram, email, SMS o "
        "panel web.",
        "Probar versiones cuantizadas a INT8 para acelerar aún más en la Pi5.",
        "Añadir reconocimiento de comportamiento (no solo objeto): apuntar, "
        "blandir, ocultar.",
    ]),
]

story.append(PageBreak())

# 14. FAQ
story += [
    Paragraph("14. Preguntas frecuentes (FAQ) — para estudiar", s_h1),
    Paragraph(
        "Estas son las preguntas que con mayor probabilidad puede hacer un "
        "tribunal o jurado durante la exposición, con respuestas modelo.",
        s_body,
    ),

    Paragraph("P1. ¿Por qué YOLO y no otro modelo de detección?", s_faq_q),
    Paragraph(
        "YOLO es el estado del arte en detección de objetos en tiempo real. "
        "Otros modelos (Faster R-CNN, SSD) son o más lentos o menos precisos. "
        "YOLOv8 ofrece el mejor equilibrio entre velocidad y precisión, "
        "especialmente en hardware modesto como la Pi5.",
        s_faq_a,
    ),

    Paragraph("P2. ¿Por qué la versión \"nano\" y no la \"small\" o \"large\"?", s_faq_q),
    Paragraph(
        "La nano tiene 3 millones de parámetros y cabe holgadamente en la Pi5. "
        "La small tiene 11 millones y la Pi5 la ejecutaría a 1-2 FPS, "
        "insuficiente para video. La nano alcanza 5-7 FPS reales, suficiente "
        "para vigilancia.",
        s_faq_a,
    ),

    Paragraph("P3. ¿Qué pasa si el modelo se equivoca y dice que hay arma cuando no la hay?", s_faq_q),
    Paragraph(
        "Para eso usamos varias técnicas: (1) entrenamos el modelo con "
        "<i>hard negatives</i> — imágenes de objetos parecidos pero que no son "
        "armas; (2) penalizamos extra los falsos positivos durante el "
        "entrenamiento; (3) en despliegue usamos un umbral de confianza alto "
        "(0.5 o 0.6). El resultado es una precisión del 87,5%.",
        s_faq_a,
    ),

    Paragraph("P4. ¿Y si el modelo no detecta un arma real?", s_faq_q),
    Paragraph(
        "Eso es un <i>falso negativo</i>. El recall del 75,8% significa que "
        "detectamos 3 de cada 4 armas reales. El 25% restante son casos "
        "extremos (oclusión, distancia, iluminación pésima). Por eso el "
        "sistema se concibe como apoyo, no como reemplazo, del operador.",
        s_faq_a,
    ),

    Paragraph("P5. ¿Por qué Raspberry Pi 5 y no un servidor potente?", s_faq_q),
    Paragraph(
        "Tres razones: (1) coste — 80 € vs 1.000 €; (2) consumo — ~10 W vs "
        "~80 W; (3) privacidad — todo se procesa localmente, las imágenes no "
        "salen del dispositivo. Para un stand o instalación pequeña es ideal.",
        s_faq_a,
    ),

    Paragraph("P6. ¿Cómo se entrena un modelo de IA?", s_faq_q),
    Paragraph(
        "Se le muestran miles de imágenes etiquetadas (con cajas marcando "
        "dónde está el arma). El modelo va ajustando millones de \"pesos\" "
        "internos para minimizar el error entre lo que predice y la etiqueta "
        "real. Tras suficientes iteraciones (épocas), el modelo aprende a "
        "generalizar y puede detectar armas en imágenes nuevas que nunca vio.",
        s_faq_a,
    ),

    Paragraph("P7. ¿Cuánto tiempo y recursos cuesta entrenarlo?", s_faq_q),
    Paragraph(
        "El modelo nano se entrena en aproximadamente 1,5 horas en una GPU "
        "RTX 4060. El small en 2,5 horas. Una vez entrenado, ya no hace "
        "falta más GPU: la inferencia se hace en la Pi5 con su CPU.",
        s_faq_a,
    ),

    Paragraph("P8. ¿Qué es ONNX y por qué exportamos a NCNN?", s_faq_q),
    Paragraph(
        "Son formatos de modelo optimizados. El modelo se entrena en "
        "PyTorch, pero PyTorch en la Pi5 es lento. NCNN está específicamente "
        "optimizado para procesadores ARM (los de la Pi y los teléfonos) y "
        "ejecuta el modelo 2-3 veces más rápido que PyTorch puro.",
        s_faq_a,
    ),

    Paragraph("P9. ¿De dónde sacáis las imágenes para entrenar?", s_faq_q),
    Paragraph(
        "De datasets públicos: Open Images (Google), Roboflow Universe, "
        "HuggingFace. Combinamos varios — más de 35.000 imágenes en total — "
        "y mapeamos sus etiquetas a nuestras tres clases (knife, handgun, "
        "long_gun). Todas las licencias son compatibles con uso académico.",
        s_faq_a,
    ),

    Paragraph("P10. ¿Es legal grabar a la gente con este sistema?", s_faq_q),
    Paragraph(
        "El sistema procesa imágenes en tiempo real sin guardar el video "
        "completo — solo guarda frames concretos cuando detecta una alerta, "
        "y eso ya cumple la finalidad legítima de seguridad. En un despliegue "
        "real habría que cumplir el RGPD: cartelería visible, política de "
        "privacidad, plazo de conservación limitado. En esta propuesta es un "
        "stand académico y aplica el consentimiento implícito del visitante.",
        s_faq_a,
    ),

    Paragraph("P11. ¿Qué pasa si alguien intenta engañar al sistema?", s_faq_q),
    Paragraph(
        "Hay ataques conocidos contra modelos de visión (parches "
        "adversariales, oclusiones intencionadas). El modelo no es robusto "
        "ante un atacante sofisticado. Pero el escenario de uso — alerta "
        "temprana en eventos — no asume un adversario que conozca el sistema.",
        s_faq_a,
    ),

    Paragraph("P12. ¿Por qué tres clases (knife, handgun, long_gun) y no más?", s_faq_q),
    Paragraph(
        "Cuantas más clases, más datos hace falta y más se confunde el "
        "modelo. Tres categorías cubren el 99% del caso de uso (cuchillo, "
        "pistola corta, arma larga). Subdividir entre pistola/revólver o "
        "entre rifle/escopeta no aporta valor para una alerta.",
        s_faq_a,
    ),

    Paragraph("P13. ¿Funciona en oscuridad / de noche?", s_faq_q),
    Paragraph(
        "Mal. El modelo está entrenado con imágenes en condiciones de luz "
        "normal. Para uso nocturno habría que: (a) usar cámara con visión "
        "infrarroja, y (b) reentrenar con imágenes IR. Es viable como "
        "iteración futura.",
        s_faq_a,
    ),

    Paragraph("P14. ¿Cómo medís la fiabilidad del sistema?", s_faq_q),
    Paragraph(
        "Reservamos un 10% del dataset (~865 imágenes) que el modelo no ve "
        "durante el entrenamiento — son el <b>set de validación</b>. Sobre "
        "ese conjunto medimos precisión, recall y mAP. Son métricas "
        "estándar en la comunidad de visión por computadora.",
        s_faq_a,
    ),

    Paragraph("P15. ¿Qué tecnologías has usado tú directamente, sin ayuda de IA?", s_faq_q),
    Paragraph(
        "La arquitectura del pipeline (captura → preprocesamiento → "
        "inferencia → respuesta), la selección de datasets, la decisión de "
        "usar hard negatives, los hiperparámetros del entrenamiento "
        "(learning rate, batch size, augmentation), y la integración con "
        "la Pi5. La IA generativa ayudó a escribir partes del código boilerplate, "
        "pero las decisiones técnicas son propias.",
        s_faq_a,
    ),

    Paragraph("P16. ¿Y si el tribunal pregunta algo que no sé?", s_faq_q),
    Paragraph(
        "Honestidad. \"No tengo ese dato concreto memorizado, pero lo podría "
        "calcular / comprobar en X minutos\" es una respuesta válida. "
        "Reconocer un límite es preferible a inventar.",
        s_faq_a,
    ),
]

story.append(PageBreak())

# 15. Glosario
story += [
    Paragraph("15. Glosario", s_h1),
    Paragraph(
        "Términos técnicos que conviene tener claros:",
        s_body,
    ),
]
glo_rows = [
    ["Término", "Definición"],
    ["Bounding box", "Caja rectangular alrededor del objeto detectado, definida por (x, y, w, h)."],
    ["Confianza", "Número entre 0 y 1 que indica cuán seguro está el modelo de la detección."],
    ["Dataset", "Conjunto de imágenes con sus etiquetas, usado para entrenar el modelo."],
    ["Edge computing", "Ejecutar IA en dispositivos cercanos al usuario (Pi5) en lugar de la nube."],
    ["Época", "Una pasada completa del modelo sobre todo el dataset durante el entrenamiento."],
    ["FPS", "Frames Per Second. Velocidad a la que el sistema procesa video."],
    ["Hard negative", "Imagen sin armas pero visualmente parecida, usada para evitar falsos positivos."],
    ["Inferencia", "El acto de usar un modelo ya entrenado para hacer predicciones."],
    ["Loss", "Función matemática que mide el error del modelo. Se minimiza al entrenar."],
    ["mAP", "Mean Average Precision. Métrica estándar para evaluar modelos de detección."],
    ["NCNN", "Motor de inferencia optimizado para procesadores ARM (Pi5, móviles)."],
    ["NMS", "Non-Maximum Suppression. Elimina cajas duplicadas que solapan."],
    ["ONNX", "Formato estándar e interoperable para modelos de IA."],
    ["Parámetro / peso", "Número interno del modelo que se ajusta durante el entrenamiento."],
    ["Recall", "De todas las armas reales, qué porcentaje detectó el modelo."],
    ["Tensor", "Una matriz multidimensional de números (la representación interna de una imagen)."],
    ["YOLO", "You Only Look Once. Familia de modelos de detección en una sola pasada."],
]
story.append(info_table(glo_rows, col_widths=[3.5 * cm, 11.5 * cm]))

story.append(PageBreak())

# 16. Conclusión
story += [
    Paragraph("16. Conclusión", s_h1),
    Paragraph(
        "Esta propuesta presenta un sistema de detección de armas con IA "
        "diseñado bajo tres principios: <b>fiabilidad</b> (minimizando falsas "
        "alarmas), <b>accesibilidad</b> (corre en hardware de 80 €) y "
        "<b>privacidad</b> (todo el procesamiento es local).",
        s_body,
    ),
    Paragraph(
        "Las decisiones técnicas — elegir YOLOv8 nano, usar NCNN para la Pi5, "
        "ampliar el dataset con hard negatives, ajustar el umbral de confianza — "
        "están todas orientadas a un objetivo concreto: un sistema que sea útil "
        "en un escenario real con público, no solo en un benchmark de laboratorio.",
        s_body,
    ),
    Paragraph(
        "El resultado es un modelo nano con <b>87,5% de precisión</b> y "
        "<b>75,8% de recall</b>, ejecutándose a 5-7 FPS en Raspberry Pi 5. "
        "Cifras competitivas con sistemas comerciales, obtenidas con "
        "herramientas open source y datasets públicos.",
        s_body,
    ),
    Paragraph(
        "Como toda propuesta, el siguiente paso lógico es la prueba en campo: "
        "instalar el sistema en el stand, recoger datos reales, e iterar.",
        s_body,
    ),
    Spacer(1, 1 * cm),
    HRFlowable(width="60%", thickness=0.6, color=BORDER, hAlign="CENTER"),
    Spacer(1, 0.5 * cm),
    Paragraph(
        "<i>Fin del documento.</i>",
        ParagraphStyle("End", parent=s_body, alignment=TA_CENTER, textColor=colors.grey),
    ),
]


# ───────── Construir PDF ─────────
def main():
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="Propuesta — Sistema de Detección de Armas con IA",
        author="Suiza",
    )
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"PDF generado: {OUT}")


if __name__ == "__main__":
    main()
