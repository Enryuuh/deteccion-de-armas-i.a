# Plan de Mejora: Poses Diversas + Anti-Alucinación (Stand)

## Problema actual
1. El modelo detecta bien pistolas **de lado** pero falla con vistas **frontales / POV / apuntando a cámara**.
2. Se presentará en un **stand** con personas, manos, teléfonos, llaves, objetos varios → el modelo NO debe alucinar armas en escenas con gente y objetos cotidianos.

## Por qué augmentar no basta
Las augmentaciones rotan/escalan imágenes existentes. Una pistola de lado rotada 25° sigue siendo una pistola de lado. Necesitamos **datos reales** de las vistas que faltan, no transformaciones.

## Estrategia (Opción B — solo datasets públicos)

### Fase 1 — Adquisición de datos (sin entrenar)
- **Positivos pose-diverso**: descargar datasets Roboflow/HF con pistolas en frontal, CCTV, POV, 3/4. Target: +1500-3000 imágenes.
- **Hard negatives tipo stand**: descargar imágenes de personas + objetos cotidianos (móviles, llaves, mandos, herramientas). Target: +1500-2500 imágenes sin etiqueta.

### Fase 2 — Fine-tune (cuando me digas)
- Partir de `best.pt` actual (NO entrenar desde cero — perderíamos lo bueno que ya tiene).
- LR muy bajo (`lr0=0.0001`) para no destruir lo aprendido.
- Aumentar **`cls` loss weight** → penaliza más fuerte la mala clasificación (reduce alucinaciones).
- **Freeze backbone** las primeras 10 epochs → solo afinar cabeza de detección.
- `close_mosaic=10` últimas epochs para fine-tune limpio.
- Augmentación con **`perspective=0.001`** y **`degrees=30`** moderada.
- 30 epochs total.

### Fase 3 — Validación
- Split de val específico de **poses frontales** y otro de **escenas con personas sin armas**.
- Medir mAP por subset + tasa de falsos positivos (`person → handgun`).
- Aceptar el run solo si:
  - mAP frontal sube ≥ +5pp.
  - mAP global no baja >2pp.
  - Falsos positivos en escenas con personas se reducen.

## Archivos creados
- `scripts/15_download_pose_diverse_weapons.py` — Roboflow datasets con poses diversas (requiere `ROBOFLOW_API_KEY`).
- `scripts/16_download_stand_hard_negatives.py` — Negativos de personas/objetos desde Open Images + HF.
- `scripts/17_finetune_anti_hallucination.py` — Fine-tune con receta anti-alucinación.

## Para correr cuando me avises
```bash
# 1) Si tienes Roboflow API key (opcional, para más poses)
set ROBOFLOW_API_KEY=tu_key
python scripts/15_download_pose_diverse_weapons.py

# 2) Hard negatives (no requiere key, usa Open Images ya cacheado)
python scripts/16_download_stand_hard_negatives.py

# 3) Fine-tune (cuando todo esté listo)
python scripts/17_finetune_anti_hallucination.py
```

## Notas técnicas
- El dataset actual ya tiene ~50% backgrounds (14k labels / 28k imgs). Sumamos más negativos específicos del escenario stand.
- Mantener proporción de clases: si añadimos 2000 imgs de handgun frontales, el modelo no se sesga porque ya hay 14k de armas en el train.
- Los hard negatives van con archivo `.txt` vacío en `labels/train/` (formato estándar de YOLO para background).
