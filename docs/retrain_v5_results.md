# Reentrenamiento v5 (mas hard negatives + armas Roboflow)

Fecha: 2026-07-18 14:39  |  Epochs: 15
Entrenamiento GRANDE ok: True  |  NANO ok: True

| Modelo | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|---|---|---|
| small_v4 | 0.8865 | 0.6575 | 0.8474 | 0.8246 |
| small_v5 | 0.8909 | 0.6537 | 0.8509 | 0.8238 |
| nano_v4 | 0.8577 | 0.6344 | 0.8927 | 0.7578 |
| nano_v5 | 0.8532 | 0.6380 | 0.9003 | 0.7407 |

## Lectura
- Small v5 vs v4: mAP50 +0.45pp, Precision +0.35pp, Recall -0.08pp
- Precision sube = menos falsos positivos (objetivo). Recall no debe caer mucho.