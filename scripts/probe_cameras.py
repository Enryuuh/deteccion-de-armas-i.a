"""
Detector de camaras: prueba los indices 0..7 y muestra cuales devuelven
imagen REAL (no negra) y a que resolucion. Sirve para identificar tu camara
integrada cuando hay camaras virtuales (Iriun, OBS, VCAMDS) ocupando indices.

Uso:
    python scripts/probe_cameras.py
"""
import cv2
import numpy as np

print("Probando camaras (indices 0..7)...\n")
print(f"{'IDX':>3}  {'ABRE':>5}  {'RESOLUCION':>12}  {'BRILLO':>7}  DIAGNOSTICO")
print("-" * 60)

found = []
for idx in range(8):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        continue
    # leer varios frames (las camaras tardan en "arrancar")
    frame = None
    for _ in range(10):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if frame is None:
        print(f"{idx:>3}  {'si':>5}  {f'{w}x{h}':>12}  {'-':>7}  abre pero no entrega frame")
        continue
    brightness = float(np.mean(frame))
    if brightness < 5:
        diag = "NEGRA (virtual sin senal)"
    else:
        diag = ">>> CAMARA REAL (usa este indice) <<<"
        found.append(idx)
    print(f"{idx:>3}  {'si':>5}  {f'{w}x{h}':>12}  {brightness:>7.1f}  {diag}")

print("-" * 60)
if found:
    print(f"\nTu(s) camara(s) real(es) esta(n) en el indice: {found}")
    print(f"Matricula con:  python scripts/20_enroll_faces.py --capture \"Tu Nombre\" --cam {found[0]}")
else:
    print("\nNinguna camara dio imagen. Cierra Iriun/OBS/apps de camara virtual e intenta de nuevo,")
    print("o revisa permisos de camara en: Configuracion > Privacidad > Camara.")
