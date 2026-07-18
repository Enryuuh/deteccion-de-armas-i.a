"""
SADEV - Interfaz grafica
=========================
Panel unico para operar el sistema sin usar la terminal:
  - Elegir el modelo: PC (YOLOv8s, preciso) o Raspberry (YOLOv8n, ligero)
  - Elegir la camara (con boton para detectar la real)
  - Guardar caras (matricular personas)
  - Iniciar la deteccion en vivo
  - Ver / gestionar personas matriculadas

Lanza los scripts existentes como procesos aparte (cada uno abre su ventana
de camara), asi la GUI no se congela.

Uso:
    python scripts/app_gui.py
"""

import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
PYEXE = sys.executable

# Rutas de los dos modelos (se resuelven contra el repo)
MODELS = {
    "PC  (YOLOv8s - preciso)": "runs/detect/models/yolov8_weapons/weapons/yolov8s_v4_pose_negs/weights/best.pt",
    "Raspberry  (YOLOv8n - ligero)": "runs/detect/models/yolov8_weapons/weapons/yolov8n_v4_pose_negs/weights/best.pt",
}

# Paleta (coherente con la presentacion SADEV)
BG = "#12161D"; PANEL = "#1C232E"; CYAN = "#2EC4B6"; RED = "#E63946"
AMBER = "#F4A259"; LIGHT = "#F4F7FB"; MUTED = "#9AA6B6"


def load_cfg():
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def db_path():
    return ROOT / load_cfg().get("face_recognition", {}).get("db_file", "models/faces/face_db.npz")


def enrolled_names():
    p = db_path()
    if not p.exists():
        return []
    d = np.load(p, allow_pickle=True)
    return list(d["names"])


class App:
    def __init__(self, root):
        self.root = root
        root.title("SADEV - Deteccion de Armas + Reconocimiento Facial")
        root.configure(bg=BG)
        root.geometry("560x620")
        root.resizable(False, False)

        self.model_var = tk.StringVar(value=list(MODELS.keys())[0])
        self.cam_var = tk.StringVar(value="0")

        self._build()

    # ---------------------------------------------------------------- UI
    def _title(self, parent, text, color=LIGHT, size=20, bold=True):
        f = ("Segoe UI", size, "bold" if bold else "normal")
        return tk.Label(parent, text=text, bg=BG, fg=color, font=f)

    def _panel(self, parent):
        return tk.Frame(parent, bg=PANEL, bd=0, highlightthickness=1,
                        highlightbackground="#33404F")

    def _build(self):
        # Cabecera
        head = tk.Frame(self.root, bg=BG)
        head.pack(fill="x", padx=20, pady=(18, 6))
        self._title(head, "SADEV", CYAN, 30).pack(anchor="w")
        self._title(head, "Sistema de deteccion de armas y reconocimiento facial",
                    MUTED, 10, False).pack(anchor="w")

        # --- Configuracion (modelo + camara) ---
        cfgp = self._panel(self.root)
        cfgp.pack(fill="x", padx=20, pady=10)
        tk.Label(cfgp, text="1 · CONFIGURACION", bg=PANEL, fg=CYAN,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=14, pady=(12, 6))

        tk.Label(cfgp, text="Modelo de deteccion:", bg=PANEL, fg=LIGHT,
                 font=("Segoe UI", 11)).pack(anchor="w", padx=14)
        for name in MODELS:
            tk.Radiobutton(cfgp, text=name, variable=self.model_var, value=name,
                           bg=PANEL, fg=LIGHT, selectcolor=BG, activebackground=PANEL,
                           activeforeground=CYAN, font=("Segoe UI", 10),
                           highlightthickness=0).pack(anchor="w", padx=28)

        camrow = tk.Frame(cfgp, bg=PANEL)
        camrow.pack(fill="x", padx=14, pady=(8, 14))
        tk.Label(camrow, text="Camara (indice):", bg=PANEL, fg=LIGHT,
                 font=("Segoe UI", 11)).pack(side="left")
        tk.Entry(camrow, textvariable=self.cam_var, width=5, font=("Segoe UI", 11),
                 justify="center").pack(side="left", padx=8)
        tk.Button(camrow, text="Detectar camara real", command=self.detect_cameras,
                  bg="#33404F", fg=LIGHT, font=("Segoe UI", 9), relief="flat",
                  activebackground=CYAN, cursor="hand2").pack(side="left", padx=6)

        # --- Guardar caras ---
        facep = self._panel(self.root)
        facep.pack(fill="x", padx=20, pady=10)
        tk.Label(facep, text="2 · CARAS (RECONOCIMIENTO)", bg=PANEL, fg=AMBER,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=14, pady=(12, 6))
        self.enrolled_lbl = tk.Label(facep, text="", bg=PANEL, fg=MUTED,
                                     font=("Segoe UI", 10), justify="left", wraplength=500)
        self.enrolled_lbl.pack(anchor="w", padx=14)
        self._refresh_enrolled()
        brow = tk.Frame(facep, bg=PANEL); brow.pack(fill="x", padx=14, pady=(8, 14))
        tk.Button(brow, text="Guardar una cara nueva", command=self.enroll_face,
                  bg=AMBER, fg="#12161D", font=("Segoe UI", 11, "bold"), relief="flat",
                  cursor="hand2", padx=12, pady=6).pack(side="left")
        tk.Button(brow, text="Actualizar lista", command=self._refresh_enrolled,
                  bg="#33404F", fg=LIGHT, font=("Segoe UI", 9), relief="flat",
                  cursor="hand2").pack(side="left", padx=8)

        # --- Iniciar deteccion ---
        runp = self._panel(self.root)
        runp.pack(fill="x", padx=20, pady=10)
        tk.Label(runp, text="3 · USO NORMAL", bg=PANEL, fg=CYAN,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=14, pady=(12, 6))
        tk.Button(runp, text="INICIAR DETECCION EN VIVO", command=self.start_detection,
                  bg=CYAN, fg="#12161D", font=("Segoe UI", 13, "bold"), relief="flat",
                  cursor="hand2", padx=12, pady=10).pack(padx=14, pady=(0, 14), fill="x")

        # Estado
        self.status = tk.Label(self.root, text="Listo.", bg=BG, fg=MUTED,
                               font=("Segoe UI", 9), anchor="w")
        self.status.pack(fill="x", padx=22, pady=(0, 10))

    # ---------------------------------------------------------------- acciones
    def _set_status(self, txt, color=MUTED):
        self.status.config(text=txt, fg=color)

    def _refresh_enrolled(self):
        names = enrolled_names()
        if names:
            self.enrolled_lbl.config(text=f"Personas matriculadas ({len(names)}): " + ", ".join(names))
        else:
            self.enrolled_lbl.config(text="Sin personas matriculadas todavia.")

    def _weights_path(self):
        return str(ROOT / MODELS[self.model_var.get()])

    def _cam(self):
        try:
            return int(self.cam_var.get())
        except ValueError:
            return 0

    def detect_cameras(self):
        self._set_status("Detectando camaras...", AMBER)
        def _run():
            try:
                out = subprocess.run([PYEXE, str(ROOT / "scripts" / "probe_cameras.py")],
                                     capture_output=True, text=True, cwd=str(ROOT), timeout=60)
                real = [l for l in out.stdout.splitlines() if "CAMARA REAL" in l]
                if real:
                    # extraer el primer indice real
                    idx = real[0].split()[0]
                    self.cam_var.set(idx)
                    self.root.after(0, lambda: self._set_status(
                        f"Camara real detectada en indice {idx}.", CYAN))
                else:
                    self.root.after(0, lambda: self._set_status(
                        "No se hallo camara real. Cierra apps de camara virtual.", RED))
            except Exception as e:
                self.root.after(0, lambda: self._set_status(f"Error detectando: {e}", RED))
        threading.Thread(target=_run, daemon=True).start()

    def enroll_face(self):
        name = simpledialog.askstring("Guardar cara",
                                      "Nombre de la persona:", parent=self.root)
        if not name or not name.strip():
            return
        name = name.strip()
        self._set_status(f"Abriendo camara para matricular a '{name}'... "
                         "ESPACIO=foto, Q=terminar", AMBER)
        cmd = [PYEXE, str(ROOT / "scripts" / "20_enroll_faces.py"),
               "--capture", name, "--cam", str(self._cam())]

        def _run():
            subprocess.run(cmd, cwd=str(ROOT))
            self.root.after(0, self._refresh_enrolled)
            self.root.after(0, lambda: self._set_status(f"Matriculacion de '{name}' finalizada.", CYAN))
        threading.Thread(target=_run, daemon=True).start()

    def start_detection(self):
        w = self._weights_path()
        if not Path(w).exists():
            messagebox.showerror("Modelo no encontrado", f"No existe:\n{w}")
            return
        self._set_status(f"Iniciando deteccion ({self.model_var.get()})... "
                         "En la ventana de camara: Q o ESC para salir.", CYAN)
        cmd = [PYEXE, str(ROOT / "scripts" / "5_inference_camera.py"),
               "--weights", w, "--cam", str(self._cam())]
        threading.Thread(target=lambda: subprocess.run(cmd, cwd=str(ROOT)), daemon=True).start()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
