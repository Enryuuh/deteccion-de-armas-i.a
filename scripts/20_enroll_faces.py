"""
Script 20: Matricular caras (enrollment)
=========================================
Registra a las personas que el sistema debe RECONOCER. Calcula el embedding
(vector 512-D) de cada persona y lo guarda en models/faces/face_db.npz.

No entrena nada: solo extrae y promedia embeddings. Segundos por persona.

Dos formas de matricular:

  A) Desde carpetas de fotos (recomendado):
       faces_db/
         Juan Perez/   foto1.jpg foto2.jpg foto3.jpg
         Ana Gomez/    a1.jpg a2.jpg
     Ejecuta:
       python scripts/20_enroll_faces.py --from-folder faces_db

  B) Capturando en vivo con la webcam:
       python scripts/20_enroll_faces.py --capture "Juan Perez"
     (pulsa ESPACIO para tomar cada foto, Q para terminar; 3-5 tomas)

Consejos: 3-5 fotos por persona, buena luz, cara de frente y algun angulo.
Matricula SOLO a personas que dieron su consentimiento (datos biometricos).
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.face_recognition import FaceRecognizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_cfg(path="config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_recognizer(cfg: dict) -> FaceRecognizer:
    """Fuerza el modelo cargado aunque face_recognition.enabled este en False."""
    fcfg = cfg.setdefault("face_recognition", {})
    fcfg["enabled"] = True
    fr = FaceRecognizer(cfg)
    return fr


def load_existing_db(path: Path):
    if path.exists():
        data = np.load(path, allow_pickle=True)
        return list(data["names"]), list(data["embeds"].astype(np.float32))
    return [], []


def save_db(path: Path, names: list, embeds: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, names=np.array(names, dtype=object),
             embeds=np.array(embeds, dtype=np.float32))
    logger.info("Base guardada: %s  (%d personas: %s)", path, len(names), ", ".join(names))


def upsert(names: list, embeds: list, name: str, emb: np.ndarray):
    """Inserta o reemplaza el embedding de una persona por su nombre."""
    if name in names:
        embeds[names.index(name)] = emb
        logger.info("Actualizada: %s", name)
    else:
        names.append(name)
        embeds.append(emb)
        logger.info("Matriculada: %s", name)


def mean_embedding(fr: FaceRecognizer, images: list[np.ndarray], who: str):
    embs = []
    for img in images:
        e = fr.embed(img)
        if e is not None:
            embs.append(e)
    if not embs:
        logger.warning("  %s: no se detecto ninguna cara en las fotos, se omite.", who)
        return None
    mean = np.mean(np.stack(embs), axis=0)
    mean = mean / (np.linalg.norm(mean) + 1e-9)   # renormalizar
    logger.info("  %s: %d/%d fotos con cara valida", who, len(embs), len(images))
    return mean.astype(np.float32)


def enroll_from_folder(fr: FaceRecognizer, root: Path, db_path: Path):
    if not root.exists():
        logger.error("No existe la carpeta %s", root)
        return
    names, embeds = load_existing_db(db_path)
    people = [d for d in sorted(root.iterdir()) if d.is_dir()]
    if not people:
        logger.error("No hay subcarpetas de personas en %s", root)
        return
    for person_dir in people:
        imgs = [cv2.imread(str(p)) for p in sorted(person_dir.iterdir())
                if p.suffix.lower() in IMG_EXT]
        imgs = [i for i in imgs if i is not None]
        if not imgs:
            logger.warning("  %s: sin imagenes validas, se omite.", person_dir.name)
            continue
        mean = mean_embedding(fr, imgs, person_dir.name)
        if mean is not None:
            upsert(names, embeds, person_dir.name, mean)
    save_db(db_path, names, embeds)


def enroll_from_webcam(fr: FaceRecognizer, name: str, db_path: Path, cam_index: int = 0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        logger.error("No se pudo abrir la camara %d", cam_index)
        return
    shots: list[np.ndarray] = []
    logger.info("Capturando a '%s'. ESPACIO=tomar foto, Q=terminar. Toma 3-5.", name)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        preview = frame.copy()
        cv2.putText(preview, f"{name}  fotos: {len(shots)}  [ESPACIO] tomar  [Q] fin",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Matricular cara", preview)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            if fr.embed(frame) is not None:
                shots.append(frame.copy())
                logger.info("  foto %d capturada", len(shots))
            else:
                logger.warning("  no se detecto cara, reintenta")
        elif k in (ord("q"), ord("Q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()
    if not shots:
        logger.error("No se capturo ninguna foto.")
        return
    # Guardar las capturas en faces_db/<nombre>/ (registro visual)
    shots_dir = Path("faces_db") / name
    shots_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(shots, 1):
        cv2.imwrite(str(shots_dir / f"captura_{i:02d}.jpg"), img)
    logger.info("Fotos guardadas en %s", shots_dir)
    names, embeds = load_existing_db(db_path)
    mean = mean_embedding(fr, shots, name)
    if mean is not None:
        upsert(names, embeds, name, mean)
        save_db(db_path, names, embeds)


def main():
    ap = argparse.ArgumentParser(description="Matricular caras para reconocimiento")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--from-folder", metavar="DIR", help="carpeta con subcarpetas por persona")
    g.add_argument("--capture", metavar="NOMBRE", help="capturar en vivo con la webcam")
    g.add_argument("--list", action="store_true", help="listar personas matriculadas")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--cam", type=int, default=0, help="indice de camara para --capture")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    db_path = Path(cfg.get("face_recognition", {}).get("db_file", "models/faces/face_db.npz"))

    if args.list:
        names, _ = load_existing_db(db_path)
        if names:
            logger.info("Matriculadas (%d): %s", len(names), ", ".join(names))
        else:
            logger.info("No hay personas matriculadas todavia (%s).", db_path)
        return

    fr = build_recognizer(cfg)
    if args.from_folder:
        enroll_from_folder(fr, Path(args.from_folder), db_path)
    else:
        enroll_from_webcam(fr, args.capture, db_path, args.cam)


if __name__ == "__main__":
    main()
