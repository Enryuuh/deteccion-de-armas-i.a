"""
Vigilante: espera a que termine la descarga de negativos y lanza el reentreno.
==============================================================================
Pensado para dejarlo corriendo mientras el usuario duerme. Cuando el proceso
de descarga (scripts/16) ya no este vivo, ejecuta el pipeline completo
(scripts/retrain_pipeline.py): entrena grande + nano, evalua y exporta.

Uso:
    python scripts/watch_and_retrain.py
"""

import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("watcher")

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def harvest_running() -> bool:
    """True si aun hay un proceso python corriendo scripts/16 (la descarga)."""
    ps = (
        "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
        "Where-Object { $_.CommandLine -like '*16_download_stand_hard_negatives*' } | "
        "Measure-Object | Select-Object -ExpandProperty Count"
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() not in ("", "0")
    except Exception as e:
        log.warning("No pude comprobar el proceso: %s (asumo terminado)", e)
        return False


def main():
    log.info("Vigilante activo. Esperando a que termine la descarga de negativos...")
    # Espera inicial para asegurar que la descarga ya arranco
    time.sleep(30)
    while harvest_running():
        time.sleep(60)
    log.info("Descarga terminada. Lanzando pipeline de reentrenamiento...")
    rc = subprocess.run([PY, "scripts/retrain_pipeline.py"], cwd=str(REPO)).returncode
    log.info("Pipeline finalizado (rc=%d).", rc)


if __name__ == "__main__":
    main()
