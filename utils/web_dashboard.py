"""
utils/web_dashboard.py
=======================
Dashboard web minimalista para monitoreo remoto desde el celular.
Stream MJPEG + estado de alertas en tiempo real.

Acceder desde LAN: http://<ip-del-pc>:8080

Corre en un hilo separado para no bloquear la inferencia.
"""

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Frame compartido entre inferencia y servidor web
_shared_state = {
    "frame": None,
    "fps": 0.0,
    "weapon_detected": False,
    "class_names": [],
    "last_alert_time": None,
    "total_alerts": 0,
    "uptime_start": time.time(),
    "lock": threading.Lock(),
}


def update_dashboard(frame: np.ndarray, fps: float, weapon_detected: bool,
                     class_names: list = None):
    """Llamar desde el loop de inferencia para actualizar el dashboard."""
    with _shared_state["lock"]:
        _shared_state["frame"] = frame.copy()
        _shared_state["fps"] = fps
        _shared_state["weapon_detected"] = weapon_detected
        _shared_state["class_names"] = class_names or []
        if weapon_detected:
            _shared_state["last_alert_time"] = datetime.now().isoformat()
            _shared_state["total_alerts"] += 1


def _generate_mjpeg(fps_limit: int = 15):
    """Generador MJPEG para streaming."""
    interval = 1.0 / max(fps_limit, 1)
    while True:
        time.sleep(interval)
        with _shared_state["lock"]:
            frame = _shared_state["frame"]
        if frame is None:
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Deteccion de Armas - Monitor</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0a0a0f;
    color: #e5e5e5;
    min-height: 100vh;
  }
  .header {
    background: #111118;
    padding: 12px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #222;
  }
  .header h1 { font-size: 18px; font-weight: 600; }
  .status-dot {
    width: 10px; height: 10px; border-radius: 50%;
    display: inline-block; margin-right: 8px;
    animation: pulse 2s infinite;
  }
  .status-dot.safe { background: #5ac86a; }
  .status-dot.alert { background: #e63946; }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
  .stream-container {
    display: flex;
    justify-content: center;
    padding: 10px;
  }
  .stream-container img {
    max-width: 100%;
    border-radius: 8px;
    border: 2px solid #222;
  }
  .stream-container img.alert-border {
    border-color: #e63946;
    box-shadow: 0 0 20px rgba(230, 57, 70, 0.3);
  }
  .stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 10px;
    padding: 10px 20px;
  }
  .stat-card {
    background: #111118;
    border-radius: 8px;
    padding: 14px;
    text-align: center;
    border: 1px solid #222;
  }
  .stat-card .value {
    font-size: 24px;
    font-weight: 700;
    color: #fff;
  }
  .stat-card .label {
    font-size: 12px;
    color: #888;
    margin-top: 4px;
  }
  .stat-card.alert-card { border-color: #e63946; }
  .stat-card.alert-card .value { color: #e63946; }
  .alert-banner {
    display: none;
    background: #e63946;
    color: white;
    text-align: center;
    padding: 10px;
    font-weight: 700;
    font-size: 16px;
    animation: flash 0.5s infinite alternate;
  }
  .alert-banner.active { display: block; }
  @keyframes flash {
    from { opacity: 1; }
    to { opacity: 0.7; }
  }
  .log-section {
    padding: 10px 20px;
  }
  .log-section h3 {
    font-size: 14px;
    color: #888;
    margin-bottom: 8px;
  }
  #log-entries {
    background: #111118;
    border-radius: 8px;
    padding: 10px;
    max-height: 200px;
    overflow-y: auto;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    border: 1px solid #222;
  }
  .log-entry { padding: 3px 0; border-bottom: 1px solid #1a1a1a; }
  .log-entry .time { color: #666; }
  .log-entry .cls { color: #e63946; font-weight: 600; }
</style>
</head>
<body>
  <div class="header">
    <h1><span class="status-dot safe" id="statusDot"></span> Deteccion de Armas</h1>
    <span id="statusText" style="font-size:13px; color:#5ac86a;">SEGURO</span>
  </div>

  <div class="alert-banner" id="alertBanner">ARMA DETECTADA</div>

  <div class="stream-container">
    <img src="/stream" id="streamImg" alt="Live Stream">
  </div>

  <div class="stats">
    <div class="stat-card">
      <div class="value" id="fpsValue">--</div>
      <div class="label">FPS</div>
    </div>
    <div class="stat-card" id="alertCard">
      <div class="value" id="alertCount">0</div>
      <div class="label">Alertas totales</div>
    </div>
    <div class="stat-card">
      <div class="value" id="uptimeValue">--</div>
      <div class="label">Uptime</div>
    </div>
    <div class="stat-card">
      <div class="value" id="lastAlert">--</div>
      <div class="label">Ultima alerta</div>
    </div>
  </div>

  <div class="log-section">
    <h3>Log de alertas</h3>
    <div id="log-entries"><div class="log-entry" style="color:#555">Sin alertas aun...</div></div>
  </div>

<script>
  const statusDot = document.getElementById('statusDot');
  const statusText = document.getElementById('statusText');
  const alertBanner = document.getElementById('alertBanner');
  const streamImg = document.getElementById('streamImg');
  const fpsValue = document.getElementById('fpsValue');
  const alertCount = document.getElementById('alertCount');
  const uptimeValue = document.getElementById('uptimeValue');
  const lastAlert = document.getElementById('lastAlert');
  const alertCard = document.getElementById('alertCard');
  const logEntries = document.getElementById('log-entries');
  let firstLog = true;

  async function pollStatus() {
    try {
      const r = await fetch('/api/status');
      const d = await r.json();

      fpsValue.textContent = d.fps.toFixed(1);
      alertCount.textContent = d.total_alerts;

      const secs = Math.floor(d.uptime);
      const h = Math.floor(secs / 3600);
      const m = Math.floor((secs % 3600) / 60);
      const s = secs % 60;
      uptimeValue.textContent = `${h}h ${m}m ${s}s`;

      if (d.weapon_detected) {
        statusDot.className = 'status-dot alert';
        statusText.textContent = 'ALERTA';
        statusText.style.color = '#e63946';
        alertBanner.classList.add('active');
        streamImg.classList.add('alert-border');
        alertCard.classList.add('alert-card');
      } else {
        statusDot.className = 'status-dot safe';
        statusText.textContent = 'SEGURO';
        statusText.style.color = '#5ac86a';
        alertBanner.classList.remove('active');
        streamImg.classList.remove('alert-border');
        alertCard.classList.remove('alert-card');
      }

      if (d.last_alert_time) {
        const t = new Date(d.last_alert_time);
        lastAlert.textContent = t.toLocaleTimeString();
      }

      if (d.weapon_detected && d.class_names && d.class_names.length > 0) {
        if (firstLog) { logEntries.innerHTML = ''; firstLog = false; }
        const now = new Date().toLocaleTimeString();
        const cls = d.class_names.join(', ');
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.innerHTML = `<span class="time">${now}</span> <span class="cls">${cls}</span>`;
        logEntries.prepend(entry);
        if (logEntries.children.length > 50) logEntries.lastChild.remove();
      }
    } catch (e) {}
  }

  setInterval(pollStatus, 500);
</script>
</body>
</html>"""


def start_dashboard(port: int = 8080, fps_limit: int = 15):
    """Arranca el servidor Flask en un hilo daemon."""
    try:
        from flask import Flask, Response, jsonify
    except ImportError:
        logger.warning("Flask no instalado. Dashboard desactivado. pip install flask")
        return

    app = Flask(__name__)
    app.logger.setLevel(logging.WARNING)

    # Silenciar logs de werkzeug
    wlog = logging.getLogger("werkzeug")
    wlog.setLevel(logging.WARNING)

    @app.route("/")
    def index():
        return _HTML_TEMPLATE

    @app.route("/stream")
    def stream():
        return Response(_generate_mjpeg(fps_limit),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/status")
    def api_status():
        with _shared_state["lock"]:
            return jsonify({
                "fps": round(_shared_state["fps"], 1),
                "weapon_detected": _shared_state["weapon_detected"],
                "class_names": _shared_state["class_names"],
                "last_alert_time": _shared_state["last_alert_time"],
                "total_alerts": _shared_state["total_alerts"],
                "uptime": time.time() - _shared_state["uptime_start"],
            })

    def _run():
        logger.info(f"Dashboard web: http://0.0.0.0:{port}")
        app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t
