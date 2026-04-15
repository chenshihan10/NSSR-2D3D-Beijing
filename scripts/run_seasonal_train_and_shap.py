from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(r"F:\project2025\wulifanyan")
XGB_DIR = ROOT / "XGBoost"
SHAP_DIR = ROOT / "shap"
OUT_DIR = XGB_DIR / "output2" / "monitor"
TRAIN_SCRIPT = XGB_DIR / "train_samples_seasonal_output2.py"
SHAP_SCRIPT = SHAP_DIR / "run_shap_seasonal_analysis.py"
TRAIN_LOG = OUT_DIR / "train.log"
SHAP_LOG = OUT_DIR / "shap.log"
STATE_JSON = OUT_DIR / "run_state.json"
LATEST_METRICS = XGB_DIR / "output2" / "model" / "metrics" / "seasonal_model_metrics_output2_latest.csv"
SHAP_SEASON_SUMMARY = SHAP_DIR / "metrics" / "shap_season_summary.csv"


@dataclass
class RuntimeState:
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    status: str = "idle"
    stage: str = "idle"
    message: str = "Not started"
    train_progress: dict[str, Any] = field(default_factory=lambda: {"season": 0, "season_total": 4, "fold": 0, "fold_total": 5})
    shap_progress: dict[str, Any] = field(default_factory=lambda: {"season": 0, "season_total": 4})
    return_codes: dict[str, int | None] = field(default_factory=lambda: {"train": None, "shap": None})
    active_pid: int | None = None
    stop_requested: bool = False

    def touch(self, message: str | None = None) -> None:
        self.updated_at = datetime.now().isoformat(timespec="seconds")
        if message is not None:
            self.message = message


STATE = RuntimeState()
LOCK = threading.Lock()


def save_state() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK:
        payload = {
            "started_at": STATE.started_at,
            "updated_at": STATE.updated_at,
            "status": STATE.status,
            "stage": STATE.stage,
            "message": STATE.message,
            "train_progress": STATE.train_progress,
            "shap_progress": STATE.shap_progress,
            "return_codes": STATE.return_codes,
            "active_pid": STATE.active_pid,
            "stop_requested": STATE.stop_requested,
        }
    STATE_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def tail_lines(path: Path, n: int = 120) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return lines[-n:]


def load_csv_rows(path: Path, max_rows: int = 20) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        import pandas as pd  # lazy import

        df = pd.read_csv(path)
        if len(df) > max_rows:
            df = df.head(max_rows)
        return df.to_dict(orient="records")
    except Exception:
        return []


def parse_train_rmse_series(path: Path, max_points: int = 800) -> dict[str, Any]:
    """
    Extract the most recent block of 'validation_0-rmse' values from the train log.
    Returns iteration + rmse series for quick plotting in the dashboard.
    """
    if not path.exists():
        return {"iters": [], "rmse": [], "meta": {}}
    text = path.read_text(encoding="utf-8", errors="ignore")
    # Keep only the last chunk to avoid scanning a huge file every refresh.
    tail = text[-600_000:] if len(text) > 600_000 else text

    season = None
    fold = None
    m_season = re.findall(r"Processing season:\s*(\w+)", tail)
    if m_season:
        season = m_season[-1]
    m_fold = re.findall(r"Fold\s+(\d+)/(\d+)", tail)
    if m_fold:
        fold = "/".join(m_fold[-1])

    pts = re.findall(r"^\[(\d+)\]\s*validation_0-rmse:([0-9.]+)\s*$", tail, flags=re.MULTILINE)
    if len(pts) > max_points:
        pts = pts[-max_points:]
    iters = [int(a) for a, _ in pts]
    rmses = [float(b) for _, b in pts]
    return {"iters": iters, "rmse": rmses, "meta": {"season": season, "fold": fold}}

def stop_process_tree(pid: int) -> None:
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        pass


def update_train_progress(line: str) -> None:
    season_match = re.search(r"\[(\d+)/(\d+)\]", line)
    fold_match = re.search(r"Fold\s+(\d+)/(\d+)", line, flags=re.IGNORECASE)
    with LOCK:
        if season_match:
            STATE.train_progress["season"] = int(season_match.group(1))
            STATE.train_progress["season_total"] = int(season_match.group(2))
        if fold_match:
            STATE.train_progress["fold"] = int(fold_match.group(1))
            STATE.train_progress["fold_total"] = int(fold_match.group(2))
        STATE.touch()


def update_shap_progress(line: str) -> None:
    season_match = re.search(r"\[(\d+)/(\d+)\]", line)
    with LOCK:
        if season_match:
            STATE.shap_progress["season"] = int(season_match.group(1))
            STATE.shap_progress["season_total"] = int(season_match.group(2))
        STATE.touch()


def run_script(script: Path, log_path: Path, stage: str, use_gpu: bool = False) -> int:
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")

    cmd = ["python", str(script)]
    if use_gpu:
        cmd.append("--use-gpu")

    with LOCK:
        STATE.stage = stage
        STATE.status = "running"
        STATE.touch(f"Running {stage} ...")
    save_state()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write(f"[{datetime.now().isoformat(timespec='seconds')}] START {stage}\n")
        lf.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(script.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        with LOCK:
            STATE.active_pid = proc.pid
            STATE.touch(f"{stage} PID={proc.pid}")
        save_state()

        assert proc.stdout is not None
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            print(f"[{stage}] {line}", end="")
            if stage == "train":
                update_train_progress(line)
            else:
                update_shap_progress(line)
            save_state()

            with LOCK:
                if STATE.stop_requested and proc.poll() is None:
                    stop_process_tree(proc.pid)
                    STATE.touch(f"Stop requested. Terminating {stage}.")
                    save_state()
                    break

        rc = proc.wait()
        lf.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] END {stage} rc={rc}\n")
        lf.flush()

    with LOCK:
        STATE.active_pid = None
        STATE.return_codes[stage] = rc
        if STATE.stop_requested:
            STATE.status = "stopped"
            STATE.touch(f"{stage} stopped by user")
        elif rc == 0:
            STATE.touch(f"{stage} finished successfully")
        else:
            STATE.status = "failed"
            STATE.touch(f"{stage} failed with code {rc}")
    save_state()
    return rc


def html_page() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Seasonal XGBoost + SHAP Monitor</title>
  <style>
    body { font-family: Segoe UI, Arial, sans-serif; margin: 16px; background: #f7f7f9; color: #111; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .card { background: #fff; border: 1px solid #ddd; border-radius: 10px; padding: 12px; }
    .head { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
    .k { color: #555; }
    pre { background: #0f172a; color: #dbeafe; padding: 10px; border-radius: 8px; overflow: auto; max-height: 360px; }
    button { padding: 8px 12px; border: none; border-radius: 8px; background: #b91c1c; color: white; cursor: pointer; }
    .small { color: #666; font-size: 12px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border-bottom: 1px solid #eee; padding: 6px 8px; text-align: left; font-size: 13px; }
    th { background: #f3f4f6; }
    canvas { width: 100%; height: 220px; border: 1px solid #eee; border-radius: 8px; }
  </style>
</head>
<body>
  <div class="head">
    <h2>Seasonal XGBoost + SHAP Monitor</h2>
    <div>
      <button onclick="stopRun()">Stop</button>
    </div>
  </div>
  <div class="card">
    <div><span class="k">Status:</span> <b id="status">-</b></div>
    <div><span class="k">Stage:</span> <b id="stage">-</b></div>
    <div><span class="k">Message:</span> <span id="msg">-</span></div>
    <div><span class="k">Updated:</span> <span id="updated">-</span></div>
    <div id="p1"></div>
    <div id="p2"></div>
  </div>
  <div class="grid">
    <div class="card">
      <h3>Season Metrics (CV)</h3>
      <table id="metrics"></table>
    </div>
    <div class="card">
      <h3>RMSE Curve (Latest)</h3>
      <div class="small" id="curve_meta"></div>
      <canvas id="curve" width="900" height="260"></canvas>
    </div>
  </div>
  <div class="grid">
    <div class="card"><h3>Train Log</h3><pre id="trainlog"></pre></div>
    <div class="card"><h3>SHAP Log</h3><pre id="shaplog"></pre></div>
  </div>
  <p class="small">Auto refresh every 5 seconds.</p>
  <script>
    async function loadJson(url) {
      const r = await fetch(url, { cache: 'no-store' });
      return await r.json();
    }
    function renderMetrics(rows) {
      const t = document.getElementById('metrics');
      if (!rows || rows.length === 0) { t.innerHTML = '<tr><td class=\"small\">No metrics yet</td></tr>'; return; }
      const head = '<tr><th>Season</th><th>R2</th><th>MAE</th><th>RMSE</th><th>Params</th></tr>';
      const body = rows.map(r => {
        const params = `lr=${r.best_learning_rate}, depth=${r.best_max_depth}, mcw=${r.best_min_child_weight}`;
        return `<tr><td>${r.Season}</td><td>${Number(r.R2_CV_Mean).toFixed(3)}</td><td>${Number(r.MAE_CV_Mean).toFixed(2)}</td><td>${Number(r.RMSE_CV_Mean).toFixed(2)}</td><td>${params}</td></tr>`;
      }).join('');
      t.innerHTML = head + body;
    }

    function drawCurve(iters, rmses) {
      const c = document.getElementById('curve');
      const ctx = c.getContext('2d');
      ctx.clearRect(0,0,c.width,c.height);
      if (!iters || iters.length < 2) {
        ctx.fillStyle = '#666';
        ctx.fillText('No curve yet', 20, 30);
        return;
      }
      const pad = 40;
      const w = c.width - pad*2;
      const h = c.height - pad*2;
      const xMin = iters[0], xMax = iters[iters.length-1];
      let yMin = Math.min(...rmses), yMax = Math.max(...rmses);
      if (yMax === yMin) { yMax = yMin + 1; }

      const x = (v)=> pad + (v - xMin) * (w / (xMax - xMin || 1));
      const y = (v)=> pad + (yMax - v) * (h / (yMax - yMin));

      ctx.strokeStyle = '#e5e7eb';
      ctx.lineWidth = 1;
      ctx.strokeRect(pad, pad, w, h);

      ctx.strokeStyle = '#2563eb';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x(iters[0]), y(rmses[0]));
      for (let i=1;i<iters.length;i++) ctx.lineTo(x(iters[i]), y(rmses[i]));
      ctx.stroke();

      ctx.fillStyle = '#111';
      ctx.font = '12px Segoe UI, Arial';
      ctx.fillText(`rmse: ${yMin.toFixed(2)} - ${yMax.toFixed(2)}`, pad, pad-10);
      ctx.fillText(`iter: ${xMin} - ${xMax}`, pad + 200, pad-10);
    }

    async function refresh() {
      try {
        const s = await loadJson('/api/status');
        document.getElementById('status').textContent = s.status;
        document.getElementById('stage').textContent = s.stage;
        document.getElementById('msg').textContent = s.message;
        document.getElementById('updated').textContent = s.updated_at;
        document.getElementById('p1').textContent =
          `Train progress: season ${s.train_progress.season}/${s.train_progress.season_total}, fold ${s.train_progress.fold}/${s.train_progress.fold_total}`;
        document.getElementById('p2').textContent =
          `SHAP progress: season ${s.shap_progress.season}/${s.shap_progress.season_total}`;

        const t = await loadJson('/api/log/train');
        const h = await loadJson('/api/log/shap');
        document.getElementById('trainlog').textContent = (t.lines || []).join('\\n');
        document.getElementById('shaplog').textContent = (h.lines || []).join('\\n');

        const m = await loadJson('/api/metrics');
        renderMetrics(m.rows || []);

        const c = await loadJson('/api/train_curve');
        document.getElementById('curve_meta').textContent = `season=${c.meta?.season || '-'} fold=${c.meta?.fold || '-'}`;
        drawCurve(c.iters || [], c.rmse || []);
      } catch (e) {}
    }
    async function stopRun() {
      await fetch('/api/control/stop', { method: 'POST' });
      await refresh();
    }
    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>"""


class MonitorHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path in ["/", "/index.html"]:
            body = html_page().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/status":
            with LOCK:
                payload = {
                    "started_at": STATE.started_at,
                    "updated_at": STATE.updated_at,
                    "status": STATE.status,
                    "stage": STATE.stage,
                    "message": STATE.message,
                    "train_progress": STATE.train_progress,
                    "shap_progress": STATE.shap_progress,
                    "return_codes": STATE.return_codes,
                    "active_pid": STATE.active_pid,
                    "stop_requested": STATE.stop_requested,
                }
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/log/train":
            body = json.dumps({"lines": tail_lines(TRAIN_LOG)}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/log/shap":
            body = json.dumps({"lines": tail_lines(SHAP_LOG)}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/metrics":
            rows = load_csv_rows(LATEST_METRICS, max_rows=12)
            body = json.dumps({"rows": rows, "path": str(LATEST_METRICS)}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/train_curve":
            body = json.dumps(parse_train_rmse_series(TRAIN_LOG), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/control/stop":
            with LOCK:
                STATE.stop_requested = True
                pid = STATE.active_pid
                STATE.touch("Stop requested")
            if pid:
                stop_process_tree(pid)
            save_state()

            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def start_server(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), MonitorHandler)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    return server


def run_pipeline(use_gpu: bool) -> None:
    rc_train = run_script(TRAIN_SCRIPT, TRAIN_LOG, "train", use_gpu=use_gpu)
    if rc_train != 0:
        return
    with LOCK:
        if STATE.stop_requested:
            return
    run_script(SHAP_SCRIPT, SHAP_LOG, "shap", use_gpu=use_gpu)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run seasonal training + SHAP with live dashboard monitor.")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="Do not auto-open monitor page.")
    p.add_argument("--use-gpu", action="store_true", help="Pass --use-gpu to train and SHAP scripts.")
    p.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Heartbeat seconds printed in terminal while waiting (default: 60).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_state()

    server = start_server(args.port)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Monitor URL: {url}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)

    worker = threading.Thread(target=run_pipeline, kwargs={"use_gpu": args.use_gpu}, daemon=True)
    worker.start()

    while worker.is_alive():
        time.sleep(max(5, args.interval))
        with LOCK:
            msg = f"[heartbeat] status={STATE.status}, stage={STATE.stage}, message={STATE.message}"
        print(msg, flush=True)
        save_state()

    worker.join()
    with LOCK:
        if not STATE.stop_requested and STATE.status != "failed":
            STATE.status = "completed"
            STATE.stage = "done"
            STATE.touch("All tasks finished")
    save_state()
    print("Pipeline finished. Monitor stays available. Press Ctrl+C to exit.", flush=True)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        with LOCK:
            pid = STATE.active_pid
            STATE.stop_requested = True
            STATE.touch("KeyboardInterrupt: stopping")
        if pid:
            stop_process_tree(pid)
        save_state()
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
