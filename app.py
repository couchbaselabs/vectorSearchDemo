import sys
import platform
import subprocess
import shlex
from pathlib import Path
from flask import Flask, render_template, request, Response, jsonify
from dotenv import load_dotenv

import config
import preflight
import dataplane

app = Flask(__name__)
BASE_DIR = Path(__file__).parent
IS_WINDOWS = platform.system() == "Windows"
ENV_PATH = str(BASE_DIR / ".env")


@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Configuration drawer: read/write .env and test connectivity without leaving
# the browser, so users don't have to hand-edit .env. Localhost demo only.
# ---------------------------------------------------------------------------

@app.route("/config", methods=["GET"])
def get_config():
    """Return the current .env values for the drawer, with secrets masked."""
    env = config.read_env_file(ENV_PATH)
    return jsonify(config.form_from_env(env))


@app.route("/config", methods=["POST"])
def save_config():
    """Write drawer values back to .env. Masked/omitted secrets keep their stored value."""
    data = request.json or {}
    existing = config.read_env_file(ENV_PATH)
    updates = config.env_updates_from_form(data, existing)
    config.write_env_file(updates, ENV_PATH)
    # Refresh this process's env so a subsequent /test-connection sees the new values;
    # scripts launched via /run read .env themselves.
    load_dotenv(ENV_PATH, override=True)
    return jsonify({"ok": True})


@app.route("/test-connection", methods=["POST"])
def test_connection():
    """Run the readiness preflight against the posted (unsaved) form values, or the saved .env
    if none were posted. Returns the structured check results as JSON."""
    data = request.json or {}
    if data:
        env = config.env_updates_from_form(data, config.read_env_file(ENV_PATH))
        settings = config.Settings.from_env_dict(env)
    else:
        load_dotenv(ENV_PATH, override=True)
        settings = config.Settings.load()
    results = preflight.run_checks(settings)
    return jsonify({"results": results, "ok": preflight.all_ok(results)})


# ---------------------------------------------------------------------------
# AI Data Plane (phase 2): live status for the diagram overlay + global toggle.
# ---------------------------------------------------------------------------

@app.route("/dataplane/status", methods=["GET"])
def dataplane_status():
    """Live status/usage for the diagram overlay (memory active/used/tokens-served + toggle)."""
    load_dotenv(ENV_PATH, override=True)
    return jsonify(dataplane.status(config.Settings.load()))


@app.route("/dataplane/toggle", methods=["POST"])
def dataplane_toggle():
    """Enable/disable the AI Data Plane for RAG. Persists AI_DATAPLANE_ENABLED to .env so the
    RAG scripts (run as subprocesses) pick it up."""
    data = request.json or {}
    enabled = bool(data.get("enabled"))
    config.write_env_file({"AI_DATAPLANE_ENABLED": "true" if enabled else "false"}, ENV_PATH)
    load_dotenv(ENV_PATH, override=True)
    return jsonify({"ok": True, "enabled": enabled})


@app.route("/run", methods=["POST"])
def run_cmd():
    cmd = request.json.get("cmd", "")
    if not cmd:
        return Response("No command provided\n", mimetype="text/plain")

    def generate():
        process = None
        try:
            cmd_args = shlex.split(cmd)

            # If user typed "python ...", force the venv interpreter
            if cmd_args and cmd_args[0].lower() == "python":
                cmd_args[0] = sys.executable

            # Linux-only: force line buffering
            if not IS_WINDOWS:
                cmd_args = ["stdbuf", "-oL"] + cmd_args

            yield f"$ {cmd}\n\n"

            process = subprocess.Popen(
                cmd_args,
                cwd=BASE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            for line in process.stdout:
                yield line

            process.wait()
            yield "\n=== finished ===\n"

        except GeneratorExit:
            # client disconnected
            if process:
                process.kill()
            raise
        except Exception as e:
            yield f"\nERROR: {e}\n"

    return Response(generate(), mimetype="text/plain")


if __name__ == "__main__":
    # debug=False disables the auto-reloader (which would otherwise drop long-running requests
    # like /test-connection mid-flight → "Failed to fetch" in the browser) and the Werkzeug
    # debugger. threaded=True lets the config/test endpoints run alongside a streaming /run.
    app.run(debug=False, threaded=True, port=8080)