"""
Host a local MLflow tracking server and expose it publicly via ngrok, so that
training runs on this machine or other machines can log to a single shared
tracking server.

Setup:
    1. Sign up at https://ngrok.com and grab an authtoken.
    2. Add it to a .env file in this directory:
         NGROK_AUTHTOKEN=xxxxxxxx
    3. (Optional) reserve a static ngrok domain and set NGROK_DOMAIN=your-domain.ngrok-free.app

Usage:
    python mlflow_server.py
    python mlflow_server.py --port=5000 --backend_store_uri=sqlite:///mlflow.db

Then point any training run at the printed public URL:
    python train_detr_api.py --mlflow_tracking_uri=<public_url> ...
    # or
    export MLFLOW_TRACKING_URI=<public_url>
    python train_detr_api.py ...

All Modal integrations are archived in deprecated/ and are no longer maintained.
Use the local DETR training entry points for new runs.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import fire


def _load_dotenv():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip("'\"")


def main(
    port: int = 5000,
    backend_store_uri: str = "sqlite:///mlflow.db",
    artifact_root: str = "./mlruns",
    domain: str = None,
):
    """Start `mlflow server` locally and tunnel it through ngrok.

    Args:
        port: Local port for the MLflow server.
        backend_store_uri: Where MLflow stores run/experiment metadata.
        artifact_root: Where MLflow stores logged artifacts (checkpoints, images).
        domain: Optional reserved ngrok static domain (e.g. 'myname.ngrok-free.app').
                Falls back to NGROK_DOMAIN env var, then a random ngrok URL.
    """
    _load_dotenv()

    from pyngrok import ngrok

    authtoken = os.environ.get("NGROK_AUTHTOKEN")
    if not authtoken:
        print("ERROR: NGROK_AUTHTOKEN not set. Add it to .env or export it.", flush=True)
        sys.exit(1)
    ngrok.set_auth_token(authtoken)

    domain = domain or os.environ.get("NGROK_DOMAIN") or None

    # Open the ngrok tunnel first so we know the public hostname to whitelist
    # for MLflow's Host-header validation (added in mlflow's server security
    # middleware — otherwise requests via the tunnel are rejected as a possible
    # DNS-rebinding attack).
    tunnel = ngrok.connect(port, "http", domain=domain) if domain else ngrok.connect(port, "http")
    public_url = tunnel.public_url
    public_host = public_url.split("://", 1)[-1]

    allowed_hosts = [
        "127.0.0.1", "127.0.0.1:*", "localhost", "localhost:*",
        "*.ngrok-free.app", "*.ngrok.app", "*.ngrok.io", "*.ngrok.dev",
        public_host,
    ]
    allowed_origins = [
        "http://127.0.0.1", f"http://127.0.0.1:{port}",
        "http://localhost", f"http://localhost:{port}",
        public_url,
    ]

    print(f"Starting MLflow server on 127.0.0.1:{port} "
          f"(backend={backend_store_uri}, artifacts={artifact_root})...", flush=True)
    env = {
        **os.environ,
        "MLFLOW_SERVER_ALLOWED_HOSTS": ",".join(allowed_hosts),
        "MLFLOW_SERVER_CORS_ALLOWED_ORIGINS": ",".join(allowed_origins),
    }
    server_proc = subprocess.Popen([
        "mlflow", "server",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--backend-store-uri", backend_store_uri,
        "--artifacts-destination", artifact_root,
    ], env=env)

    # Give the server a moment to bind before it starts receiving tunneled traffic
    time.sleep(3)

    print("=" * 70, flush=True)
    print(f"MLflow tracking server public URL: {public_url}", flush=True)
    print(f"MLflow UI:                         {public_url}", flush=True)
    print("Point training runs at this URL via --mlflow_tracking_uri="
          f"{public_url}  (or export MLFLOW_TRACKING_URI={public_url})", flush=True)
    print("=" * 70, flush=True)

    def _shutdown(*_):
        print("\nShutting down ngrok tunnel and MLflow server...", flush=True)
        ngrok.disconnect(public_url)
        ngrok.kill()
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Block until the mlflow server process exits (or we're interrupted above)
    server_proc.wait()


if __name__ == "__main__":
    fire.Fire(main)
