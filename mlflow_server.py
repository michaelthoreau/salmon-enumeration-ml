"""
Host a loopback-only MLflow tracking server, so that
training runs on this machine or other machines can log to a single shared
tracking server.

Setup:
    Set MLFLOW_TRACKING_URI=http://127.0.0.1:5000 in .env.

Usage:
    python mlflow_server.py
    python mlflow_server.py --port=5000 --backend_store_uri=sqlite:///mlflow.db

Then point any local training run at the loopback URL:
    python train_detr_api.py --mlflow_tracking_uri=http://127.0.0.1:5000 ...
    # or
    export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
    python train_detr_api.py ...

All Modal integrations are archived in deprecated/ and are no longer maintained.
Use the local DETR training entry points for new runs.
"""

import os
import signal
import subprocess
import sys

import fire


def main(
    port: int = 5000,
    backend_store_uri: str = "sqlite:///mlflow.db",
    artifact_root: str = "./mlruns",
):
    """Start a loopback-only MLflow server without a public tunnel.

    Args:
        port: Local port for the MLflow server.
        backend_store_uri: Where MLflow stores run/experiment metadata.
        artifact_root: Where MLflow stores logged artifacts (checkpoints, images).
    """
    tracking_uri = f"http://127.0.0.1:{port}"

    allowed_hosts = [
        "127.0.0.1", "127.0.0.1:*", "localhost", "localhost:*",
    ]
    allowed_origins = [
        "http://127.0.0.1", f"http://127.0.0.1:{port}",
        "http://localhost", f"http://localhost:{port}",
    ]

    print(f"Starting MLflow server on 127.0.0.1:{port} "
          f"(backend={backend_store_uri}, artifacts={artifact_root})...", flush=True)
    env = {
        **os.environ,
        "MLFLOW_SERVER_ALLOWED_HOSTS": ",".join(allowed_hosts),
        "MLFLOW_SERVER_CORS_ALLOWED_ORIGINS": ",".join(allowed_origins),
    }
    server_proc = subprocess.Popen([
        sys.executable, "-m", "mlflow", "server",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--backend-store-uri", backend_store_uri,
        "--artifacts-destination", artifact_root,
    ], env=env)

    print("=" * 70, flush=True)
    print(f"MLflow tracking server URL: {tracking_uri}", flush=True)
    print(f"MLflow UI:                 {tracking_uri}", flush=True)
    print("Point training runs at this URL via --mlflow_tracking_uri="
          f"{tracking_uri}  (or export MLFLOW_TRACKING_URI={tracking_uri})", flush=True)
    print("=" * 70, flush=True)

    def _shutdown(*_):
        print("\nShutting down local MLflow server...", flush=True)
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Block until the mlflow server process exits (or we're interrupted above)
    sys.exit(server_proc.wait())


if __name__ == "__main__":
    fire.Fire(main)
