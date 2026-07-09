import modal

# Define a separate Modal App for TensorBoard
app = modal.App("salmon-tensorboard")

# Lightweight container image for running TensorBoard
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("tensorboard")
)

# Reference the runs volume
runs_volume = modal.Volume.from_name("salmon-runs-volume", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/root/salmon-ml/runs": runs_volume},
)
@modal.web_server(6006)
def run_tensorboard():
    import subprocess
    import threading
    import time
    
    print("Starting TensorBoard server on port 6006...", flush=True)
    cmd = "tensorboard --logdir /root/salmon-ml/runs --port 6006 --bind_all"
    subprocess.Popen(cmd, shell=True)
    
    # Run the volume reloader in a background thread so the function can return immediately
    def reloader():
        while True:
            time.sleep(15)
            try:
                runs_volume.reload()
            except Exception:
                pass
                
    threading.Thread(target=reloader, daemon=True).start()
    print("TensorBoard started, returning control to Modal.", flush=True)
