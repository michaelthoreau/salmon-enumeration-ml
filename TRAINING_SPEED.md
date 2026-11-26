# Training Speed Measurements

Measured on 2026-10-12 using Python 3.11.15, PyTorch 2.5.1+cu121,
Transformers 5.12.1, and an RTX 2060 SUPER with 8 GB VRAM. Runs used the
Sockeye API dataset, 512px crops, two context frames at stride 20, and the
existing remote MLflow tracking server. All runs completed; no model
checkpoints were written. Results are speed checks, not accuracy comparisons.

## Results

The short runs excluded four initial steps; the 128-step runs excluded sixteen.
Step time includes metric reporting but excludes initialization, validation,
checkpoint writes, and final speed-summary logging. GPU phases were synchronized
for measurement. Peak memory is PyTorch allocated memory, not total GPU usage.

| Setup | Steps | Mean Seconds/Step | Augmented Crops/Second | Peak Allocated VRAM |
|---|---|---|---|---|
| Original six MLflow requests/step, FP32, batch 4, workers 8 | 16 | 1.219 | 3.28 | 1.51 GB |
| Batched MLflow request/step, FP32, batch 4, workers 8 | 16 | 0.332 | 12.05 | 1.51 GB |
| Log every 10 steps, FP32, batch 4, workers 4 | 32 | 0.175 | 22.80 | 1.51 GB |
| Log every 10 steps, AMP, batch 4, workers 4 | 32 | 0.135 | 29.55 | 1.94 GB |
| Log every 10 steps, AMP, batch 8, workers 4 | 32 | 0.221 | 36.19 | 3.30 GB |
| Log every 10 steps, AMP, frozen backbone, batch 4, workers 4 | 32 | 0.082 | 48.78 | 0.62 GB |
| AMP, batch 4, workers 4, reaching uncached frames | 128 | 0.657 | 6.08 | 1.95 GB |
| AMP, batch 4, workers 4, warmed cache | 128 | 0.135 | 29.66 | 1.95 GB |
| AMP, batch 4, workers 8, warmed cache | 128 | 0.135 | 29.68 | 1.94 GB |

The baseline spent about 1.057 seconds/step reporting metrics and about 0.16
seconds on transfer and model computation. Batching requests alone reduced step
time by about 3.7x without changing the model or loss calculation. Logging every
ten steps reduced reporting cost further. The longer uncached run spent about
0.520 seconds/step waiting for data; the warmed repeat spent about 0.0002 seconds.
Short warm-cache timings should not be used to predict first-epoch runtime.

Changing worker counts changes worker-local augmentation RNG streams, so those
runs do not have identical crops. Batch size and backbone freezing also change
training behavior. The original versus batched-request runs kept eight workers
and matched the observed short-run losses.

## Recommended Starting Point

Keep the existing model, batch size four, and FP32. Use `--workers=4
--log_every=10`. This preserves the training objective while reducing overhead.
Eight workers may still be useful for a cold cache; the worker comparison here
only establishes equal throughput on warmed data.

AMP is opt-in. Losses stayed finite, but the warmed 128-step runs skipped three
optimizer updates after the sixteen-step timing warmup while adjusting the
gradient scale. The scheduler does not advance on skipped updates, and scaler
state is saved in new checkpoints. `global_step` still counts attempted batches.
Compare held-out count MAE before adopting AMP for production training.

Freezing reduces trainable parameters from 41.5M to 18.0M, not total model size
or inference cost. In the two-context-frame setup it also freezes the expanded
first convolution, preventing adaptation of its new temporal-channel weights.
Treat this as a separate accuracy experiment, not a default optimization.

## Small Dataset And Smaller Models

The API returned 1,101 labeled tasks from 19 flights, split into 939 training
tasks across 17 flights and 162 validation tasks across two flights. Tasks from
the same flight can be highly correlated; the number of independent scenes is
much smaller than the task count. Keep flight-level splits for comparisons.

`SameFrameBatchSampler` emits a batch of independently augmented crops from one
task, then moves to the next task. Thus an epoch contains 939 steps even with
batch size four. Larger batches add crops per task, not more independent tasks
per step. The existing `min_batches` option does not control this sampler.

Prioritize pretrained weights, held-out flight metrics, and shorter runs selected
by validation performance over training from scratch or assuming 100 epochs are
necessary. Early stopping and validation frequency are worthwhile follow-ups.

For an actually smaller DETR, compare fewer encoder/decoder layers or a smaller
pretrained backbone. Those changes were not implemented or benchmarked here:
they require configuration-aware checkpoint loading and a controlled accuracy
comparison. Reducing object queries also sets a lower ceiling on detections per
crop; inspect crop label counts before lowering the current 100 queries.

## Reproducing A Profile

```bash
python train_detr_api.py \
  --num_context_frames=2 --context_stride=20 \
  --benchmark_batches=128 --profile_warmup=16 \
  --workers=4 --log_every=10 --use_ortho_data=False \
  --mlflow=speed_test --save_path=runs/speed_test
```

Benchmark mode requires a fresh run, overrides epochs to one, and skips validation
and checkpoints. Use a unique name and output directory for each comparison.
It still fetches API data, fills the frame cache, and logs metrics. The dotenv
loader can override an environment-only `MLFLOW_TRACKING_URI`; pass
`--mlflow_tracking_uri=...` explicitly when isolating tracking storage.

The measured JSON summaries are under `runs/speed_profiles/`. Normal full runs
can use `--profile=True` to emit the same phase timings without enabling
benchmark mode. Disable profiling for production throughput because synchronizing
CUDA phases prevents some CPU/GPU overlap.