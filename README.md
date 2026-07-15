# TEDS GNN Reviewer Experiments

PyG-based k-fold baselines for reviewer responses.

## Local smoke

```bash
python run_tensor_kfold.py \
  --backend pyg \
  --mode full \
  --max-rows 500 \
  --max-epochs 1 \
  --models gcn \
  --graph-types statistical \
  --folds-to-run 1 \
  --device cpu
```

## Full reviewer run

```bash
python run_tensor_kfold.py \
  --backend pyg \
  --mode full \
  --models gcn gin gat \
  --graph-types statistical fully_connected \
  --output-dir results_full_pyg
```

## Vast.ai

Set `TEDS_GDRIVE_FILE_ID` for `TEDS_Discharge.csv`.
Set `DISCORD_WEBHOOK_URL` to receive start, failure, upload, and completion notifications.
Notifications are best-effort and do not fail the run if the webhook is missing or unreachable.
Result upload is disabled by default; set `UPLOAD_RESULTS=1` with `RCLONE_REMOTE` to enable rclone uploads.

Single job:

```bash
TEDS_GDRIVE_FILE_ID=<file_id> bash run_vast_teds_job.sh gcn statistical 1
```

Thirty single-job instances:

```bash
JOB_INDEX=<0..29> TEDS_GDRIVE_FILE_ID=<file_id> bash run_vast_teds_job.sh
```

`JOB_INDEX` order is `gcn`, `gin`, `gat` x `statistical`, `fully_connected` x folds `1..5`.
Set `SKIP_SETUP=1` only on prepared images that already have the conda env, code, and dataset.
Progress bars are shown only for interactive terminals and are automatically omitted from `train.log`.
Set `NO_PROGRESS=1` to disable tqdm even in an interactive terminal.
Vast scripts default to `PRELOAD_DEVICE=cuda`, which preloads fold tensors into GPU memory; set `PRELOAD_DEVICE=none` to disable it.
Set `DISCORD_NOTIFY=0` to disable Discord notifications, or `DISCORD_BOT_NAME="..."` to change the bot name.

Parallel jobs on one multi-GPU instance:

```bash
TEDS_GDRIVE_FILE_ID=<file_id> bash run_vast_teds_parallel.sh
```

Parallel range on one multi-GPU instance:

```bash
JOB_INDEX_RANGE=0-9 TEDS_GDRIVE_FILE_ID=<file_id> bash run_vast_teds_parallel.sh --batch-size 1024
```

`JOB_INDEX_RANGE` accepts `0-9`, `0..9`, or `0:9`.
For non-contiguous selections, use `JOB_INDEXES='0 1 2 10-14'`.
Range mode auto-detects GPU count and schedules selected jobs across GPUs.
If `UPLOAD_RESULTS=1` and `RCLONE_REMOTE` is set, raw selected results upload to `${RCLONE_DEST_DIR}/raw`, and partial merged summaries upload to `${RCLONE_DEST_DIR}/merged_ranges/<range>`.
Set `UPLOAD_RAW_RESULTS=0` to skip raw result upload in range mode.

Aggregate sharded results:

```bash
python aggregate_vast_results.py \
  --results-root results_vast \
  --output-dir results_vast_merged
```

The final reviewer table is `results_vast_merged/summary_mean_sd.csv`.

## State-generalization rerun for Table 3

This reruns Table 3 with the reviewer experiment preprocessing and hyperparameters.
It includes `GCN`, `GIN`, and `GAT`; `T-GCN` is intentionally excluded from this
state-generalization analysis.

Local smoke:

```bash
python run_state_generalization.py \
  --backend dense \
  --mode full \
  --max-rows 5000 \
  --max-epochs 1 \
  --models gcn \
  --scenarios scenario01 \
  --graph-types statistical \
  --folds-to-run 1 \
  --device cpu
```

Full Vast.ai run:

```bash
SCENARIOS="scenario01 scenario02" \
MODELS="gcn gin gat" \
GRAPH_TYPES="statistical" \
FOLDS="1 2 3 4 5" \
bash run_vast_teds_state_generalization_parallel.sh --batch-size 1024
```

The state-generalization matrix order is `scenario01`, `scenario02` x `gcn`,
`gin`, `gat` x `statistical` x folds `1..5`. The final Table 3 candidate is
`results_state_generalization_vast_merged/summary_mean_sd.csv`.
