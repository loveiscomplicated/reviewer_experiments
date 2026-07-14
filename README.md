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
Set `DISCORD_NOTIFY=0` to disable Discord notifications, or `DISCORD_BOT_NAME="..."` to change the bot name.

Parallel jobs on one multi-GPU instance:

```bash
TEDS_GDRIVE_FILE_ID=<file_id> bash run_vast_teds_parallel.sh
```

Aggregate sharded results:

```bash
python aggregate_vast_results.py \
  --results-root results_vast \
  --output-dir results_vast_merged
```

The final reviewer table is `results_vast_merged/summary_mean_sd.csv`.
