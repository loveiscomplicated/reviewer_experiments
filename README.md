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

Single job:

```bash
TEDS_GDRIVE_FILE_ID=<file_id> bash run_vast_teds_job.sh gcn statistical 1
```

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
