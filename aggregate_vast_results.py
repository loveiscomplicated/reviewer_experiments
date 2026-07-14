from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from reviewer_experiments.run_tensor_kfold import build_summary
except ModuleNotFoundError:
    from run_tensor_kfold import build_summary  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate sharded Vast.ai TEDS k-fold results.")
    parser.add_argument("--results-root", default="reviewer_experiments/results_vast")
    parser.add_argument("--output-dir", default="reviewer_experiments/results_vast_merged")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_paths = sorted(results_root.rglob("fold_metrics.csv"))
    if not metric_paths:
        raise FileNotFoundError(f"No fold_metrics.csv files found under {results_root}")

    metrics = []
    for path in metric_paths:
        df = pd.read_csv(path)
        df["source_file"] = str(path)
        metrics.append(df)
    metrics_df = pd.concat(metrics, ignore_index=True)
    metrics_df = metrics_df.drop_duplicates(
        subset=["backend", "fold", "model", "graph_type"],
        keep="last",
    ).sort_values(["backend", "model", "graph_type", "fold"])
    metrics_df.to_csv(output_dir / "fold_metrics_all.csv", index=False)
    build_summary(metrics_df).to_csv(output_dir / "summary_mean_sd.csv", index=False)

    epoch_paths = sorted(results_root.rglob("epoch_logs.jsonl"))
    if epoch_paths:
        with (output_dir / "epoch_logs_all.jsonl").open("w", encoding="utf-8") as out:
            for path in epoch_paths:
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        out.write(line + "\n")

    print(f"Aggregated {len(metrics_df)} fold/model/graph rows")
    print(f"Wrote: {output_dir / 'fold_metrics_all.csv'}")
    print(f"Wrote: {output_dir / 'summary_mean_sd.csv'}")


if __name__ == "__main__":
    main()
