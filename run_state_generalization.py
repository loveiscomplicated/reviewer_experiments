from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from reviewer_experiments.run_tensor_kfold import (
        build_experiment_model,
        choose_device,
        evaluate,
        format_metric,
        make_loader,
        move_split_to_device,
        set_seed,
        subsample_split,
        train_epoch,
    )
except ModuleNotFoundError:
    from run_tensor_kfold import (  # type: ignore
        build_experiment_model,
        choose_device,
        evaluate,
        format_metric,
        make_loader,
        move_split_to_device,
        set_seed,
        subsample_split,
        train_epoch,
    )

try:
    from reviewer_experiments.teds_tensor_data import (
        STATE_GENERALIZATION_SD_MULTIPLIERS,
        StateGeneralizationTensors,
        StateMissingnessScenario,
        build_state_generalization_tensors,
        build_state_missingness_scenarios,
        iter_state_generalization_indices,
        load_teds_main,
    )
except ModuleNotFoundError:
    from teds_tensor_data import (  # type: ignore
        STATE_GENERALIZATION_SD_MULTIPLIERS,
        StateGeneralizationTensors,
        StateMissingnessScenario,
        build_state_generalization_tensors,
        build_state_missingness_scenarios,
        iter_state_generalization_indices,
        load_teds_main,
    )


STATE_MODEL_CHOICES = ("gcn", "gin", "gat")
GRAPH_TYPE_CHOICES = ("statistical", "fully_connected")


@dataclass
class StateRunResult:
    mode: str
    backend: str
    scenario: str
    sd_multiplier: float
    state_missingness_mean: float
    state_missingness_sd: float
    missingness_threshold: float
    partial_states: str
    partial_state_count: int
    partial_episodes: int
    partial_episode_fraction: float
    fold: int
    model: str
    graph_type: str
    eval_group: str
    train_samples: int
    val_samples: int
    test_samples: int
    full_train_samples: int
    full_val_samples: int
    full_test_samples: int
    epochs_ran: int
    best_epoch: int
    best_val_loss: float
    test_loss: float
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float
    test_auc: float
    elapsed_seconds: float
    seconds_per_epoch_observed: float
    estimated_seconds_per_epoch_full: Optional[float] = None
    estimated_max_run_seconds: Optional[float] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TEDS state-generalization reruns using reviewer experiment settings."
    )
    parser.add_argument("--mode", choices=("estimate", "full"), default="estimate")
    parser.add_argument("--backend", choices=("pyg", "dense"), default="pyg")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--csv-path", default="TEDS_Discharge.csv")
    parser.add_argument("--output-dir", default="reviewer_experiments/results_state_generalization")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=tuple(STATE_GENERALIZATION_SD_MULTIPLIERS),
        default=list(STATE_GENERALIZATION_SD_MULTIPLIERS),
    )
    parser.add_argument("--models", nargs="+", choices=STATE_MODEL_CHOICES + ("all",), default=None)
    parser.add_argument("--graph-types", nargs="+", choices=GRAPH_TYPE_CHOICES, default=["statistical"])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None, help="Stratified row cap for debugging only.")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--estimate-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--proj-dim", type=int, default=7)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--scheduler-patience", type=int, default=7)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm batch progress bars.")
    parser.add_argument(
        "--preload-device",
        choices=("none", "cuda"),
        default="none",
        help="Move split tensors to GPU before DataLoader iteration to reduce per-batch host transfers.",
    )
    parser.add_argument("--estimate-max-train", type=int, default=50_000)
    parser.add_argument("--estimate-max-val", type=int, default=10_000)
    parser.add_argument("--estimate-max-test", type=int, default=10_000)
    parser.add_argument(
        "--folds-to-run",
        nargs="+",
        type=int,
        default=None,
        help="Optional fold ids to run. Useful for resuming full experiments.",
    )
    parser.add_argument(
        "--state-sd-ddof",
        type=int,
        default=0,
        help="Delta degrees of freedom for across-state SD; 0 reproduces the reviewer threshold values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = resolve_models(args.models)
    device = choose_device(args.backend, args.device)
    print(f"Device: {device}")
    print(
        f"Mode: {args.mode} | Backend: {args.backend} | "
        f"Models: {models} | Graphs: {args.graph_types} | Scenarios: {args.scenarios}"
    )

    df = load_teds_main(args.csv_path, max_rows=args.max_rows, seed=args.seed)
    y = df["REASONb"].to_numpy()
    print(f"Loaded rows: {len(df):,} | positive ratio: {y.mean():.4f}")

    state_missingness, scenarios = build_state_missingness_scenarios(df, sd_ddof=args.state_sd_ddof)
    write_state_audit(output_dir, state_missingness, scenarios, args.state_sd_ddof)

    metric_rows: List[StateRunResult] = []
    epoch_rows: List[Dict[str, object]] = []
    runtime_estimates: List[Dict[str, object]] = []

    for scenario_name in args.scenarios:
        scenario = scenarios[scenario_name]
        print_scenario(scenario)
        for fold, train_idx, val_idx, comprehensive_test_idx, partial_test_idx in iter_state_generalization_indices(
            df=df,
            partial_states=scenario.partial_states,
            n_splits=args.n_splits,
            seed=args.seed,
        ):
            if args.folds_to_run and fold not in set(args.folds_to_run):
                continue
            if args.mode == "estimate" and fold != 1:
                break

            print(
                f"\nPreparing {scenario_name} fold {fold}: "
                f"train={len(train_idx):,}, val={len(val_idx):,}, "
                f"comprehensive_test={len(comprehensive_test_idx):,}, "
                f"partial_test={len(partial_test_idx):,}"
            )
            split_data = build_state_generalization_tensors(
                df=df,
                scenario=scenario_name,
                fold=fold,
                train_idx=train_idx,
                val_idx=val_idx,
                comprehensive_test_idx=comprehensive_test_idx,
                partial_test_idx=partial_test_idx,
                graph_types=args.graph_types,
            )
            save_state_edge_audits(split_data, output_dir)

            for model_name in models:
                for graph_type in args.graph_types:
                    print(f"\nRunning scenario={scenario_name} fold={fold} model={model_name} graph={graph_type}")
                    results, epochs, runtime = run_one_state_configuration(
                        args=args,
                        scenario=scenario,
                        split_data=split_data,
                        model_name=model_name,
                        graph_type=graph_type,
                        device=device,
                    )
                    metric_rows.extend(results)
                    epoch_rows.extend(epochs)
                    if runtime:
                        runtime_estimates.append(runtime)

                    write_outputs(metric_rows, epoch_rows, output_dir)
                    if runtime_estimates:
                        (output_dir / "runtime_estimate.json").write_text(
                            json.dumps(runtime_estimates, indent=2),
                            encoding="utf-8",
                        )

    write_outputs(metric_rows, epoch_rows, output_dir)
    if runtime_estimates:
        total_estimated = sum(
            row.get("estimated_max_run_seconds", 0.0) or 0.0 for row in runtime_estimates
        )
        runtime_estimates.append(
            {
                "mode": args.mode,
                "note": "Sum assumes max epochs for every full-run configuration; early stopping usually shortens this.",
                "estimated_total_max_run_seconds": total_estimated,
                "estimated_total_max_run_hours": total_estimated / 3600.0,
            }
        )
        (output_dir / "runtime_estimate.json").write_text(
            json.dumps(runtime_estimates, indent=2),
            encoding="utf-8",
        )
        print(f"\nEstimated max full-run time: {total_estimated / 3600.0:.2f} hours")

    print(f"\nDone. Results written to {output_dir}")


def resolve_models(requested: Optional[Sequence[str]]) -> List[str]:
    if not requested:
        return list(STATE_MODEL_CHOICES)
    if "all" in requested:
        return list(STATE_MODEL_CHOICES)
    return list(dict.fromkeys(requested))


def run_one_state_configuration(
    args: argparse.Namespace,
    scenario: StateMissingnessScenario,
    split_data: StateGeneralizationTensors,
    model_name: str,
    graph_type: str,
    device: torch.device,
) -> Tuple[List[StateRunResult], List[Dict[str, object]], Optional[Dict[str, object]]]:
    train_split = split_data.train
    val_split = split_data.val
    comprehensive_split = split_data.comprehensive_test
    partial_split = split_data.partial_test
    full_sizes = (
        len(train_split.y),
        len(val_split.y),
        len(comprehensive_split.y),
        len(partial_split.y),
    )

    if args.mode == "estimate":
        train_split = subsample_split(train_split, args.estimate_max_train, seed=args.seed + split_data.fold)
        val_split = subsample_split(val_split, args.estimate_max_val, seed=args.seed + split_data.fold + 100)
        comprehensive_split = subsample_split(
            comprehensive_split,
            args.estimate_max_test,
            seed=args.seed + split_data.fold + 200,
        )
        partial_split = subsample_split(partial_split, args.estimate_max_test, seed=args.seed + split_data.fold + 300)
        max_epochs = args.estimate_epochs
    else:
        max_epochs = args.max_epochs

    if args.preload_device != "none":
        if args.num_workers != 0:
            raise ValueError("--preload-device requires --num-workers 0.")
        if args.preload_device == "cuda":
            if device.type != "cuda":
                raise ValueError("--preload-device cuda requires --device cuda/auto with CUDA available.")
            train_split = move_split_to_device(train_split, device)
            val_split = move_split_to_device(val_split, device)
            comprehensive_split = move_split_to_device(comprehensive_split, device)
            partial_split = move_split_to_device(partial_split, device)

    train_loader = make_loader(train_split, args.batch_size, shuffle=True, args=args)
    val_loader = make_loader(val_split, args.batch_size, shuffle=False, args=args)
    comprehensive_loader = make_loader(comprehensive_split, args.batch_size, shuffle=False, args=args)
    partial_loader = make_loader(partial_split, args.batch_size, shuffle=False, args=args)

    model = build_experiment_model(
        backend=args.backend,
        model_name=model_name,
        cat_dims=split_data.cat_dims,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        dropout=args.dropout,
        proj_dim=args.proj_dim,
    ).to(device)
    adjacency = split_data.adjacency[graph_type].to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )
    patience = max(args.patience, max_epochs) if args.mode == "estimate" else args.patience

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    epoch_rows: List[Dict[str, object]] = []
    start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            adjacency,
            device,
            desc=f"{split_data.scenario} fold {split_data.fold} {model_name}/{graph_type} epoch {epoch} train",
            show_progress=not args.no_progress,
        )
        val_loss, val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            adjacency,
            device,
            desc=f"{split_data.scenario} fold {split_data.fold} {model_name}/{graph_type} epoch {epoch} val",
            show_progress=not args.no_progress,
        )
        epoch_seconds = time.perf_counter() - epoch_start
        epoch_samples = len(train_split.y) + len(val_split.y)
        scheduler.step(val_loss)

        improved = val_loss < best_val_loss - 1e-8
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1

        row = {
            "mode": args.mode,
            "backend": args.backend,
            "scenario": split_data.scenario,
            "fold": split_data.fold,
            "model": model_name,
            "graph_type": graph_type,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "train_auc": train_metrics["auc"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": epoch_seconds,
            "samples_per_second": epoch_samples / epoch_seconds if epoch_seconds > 0 else float("nan"),
        }
        epoch_rows.append(row)
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} acc={train_metrics['accuracy']:.4f} "
            f"P={train_metrics['precision']:.4f} R={train_metrics['recall']:.4f} "
            f"F1={train_metrics['f1']:.4f} AUC={format_metric(train_metrics['auc'])} | "
            f"val_loss={val_loss:.4f} acc={val_metrics['accuracy']:.4f} "
            f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} "
            f"F1={val_metrics['f1']:.4f} AUC={format_metric(val_metrics['auc'])} | "
            f"{epoch_seconds:.1f}s {row['samples_per_second']:.1f} samples/s"
        )

        if stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    elapsed = time.perf_counter() - start_time
    epochs_ran = len(epoch_rows)
    if best_state is not None:
        model.load_state_dict(best_state)

    eval_specs = [
        ("comprehensive", comprehensive_loader, comprehensive_split, full_sizes[2]),
        ("partial", partial_loader, partial_split, full_sizes[3]),
    ]
    seconds_per_epoch_observed = elapsed / max(1, epochs_ran)
    runtime = estimate_runtime(
        args=args,
        scenario=split_data.scenario,
        model_name=model_name,
        graph_type=graph_type,
        fold=split_data.fold,
        full_train_samples=full_sizes[0],
        full_val_samples=full_sizes[1],
        observed_train_samples=len(train_split.y),
        observed_val_samples=len(val_split.y),
        seconds_per_epoch_observed=seconds_per_epoch_observed,
    )

    results: List[StateRunResult] = []
    for eval_group, loader, split, full_test_samples in eval_specs:
        test_loss, test_metrics = evaluate(
            model,
            loader,
            criterion,
            adjacency,
            device,
            desc=f"{split_data.scenario} fold {split_data.fold} {model_name}/{graph_type} {eval_group} test",
            show_progress=not args.no_progress,
        )
        results.append(
            StateRunResult(
                mode=args.mode,
                backend=args.backend,
                scenario=split_data.scenario,
                sd_multiplier=scenario.sd_multiplier,
                state_missingness_mean=scenario.mean_missingness,
                state_missingness_sd=scenario.sd_missingness,
                missingness_threshold=scenario.threshold,
                partial_states=" ".join(str(state) for state in scenario.partial_states),
                partial_state_count=len(scenario.partial_states),
                partial_episodes=scenario.partial_episodes,
                partial_episode_fraction=scenario.partial_episodes / scenario.total_episodes,
                fold=split_data.fold,
                model=model_name,
                graph_type=graph_type,
                eval_group=eval_group,
                train_samples=len(train_split.y),
                val_samples=len(val_split.y),
                test_samples=len(split.y),
                full_train_samples=full_sizes[0],
                full_val_samples=full_sizes[1],
                full_test_samples=full_test_samples,
                epochs_ran=epochs_ran,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                test_loss=test_loss,
                test_accuracy=test_metrics["accuracy"],
                test_precision=test_metrics["precision"],
                test_recall=test_metrics["recall"],
                test_f1=test_metrics["f1"],
                test_auc=test_metrics["auc"],
                elapsed_seconds=elapsed,
                seconds_per_epoch_observed=seconds_per_epoch_observed,
                estimated_seconds_per_epoch_full=runtime.get("estimated_seconds_per_epoch_full") if runtime else None,
                estimated_max_run_seconds=runtime.get("estimated_max_run_seconds") if runtime else None,
            )
        )
        print(
            f"{eval_group} test complete | scenario={split_data.scenario} fold={split_data.fold} "
            f"model={model_name} graph={graph_type} | loss={test_loss:.4f} | "
            f"acc={test_metrics['accuracy']:.4f} | P={test_metrics['precision']:.4f} | "
            f"R={test_metrics['recall']:.4f} | F1={test_metrics['f1']:.4f} | "
            f"AUC={format_metric(test_metrics['auc'])}"
        )

    return results, epoch_rows, runtime


def estimate_runtime(
    args: argparse.Namespace,
    scenario: str,
    model_name: str,
    graph_type: str,
    fold: int,
    full_train_samples: int,
    full_val_samples: int,
    observed_train_samples: int,
    observed_val_samples: int,
    seconds_per_epoch_observed: float,
) -> Optional[Dict[str, object]]:
    if args.mode != "estimate":
        return None
    observed_work = max(1, observed_train_samples + observed_val_samples)
    full_work = full_train_samples + full_val_samples
    estimated_seconds_per_epoch_full = seconds_per_epoch_observed * (full_work / observed_work)
    return {
        "mode": args.mode,
        "backend": args.backend,
        "scenario": scenario,
        "fold": fold,
        "model": model_name,
        "graph_type": graph_type,
        "observed_train_samples": observed_train_samples,
        "observed_val_samples": observed_val_samples,
        "full_train_samples": full_train_samples,
        "full_val_samples": full_val_samples,
        "seconds_per_epoch_observed": seconds_per_epoch_observed,
        "estimated_seconds_per_epoch_full": estimated_seconds_per_epoch_full,
        "estimated_max_run_seconds": estimated_seconds_per_epoch_full * args.max_epochs,
        "estimated_max_run_hours": estimated_seconds_per_epoch_full * args.max_epochs / 3600.0,
    }


def write_outputs(
    metric_rows: Sequence[StateRunResult],
    epoch_rows: Sequence[Dict[str, object]],
    output_dir: Path,
) -> None:
    if metric_rows:
        metrics_df = pd.DataFrame([asdict(row) for row in metric_rows])
        metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)
        build_state_summary(metrics_df).to_csv(output_dir / "summary_mean_sd.csv", index=False)
    if epoch_rows:
        pd.DataFrame(epoch_rows).to_json(
            output_dir / "epoch_logs.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )


def build_state_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_auc", "test_loss"]
    grouped = metrics_df.groupby(
        ["backend", "scenario", "model", "graph_type", "eval_group"],
        as_index=False,
    )[metric_cols].agg(["mean", "std"])
    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col for col in grouped.columns.to_flat_index()
    ]
    for metric in metric_cols:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        grouped[std_col] = grouped[std_col].fillna(0.0)
        grouped[f"{metric}_mean_sd"] = grouped.apply(
            lambda row, m=mean_col, s=std_col: f"{row[m]:.4f} +/- {row[s]:.4f}",
            axis=1,
        )
    return grouped


def write_state_audit(
    output_dir: Path,
    state_missingness: pd.DataFrame,
    scenarios: Dict[str, StateMissingnessScenario],
    sd_ddof: int,
) -> None:
    audit = state_missingness.copy()
    for scenario in scenarios.values():
        partial_set = set(scenario.partial_states)
        audit[f"{scenario.name}_is_partial"] = audit["STFIPS"].isin(partial_set)
    audit.to_csv(output_dir / "state_missingness_audit.csv", index=False)

    rows = []
    for scenario in scenarios.values():
        row = asdict(scenario)
        row["partial_states"] = " ".join(str(state) for state in scenario.partial_states)
        row["comprehensive_states"] = " ".join(str(state) for state in scenario.comprehensive_states)
        row["partial_episode_fraction"] = scenario.partial_episodes / scenario.total_episodes
        row["state_sd_ddof"] = sd_ddof
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "state_scenarios.csv", index=False)


def save_state_edge_audits(split_data: StateGeneralizationTensors, output_dir: Path) -> None:
    for graph_type, audit in split_data.edge_audit.items():
        audit.to_csv(
            output_dir / f"edge_audit_{split_data.scenario}_fold{split_data.fold}_{graph_type}.csv",
            index=False,
        )


def print_scenario(scenario: StateMissingnessScenario) -> None:
    print(
        f"\nScenario {scenario.name}: mean={scenario.mean_missingness:.4f}, "
        f"sd={scenario.sd_missingness:.4f}, threshold={scenario.threshold:.4f}, "
        f"partial_states={list(scenario.partial_states)}, "
        f"partial_episodes={scenario.partial_episodes:,}/"
        f"{scenario.total_episodes:,}"
    )


if __name__ == "__main__":
    main()
