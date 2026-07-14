from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

try:
    from reviewer_experiments.teds_tensor_data import (
        FoldTensors,
        TensorSplit,
        build_fold_tensors,
        iter_fold_indices,
        load_teds_main,
        save_edge_audits,
    )
    from reviewer_experiments.tensor_models import build_model as build_dense_model
except ModuleNotFoundError:
    from teds_tensor_data import (  # type: ignore
        FoldTensors,
        TensorSplit,
        build_fold_tensors,
        iter_fold_indices,
        load_teds_main,
        save_edge_audits,
    )
    from tensor_models import build_model as build_dense_model  # type: ignore


MODEL_CHOICES = ("gcn", "gin", "gat", "tgcn")
GRAPH_CHOICES = ("statistical", "fully_connected")


@dataclass
class RunResult:
    mode: str
    backend: str
    fold: int
    model: str
    graph_type: str
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
        description="TEDS k-fold GNN baselines for reviewer responses."
    )
    parser.add_argument("--mode", choices=("estimate", "full"), default="estimate")
    parser.add_argument("--backend", choices=("pyg", "dense"), default="pyg")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--csv-path", default="TEDS_Discharge.csv")
    parser.add_argument("--output-dir", default="reviewer_experiments/results")
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES + ("all",), default=None)
    parser.add_argument("--graph-types", nargs="+", choices=GRAPH_CHOICES, default=list(GRAPH_CHOICES))
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
    parser.add_argument("--tgcn-patience", type=int, default=10)
    parser.add_argument("--scheduler-patience", type=int, default=7)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm batch progress bars.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = resolve_models(args.mode, args.models)
    include_temporal = "tgcn" in models
    device = choose_device(args.backend, args.device)

    print(f"Device: {device}")
    print(f"Mode: {args.mode} | Backend: {args.backend} | Models: {models} | Graphs: {args.graph_types}")
    df = load_teds_main(args.csv_path, max_rows=args.max_rows, seed=args.seed)
    y = df["REASONb"].to_numpy()
    print(f"Loaded rows: {len(df):,} | positive ratio: {y.mean():.4f}")

    metric_rows: List[RunResult] = []
    epoch_rows: List[Dict[str, object]] = []
    runtime_estimates: List[Dict[str, object]] = []

    for fold, train_idx, val_idx, test_idx in iter_fold_indices(y, n_splits=args.n_splits, seed=args.seed):
        if args.folds_to_run and fold not in set(args.folds_to_run):
            continue
        if args.mode == "estimate" and fold != 1:
            break

        print(
            f"\nPreparing fold {fold}: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}"
        )
        fold_data = build_fold_tensors(
            df=df,
            fold=fold,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            graph_types=args.graph_types,
            include_temporal=include_temporal,
        )
        save_edge_audits(fold_data, output_dir)

        for model_name in models:
            for graph_type in args.graph_types:
                print(f"\nRunning fold={fold} model={model_name} graph={graph_type}")
                result, epochs, runtime = run_one_configuration(
                    args=args,
                    fold_data=fold_data,
                    model_name=model_name,
                    graph_type=graph_type,
                    device=device,
                )
                metric_rows.append(result)
                epoch_rows.extend(epochs)
                if runtime:
                    runtime_estimates.append(runtime)

                write_outputs(metric_rows, epoch_rows, output_dir)
                if runtime_estimates:
                    (output_dir / "runtime_estimate.json").write_text(
                        json.dumps(runtime_estimates, indent=2), encoding="utf-8"
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
            json.dumps(runtime_estimates, indent=2), encoding="utf-8"
        )
        print(f"\nEstimated max full-run time: {total_estimated / 3600.0:.2f} hours")

    print(f"\nDone. Results written to {output_dir}")


def resolve_models(mode: str, requested: Optional[Sequence[str]]) -> List[str]:
    if not requested:
        return ["gcn", "gin", "gat", "tgcn"] if mode == "estimate" else ["gcn", "gin", "gat"]
    if "all" in requested:
        return list(MODEL_CHOICES)
    return list(dict.fromkeys(requested))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(backend: str, requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if backend == "dense" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_experiment_model(
    backend: str,
    model_name: str,
    cat_dims: Sequence[int],
    hidden_dim: int,
    num_classes: int,
    dropout: float,
    proj_dim: int,
) -> nn.Module:
    if backend == "dense":
        return build_dense_model(
            model_name,
            cat_dims=cat_dims,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            proj_dim=proj_dim,
        )
    if backend == "pyg":
        try:
            from reviewer_experiments.pyg_models import build_pyg_model
        except ModuleNotFoundError:
            from pyg_models import build_pyg_model  # type: ignore
        return build_pyg_model(
            model_name,
            cat_dims=cat_dims,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            proj_dim=proj_dim,
        )
    raise ValueError(f"Unknown backend: {backend}")


def run_one_configuration(
    args: argparse.Namespace,
    fold_data: FoldTensors,
    model_name: str,
    graph_type: str,
    device: torch.device,
) -> Tuple[RunResult, List[Dict[str, object]], Optional[Dict[str, object]]]:
    train_split, val_split, test_split, cat_dims, adjacency = select_splits_and_graph(
        fold_data, model_name, graph_type
    )
    full_sizes = (len(train_split.y), len(val_split.y), len(test_split.y))

    if args.mode == "estimate":
        train_split = subsample_split(train_split, args.estimate_max_train, seed=args.seed + fold_data.fold)
        val_split = subsample_split(val_split, args.estimate_max_val, seed=args.seed + fold_data.fold + 100)
        test_split = subsample_split(test_split, args.estimate_max_test, seed=args.seed + fold_data.fold + 200)
        max_epochs = args.estimate_epochs
    else:
        max_epochs = args.max_epochs

    train_loader = make_loader(train_split, args.batch_size, shuffle=True, args=args)
    val_loader = make_loader(val_split, args.batch_size, shuffle=False, args=args)
    test_loader = make_loader(test_split, args.batch_size, shuffle=False, args=args)

    model = build_experiment_model(
        backend=args.backend,
        model_name=model_name,
        cat_dims=cat_dims,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        dropout=args.dropout,
        proj_dim=args.proj_dim,
    ).to(device)
    adjacency = adjacency.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
        min_lr=args.scheduler_min_lr,
    )
    patience = args.tgcn_patience if model_name == "tgcn" else args.patience
    if args.mode == "estimate":
        patience = max(patience, max_epochs)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    epoch_rows: List[Dict[str, object]] = []
    start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        train_loss, train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            adjacency,
            device,
            desc=f"fold {fold_data.fold} {model_name}/{graph_type} epoch {epoch} train",
            show_progress=not args.no_progress,
        )
        val_loss, val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            adjacency,
            device,
            desc=f"fold {fold_data.fold} {model_name}/{graph_type} epoch {epoch} val",
            show_progress=not args.no_progress,
        )
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
            "backend": args.backend,
            "fold": fold_data.fold,
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
        }
        epoch_rows.append(row)
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} acc={train_metrics['accuracy']:.4f} "
            f"P={train_metrics['precision']:.4f} R={train_metrics['recall']:.4f} "
            f"F1={train_metrics['f1']:.4f} AUC={format_metric(train_metrics['auc'])} | "
            f"val_loss={val_loss:.4f} acc={val_metrics['accuracy']:.4f} "
            f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} "
            f"F1={val_metrics['f1']:.4f} AUC={format_metric(val_metrics['auc'])}"
        )

        if stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    elapsed = time.perf_counter() - start_time
    epochs_ran = len(epoch_rows)
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        adjacency,
        device,
        desc=f"fold {fold_data.fold} {model_name}/{graph_type} test",
        show_progress=not args.no_progress,
    )

    seconds_per_epoch_observed = elapsed / max(1, epochs_ran)
    estimate = estimate_runtime(
        args=args,
        mode=args.mode,
        model_name=model_name,
        graph_type=graph_type,
        fold=fold_data.fold,
        full_sizes=full_sizes,
        observed_sizes=(len(train_split.y), len(val_split.y), len(test_split.y)),
        seconds_per_epoch_observed=seconds_per_epoch_observed,
    )

    result = RunResult(
        mode=args.mode,
        backend=args.backend,
        fold=fold_data.fold,
        model=model_name,
        graph_type=graph_type,
        train_samples=len(train_split.y),
        val_samples=len(val_split.y),
        test_samples=len(test_split.y),
        full_train_samples=full_sizes[0],
        full_val_samples=full_sizes[1],
        full_test_samples=full_sizes[2],
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
        estimated_seconds_per_epoch_full=estimate.get("estimated_seconds_per_epoch_full") if estimate else None,
        estimated_max_run_seconds=estimate.get("estimated_max_run_seconds") if estimate else None,
    )
    print(
        f"Fold {fold_data.fold} test complete | model={model_name} graph={graph_type} | "
        f"loss={test_loss:.4f} | acc={test_metrics['accuracy']:.4f} | "
        f"P={test_metrics['precision']:.4f} | R={test_metrics['recall']:.4f} | "
        f"F1={test_metrics['f1']:.4f} | AUC={format_metric(test_metrics['auc'])}"
    )
    return result, epoch_rows, estimate


def select_splits_and_graph(
    fold_data: FoldTensors,
    model_name: str,
    graph_type: str,
) -> Tuple[TensorSplit, TensorSplit, TensorSplit, List[int], torch.Tensor]:
    if model_name == "tgcn":
        if (
            fold_data.temporal_train is None
            or fold_data.temporal_val is None
            or fold_data.temporal_test is None
            or fold_data.temporal_cat_dims is None
            or fold_data.temporal_adjacency is None
        ):
            raise ValueError("T-GCN requested but temporal fold tensors were not built.")
        return (
            fold_data.temporal_train,
            fold_data.temporal_val,
            fold_data.temporal_test,
            fold_data.temporal_cat_dims,
            fold_data.temporal_adjacency[graph_type],
        )
    return (
        fold_data.train,
        fold_data.val,
        fold_data.test,
        fold_data.cat_dims,
        fold_data.adjacency[graph_type],
    )


def make_loader(split: TensorSplit, batch_size: int, shuffle: bool, args: argparse.Namespace) -> DataLoader:
    dataset = TensorDataset(split.x_id, split.x_missing, split.y)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def subsample_split(split: TensorSplit, max_samples: int, seed: int) -> TensorSplit:
    if max_samples is None or len(split.y) <= max_samples:
        return split
    y = split.y.numpy()
    idx = np.arange(len(y))
    if np.unique(y).size > 1:
        _, selected = train_test_split(idx, test_size=max_samples, random_state=seed, stratify=y)
    else:
        rng = np.random.default_rng(seed)
        selected = rng.choice(idx, size=max_samples, replace=False)
    selected = np.sort(selected)
    return TensorSplit(split.x_id[selected], split.x_missing[selected], split.y[selected])


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    adjacency: torch.Tensor,
    device: torch.device,
    desc: str,
    show_progress: bool,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    metrics = BinaryMetricAccumulator()
    progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, disable=not show_progress)
    for x_id, x_missing, y in progress:
        x_id = x_id.to(device, non_blocking=True)
        x_missing = x_missing.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_id, x_missing, adjacency)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        metrics.update(logits.detach(), y)
        progress.set_postfix(loss=f"{total_loss / max(1, total_samples):.4f}")
    return total_loss / max(1, total_samples), metrics.compute()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    adjacency: torch.Tensor,
    device: torch.device,
    desc: str,
    show_progress: bool,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    metrics = BinaryMetricAccumulator()

    progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, disable=not show_progress)
    for x_id, x_missing, y in progress:
        x_id = x_id.to(device, non_blocking=True)
        x_missing = x_missing.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x_id, x_missing, adjacency)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        metrics.update_from_prediction(pred, logits, y)
        progress.set_postfix(loss=f"{total_loss / max(1, total_samples):.4f}")

    return total_loss / max(1, total_samples), metrics.compute()


class BinaryMetricAccumulator:
    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.y_true_parts: List[np.ndarray] = []
        self.y_score_parts: List[np.ndarray] = []

    def update(self, logits: torch.Tensor, y: torch.Tensor) -> None:
        pred = logits.argmax(dim=1)
        self.update_from_prediction(pred, logits, y)

    def update_from_prediction(self, pred: torch.Tensor, logits: torch.Tensor, y: torch.Tensor) -> None:
        self.tp += int(((pred == 1) & (y == 1)).sum().item())
        self.fp += int(((pred == 1) & (y == 0)).sum().item())
        self.tn += int(((pred == 0) & (y == 0)).sum().item())
        self.fn += int(((pred == 0) & (y == 1)).sum().item())
        prob_pos = torch.softmax(logits, dim=1)[:, 1]
        self.y_true_parts.append(y.detach().cpu().numpy())
        self.y_score_parts.append(prob_pos.detach().cpu().numpy())

    def compute(self) -> Dict[str, float]:
        total = self.tp + self.fp + self.tn + self.fn
        accuracy = (self.tp + self.tn) / total if total else 0.0
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        auc = float("nan")
        if self.y_true_parts:
            y_true = np.concatenate(self.y_true_parts)
            y_score = np.concatenate(self.y_score_parts)
            if np.unique(y_true).size == 2:
                auc = float(roc_auc_score(y_true, y_score))
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": auc,
        }


def format_metric(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.4f}"


def estimate_runtime(
    args: argparse.Namespace,
    mode: str,
    model_name: str,
    graph_type: str,
    fold: int,
    full_sizes: Tuple[int, int, int],
    observed_sizes: Tuple[int, int, int],
    seconds_per_epoch_observed: float,
) -> Optional[Dict[str, object]]:
    if mode != "estimate":
        return None
    observed_work = max(1, observed_sizes[0] + observed_sizes[1])
    full_work = full_sizes[0] + full_sizes[1]
    estimated_seconds_per_epoch_full = seconds_per_epoch_observed * (full_work / observed_work)
    max_epochs = args.max_epochs
    return {
        "mode": mode,
        "backend": args.backend,
        "fold": fold,
        "model": model_name,
        "graph_type": graph_type,
        "observed_train_samples": observed_sizes[0],
        "observed_val_samples": observed_sizes[1],
        "full_train_samples": full_sizes[0],
        "full_val_samples": full_sizes[1],
        "seconds_per_epoch_observed": seconds_per_epoch_observed,
        "estimated_seconds_per_epoch_full": estimated_seconds_per_epoch_full,
        "estimated_max_run_seconds": estimated_seconds_per_epoch_full * max_epochs,
        "estimated_max_run_hours": estimated_seconds_per_epoch_full * max_epochs / 3600.0,
    }


def write_outputs(
    metric_rows: Sequence[RunResult],
    epoch_rows: Sequence[Dict[str, object]],
    output_dir: Path,
) -> None:
    if metric_rows:
        metrics_df = pd.DataFrame([asdict(row) for row in metric_rows])
        metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)
        summary = build_summary(metrics_df)
        summary.to_csv(output_dir / "summary_mean_sd.csv", index=False)
    if epoch_rows:
        pd.DataFrame(epoch_rows).to_json(
            output_dir / "epoch_logs.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )


def build_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_auc", "test_loss"]
    grouped = metrics_df.groupby(["backend", "model", "graph_type"], as_index=False)[metric_cols].agg(["mean", "std"])
    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col for col in grouped.columns.to_flat_index()
    ]
    for metric in metric_cols:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        grouped[std_col] = grouped[std_col].fillna(0.0)
        grouped[f"{metric}_mean_sd"] = grouped.apply(
            lambda row, m=mean_col, s=std_col: f"{row[m]:.4f} ± {row[s]:.4f}",
            axis=1,
        )
    return grouped


if __name__ == "__main__":
    main()
