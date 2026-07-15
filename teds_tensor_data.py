from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2_contingency
from sklearn.model_selection import StratifiedKFold, train_test_split


MISSING_CODE = -9
LABEL_COLUMN = "REASONb"

FEATURE_COLUMNS: List[str] = [
    "EDUC",
    "MARSTAT",
    "PSOURCE",
    "NOPRIOR",
    "ARRESTS",
    "EMPLOY",
    "METHUSE",
    "PSYPROB",
    "GENDER",
    "LIVARAG",
    "SERVICES_D",
    "EMPLOY_D",
    "LIVARAG_D",
    "ARRESTS_D",
    "DSMCRIT",
    "AGE",
    "RACE",
    "LOS",
    "STFIPS",
    "SERVICES",
]

VAR_TYPES: Dict[str, str] = {
    "EDUC": "nominal",
    "MARSTAT": "nominal",
    "PSOURCE": "nominal",
    "NOPRIOR": "nominal",
    "ARRESTS": "ordinal",
    "EMPLOY": "nominal",
    "METHUSE": "nominal",
    "PSYPROB": "nominal",
    "GENDER": "nominal",
    "LIVARAG": "nominal",
    "SERVICES_D": "nominal",
    "EMPLOY_D": "nominal",
    "LIVARAG_D": "nominal",
    "ARRESTS_D": "ordinal",
    "DSMCRIT": "nominal",
    "AGE": "ordinal",
    "RACE": "nominal",
    "LOS": "ordinal",
    "STFIPS": "nominal",
    "SERVICES": "nominal",
}

TEMPORAL_DYNAMIC_PAIRS: Mapping[str, str] = {
    "SERVICES": "SERVICES_D",
    "EMPLOY": "EMPLOY_D",
    "LIVARAG": "LIVARAG_D",
    "ARRESTS": "ARRESTS_D",
}

TEMPORAL_NODE_COLUMNS: List[str] = [
    col for col in FEATURE_COLUMNS if col not in set(TEMPORAL_DYNAMIC_PAIRS.values())
]

STATE_GENERALIZATION_SD_MULTIPLIERS: Dict[str, float] = {
    "scenario01": 1.0,
    "scenario02": 2.0,
}


@dataclass(frozen=True)
class TensorSplit:
    x_id: torch.Tensor
    x_missing: torch.Tensor
    y: torch.Tensor


@dataclass(frozen=True)
class FoldTensors:
    fold: int
    train: TensorSplit
    val: TensorSplit
    test: TensorSplit
    cat_dims: List[int]
    adjacency: Dict[str, torch.Tensor]
    edge_audit: Dict[str, pd.DataFrame]
    temporal_train: Optional[TensorSplit] = None
    temporal_val: Optional[TensorSplit] = None
    temporal_test: Optional[TensorSplit] = None
    temporal_cat_dims: Optional[List[int]] = None
    temporal_adjacency: Optional[Dict[str, torch.Tensor]] = None
    temporal_edge_audit: Optional[Dict[str, pd.DataFrame]] = None


@dataclass(frozen=True)
class StateMissingnessScenario:
    name: str
    sd_multiplier: float
    mean_missingness: float
    sd_missingness: float
    threshold: float
    partial_states: Tuple[int, ...]
    comprehensive_states: Tuple[int, ...]
    partial_episodes: int
    total_episodes: int


@dataclass(frozen=True)
class StateGeneralizationTensors:
    scenario: str
    fold: int
    train: TensorSplit
    val: TensorSplit
    comprehensive_test: TensorSplit
    partial_test: TensorSplit
    cat_dims: List[int]
    adjacency: Dict[str, torch.Tensor]
    edge_audit: Dict[str, pd.DataFrame]


def load_teds_main(csv_path: str | Path, max_rows: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    """Load the main TEDS variables and create the binary REASON label."""
    csv_path = Path(csv_path)
    usecols = FEATURE_COLUMNS + ["REASON"]
    df = pd.read_csv(csv_path, usecols=usecols)
    df[LABEL_COLUMN] = (df["REASON"] == 1).astype(np.int64)
    df = df.drop(columns=["REASON"])
    df = df[FEATURE_COLUMNS + [LABEL_COLUMN]]

    if max_rows is not None and max_rows < len(df):
        _, sample_idx = train_test_split(
            np.arange(len(df)),
            test_size=max_rows,
            random_state=seed,
            stratify=df[LABEL_COLUMN].to_numpy(),
        )
        df = df.iloc[np.sort(sample_idx)].reset_index(drop=True)

    return df.reset_index(drop=True)


def iter_fold_indices(
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
    val_fraction_within_trainval: float = 0.15,
) -> Iterable[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_idx = np.arange(len(y))
    for fold, train_val_idx, test_idx in (
        (fold, train_val_idx, test_idx)
        for fold, (train_val_idx, test_idx) in enumerate(outer.split(all_idx, y), start=1)
    ):
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_fraction_within_trainval,
            random_state=seed + fold,
            stratify=y[train_val_idx],
        )
        yield fold, np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def compute_state_missingness(
    df: pd.DataFrame,
    columns: Sequence[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    row_missingness = (df[list(columns)] == MISSING_CODE).mean(axis=1)
    state_df = pd.DataFrame(
        {
            "STFIPS": df["STFIPS"].to_numpy(),
            "row_missingness": row_missingness.to_numpy(dtype=np.float64),
        }
    )
    return (
        state_df.groupby("STFIPS", as_index=False)
        .agg(missing_rate=("row_missingness", "mean"), episodes=("row_missingness", "size"))
        .sort_values(["missing_rate", "STFIPS"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_state_missingness_scenarios(
    df: pd.DataFrame,
    sd_ddof: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, StateMissingnessScenario]]:
    state_missingness = compute_state_missingness(df)
    mean_missingness = float(state_missingness["missing_rate"].mean())
    sd_missingness = float(state_missingness["missing_rate"].std(ddof=sd_ddof))
    total_episodes = int(state_missingness["episodes"].sum())
    all_states = tuple(int(state) for state in sorted(state_missingness["STFIPS"].tolist()))

    scenarios: Dict[str, StateMissingnessScenario] = {}
    for name, sd_multiplier in STATE_GENERALIZATION_SD_MULTIPLIERS.items():
        threshold = mean_missingness + sd_multiplier * sd_missingness
        partial_df = state_missingness[state_missingness["missing_rate"] >= threshold]
        partial_states = tuple(int(state) for state in sorted(partial_df["STFIPS"].tolist()))
        partial_set = set(partial_states)
        comprehensive_states = tuple(state for state in all_states if state not in partial_set)
        scenarios[name] = StateMissingnessScenario(
            name=name,
            sd_multiplier=sd_multiplier,
            mean_missingness=mean_missingness,
            sd_missingness=sd_missingness,
            threshold=float(threshold),
            partial_states=partial_states,
            comprehensive_states=comprehensive_states,
            partial_episodes=int(partial_df["episodes"].sum()),
            total_episodes=total_episodes,
        )
    return state_missingness, scenarios


def iter_state_generalization_indices(
    df: pd.DataFrame,
    partial_states: Sequence[int],
    n_splits: int = 5,
    seed: int = 42,
    val_fraction_within_trainval: float = 0.15,
) -> Iterable[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    partial_mask = df["STFIPS"].isin(set(partial_states)).to_numpy()
    comprehensive_idx = np.flatnonzero(~partial_mask)
    partial_idx = np.flatnonzero(partial_mask)
    if len(partial_idx) == 0:
        raise ValueError("No partial-reporting rows selected for state-generalization.")
    if len(comprehensive_idx) == 0:
        raise ValueError("No comprehensive-reporting rows remain for state-generalization.")

    y = df[LABEL_COLUMN].to_numpy()
    y_comprehensive = y[comprehensive_idx]
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_val_pos, comprehensive_test_pos) in enumerate(
        outer.split(comprehensive_idx, y_comprehensive),
        start=1,
    ):
        train_val_idx = comprehensive_idx[train_val_pos]
        comprehensive_test_idx = comprehensive_idx[comprehensive_test_pos]
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_fraction_within_trainval,
            random_state=seed + fold,
            stratify=y[train_val_idx],
        )
        yield (
            fold,
            np.sort(train_idx),
            np.sort(val_idx),
            np.sort(comprehensive_test_idx),
            np.sort(partial_idx),
        )


def build_state_generalization_tensors(
    df: pd.DataFrame,
    scenario: str,
    fold: int,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    comprehensive_test_idx: Sequence[int],
    partial_test_idx: Sequence[int],
    graph_types: Sequence[str],
) -> StateGeneralizationTensors:
    train_raw = df.iloc[list(train_idx)][FEATURE_COLUMNS].copy()
    val_raw = df.iloc[list(val_idx)][FEATURE_COLUMNS].copy()
    comprehensive_test_raw = df.iloc[list(comprehensive_test_idx)][FEATURE_COLUMNS].copy()
    partial_test_raw = df.iloc[list(partial_test_idx)][FEATURE_COLUMNS].copy()

    (
        train_filled,
        split_filled,
        train_flags,
        split_flags,
    ) = _fit_impute_and_flags_many(
        train_raw,
        [val_raw, comprehensive_test_raw, partial_test_raw],
        FEATURE_COLUMNS,
    )
    val_filled, comprehensive_test_filled, partial_test_filled = split_filled
    val_flags, comprehensive_test_flags, partial_test_flags = split_flags

    vocabs = _fit_vocabs(train_filled, FEATURE_COLUMNS)
    cat_dims = [len(vocabs[col]) + 1 for col in FEATURE_COLUMNS]
    adjacency, audits = build_adjacencies(train_filled, FEATURE_COLUMNS, VAR_TYPES, graph_types)

    return StateGeneralizationTensors(
        scenario=scenario,
        fold=fold,
        train=_make_split(train_filled, train_flags, df.iloc[list(train_idx)][LABEL_COLUMN], FEATURE_COLUMNS, vocabs),
        val=_make_split(val_filled, val_flags, df.iloc[list(val_idx)][LABEL_COLUMN], FEATURE_COLUMNS, vocabs),
        comprehensive_test=_make_split(
            comprehensive_test_filled,
            comprehensive_test_flags,
            df.iloc[list(comprehensive_test_idx)][LABEL_COLUMN],
            FEATURE_COLUMNS,
            vocabs,
        ),
        partial_test=_make_split(
            partial_test_filled,
            partial_test_flags,
            df.iloc[list(partial_test_idx)][LABEL_COLUMN],
            FEATURE_COLUMNS,
            vocabs,
        ),
        cat_dims=cat_dims,
        adjacency=adjacency,
        edge_audit=audits,
    )


def build_fold_tensors(
    df: pd.DataFrame,
    fold: int,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    graph_types: Sequence[str],
    include_temporal: bool = False,
) -> FoldTensors:
    train_raw = df.iloc[list(train_idx)][FEATURE_COLUMNS].copy()
    val_raw = df.iloc[list(val_idx)][FEATURE_COLUMNS].copy()
    test_raw = df.iloc[list(test_idx)][FEATURE_COLUMNS].copy()

    train_filled, val_filled, test_filled, train_flags, val_flags, test_flags = (
        _fit_impute_and_flags(train_raw, val_raw, test_raw, FEATURE_COLUMNS)
    )
    vocabs = _fit_vocabs(train_filled, FEATURE_COLUMNS)
    cat_dims = [len(vocabs[col]) + 1 for col in FEATURE_COLUMNS]

    fold_tensors = FoldTensors(
        fold=fold,
        train=_make_split(train_filled, train_flags, df.iloc[list(train_idx)][LABEL_COLUMN], FEATURE_COLUMNS, vocabs),
        val=_make_split(val_filled, val_flags, df.iloc[list(val_idx)][LABEL_COLUMN], FEATURE_COLUMNS, vocabs),
        test=_make_split(test_filled, test_flags, df.iloc[list(test_idx)][LABEL_COLUMN], FEATURE_COLUMNS, vocabs),
        cat_dims=cat_dims,
        adjacency={},
        edge_audit={},
    )

    adjacency, audits = build_adjacencies(train_filled, FEATURE_COLUMNS, VAR_TYPES, graph_types)
    object.__setattr__(fold_tensors, "adjacency", adjacency)
    object.__setattr__(fold_tensors, "edge_audit", audits)

    if include_temporal:
        temporal = _build_temporal_parts(
            df=df,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            graph_types=graph_types,
            train_filled=train_filled,
            val_filled=val_filled,
            test_filled=test_filled,
            train_flags=train_flags,
            val_flags=val_flags,
            test_flags=test_flags,
        )
        for name, value in temporal.items():
            object.__setattr__(fold_tensors, name, value)

    return fold_tensors


def save_edge_audits(fold_data: FoldTensors, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for graph_type, audit in fold_data.edge_audit.items():
        audit.to_csv(output_dir / f"edge_audit_fold{fold_data.fold}_{graph_type}.csv", index=False)
    if fold_data.temporal_edge_audit:
        for graph_type, audit in fold_data.temporal_edge_audit.items():
            audit.to_csv(output_dir / f"edge_audit_fold{fold_data.fold}_tgcn_{graph_type}.csv", index=False)


def build_adjacencies(
    train_filled: pd.DataFrame,
    node_columns: Sequence[str],
    var_types: Mapping[str, str],
    graph_types: Sequence[str],
    topk: int = 3,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, pd.DataFrame]]:
    adjacency: Dict[str, torch.Tensor] = {}
    audits: Dict[str, pd.DataFrame] = {}
    n = len(node_columns)

    for graph_type in graph_types:
        if graph_type == "fully_connected":
            adj = np.ones((n, n), dtype=np.float32)
            np.fill_diagonal(adj, 0.0)
            audit_rows = [
                {"source": node_columns[i], "target": node_columns[j], "weight": 1.0}
                for i in range(n)
                for j in range(n)
                if i != j
            ]
        elif graph_type == "statistical":
            cols, assoc = build_association_matrix(train_filled, node_columns, var_types)
            adj, audit_rows = topk_adjacency(assoc, cols, topk=topk)
        else:
            raise ValueError(f"Unknown graph_type: {graph_type}")

        adjacency[graph_type] = torch.tensor(adj, dtype=torch.float32)
        audits[graph_type] = pd.DataFrame(audit_rows)

    return adjacency, audits


def build_association_matrix(
    train_filled: pd.DataFrame,
    node_columns: Sequence[str],
    var_types: Mapping[str, str],
) -> Tuple[List[str], np.ndarray]:
    cols = [c for c in node_columns if c in var_types]
    assoc = np.zeros((len(cols), len(cols)), dtype=np.float64)

    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            b = cols[j]
            ta, tb = var_types[a], var_types[b]
            if ta == "nominal" and tb == "nominal":
                weight = cramers_v(train_filled[a], train_filled[b])
            elif ta == "ordinal" and tb == "ordinal":
                weight = spearman_abs(train_filled[a], train_filled[b])
            elif ta == "ordinal":
                weight = correlation_ratio(train_filled[a], train_filled[b])
            else:
                weight = correlation_ratio(train_filled[b], train_filled[a])
            assoc[i, j] = assoc[j, i] = 0.0 if not np.isfinite(weight) else float(weight)

    return cols, assoc


def topk_adjacency(
    assoc: np.ndarray,
    node_columns: Sequence[str],
    topk: int = 3,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    n = len(node_columns)
    undirected_edges: set[Tuple[int, int]] = set()
    for i in range(n):
        ranked = np.argsort(-assoc[i])
        count = 0
        for j in ranked:
            if i == j or assoc[i, j] <= 0:
                continue
            undirected_edges.add(tuple(sorted((i, j))))
            count += 1
            if count >= topk:
                break

    adj = np.zeros((n, n), dtype=np.float32)
    rows: List[Dict[str, object]] = []
    for i, j in sorted(undirected_edges):
        weight = float(assoc[i, j])
        adj[i, j] = adj[j, i] = weight
        rows.append({"source": node_columns[i], "target": node_columns[j], "weight": weight})

    return adj, rows


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x, y)
    if table.empty or min(table.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    if n <= 1:
        return 0.0
    phi2 = chi2 / n
    r, k = table.shape
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return math.sqrt(phi2_corr / denom) if denom > 0 else 0.0


def spearman_abs(x: pd.Series, y: pd.Series) -> float:
    corr = pd.to_numeric(x, errors="coerce").corr(pd.to_numeric(y, errors="coerce"), method="spearman")
    return abs(float(corr)) if np.isfinite(corr) else 0.0


def correlation_ratio(ordinal: pd.Series, nominal: pd.Series) -> float:
    values = pd.to_numeric(ordinal, errors="coerce").to_numpy(dtype=np.float64)
    groups = nominal.to_numpy()
    valid = np.isfinite(values)
    values = values[valid]
    groups = groups[valid]
    if values.size == 0:
        return 0.0
    grand_mean = values.mean()
    denom = np.square(values - grand_mean).sum()
    if denom <= 0:
        return 0.0
    numerator = 0.0
    for group in pd.unique(groups):
        group_values = values[groups == group]
        if group_values.size:
            numerator += group_values.size * float((group_values.mean() - grand_mean) ** 2)
    return math.sqrt(numerator / denom)


def _fit_impute_and_flags(
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    columns: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_flags = (train_raw[columns] == MISSING_CODE).astype(np.float32)
    val_flags = (val_raw[columns] == MISSING_CODE).astype(np.float32)
    test_flags = (test_raw[columns] == MISSING_CODE).astype(np.float32)

    fill_values = {}
    for col in columns:
        nonmissing = train_raw.loc[train_raw[col] != MISSING_CODE, col]
        fill_values[col] = nonmissing.mode(dropna=True).iloc[0] if not nonmissing.empty else MISSING_CODE

    def fill(df: pd.DataFrame) -> pd.DataFrame:
        out = df[columns].copy()
        for col, value in fill_values.items():
            out.loc[out[col] == MISSING_CODE, col] = value
        return out

    return fill(train_raw), fill(val_raw), fill(test_raw), train_flags, val_flags, test_flags


def _fit_impute_and_flags_many(
    train_raw: pd.DataFrame,
    split_raws: Sequence[pd.DataFrame],
    columns: Sequence[str],
) -> Tuple[pd.DataFrame, List[pd.DataFrame], pd.DataFrame, List[pd.DataFrame]]:
    train_flags = (train_raw[columns] == MISSING_CODE).astype(np.float32)
    split_flags = [(split_raw[columns] == MISSING_CODE).astype(np.float32) for split_raw in split_raws]

    fill_values = {}
    for col in columns:
        nonmissing = train_raw.loc[train_raw[col] != MISSING_CODE, col]
        fill_values[col] = nonmissing.mode(dropna=True).iloc[0] if not nonmissing.empty else MISSING_CODE

    def fill(df: pd.DataFrame) -> pd.DataFrame:
        out = df[columns].copy()
        for col, value in fill_values.items():
            out.loc[out[col] == MISSING_CODE, col] = value
        return out

    return fill(train_raw), [fill(split_raw) for split_raw in split_raws], train_flags, split_flags


def _fit_vocabs(df: pd.DataFrame, columns: Sequence[str]) -> Dict[str, Dict[object, int]]:
    vocabs: Dict[str, Dict[object, int]] = {}
    for col in columns:
        values = pd.unique(df[col])
        values = sorted(values.tolist())
        vocabs[col] = {value: idx + 1 for idx, value in enumerate(values)}
    return vocabs


def _make_split(
    filled: pd.DataFrame,
    flags: pd.DataFrame,
    y: pd.Series,
    columns: Sequence[str],
    vocabs: Mapping[str, Mapping[object, int]],
) -> TensorSplit:
    x_id = np.zeros((len(filled), len(columns)), dtype=np.int64)
    for col_idx, col in enumerate(columns):
        mapping = vocabs[col]
        x_id[:, col_idx] = filled[col].map(mapping).fillna(0).to_numpy(dtype=np.int64)

    x_missing = flags[columns].to_numpy(dtype=np.float32)[..., None]
    return TensorSplit(
        x_id=torch.from_numpy(x_id),
        x_missing=torch.from_numpy(x_missing),
        y=torch.tensor(y.to_numpy(dtype=np.int64), dtype=torch.long),
    )


def _build_temporal_parts(
    df: pd.DataFrame,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    graph_types: Sequence[str],
    train_filled: pd.DataFrame,
    val_filled: pd.DataFrame,
    test_filled: pd.DataFrame,
    train_flags: pd.DataFrame,
    val_flags: pd.DataFrame,
    test_flags: pd.DataFrame,
) -> Dict[str, object]:
    temporal_train_filled = _make_temporal_frame(train_filled, timestep=0)
    temporal_train_filled_t2 = _make_temporal_frame(train_filled, timestep=1)
    stacked_train = pd.concat([temporal_train_filled, temporal_train_filled_t2], axis=0, ignore_index=True)

    temporal_vocabs = _fit_vocabs(stacked_train, TEMPORAL_NODE_COLUMNS)
    temporal_cat_dims = [len(temporal_vocabs[col]) + 1 for col in TEMPORAL_NODE_COLUMNS]

    temporal_var_types = {col: VAR_TYPES[col] for col in TEMPORAL_NODE_COLUMNS}
    temporal_adjacency, temporal_audits = build_adjacencies(
        stacked_train,
        TEMPORAL_NODE_COLUMNS,
        temporal_var_types,
        graph_types,
    )

    labels_train = df.iloc[list(train_idx)][LABEL_COLUMN]
    labels_val = df.iloc[list(val_idx)][LABEL_COLUMN]
    labels_test = df.iloc[list(test_idx)][LABEL_COLUMN]

    return {
        "temporal_train": _make_temporal_split(train_filled, train_flags, labels_train, temporal_vocabs),
        "temporal_val": _make_temporal_split(val_filled, val_flags, labels_val, temporal_vocabs),
        "temporal_test": _make_temporal_split(test_filled, test_flags, labels_test, temporal_vocabs),
        "temporal_cat_dims": temporal_cat_dims,
        "temporal_adjacency": temporal_adjacency,
        "temporal_edge_audit": temporal_audits,
    }


def _make_temporal_frame(filled: pd.DataFrame, timestep: int) -> pd.DataFrame:
    data = {}
    for col in TEMPORAL_NODE_COLUMNS:
        discharge_col = TEMPORAL_DYNAMIC_PAIRS.get(col)
        if timestep == 1 and discharge_col is not None:
            data[col] = filled[discharge_col].to_numpy()
        else:
            data[col] = filled[col].to_numpy()
    return pd.DataFrame(data, index=filled.index)


def _make_temporal_flags(flags: pd.DataFrame, timestep: int) -> pd.DataFrame:
    data = {}
    for col in TEMPORAL_NODE_COLUMNS:
        discharge_col = TEMPORAL_DYNAMIC_PAIRS.get(col)
        if timestep == 1 and discharge_col is not None:
            data[col] = flags[discharge_col].to_numpy()
        else:
            data[col] = flags[col].to_numpy()
    return pd.DataFrame(data, index=flags.index)


def _make_temporal_split(
    filled: pd.DataFrame,
    flags: pd.DataFrame,
    y: pd.Series,
    vocabs: Mapping[str, Mapping[object, int]],
) -> TensorSplit:
    frames = [_make_temporal_frame(filled, timestep=0), _make_temporal_frame(filled, timestep=1)]
    flag_frames = [_make_temporal_flags(flags, timestep=0), _make_temporal_flags(flags, timestep=1)]
    x_ids = []
    x_flags = []
    for frame, flag_frame in zip(frames, flag_frames):
        split = _make_split(frame, flag_frame, y, TEMPORAL_NODE_COLUMNS, vocabs)
        x_ids.append(split.x_id)
        x_flags.append(split.x_missing)
    return TensorSplit(
        x_id=torch.stack(x_ids, dim=1),
        x_missing=torch.stack(x_flags, dim=1),
        y=torch.tensor(y.to_numpy(dtype=np.int64), dtype=torch.long),
    )
