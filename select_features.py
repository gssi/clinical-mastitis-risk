"""
Select robust predictive features through a cross-model explainability consensus.

This script combines feature-importance evidence from multiple machine learning
models to identify a compact and stable feature subset for downstream modeling.
For each model, it reads two complementary explainability reports:

1. permutation-importance rankings, which estimate how much validation
   performance decreases when a feature is perturbed;
2. SHAP global-importance rankings, which estimate the average contribution of
   each feature to model predictions.

The script also uses SHAP-derived monotonicity information and interaction
partners to support features that may not be top-ranked in both views but still
show consistent predictive evidence.

Main inputs:
    perm_csvs:
        List of permutation-importance CSV files, one for each trained model.
    shap_csvs:
        List of SHAP global-importance CSV files, one for each trained model.
    model_names:
        Optional names associated with the input models.
    output_csv:
        Path where the final consensus feature-selection table is saved.
    details_dir:
        Optional directory where per-model detailed consensus matrices are
        saved.
    core_top_k:
        Rank threshold used to assign core status when a feature is highly
        ranked in both permutation and SHAP views.
    support_top_k:
        Rank threshold used for single-view support and for defining the
        interaction top-feature pool.
    min_interactions_support:
        Minimum number of interactions with top-ranked features required to
        support a feature.
    min_core_models:
        Minimum number of models that must label a feature as core for final
        core selection.
    min_support_models:
        Minimum number of models that must label a feature as core or
        supportive for final selection.
    spearman_monotonicity_thr:
        Threshold used to identify clear monotonic directional agreement
        between feature values and SHAP contributions.

Main process:
    For each model, the script standardizes the permutation and SHAP reports,
    ranks features in both views, detects monotonic SHAP directionality, counts
    interactions with top-ranked features, and assigns each feature a per-model
    label: core, supportive, or peripheral. The per-model labels are then
    aggregated by voting across models to determine the final consensus label
    and whether each feature should be selected.

Main outputs:
    A CSV file containing one row per feature with:
        feature:
            Feature name.
        to_select:
            Whether the feature is retained in the consensus subset.
        label:
            Final consensus label: core, support, or peripheral.
        core_votes:
            Number of models labeling the feature as core.
        supportive_votes:
            Number of models labeling the feature as supportive.
        peripheral_votes:
            Number of models labeling the feature as peripheral.

    Optional per-model consensus matrices describing ranks, monotonicity,
    interaction support, model-level labels, and labeling reasons.

The resulting feature subset is intended to capture stable mammary pathology
risk signals across models and explanation views, reducing dependence on a
single classifier or a single importance estimator.
"""


from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Iterable
import pandas as pd

# Support functions

def normalize_name(name: str) -> str:
    """Normalize a column name into a lowercase snake-like format."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")

def flatten_cli_list(values: list[str]) -> list[str]:
    """Flatten CLI values by splitting comma-separated items."""
    out = []
    for value in values:
        out.extend([item.strip() for item in str(value).split(",") if item.strip()])
    return out

def pick_column(columns: Iterable[str], candidates: list[str], what: str) -> str:
    """Return the first matching column name from a candidate list."""
    cols = list(columns)
    for candidate in candidates:
        if candidate in cols:
            del cols
            return candidate
    raise ValueError(f"Missing required column for {what}. Available columns: {cols}")

def read_header(csv_path: str) -> list[str]:
    """Read only the CSV header and return normalized column names."""
    raw = pd.read_csv(csv_path, nrows=0)
    normalized_columns = [normalize_name(col) for col in raw.columns]
    del raw
    return normalized_columns

def read_csv_with_columns(csv_path: str, usecols_norm: set[str]) -> pd.DataFrame:
    """Read only selected columns from a CSV using normalized names."""
    raw = pd.read_csv(csv_path, nrows=0)
    orig_cols = list(raw.columns)
    norm_cols = [normalize_name(col) for col in orig_cols]
    keep = [orig for orig, norm in zip(orig_cols, norm_cols) if norm in usecols_norm]
    del raw, norm_cols
    df = pd.read_csv(csv_path, usecols=keep)
    df.columns = [normalize_name(col) for col in df.columns]
    del orig_cols, keep
    return df

def split_partners(value: object) -> list[str]:
    """Split a comma-separated interaction partner string into a list."""
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]

def sign_from_spearman(value: float, monotonicity_thr: float) -> str:
    """Map a Spearman value to pos, neg, or zero based on a threshold."""
    if pd.isna(value) or abs(value) <= monotonicity_thr:
        return "zero"
    return "pos" if value > 0 else "neg"

def read_permutation_csv(csv_path: str) -> pd.DataFrame:
    """Read and standardize the permutation importance CSV."""
    # Read and identify the required columns
    cols = read_header(csv_path)
    feature_col = pick_column(cols, ["feature"], "feature")
    score_col = pick_column(cols, ["perm_importance_norm", "perm_norm", "perm_importance_mean", "perm_mean"], "permutation score")
    partners_col = pick_column(cols, ["top_interaction_partners", "interaction_partners"], "interaction partners")
    del cols
    # Load and standardize the selected columns
    df = read_csv_with_columns(csv_path, {feature_col, score_col, partners_col}).copy()
    df = df.rename(columns={ feature_col: "feature", score_col: "perm_score", partners_col: "interaction_partners"})
    del feature_col, score_col, partners_col
    # Clean and derive ranking information
    df["feature"] = df["feature"].astype(str).str.strip()
    df["perm_score"] = pd.to_numeric(df["perm_score"], errors="coerce").fillna(0.0)
    df["interaction_partners"] = df["interaction_partners"].map(split_partners)
    df = df.drop_duplicates(subset=["feature"], keep="first")
    df["rank_perm"] = df["perm_score"].rank(method="dense", ascending=False).astype(int)
    return df[["feature", "perm_score", "rank_perm", "interaction_partners"]]

def read_shap_csv(csv_path: str, spearman_monotonicity_thr: float) -> pd.DataFrame:
    """Read and standardize the SHAP global importance CSV."""
    # Read and identify the required columns
    cols = read_header(csv_path)
    feature_col = pick_column(cols, ["feature"], "feature")
    shap_col = pick_column(cols, ["mean_abs_shap", "shap_mean_abs", "mean_abs"], "mean_abs_shap")
    spearman_col = pick_column(cols, ["spearman_feature_vs_shap", "spearman", "spearman_corr"], "spearman")
    del cols
    # Load and standardize the selected columns
    df = read_csv_with_columns(csv_path, {feature_col, shap_col, spearman_col}).copy()
    df = df.rename(
        columns={feature_col: "feature", shap_col: "mean_abs_shap", spearman_col: "spearman"})
    del feature_col, shap_col, spearman_col
    # Clean and derive ranking and monotonicity information
    df["feature"] = df["feature"].astype(str).str.strip()
    df["mean_abs_shap"] = pd.to_numeric(df["mean_abs_shap"], errors="coerce").fillna(0.0)
    df["spearman"] = pd.to_numeric(df["spearman"], errors="coerce").fillna(0.0)
    df = df.drop_duplicates(subset=["feature"], keep="first")
    df["rank_shap"] = df["mean_abs_shap"].rank(method="dense", ascending=False).astype(int)
    df["spearman_abs"] = df["spearman"].abs()
    df["spearman_sign"] = df["spearman"].map(lambda x: sign_from_spearman(x, spearman_monotonicity_thr))
    return df[["feature", "mean_abs_shap", "rank_shap", "spearman", "spearman_abs", "spearman_sign"]]

def count_intersections(partners: list[str], top_pool: set[str], top_k: int) -> int:
    """Count how many top-k partners belong to the selected top pool."""
    if not partners:
        return 0
    return sum(1 for partner in partners[:top_k] if partner in top_pool)

def validate_thresholds(n_models: int, core_top_k: int, support_top_k: int, min_interactions_support: int, min_core_models: int, min_support_models: int,
                        spearman_monotonicity_thr: float) -> None:
    """Validate all threshold values before running the selection."""
    if n_models <= 0:
        raise ValueError("At least one model is required.")
    if core_top_k <= 0:
        raise ValueError("core_top_k must be > 0.")
    if support_top_k <= 0:
        raise ValueError("support_top_k must be > 0.")
    if support_top_k > core_top_k:
        raise ValueError("support_top_k should be <= core_top_k.")
    if min_interactions_support <= 0:
        raise ValueError("min_interactions_support must be > 0.")
    if min_core_models <= 0 or min_core_models > n_models:
        raise ValueError("min_core_models must be between 1 and the number of models.")
    if min_support_models <= 0 or min_support_models > n_models:
        raise ValueError("min_support_models must be between 1 and the number of models.")
    if min_core_models > min_support_models:
        raise ValueError("min_core_models should be <= min_support_models.")
    if not (0.0 <= spearman_monotonicity_thr <= 1.0):
        raise ValueError("spearman_monotonicity_thr must be in [0,1].")

def label_feature(rank_perm: int, rank_shap: int, n_interactions_with_top: int, spearman_sign: str, core_top_k: int, support_top_k: int, 
                  min_interactions_support: int) -> tuple[str, str]:
    """Assign a per-model label and the reason behind it."""
    if rank_perm <= core_top_k and rank_shap <= core_top_k:
        return "core", "two_view_rank_agreement"
    if rank_perm <= support_top_k or rank_shap <= support_top_k:
        return "supportive", "single_view_rank"
    if n_interactions_with_top >= min_interactions_support:
        return "supportive", "interaction_support"
    if (rank_perm <= core_top_k or rank_shap <= core_top_k) and spearman_sign != "zero":
        return "supportive", "monotonic_direction_support"
    return "peripheral", "weak_support"

def build_model_matrix(model_name: str, perm_csv: str, shap_csv: str, core_top_k: int, support_top_k: int, min_interactions_support: int,
                       spearman_monotonicity_thr: float) -> pd.DataFrame:
    """Build the per-model consensus matrix from permutation and SHAP files."""
    # Read the two explainability sources
    perm_df = read_permutation_csv(perm_csv)
    shap_df = read_shap_csv(shap_csv, spearman_monotonicity_thr)
    # Build the top-feature pool used for interaction support
    top_perm = set(perm_df.nsmallest(support_top_k, "rank_perm")["feature"].tolist())
    top_shap = set(shap_df.nsmallest(support_top_k, "rank_shap")["feature"].tolist())
    top_pool = top_perm | top_shap
    del top_perm, top_shap
    # Count how many top interaction partners each feature has
    perm_df = perm_df.copy()
    perm_df["n_interactions_with_top"] = perm_df["interaction_partners"].map(lambda x: count_intersections(x, top_pool, support_top_k))
    del top_pool
    # Merge the two views into a single per-model matrix
    merged = perm_df[["feature", "rank_perm", "n_interactions_with_top"]].merge(shap_df[["feature", "rank_shap", "spearman_sign", "spearman_abs"]], how="outer",
                                                                                on="feature")
    worst_perm = (int(perm_df["rank_perm"].max()) if not perm_df.empty else 0) + 1
    worst_shap = (int(shap_df["rank_shap"].max()) if not shap_df.empty else 0) + 1
    del perm_df, shap_df
    # Fill missing values and standardize types
    merged["rank_perm"] = pd.to_numeric(merged["rank_perm"], errors="coerce").fillna(worst_perm).astype(int)
    merged["rank_shap"] = pd.to_numeric(merged["rank_shap"], errors="coerce").fillna(worst_shap).astype(int)
    merged["n_interactions_with_top"] = pd.to_numeric(merged["n_interactions_with_top"], errors="coerce").fillna(0).astype(int)
    merged["spearman_sign"] = merged["spearman_sign"].fillna("zero")
    merged["spearman_abs"] = pd.to_numeric(merged["spearman_abs"], errors="coerce").fillna(0.0)
    merged["model"] = model_name
    # Assign the model-level label and the associated reason
    labels = merged.apply(
        lambda row: label_feature(int(row["rank_perm"]), int(row["rank_shap"]), int(row["n_interactions_with_top"]), str(row["spearman_sign"]),
                                  core_top_k, support_top_k, min_interactions_support), axis=1)
    merged["model_label"] = labels.map(lambda x: x[0])
    merged["label_reason"] = labels.map(lambda x: x[1])
    del labels, worst_perm, worst_shap
    return (merged[["feature", "model", "rank_perm", "rank_shap", "spearman_sign", "spearman_abs", "n_interactions_with_top", "model_label",
                    "label_reason"]].sort_values(["model_label", "rank_perm", "rank_shap", "feature"], kind="stable").reset_index(drop=True))

def vote_final_labels(model_matrices: list[pd.DataFrame], model_names: list[str], min_core_models: int, min_support_models: int) -> pd.DataFrame:
    """Aggregate per-model labels into a final cross-model decision."""
    # Build a feature-indexed view for each model
    per_model = {(df["model"].iloc[0] if not df.empty else model_name): df.set_index("feature") for df, model_name in zip(model_matrices, model_names)}
    # Collect all features observed across the input models
    all_features = sorted(set().union(*(set(df["feature"].tolist()) for df in model_matrices)))
    rows = []
    # Vote the final label feature by feature
    for feature in all_features:
        labels = []
        for model in model_names:
            if model in per_model and feature in per_model[model].index:
                labels.append(str(per_model[model].at[feature, "model_label"]))
            else:
                labels.append("peripheral")
        core_count = sum(1 for label in labels if label == "core")
        support_count = sum(1 for label in labels if label == "supportive")
        peripheral_count = sum(1 for label in labels if label == "peripheral")
        del labels
        if core_count >= min_core_models:
            final_label = "core"
            to_select = "yes"
        elif core_count + support_count >= min_support_models:
            final_label = "support"
            to_select = "yes"
        else:
            final_label = "peripheral"
            to_select = "no"
        rows.append({"feature": feature, "to_select": to_select, "label": final_label, "core_votes": core_count, "supportive_votes": support_count, 
                     "peripheral_votes": peripheral_count})
    out = pd.DataFrame(rows)
    del rows, all_features, per_model
    # Sort output so selected and stronger labels appear first
    label_order = {"core": 0, "support": 1, "peripheral": 2}
    out["_sel"] = out["to_select"].map({"yes": 0, "no": 1})
    out["_lab"] = out["label"].map(label_order)
    out = out.sort_values(["_sel", "_lab", "feature"], kind="stable").drop(columns=["_sel", "_lab"]).reset_index(drop=True)
    del label_order
    return out

# Parsing

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Select features by cross-model consensus from permutation and SHAP CSV files.")
    # Define input files and output paths
    parser.add_argument("--perm-csvs", nargs="+", required=True, help="List of permutation-ranking CSV files, one per model.")
    parser.add_argument("--shap-csvs", nargs="+", required=True, help="List of SHAP-global CSV files, one per model.")
    parser.add_argument("--model-names", nargs="+", default=None, help="Optional model names. If omitted, names are auto-generated.")
    parser.add_argument("--output-csv", required=True, help="Path to the final output CSV.")
    parser.add_argument("--details-dir", default=None, help="Optional directory where per-model detailed matrices are saved.")
    # Define consensus thresholds
    parser.add_argument("--core-top-k", type=int, default=10, help="Core if a feature is in the top-k of both permutation and SHAP rankings.")
    parser.add_argument("--support-top-k", type=int, default=3, help="Support threshold for single-view evidence and size of the top-feature interaction pool.")
    parser.add_argument("--min-interactions-support", type=int, default=2, help="Minimum number of top-pool interaction partners needed for supportive status.")
    parser.add_argument("--min-core-models", type=int, default=2, help="Minimum number of models labeling a feature as core for final yes/core.")
    parser.add_argument("--min-support-models", type=int, default=2, help="Minimum number of models labeling a feature as core or supportive for final yes/support.")
    parser.add_argument("--spearman-monotonicity-thr", type=float, default=0.5, help="If |Spearman| is above this threshold, the feature is treated as having a clear monotonic directional signal.")
    return parser.parse_args()

# Main

def main() -> None:
    """Run the feature selection workflow from CLI inputs."""
    args = parse_args()
    # Read and validate the main input lists
    perm_csvs = flatten_cli_list(args.perm_csvs)
    shap_csvs = flatten_cli_list(args.shap_csvs)
    if len(perm_csvs) != len(shap_csvs):
        raise ValueError(f"--perm-csvs and --shap-csvs must have the same length. Got {len(perm_csvs)} and {len(shap_csvs)}.")
    # Resolve model names
    if args.model_names is None:
        model_names = [f"model_{i + 1}" for i in range(len(perm_csvs))]
    else:
        model_names = flatten_cli_list(args.model_names)
        if len(model_names) != len(perm_csvs):
            raise ValueError(f"--model-names must have the same length as input CSV lists. Got {len(model_names)} and {len(perm_csvs)}.")
    # Validate thresholds before processing data
    validate_thresholds(len(model_names), args.core_top_k, args.support_top_k, args.min_interactions_support, args.min_core_models, args.min_support_models,
                        args.spearman_monotonicity_thr)
    # Build one consensus matrix for each model
    model_matrices = []
    for model_name, perm_csv, shap_csv in zip(model_names, perm_csvs, shap_csvs):
        model_matrix = build_model_matrix(model_name=model_name, perm_csv=perm_csv, shap_csv=shap_csv, core_top_k=args.core_top_k, support_top_k=args.support_top_k,
                                          min_interactions_support=args.min_interactions_support, spearman_monotonicity_thr=args.spearman_monotonicity_thr)
        model_matrices.append(model_matrix)
        del model_matrix
    # Compute the final voted feature labels
    final_df = vote_final_labels(model_matrices=model_matrices, model_names=model_names, min_core_models=args.min_core_models, min_support_models=args.min_support_models)
    # Save the main output file
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_path, index=False)
    # Optionally save one detailed matrix per model
    if args.details_dir is not None:
        details_dir = Path(args.details_dir)
        details_dir.mkdir(parents=True, exist_ok=True)
        for df, model_name in zip(model_matrices, model_names):
            df.to_csv(details_dir / f"{model_name}_consensus_matrix.csv", index=False)
        del details_dir
    # Free references that are no longer needed
    del args, perm_csvs, shap_csvs, model_names, model_matrices, final_df, out_path

if __name__ == "__main__":
    main()


"""
python3 select_features.py \
  --perm-csvs workspace2/models/cat_run/cat_feature_ranking.csv workspace2/models/xgb_run/xgb_feature_ranking.csv workspace2/models/lgbm_run/lgbm_feature_ranking.csv \
  --shap-csvs workspace2/models/cat_run/cat_shap_global.csv workspace2/models/xgb_run/xgb_shap_global.csv workspace2/models/lgbm_run/lgbm_shap_global.csv \
  --model-names cat xgb lgbm \
  --output-csv workspace2/artifacts/selected_features.csv
"""