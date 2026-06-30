from libraries import pd, Path, log, gc, json
import argparse

# Global variables

DEFAULT_STRATIFY_COLS = ["age", "lactation_phase", "season"]

# Support functions

def parse_csv_list(values: list[str] | tuple[str, ...] | str | None) -> list[str]:
    """Parse comma-separated CLI values into a clean list."""
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return [str(item).strip() for item in values if str(item).strip()]
    return [item.strip() for item in str(values).split(",") if item.strip()]

def save_json(obj: dict, path: Path) -> Path:
    """Save a dictionary as a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

def parse_pattern(pattern: str) -> list[int]:
    """Parse a binary temporal pattern written as comma-separated values."""
    values = [int(item.strip()) for item in str(pattern).split(",") if item.strip()]
    if not values:
        raise ValueError("Pattern must contain at least one binary value.")
    if any(value not in {0, 1} for value in values):
        raise ValueError(f"Invalid pattern '{pattern}'. Pattern values must be binary 0/1.")
    return values

def parse_class_ratio(class_ratio: str) -> tuple[int, int]:
    """Parse a positive:negative class ratio such as 1:1 or 1:5."""
    parts = [item.strip() for item in str(class_ratio).split(":")]
    if len(parts) != 2:
        raise ValueError(f"Invalid class ratio '{class_ratio}'. Expected format like '1:1' or '1:5'.")
    pos_units = int(parts[0])
    neg_units = int(parts[1])
    if pos_units <= 0 or neg_units <= 0:
        raise ValueError(f"Invalid class ratio '{class_ratio}'. Ratio units must be positive integers.")
    if neg_units < pos_units:
        raise ValueError(f"Invalid class ratio '{class_ratio}'. Negative units must be >= positive units.")
    return pos_units, neg_units

def infer_lag_steps_from_pattern(pattern: list[int]) -> list[int]:
    """Infer lag steps from a temporal pattern length."""
    return list(range(1, len(pattern)))

def validate_patterns(positive_pattern: list[int], negative_pattern: list[int]) -> list[int]:
    """Validate that positive and negative patterns have the same temporal length."""
    if len(positive_pattern) != len(negative_pattern):
        raise ValueError("positive_pattern and negative_pattern must have the same length.")
    return infer_lag_steps_from_pattern(positive_pattern)

def validate_required_columns(df: pd.DataFrame, target_col: str, healthy_col: str, id_col: str, year_col: str, month_col: str, 
                              lag_steps: list[int], stratify_cols: list[str]) -> None:
    """Validate that required columns exist in the validated anchor dataset."""
    required = [id_col, year_col, month_col, target_col, healthy_col]
    required.extend([f"{target_col}_t-{lag}" for lag in lag_steps])
    required.extend(stratify_cols)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

def normalize_binary_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Normalize selected columns to binary int8 values."""
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype("int8")
    return out

def build_candidate_masks(df: pd.DataFrame, target_col: str, healthy_col: str, positive_pattern: list[int], negative_pattern: list[int]) -> tuple[pd.Series, pd.Series]:
    """Build discovery candidate masks from target history and healthy profile rules."""
    positive_mask = df[target_col].eq(positive_pattern[0])
    negative_mask = df[target_col].eq(negative_pattern[0])
    for lag in range(1, len(positive_pattern)):
        lag_col = f"{target_col}_t-{lag}"
        positive_mask &= df[lag_col].eq(positive_pattern[lag])
        negative_mask &= df[lag_col].eq(negative_pattern[lag])
    positive_mask &= df[healthy_col].eq(0)
    negative_mask &= df[healthy_col].eq(1)
    return positive_mask, negative_mask

def build_stratum_key(df: pd.DataFrame, stratify_cols: list[str]) -> pd.Series:
    """Build a unified stratification key from the requested biological strata."""
    if not stratify_cols:
        return pd.Series([("__all__",)] * len(df), index=df.index)
    safe = df[stratify_cols].copy()
    for column in stratify_cols:
        safe[column] = safe[column].astype("string")
    return pd.Series(list(map(tuple, safe.to_numpy())), index=df.index)

def sample_exact_ratio_by_stratum(positives: pd.DataFrame, negatives: pd.DataFrame, class_ratio: str, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Sample an exact positive:negative ratio independently inside each shared stratum."""
    pos_units, neg_units = parse_class_ratio(class_ratio)
    pos_counts = positives["__stratum"].value_counts().to_dict()
    neg_counts = negatives["__stratum"].value_counts().to_dict()
    overlap_strata = sorted(set(pos_counts).intersection(set(neg_counts)))
    if not overlap_strata:
        raise ValueError("No overlapping strata between positive and negative candidates.")
    pos_samples = []
    neg_samples = []
    insufficient_overlap_strata = 0
    for stratum in overlap_strata:
        n_blocks = min(pos_counts[stratum] // pos_units, neg_counts[stratum] // neg_units)
        if n_blocks <= 0:
            insufficient_overlap_strata += 1
            continue
        n_pos = n_blocks * pos_units
        n_neg = n_blocks * neg_units
        pos_current = positives.loc[positives["__stratum"] == stratum]
        neg_current = negatives.loc[negatives["__stratum"] == stratum]
        pos_samples.append(pos_current.sample(n=n_pos, replace=False, random_state=random_state))
        neg_samples.append(neg_current.sample(n=n_neg, replace=False, random_state=random_state))
    if not pos_samples or not neg_samples:
        raise ValueError("Sampling failed: no shared stratum retained for the requested ratio.")
    pos_final = pd.concat(pos_samples, axis=0, ignore_index=True)
    neg_final = pd.concat(neg_samples, axis=0, ignore_index=True)
    report = {
        "positive_strata": len(pos_counts),
        "negative_strata": len(neg_counts),
        "overlap_strata": len(overlap_strata),
        "dropped_positive_only_strata": len(set(pos_counts) - set(overlap_strata)),
        "dropped_negative_only_strata": len(set(neg_counts) - set(overlap_strata)),
        "dropped_insufficient_overlap_strata": int(insufficient_overlap_strata),
    }
    return pos_final, neg_final, report

def create_sampled_ids(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Extract unique animal identifiers from the sampled anchor dataset."""
    return df[[id_col]].drop_duplicates().rename(columns={id_col: "id"}).reset_index(drop=True)

def run_temporal_sampling(input_path: Path, output_path: Path, output_ids_path: Path, report_path: Path, target_col: str = "disease", healthy_col: str = "healthy",
                          id_col: str = "id", time_cols: tuple[str, str] = ("year", "month"), positive_pattern: str = "1,0,0", negative_pattern: str = "0,0,0",
                          class_ratio: str = "1:1", stratify_cols: list[str] | None = None, random_state: int = 42) -> None:
    """Build a sampled discovery anchor set from the validated anchor artifact."""
    year_col, month_col = time_cols
    requested_stratify_cols = parse_csv_list(stratify_cols) or DEFAULT_STRATIFY_COLS.copy()
    positive_pattern_values = parse_pattern(positive_pattern)
    negative_pattern_values = parse_pattern(negative_pattern)
    lag_steps = validate_patterns(positive_pattern_values, negative_pattern_values)
    parse_class_ratio(class_ratio)
    log.info("Loading validated anchors: %s", input_path)
    df = pd.read_parquet(input_path)
    log.info(
        "Input loaded | rows=%d cols=%d | positive_pattern=%s | negative_pattern=%s | class_ratio=%s",
        len(df),
        df.shape[1],
        positive_pattern,
        negative_pattern,
        class_ratio,
    )
    validate_required_columns(
        df=df,
        target_col=target_col,
        healthy_col=healthy_col,
        id_col=id_col,
        year_col=year_col,
        month_col=month_col,
        lag_steps=lag_steps,
        stratify_cols=requested_stratify_cols,
    )
    binary_cols = [target_col, healthy_col] + [f"{target_col}_t-{lag}" for lag in lag_steps]
    df = normalize_binary_columns(df, binary_cols)
    before = len(df)
    for column in requested_stratify_cols:
        df = df[df[column].notna()].copy()
    log.info("Rows kept after valid stratification filtering: %d -> %d", before, len(df))
    if len(df) == 0:
        raise ValueError("No rows left after valid stratification filtering.")
    positive_mask, negative_mask = build_candidate_masks(
        df=df,
        target_col=target_col,
        healthy_col=healthy_col,
        positive_pattern=positive_pattern_values,
        negative_pattern=negative_pattern_values,
    )
    positives = df.loc[positive_mask].copy()
    negatives = df.loc[negative_mask].copy()
    log.info("Positive candidates: %d", len(positives))
    log.info("Negative candidates: %d", len(negatives))
    log.info("Stratification columns: %s", requested_stratify_cols)
    if len(positives) == 0:
        raise ValueError("No positive candidates found for the requested pattern/profile.")
    if len(negatives) == 0:
        raise ValueError("No negative candidates found for the requested pattern/profile.")
    positives["__stratum"] = build_stratum_key(positives, requested_stratify_cols)
    negatives["__stratum"] = build_stratum_key(negatives, requested_stratify_cols)
    pos_final, neg_final, stratum_report = sample_exact_ratio_by_stratum(
        positives=positives,
        negatives=negatives,
        class_ratio=class_ratio,
        random_state=random_state,
    )
    sampled = pd.concat([pos_final, neg_final], axis=0, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    sampled = sampled.drop(columns=["__stratum"], errors="ignore")
    sampled_ids = create_sampled_ids(sampled, id_col=id_col)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(output_path, index=False)
    output_ids_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_ids.to_parquet(output_ids_path, index=False)
    report = {
        "step": "temporal_sampler",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "output_ids_path": str(output_ids_path),
        "target_column": target_col,
        "healthy_column": healthy_col,
        "id_column": id_col,
        "time_columns": [year_col, month_col],
        "positive_pattern": positive_pattern_values,
        "negative_pattern": negative_pattern_values,
        "lag_steps": lag_steps,
        "class_ratio": class_ratio,
        "stratify_columns": requested_stratify_cols,
        "random_state": int(random_state),
        "input_rows": int(len(df)),
        "positive_candidates": int(len(positives)),
        "negative_candidates": int(len(negatives)),
        "sampled_rows": int(len(sampled)),
        "sampled_positives": int((sampled[target_col] == 1).sum()),
        "sampled_negatives": int((sampled[target_col] == 0).sum()),
        "unique_ids": int(sampled_ids["id"].nunique()),
        **stratum_report,
    }
    save_json(report, report_path)
    log.info(
        "Sampled anchors saved: %s | rows=%d positives=%d negatives=%d unique_ids=%d",
        output_path,
        len(sampled),
        int((sampled[target_col] == 1).sum()),
        int((sampled[target_col] == 0).sum()),
        int(sampled_ids["id"].nunique()),
    )
    log.info("Sampled ids saved: %s", output_ids_path)
    log.info("Sampling report saved: %s", report_path)
    del df, positives, negatives, pos_final, neg_final, sampled, sampled_ids
    gc.collect()
    log.info("Temporal sampling completed successfully.")

# Parsing

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for discovery temporal sampling."""
    parser = argparse.ArgumentParser(description="Build a sampled discovery anchor set from validated temporal anchors.")
    parser.add_argument("--input-path", type=Path, required=True, help="Input validated anchors parquet path.")
    parser.add_argument("--output-path", type=Path, required=True, help="Output sampled anchors parquet path.")
    parser.add_argument("--output-ids-path", type=Path, required=True, help="Output parquet path for sampled unique animal IDs.")
    parser.add_argument("--report-path", type=Path, required=True, help="Output JSON path for the temporal sampling report.")
    parser.add_argument("--target-col", type=str, default="disease", help="Target column.")
    parser.add_argument("--healthy-col", type=str, default="healthy", help="Healthy indicator column.")
    parser.add_argument("--id-col", type=str, default="id", help="Entity identifier column.")
    parser.add_argument("--year-col", type=str, default="year", help="Year column.")
    parser.add_argument("--month-col", type=str, default="month", help="Month column.")
    parser.add_argument("--positive-pattern", type=str, default="1,0,0", help="Positive target pattern written as current,t-1,...")
    parser.add_argument("--negative-pattern", type=str, default="0,0,0", help="Negative target pattern written as current,t-1,...")
    parser.add_argument("--class-ratio", type=str, default="1:1", help="Requested positive:negative ratio, for example 1:1 or 1:5.")
    parser.add_argument("--stratify-cols", type=str, default="age,lactation_phase,season", help="Comma-separated biological strata.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducible sampling.")
    return parser.parse_args()

# Main call

if __name__ == "__main__":
    args = parse_args()
    stratify_cols = parse_csv_list(args.stratify_cols)
    log.info(
        "CLI parsed | input=%s output=%s output_ids=%s report=%s positive_pattern=%s negative_pattern=%s class_ratio=%s stratify_cols=%s",
        args.input_path,
        args.output_path,
        args.output_ids_path,
        args.report_path,
        args.positive_pattern,
        args.negative_pattern,
        args.class_ratio,
        stratify_cols,
    )
    run_temporal_sampling(
        input_path=args.input_path,
        output_path=args.output_path,
        output_ids_path=args.output_ids_path,
        report_path=args.report_path,
        target_col=args.target_col,
        healthy_col=args.healthy_col,
        id_col=args.id_col,
        time_cols=(args.year_col, args.month_col),
        positive_pattern=args.positive_pattern,
        negative_pattern=args.negative_pattern,
        class_ratio=args.class_ratio,
        stratify_cols=stratify_cols,
        random_state=args.random_state,
    )


