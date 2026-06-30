"""
Build validated temporal anchors for short-horizon clinical mammary pathology
risk modeling.

This script takes as input an imputed longitudinal tabular dataset containing
animal-level monthly observations. The expected main inputs are:

- an input Parquet file with at least animal identifier, year, and month columns;
- optional target information, typically the binary `disease` variable;
- optional support variables describing animal status, health condition,
  lactation phase, seasonality, and temporal context;
- lag steps defining the required temporal history, by default t-1 and t-2.

The process constructs a temporal month index, adds previous-month history for
each animal, and keeps only those rows whose lagged observations are available
and strictly consecutive. Each retained row is a valid temporal anchor at time t,
representing an animal-month observation with a complete recent history.

The main outputs are:

- a Parquet file containing the validated temporal anchors;
- a JSON schema describing identifier, time, target, target-history, and support
  columns retained in the anchor dataset;
- a JSON metadata file summarizing the temporal construction step.

The resulting dataset is designed to support downstream machine learning and
deep learning workflows based on coherent three-month temporal windows.
"""

from libraries import pd, Path, log, gc, json
import argparse

# Global variables

DEFAULT_SUPPORT_COLS = ["cf_date", "disease", "healthy", "age", "age_class", "season", "lactation_phase", "months_since_calving", "month_sin", "month_cos"]

DEFAULT_AGE_BINS = [1.5, 3.5, 5.5, 7.5]
DEFAULT_AGE_LABELS = ["2_3", "4_5", "6_7"]

# Support functions

def parse_csv_list(values: list[str] | tuple[str, ...] | str | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return [str(item).strip() for item in values if str(item).strip()]
    return [item.strip() for item in str(values).split(",") if item.strip()]


def save_json(obj: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def month_index(year: pd.Series, month: pd.Series) -> pd.Series:
    y = pd.to_numeric(year, errors="coerce").astype("Int32")
    m = pd.to_numeric(month, errors="coerce").astype("Int32")
    return (y * 12 + m).astype("Int32")


def build_age_class(df: pd.DataFrame, age_col: str = "age", bins: list[float] | None = None, labels: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    if "age_class" in out.columns or age_col not in out.columns:
        return out
    age_num = pd.to_numeric(out[age_col], errors="coerce")
    out["age_class"] = pd.cut(age_num, bins=bins or DEFAULT_AGE_BINS, labels=labels or DEFAULT_AGE_LABELS)
    return out


def validate_required_columns(df: pd.DataFrame, id_col: str, year_col: str, month_col: str) -> None:
    missing = [c for c in [id_col, year_col, month_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def add_temporal_history(df: pd.DataFrame, id_col: str, year_col: str, month_col: str, target_col: str, lag_steps: list[int]) -> pd.DataFrame:
    out = df.copy()
    out[year_col] = pd.to_numeric(out[year_col], errors="coerce").astype("Int16")
    out[month_col] = pd.to_numeric(out[month_col], errors="coerce").astype("Int8")
    out["__month_idx"] = month_index(out[year_col], out[month_col]).astype("Int32")
    out = out.sort_values([id_col, year_col, month_col], kind="mergesort").reset_index(drop=True)
    grouped = out.groupby(id_col, observed=True, sort=False)
    has_target = target_col in out.columns
    for lag in lag_steps:
        out[f"__month_idx_t-{lag}"] = grouped["__month_idx"].shift(lag)
        if has_target:
            out[f"{target_col}_t-{lag}"] = grouped[target_col].shift(lag)
    return out


def build_valid_anchor_mask(df: pd.DataFrame, lag_steps: list[int]) -> pd.Series:
    current_month = pd.to_numeric(df["__month_idx"], errors="coerce")
    valid_mask = current_month.notna().copy()
    for lag in lag_steps:
        lag_month = pd.to_numeric(df[f"__month_idx_t-{lag}"], errors="coerce")
        valid_mask &= lag_month.notna()
        valid_mask &= current_month.sub(lag_month).eq(lag)
    return valid_mask


def build_anchor_schema(df: pd.DataFrame, id_col: str, year_col: str, month_col: str, target_col: str, lag_steps: list[int], support_cols: list[str]) -> dict:
    target_history_cols = [
        f"{target_col}_t-{lag}"
        for lag in lag_steps
        if f"{target_col}_t-{lag}" in df.columns]
    support_present = [c for c in support_cols if c in df.columns]
    return {
        "id_column": id_col,
        "time_columns": [year_col, month_col],
        "target_column": target_col if target_col in df.columns else None,
        "month_index_column": "__month_idx",
        "target_history_columns": target_history_cols,
        "support_columns": support_present,
        "anchor_columns": list(
            dict.fromkeys(
                [id_col, year_col, month_col, "__month_idx"]
                + ([target_col] if target_col in df.columns else [])
                + target_history_cols
                + support_present
            )
        ),
    }


def create_validated_anchors(input_path: Path, output_path: Path, anchor_schema_path: Path, anchor_meta_path: Path, lag_steps: list[int] | None = None, id_col: str = "id", time_cols: tuple[str, str] = ("year", "month"), target_col: str = "disease", support_cols: list[str] | None = None, age_col: str = "age",
                             min_year: int = 2020, keep_month_index: bool = True) -> None:

    lag_steps = sorted(set(lag_steps or [1, 2]))
    if not lag_steps or not all(isinstance(s, int) and s > 0 for s in lag_steps):
        raise ValueError("lag_steps must be positive integers.")
    year_col, month_col = time_cols
    requested_support_cols = parse_csv_list(support_cols) or DEFAULT_SUPPORT_COLS.copy()
    log.info("Loading dataset: %s", input_path)
    df = pd.read_parquet(input_path)
    has_target = target_col in df.columns
    validate_required_columns(df, id_col, year_col, month_col)
    df = build_age_class(df, age_col=age_col)
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df[month_col] = pd.to_numeric(df[month_col], errors="coerce")
    df = df[df[year_col].notna() & df[month_col].notna()].copy()
    df = df[df[year_col] >= min_year].copy()
    if has_target:
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype("int8")
    if "healthy" in df.columns:
        df["healthy"] = pd.to_numeric(df["healthy"], errors="coerce").fillna(0).astype("int8")
    df = add_temporal_history(
        df=df,
        id_col=id_col,
        year_col=year_col,
        month_col=month_col,
        target_col=target_col,
        lag_steps=lag_steps,
    )
    valid_mask = build_valid_anchor_mask(df, lag_steps)
    anchors = df.loc[valid_mask].copy().reset_index(drop=True)
    if len(anchors) == 0:
        raise ValueError("No validated anchors left.")
    support_present = [c for c in requested_support_cols if c in anchors.columns]
    target_history_cols = [
        f"{target_col}_t-{lag}"
        for lag in lag_steps
        if f"{target_col}_t-{lag}" in anchors.columns]
    final_cols = [id_col, year_col, month_col]
    if keep_month_index:
        final_cols.append("__month_idx")
    if has_target:
        final_cols.append(target_col)
    final_cols.extend(target_history_cols)
    final_cols.extend(support_present)
    final_cols = list(dict.fromkeys([c for c in final_cols if c in anchors.columns]))
    anchors = anchors[final_cols].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anchors.to_parquet(output_path, index=False)
    anchor_schema = build_anchor_schema(
        anchors, id_col, year_col, month_col, target_col, lag_steps, support_present
    )
    save_json(anchor_schema, anchor_schema_path)
    anchor_meta = {
        "step": "temporal_anchor_builder",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "lag_steps": lag_steps,
        "target_available": bool(has_target),
        "validated_anchors": int(len(anchors)),
    }
    save_json(anchor_meta, anchor_meta_path)
    del df, anchors
    gc.collect()
    log.info("Temporal anchor construction completed.")

# Parsing

def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description="Build validated temporal anchors from an imputed longitudinal animal-level dataset.")
    parser.add_argument("--input-path", type=Path, required=True, help="Input imputed longitudinal Parquet dataset path.")
    parser.add_argument("--output-path", type=Path, required=True, help="Output validated temporal anchors Parquet path.")
    parser.add_argument("--anchor-schema-path", type=Path, required=True, help="Output JSON path for the validated anchor schema.")
    parser.add_argument("--anchor-meta-path", type=Path, required=True, help="Output JSON path for the validated anchor metadata.")
    parser.add_argument("--lag-steps", type=str, default="1,2", help="Comma-separated positive lag steps required for valid anchors.")
    parser.add_argument("--id-col", type=str, default="id", help="Animal identifier column.")
    parser.add_argument("--year-col", type=str, default="year", help="Year column.")
    parser.add_argument("--month-col", type=str, default="month", help="Month column.")
    parser.add_argument("--target-col", type=str, default="disease", help="Binary target column used for clinical risk labeling.")
    parser.add_argument("--support-cols", type=str, default=",".join(DEFAULT_SUPPORT_COLS), help="Comma-separated support columns to retain in the anchor dataset.")
    parser.add_argument("--age-col", type=str, default="age", help="Age column used to derive age_class when needed.")
    parser.add_argument("--min-year", type=int, default=2020, help="Minimum year retained during temporal anchor construction.")
    parser.add_argument("--keep-month-index", action="store_true", default=True, help="Keep the internal continuous month index in the output dataset.")
    parser.add_argument("--drop-month-index", action="store_false", dest="keep_month_index", help="Drop the internal continuous month index from the output dataset.")
    return parser.parse_args()

# Main call

if __name__ == "__main__":

    args = parse_args()
    lag_steps = [int(item) for item in parse_csv_list(args.lag_steps)]
    support_cols = parse_csv_list(args.support_cols)
    log.info(
        (
            "CLI parsed | input=%s output=%s schema=%s meta=%s "
            "lag_steps=%s target_col=%s support_cols=%s min_year=%s"
        ),
        args.input_path,
        args.output_path,
        args.anchor_schema_path,
        args.anchor_meta_path,
        lag_steps,
        args.target_col,
        support_cols,
        args.min_year,
    )
    create_validated_anchors(
        input_path=args.input_path,
        output_path=args.output_path,
        anchor_schema_path=args.anchor_schema_path,
        anchor_meta_path=args.anchor_meta_path,
        lag_steps=lag_steps,
        id_col=args.id_col,
        time_cols=(args.year_col, args.month_col),
        target_col=args.target_col,
        support_cols=support_cols,
        age_col=args.age_col,
        min_year=args.min_year,
        keep_month_index=args.keep_month_index,
    )


