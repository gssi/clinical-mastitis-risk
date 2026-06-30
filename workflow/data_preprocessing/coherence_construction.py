"""
Build the shared temporal anchor set used to ensure coherence between the
machine learning and deep learning modeling branches.

This script takes as input an imputed longitudinal dataset and a previously
validated or sampled anchor dataset. Each anchor represents an animal-month
observation that may be used for clinical mammary pathology risk modeling.
The script verifies whether each anchor can be represented consistently in
both modeling paradigms:

1. the machine learning branch, where temporal information is encoded through
   explicit lagged tabular variables;
2. the deep learning branch, where the same information is encoded as a
   fixed-length sequential tensor.

The process removes invalid identifiers, normalizes numeric columns, 
removes duplicate animal-month rows, checks lag completeness for the 
machine learning representation, checks sequence completeness for the
deep learning representation, and finally keeps only the intersection of the
two eligible anchor sets.

Main inputs:
    input_long_path:
        Path to the imputed longitudinal dataset containing animal-level
        monthly observations and modeling variables.
    anchors_path:
        Path to the candidate anchor dataset, usually produced after temporal
        window construction or sampling.
    output_path:
        Path where the final shared anchor parquet file is saved.
    output_ids_path:
        Path where the unique animal identifiers retained in the shared anchor
        set are saved.
    report_path:
        Path where a JSON report describing filtering, eligibility, and output
        statistics is saved.
    lag_steps:
        Lag offsets required by the machine learning tabular representation.
    lag_variables:
        Longitudinal variables for which lagged features are required.
    keep_variables:
        Current-time contextual variables retained in the machine learning
        representation.
    dl_feature_cols:
        Longitudinal variables required by the deep learning sequence
        representation.
    dl_schema_path:
        Optional JSON schema used to validate the deep learning feature list.
    support_cols:
        Additional descriptive or target-related columns preserved in the final
        shared anchor artifact.
    id_col, year_col, month_col:
        Columns defining the animal identifier and monthly time reference.
    target_col:
        Binary target variable, typically indicating mammary pathology risk or
        treatment occurrence within the prediction horizon.
    seq_len:
        Sequence length required by the deep learning branch.
    min_year:
        Minimum year retained in the construction process.

Main outputs:
    A parquet file containing the final shared anchor set, compatible with both
    the lagged tabular machine learning branch and the sequential deep learning
    branch.
    A parquet file containing the unique retained animal identifiers.
    A JSON report summarizing input size, duplicate handling, ML eligibility,
    DL eligibility, final intersection size, class distribution, selected
    variables, and configuration metadata.

The resulting artifact supports a fair comparison between ML and DL models by
ensuring that both branches operate on the same animal-month instances and on
coherent temporal information.
"""


from libraries import pd, np, Path, log, gc, json
import argparse

# Global variables

DEFAULT_LAG_VARIABLES = ["scs", "fat", "lactose", "milk", "protein", "ec"]
DEFAULT_KEEP_VARIABLES = ["age", "months_since_calving", "month_sin", "month_cos"]
DEFAULT_DL_FEATURES = ["scs", "milk", "fat", "protein", "lactose", "ec", "age", "months_since_calving", "month_sin", "month_cos",]
DEFAULT_SUPPORT_COLS = [ "cf_date", "disease", "healthy", "age", "age_class", "season", "lactation_phase", "months_since_calving", "month_sin", "month_cos",]
DROP_META_COLS = ["calving_date", "calving", "diagnosis", "t_date", "birth_date", "breed"]

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

def read_table(path: Path) -> pd.DataFrame:
    """Read a parquet or CSV table from disk."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported format: {path.suffix}")

def read_json_schema(path: Path, key: str) -> list[str]:
    """Load a feature list from a JSON schema file by key."""
    schema = json.loads(path.read_text(encoding="utf-8"))
    features = schema.get(key, [])
    if not features:
        raise ValueError(f"{key} missing or empty in schema: {path}")
    return features

def month_index(year: pd.Series, month: pd.Series) -> pd.Series:
    """Convert year and month into a monotonic monthly index."""
    y = pd.to_numeric(year, errors="coerce").astype("Int32")
    m = pd.to_numeric(month, errors="coerce").astype("Int32")
    return (y * 12 + m).astype("Int32")

def validate_required_columns(df: pd.DataFrame, columns: list[str], what: str) -> None:
    """Validate that required columns exist in a dataframe."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {what}: {missing}")

def enforce_unique_monthly_rows(df: pd.DataFrame, id_col: str, year_col: str, month_col: str) -> tuple[pd.DataFrame, dict]:
    """Ensure one unique row per entity-month before eligibility checks."""
    out = df.copy()
    dup_mask = out.duplicated([id_col, year_col, month_col], keep=False)
    duplicate_rows = int(dup_mask.sum())
    duplicate_keys = int(out.loc[dup_mask, [id_col, year_col, month_col]].drop_duplicates().shape[0])
    if duplicate_rows > 0:
        log.warning(
            "Duplicate monthly rows detected in long dataset: duplicate_rows=%d duplicate_keys=%d. Keeping first row per key.",
            duplicate_rows,
            duplicate_keys,
        )
        out = out.sort_values([id_col, year_col, month_col], kind="mergesort")
        out = out.drop_duplicates([id_col, year_col, month_col], keep="first").reset_index(drop=True)
    report = {
        "duplicate_rows_detected": duplicate_rows,
        "duplicate_keys_detected": duplicate_keys,
        "deduplicated_rows_kept": int(len(out)),
    }
    return out, report

def extract_anchor_keys(anchors: pd.DataFrame, id_col: str, year_col: str, month_col: str) -> pd.DataFrame:
    """Extract unique anchor keys from the anchor dataset."""
    out = anchors[[id_col, year_col, month_col]].copy()
    out[id_col] = pd.to_numeric(out[id_col], errors="coerce")
    out[year_col] = pd.to_numeric(out[year_col], errors="coerce").astype("Int16")
    out[month_col] = pd.to_numeric(out[month_col], errors="coerce").astype("Int8")
    out = out.dropna(subset=[id_col, year_col, month_col]).drop_duplicates([id_col, year_col, month_col], keep="first")
    return out.reset_index(drop=True)

def apply_requested_ohe(df: pd.DataFrame, ohe_features: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Apply one-hot encoding to selected columns while preserving original columns."""
    out = df.copy()
    present = [column for column in ohe_features if column in out.columns]
    if not present:
        return out, [], []
    dummies = pd.get_dummies(out[present], prefix=present, prefix_sep="_", dtype="int8")
    dummies = dummies.loc[:, ~dummies.columns.duplicated()].copy()
    out = pd.concat([out, dummies], axis=1)
    return out, present, dummies.columns.tolist()

def add_lagged_columns(df: pd.DataFrame, id_col: str, year_col: str, month_col: str, lag_steps: list[int], lag_variables: list[str], target_col: str) -> pd.DataFrame:
    """Add lagged feature and target columns to the longitudinal dataset."""
    out = df.copy()
    out[year_col] = pd.to_numeric(out[year_col], errors="coerce").astype("Int16")
    out[month_col] = pd.to_numeric(out[month_col], errors="coerce").astype("Int8")
    out["__month_idx"] = month_index(out[year_col], out[month_col]).astype("Int32")
    out = out.sort_values([id_col, year_col, month_col], kind="mergesort").reset_index(drop=True)
    grouped = out.groupby(id_col, observed=True, sort=False)
    lag_candidates = list(dict.fromkeys([column for column in lag_variables if column in out.columns] + ([target_col] if target_col in out.columns else [])))
    for lag in lag_steps:
        out[f"__month_idx_t-{lag}"] = grouped["__month_idx"].shift(lag)
        if lag_candidates:
            lagged = grouped[lag_candidates].shift(lag)
            lagged.columns = [f"{column}_t-{lag}" for column in lag_candidates]
            out = pd.concat([out, lagged], axis=1)
    return out

def attach_anchor_rows(long_df: pd.DataFrame, anchor_keys: pd.DataFrame, id_col: str, year_col: str, month_col: str) -> pd.DataFrame:
    """Attach anchor rows to an enriched long dataframe using exact temporal keys."""
    return anchor_keys.merge(long_df, on=[id_col, year_col, month_col], how="left", validate="one_to_one")

def build_wide_feature_cols(df: pd.DataFrame, lag_steps: list[int], lag_variables: list[str], keep_variables: list[str], ohe_features_for_schema: list[str]) -> list[str]:
    """Build the ML feature schema used for eligibility checks."""
    cols = []
    cols.extend([column for column in lag_variables if column in df.columns])
    for column in keep_variables:
        if column not in df.columns:
            continue
        if column in ohe_features_for_schema:
            cols.extend(sorted([dummy for dummy in df.columns if dummy.startswith(f"{column}_") and dummy != column]))
        else:
            cols.append(column)
    for base in lag_variables:
        for lag in lag_steps:
            lag_col = f"{base}_t-{lag}"
            if lag_col in df.columns:
                cols.append(lag_col)
    return list(dict.fromkeys(cols))

def build_ml_eligibility_mask(long_df: pd.DataFrame, anchor_keys: pd.DataFrame, id_col: str, year_col: str, month_col: str, target_col: str, lag_steps: list[int],
                              lag_variables: list[str], keep_variables: list[str], ohe_features: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build the ML-eligible anchor set from lagged tabular completeness rules."""
    lag_present = [column for column in lag_variables if column in long_df.columns]
    keep_present = [column for column in keep_variables if column in long_df.columns]
    if not lag_present:
        raise ValueError("No lag variables found in the long dataset for ML eligibility.")
    enriched, ohe_present, _ = apply_requested_ohe(long_df, ohe_features)
    schema_ohe_features = [column for column in ohe_present if column in keep_present]
    enriched = add_lagged_columns(
        df=enriched,
        id_col=id_col,
        year_col=year_col,
        month_col=month_col,
        lag_steps=lag_steps,
        lag_variables=lag_present,
        target_col=target_col,
    )
    merged = attach_anchor_rows(enriched, anchor_keys, id_col=id_col, year_col=year_col, month_col=month_col)
    current_month = pd.to_numeric(merged["__month_idx"], errors="coerce")
    valid_mask = pd.Series(True, index=merged.index)
    for lag in lag_steps:
        lag_target_col = f"{target_col}_t-{lag}"
        lag_month_col = f"__month_idx_t-{lag}"
        lag_target = pd.to_numeric(merged[lag_target_col], errors="coerce") if lag_target_col in merged.columns else pd.Series(np.nan, index=merged.index)
        lag_month = pd.to_numeric(merged[lag_month_col], errors="coerce") if lag_month_col in merged.columns else pd.Series(np.nan, index=merged.index)
        valid_mask &= lag_target.notna()
        valid_mask &= current_month.sub(lag_month).eq(lag)
    wide_feature_cols = build_wide_feature_cols(
        df=merged,
        lag_steps=lag_steps,
        lag_variables=lag_present,
        keep_variables=keep_present,
        ohe_features_for_schema=schema_ohe_features,
    )
    target_lag_cols = [f"{target_col}_t-{lag}" for lag in lag_steps if f"{target_col}_t-{lag}" in merged.columns]
    required_cols = [column for column in wide_feature_cols + [target_col] + target_lag_cols if column in merged.columns]
    if required_cols:
        valid_mask &= merged[required_cols].notna().all(axis=1)
    ml_ok = merged.loc[valid_mask, [id_col, year_col, month_col]].copy().drop_duplicates([id_col, year_col, month_col], keep="first")
    return ml_ok.reset_index(drop=True), lag_present, keep_present

def build_lookup_frame(df: pd.DataFrame, id_col: str, month_idx_col: str, feat_cols: list[str]) -> pd.DataFrame:
    """Build a MultiIndex lookup frame for exact DL sequence retrieval."""
    base = df[[id_col, month_idx_col] + feat_cols].copy()
    base = base.sort_values([id_col, month_idx_col], kind="mergesort")
    base = base.drop_duplicates([id_col, month_idx_col], keep="first").reset_index(drop=True)
    index = pd.MultiIndex.from_frame(base[[id_col, month_idx_col]])
    lookup = base[feat_cols].copy()
    lookup.index = index
    return lookup

def build_dl_eligibility_mask(long_df: pd.DataFrame, anchor_keys: pd.DataFrame, id_col: str, year_col: str, month_col: str, feature_cols: list[str], seq_len: int) -> pd.DataFrame:
    """Build the DL-eligible anchor set from sequence completeness rules."""
    validate_required_columns(long_df, [id_col, year_col, month_col] + feature_cols, what="long dataset for DL eligibility")
    out = long_df.copy()
    for column in feature_cols:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out[feature_cols].isna().any().any():
        bad_cols = out[feature_cols].columns[out[feature_cols].isna().any()].tolist()
        raise ValueError(f"NaNs found in longitudinal features for DL eligibility: {bad_cols}")
    out[year_col] = pd.to_numeric(out[year_col], errors="coerce").astype("Int16")
    out[month_col] = pd.to_numeric(out[month_col], errors="coerce").astype("Int8")
    out["__month_idx"] = month_index(out[year_col], out[month_col]).astype("Int32")
    anchor_df = anchor_keys.copy()
    anchor_df[year_col] = pd.to_numeric(anchor_df[year_col], errors="coerce").astype("Int16")
    anchor_df[month_col] = pd.to_numeric(anchor_df[month_col], errors="coerce").astype("Int8")
    anchor_df["__month_idx"] = month_index(anchor_df[year_col], anchor_df[month_col]).astype("Int32")
    feat_lookup = build_lookup_frame(out, id_col=id_col, month_idx_col="__month_idx", feat_cols=feature_cols)
    lookups = []
    for offset in range(seq_len - 1, -1, -1):
        query = anchor_df[[id_col, "__month_idx"]].copy()
        query["__month_idx"] = query["__month_idx"] - offset
        query_index = pd.MultiIndex.from_frame(query[[id_col, "__month_idx"]])
        step_frame = feat_lookup.reindex(query_index)
        lookups.append(step_frame.reset_index(drop=True))
    wide_seq = pd.concat(lookups, axis=1)
    valid_mask = ~wide_seq.isna().any(axis=1)
    dl_ok = anchor_df.loc[valid_mask, [id_col, year_col, month_col]].copy().drop_duplicates([id_col, year_col, month_col], keep="first")
    return dl_ok.reset_index(drop=True)

def intersect_anchor_sets(anchor_keys: pd.DataFrame, ml_ok: pd.DataFrame, dl_ok: pd.DataFrame, id_col: str, year_col: str, month_col: str) -> pd.DataFrame:
    """Intersect ML-eligible and DL-eligible anchor sets."""
    merged = anchor_keys.merge(ml_ok.assign(__ml_ok=1), on=[id_col, year_col, month_col], how="left")
    merged = merged.merge(dl_ok.assign(__dl_ok=1), on=[id_col, year_col, month_col], how="left")
    final_mask = merged["__ml_ok"].eq(1) & merged["__dl_ok"].eq(1)
    final_anchors = merged.loc[final_mask, [id_col, year_col, month_col]].copy()
    return final_anchors.drop_duplicates([id_col, year_col, month_col], keep="first").reset_index(drop=True)

def create_shared_ids(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Extract unique entity identifiers from the final shared anchors."""
    return df[[id_col]].drop_duplicates().rename(columns={id_col: "id"}).reset_index(drop=True)

def build_shared_report(input_anchor_rows: int, ml_ok_rows: int, dl_ok_rows: int, final_rows: int, source_name: str, duplicate_report: dict, final_df: pd.DataFrame,
                        target_col: str) -> dict:
    """Build the structural report of the shared final anchor artifact."""
    out = {
        "source_anchor_set": source_name,
        "input_anchor_rows": int(input_anchor_rows),
        "ml_eligible_rows": int(ml_ok_rows),
        "dl_eligible_rows": int(dl_ok_rows),
        "shared_final_rows": int(final_rows),
        "shared_unique_ids": int(final_df["id"].nunique()) if len(final_df) and "id" in final_df.columns else 0,
    }
    out.update(duplicate_report)
    if target_col in final_df.columns:
        out["positive_rows"] = int((pd.to_numeric(final_df[target_col], errors="coerce") == 1).sum())
        out["negative_rows"] = int((pd.to_numeric(final_df[target_col], errors="coerce") == 0).sum())
    return out

def create_shared_final_anchors(input_long_path: Path, anchors_path: Path, output_path: Path, output_ids_path: Path, report_path: Path, lag_steps: list[int] | None = None,
                                lag_variables: list[str] | None = None, keep_variables: list[str] | None = None, ohe_features: list[str] | None = None, dl_feature_cols: list[str] | None = None,
                                dl_schema_path: Path | None = None, support_cols: list[str] | None = None, id_col: str = "id", time_cols: tuple[str, str] = ("year", "month"),
                                target_col: str = "disease", seq_len: int = 3, min_year: int = 2020) -> None:
    
    """Build a final shared anchor set compatible with both ML and DL branches.
    The CLI-defined ML feature space is the source of truth:
    dl_feature_columns must correspond exactly to lag_variables + keep_variables.
    """

    lag_steps = sorted(set(lag_steps or [1, 2]))
    if not lag_steps or not all(isinstance(step, int) and step > 0 for step in lag_steps):
        raise ValueError("lag_steps must be a non-empty list of positive integers.")
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1.")
    year_col, month_col = time_cols
    lag_variables = parse_csv_list(lag_variables) or DEFAULT_LAG_VARIABLES.copy()
    keep_variables = parse_csv_list(keep_variables) or DEFAULT_KEEP_VARIABLES.copy()
    ohe_features = parse_csv_list(ohe_features)
    requested_support_cols = parse_csv_list(support_cols) or DEFAULT_SUPPORT_COLS.copy()
    cli_feature_columns = list(dict.fromkeys(lag_variables + keep_variables))

    if dl_schema_path is not None:
        schema_dl_feature_cols = read_json_schema(
            dl_schema_path,
            "longitudinal_feature_columns",
        )
        if schema_dl_feature_cols != cli_feature_columns:
            raise ValueError(
                "DL schema feature columns do not match CLI lag_variables + keep_variables.\n"
                f"Expected from CLI: {cli_feature_columns}\n"
                f"Found in schema: {schema_dl_feature_cols}"
            )
        dl_feature_cols = schema_dl_feature_cols
    else:
        parsed_dl_feature_cols = parse_csv_list(dl_feature_cols)
        if parsed_dl_feature_cols and parsed_dl_feature_cols != cli_feature_columns:
            raise ValueError(
                "CLI --dl-feature-cols does not match --lag-variables + --keep-variables.\n"
                f"Expected: {cli_feature_columns}\n"
                f"Got: {parsed_dl_feature_cols}"
            )
        dl_feature_cols = cli_feature_columns
    log.info("Loading imputed longitudinal dataset: %s", input_long_path)
    long_df = read_table(input_long_path)
    log.info("Loading anchor dataset: %s", anchors_path)
    anchors_df = read_table(anchors_path)

    validate_required_columns(
        long_df,
        [id_col, year_col, month_col],
        what="long dataset",
    )
    validate_required_columns(
        anchors_df,
        [id_col, year_col, month_col],
        what="anchor dataset",
    )

    drop_cols = [column for column in DROP_META_COLS if column in long_df.columns]
    if drop_cols:
        long_df = long_df.drop(columns=drop_cols, errors="ignore")
        log.info("Dropped metadata columns from long dataset: %s", drop_cols)
    long_df[id_col] = pd.to_numeric(long_df[id_col], errors="coerce")
    anchors_df[id_col] = pd.to_numeric(anchors_df[id_col], errors="coerce")
    before_long = len(long_df)
    before_anchors = len(anchors_df)
    long_df = long_df.dropna(subset=[id_col]).copy()
    anchors_df = anchors_df.dropna(subset=[id_col]).copy()
    log.info(
        "Rows kept after valid id filtering | long: %d -> %d | anchors: %d -> %d",
        before_long,
        len(long_df),
        before_anchors,
        len(anchors_df),
    )
    long_df[year_col] = pd.to_numeric(long_df[year_col], errors="coerce")
    long_df[month_col] = pd.to_numeric(long_df[month_col], errors="coerce")
    anchors_df[year_col] = pd.to_numeric(anchors_df[year_col], errors="coerce")
    anchors_df[month_col] = pd.to_numeric(anchors_df[month_col], errors="coerce")
    before_long = len(long_df)
    before_anchors = len(anchors_df)
    long_df = long_df[
        long_df[year_col].notna() & long_df[month_col].notna()
    ].copy()
    anchors_df = anchors_df[
        anchors_df[year_col].notna() & anchors_df[month_col].notna()
    ].copy()
    log.info(
        "Rows kept after valid year/month filtering | long: %d -> %d | anchors: %d -> %d",
        before_long,
        len(long_df),
        before_anchors,
        len(anchors_df),
    )
    before_long = len(long_df)
    before_anchors = len(anchors_df)
    long_df = long_df[long_df[year_col] >= min_year].copy()
    anchors_df = anchors_df[anchors_df[year_col] >= min_year].copy()
    log.info(
        "Rows kept after min_year >= %d filtering | long: %d -> %d | anchors: %d -> %d",
        min_year,
        before_long,
        len(long_df),
        before_anchors,
        len(anchors_df),
    )
    normalize_cols = list(
        dict.fromkeys(
            lag_variables
            + keep_variables
            + dl_feature_cols
            + [target_col, "healthy"]
        )
    )
    for column in normalize_cols:
        if column in long_df.columns and column not in ohe_features:
            long_df[column] = pd.to_numeric(long_df[column], errors="coerce")
    if target_col in long_df.columns:
        long_df[target_col] = (
            pd.to_numeric(long_df[target_col], errors="coerce")
            .fillna(0)
            .astype("int8")
        )
    if "healthy" in long_df.columns:
        long_df["healthy"] = (
            pd.to_numeric(long_df["healthy"], errors="coerce")
            .fillna(0)
            .astype("int8")
        )
    long_df, duplicate_report = enforce_unique_monthly_rows(
        df=long_df,
        id_col=id_col,
        year_col=year_col,
        month_col=month_col,
    )
    anchor_keys = extract_anchor_keys(
        anchors_df,
        id_col=id_col,
        year_col=year_col,
        month_col=month_col,
    )
    source_name = anchors_path.name
    log.info("Anchor keys prepared: %d", len(anchor_keys))
    ml_ok, lag_present, keep_present = build_ml_eligibility_mask(
        long_df=long_df,
        anchor_keys=anchor_keys,
        id_col=id_col,
        year_col=year_col,
        month_col=month_col,
        target_col=target_col,
        lag_steps=lag_steps,
        lag_variables=lag_variables,
        keep_variables=keep_variables,
        ohe_features=ohe_features,
    )
    log.info("ML-eligible anchors: %d", len(ml_ok))
    dl_ok = build_dl_eligibility_mask(
        long_df=long_df,
        anchor_keys=anchor_keys,
        id_col=id_col,
        year_col=year_col,
        month_col=month_col,
        feature_cols=dl_feature_cols,
        seq_len=seq_len,
    )
    log.info("DL-eligible anchors: %d", len(dl_ok))
    final_anchor_keys = intersect_anchor_sets(
        anchor_keys=anchor_keys,
        ml_ok=ml_ok,
        dl_ok=dl_ok,
        id_col=id_col,
        year_col=year_col,
        month_col=month_col,
    )
    log.info("Shared final anchors: %d", len(final_anchor_keys))
    if len(final_anchor_keys) == 0:
        raise ValueError("No shared final anchors left after ML/DL intersection.")
    final_df = final_anchor_keys.merge(
        anchors_df,
        on=[id_col, year_col, month_col],
        how="left",
        validate="one_to_one",
    )
    support_present = [
        column for column in requested_support_cols
        if column in final_df.columns
    ]
    target_history_cols = [
        f"{target_col}_t-{lag}"
        for lag in lag_steps
        if f"{target_col}_t-{lag}" in final_df.columns
    ]
    final_cols = [id_col, year_col, month_col]
    final_cols.extend(
        [
            column for column in [target_col, "healthy", "__month_idx"]
            if column in final_df.columns
        ]
    )
    final_cols.extend(target_history_cols)
    final_cols.extend(support_present)
    final_cols = list(
        dict.fromkeys(
            [column for column in final_cols if column in final_df.columns]
        )
    )
    final_df = final_df[final_cols].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_path, index=False)
    final_ids = create_shared_ids(final_df, id_col=id_col)
    output_ids_path.parent.mkdir(parents=True, exist_ok=True)
    final_ids.to_parquet(output_ids_path, index=False)
    report = build_shared_report(
        input_anchor_rows=len(anchor_keys),
        ml_ok_rows=len(ml_ok),
        dl_ok_rows=len(dl_ok),
        final_rows=len(final_df),
        source_name=source_name,
        duplicate_report=duplicate_report,
        final_df=final_df,
        target_col=target_col,
    )
    report.update(
        {
            "step": "shared_final_anchor_builder",
            "input_long_path": str(input_long_path),
            "anchors_path": str(anchors_path),
            "output_path": str(output_path),
            "output_ids_path": str(output_ids_path),
            "lag_steps": lag_steps,
            "seq_len": int(seq_len),
            "lag_variables_requested": lag_variables,
            "lag_variables_used": lag_present,
            "keep_variables_requested": keep_variables,
            "keep_variables_used": keep_present,
            "ohe_features_requested": ohe_features,
            "dl_feature_columns": dl_feature_cols,
            "feature_coherence_policy": "dl_feature_columns == lag_variables_requested + keep_variables_requested",
            "support_columns_saved": [
                column
                for column in final_df.columns
                if column not in [id_col, year_col, month_col]
            ],
            "min_year": int(min_year),
        }
    )
    save_json(report, report_path)
    log.info(
        "Shared final anchors saved: %s | rows=%d unique_ids=%d",
        output_path,
        len(final_df),
        int(final_ids["id"].nunique()),
    )
    log.info("Shared final ids saved: %s", output_ids_path)
    log.info("Shared final anchor report saved: %s", report_path)
    del long_df, anchors_df, anchor_keys, ml_ok, dl_ok
    del final_anchor_keys, final_df, final_ids
    gc.collect()
    log.info("Shared final anchor construction completed successfully.")

# Parsing

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for shared final anchor construction."""
    parser = argparse.ArgumentParser(description="Build a final shared anchor set compatible with both ML and DL branches.")
    parser.add_argument("--input-long-path", type=Path, required=True, help="Input imputed long dataset path.")
    parser.add_argument("--anchors-path", type=Path, required=True, help="Input validated or sampled anchors path.")
    parser.add_argument("--output-path", type=Path, required=True, help="Output parquet path for shared final anchors.")
    parser.add_argument("--output-ids-path", type=Path, required=True, help="Output parquet path for shared final IDs.")
    parser.add_argument("--report-path", type=Path, required=True, help="Output JSON path for the shared final anchor report.")
    parser.add_argument("--lag-steps", type=str, default="1,2", help="Comma-separated positive lag steps for ML compatibility checks.")
    parser.add_argument("--lag-variables", type=str, default="scs,fat,lactose,milk,protein,ec", help="Comma-separated lag variables for ML compatibility checks.")
    parser.add_argument("--keep-variables", type=str, default="age,months_since_calving,month_sin,month_cos", help="Comma-separated keep variables for ML compatibility checks.")
    parser.add_argument("--ohe-features", type=str, default="", help="Comma-separated columns to one-hot encode for ML compatibility checks.")
    parser.add_argument("--dl-feature-cols", type=str, default="", help="Comma-separated DL feature columns when no schema file is provided.")
    parser.add_argument("--dl-schema-path", type=Path, default=None, help="Optional longitudinal schema JSON with longitudinal_feature_columns.")
    parser.add_argument("--support-cols", type=str, default="cf_date,disease,healthy,age,age_class,season,lactation_phase,months_since_calving,month_sin,month_cos", help="Comma-separated support columns to preserve in the final shared anchors.")
    parser.add_argument("--id-col", type=str, default="id", help="Entity identifier column.")
    parser.add_argument("--year-col", type=str, default="year", help="Year column.")
    parser.add_argument("--month-col", type=str, default="month", help="Month column.")
    parser.add_argument("--target-col", type=str, default="disease", help="Target column.")
    parser.add_argument("--seq-len", type=int, default=3, help="Sequence length required by the DL branch.")
    parser.add_argument("--min-year", type=int, default=2020, help="Minimum year to keep.")
    return parser.parse_args()

# Main call

if __name__ == "__main__":
    args = parse_args()
    lag_steps = [int(item.strip()) for item in args.lag_steps.split(",") if item.strip()]
    lag_variables = parse_csv_list(args.lag_variables)
    keep_variables = parse_csv_list(args.keep_variables)
    ohe_features = parse_csv_list(args.ohe_features)
    dl_feature_cols = parse_csv_list(args.dl_feature_cols)
    support_cols = parse_csv_list(args.support_cols)
    log.info(
        "CLI parsed | long=%s anchors=%s output=%s output_ids=%s report=%s lag_steps=%s seq_len=%d",
        args.input_long_path,
        args.anchors_path,
        args.output_path,
        args.output_ids_path,
        args.report_path,
        lag_steps,
        args.seq_len,
    )
    create_shared_final_anchors(
        input_long_path=args.input_long_path,
        anchors_path=args.anchors_path,
        output_path=args.output_path,
        output_ids_path=args.output_ids_path,
        report_path=args.report_path,
        lag_steps=lag_steps,
        lag_variables=lag_variables,
        keep_variables=keep_variables,
        ohe_features=ohe_features,
        dl_feature_cols=dl_feature_cols,
        dl_schema_path=args.dl_schema_path,
        support_cols=support_cols,
        id_col=args.id_col,
        time_cols=(args.year_col, args.month_col),
        target_col=args.target_col,
        seq_len=args.seq_len,
        min_year=args.min_year,
    )

