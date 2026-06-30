"""
Lactose (Lattosio) pipeline.

This script:
- Loads lactose measurements from CSV.
- Restricts to animals present in the functional control (CF) universe.
- Keeps recent years only (>2018), removes duplicates.
- Fills missing lactose values from 'valoreMisura' when available.
- Applies positivity and IQR-based outlier filtering.
- Aggregates to one measurement per animal-day (first observation).
- Saves a compact Parquet dataset.

Context:
- Part of the mammary diseases indicators workflow.
- Mirrors filtering choices used in other CF-derived pipelines for consistency.

Outputs:
- ltts_agg.parquet: daily lactose values per animal.
"""

from libraries import pd, Path, log, gc
import argparse

# Global variables

VARIABILE = "Lattosio"
RENAME_MAP = {"idAnimale": "id", "Lattosio": "lactose", "giorno": "day", "mese": "month", "anno": "year"}

# Support functions

def IQR_filtering(df: pd.DataFrame, col: str, iqr_k: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from a numeric column using Tukey's IQR rule.
    """
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
    before = len(df)
    df = df[(df[col] >= lo) & (df[col] <= hi)]
    log.info("IQR filter on '%s': %d -> %d records", col, before, len(df))
    return df

def build_lactose_agg(ids_cf: set[int], chunksize: int) -> pd.DataFrame:
    """
    Read lactose CSV in chunks, apply base filters, fill lactose values when needed,
    aggregate one record per animal-day inside each chunk, then re-aggregate globally.
    """
    usecols = ["idAnimale", "anno", "mese", "giorno", VARIABILE, "valoreMisura"]
    dtype = {"idAnimale": "int64", "anno": "int16", "mese": "int8", "giorno": "int8", VARIABILE: "float32", "valoreMisura": "float32"}
    parts = []
    total_rows = 0
    reader = pd.read_csv(RAW_LTTS_CSV, low_memory=False, usecols=lambda c: c in usecols, dtype=dtype, chunksize=chunksize)
    for i, chunk in enumerate(reader, start=1):
        log.info("Processing chunk %d...", i)
        total_rows += len(chunk)
        chunk = chunk[chunk["anno"] > 2018]
        chunk = chunk.drop_duplicates()
        chunk = chunk[chunk["idAnimale"].isin(ids_cf)]
        if len(chunk) == 0:
            del chunk
            gc.collect()
            continue
        chunk = chunk.sort_values(["idAnimale", "anno", "mese", "giorno"], kind="mergesort")
        if "valoreMisura" in chunk.columns and chunk[VARIABILE].isna().sum() > 0:
            chunk[VARIABILE] = chunk[VARIABILE].fillna(chunk["valoreMisura"])
        chunk = chunk[chunk[VARIABILE].notna() & (chunk[VARIABILE] > 0)]
        if len(chunk) == 0:
            del chunk
            gc.collect()
            continue
        chunk_agg = chunk.groupby(["idAnimale", "anno", "mese", "giorno"], observed=True, as_index=False).agg({VARIABILE: "first"})
        parts.append(chunk_agg)
        del chunk, chunk_agg
        gc.collect()
    log.info("Total raw rows processed: %s", f"{total_rows:,}")
    if not parts:
        return pd.DataFrame(columns=["idAnimale", "anno", "mese", "giorno", VARIABILE])
    log.info("Concatenating chunk-level aggregates...")
    df_agg = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    log.info("Re-aggregating globally by animal-day...")
    df_agg = df_agg.groupby(["idAnimale", "anno", "mese", "giorno"], observed=True, as_index=False).agg({VARIABILE: "first"})
    return df_agg

def ltts_main(chunksize: int) -> None:
    """
    Load CF IDs, process lactose data in chunks, apply quality filtering,
    aggregate one daily value per animal, and save the final parquet output.
    """
    ids_cf = set(pd.read_parquet(CF_IDS_PARQUET)["id"].astype("int64").unique())
    log.info("Unique IDs from functional control: %d", len(ids_cf))
    df_agg = build_lactose_agg(ids_cf=ids_cf, chunksize=chunksize)
    log.info("Aggregation completed - rows before IQR: %s, columns: %s", f"{df_agg.shape[0]:,}", df_agg.shape[1])
    if len(df_agg) > 0:
        df_agg = IQR_filtering(df_agg, VARIABILE)
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_agg = df_agg.rename(columns=RENAME_MAP)
    df_agg.to_parquet(OUTPUT_PARQUET, index=False)
    log.info("File saved -> %s (%d rows)", OUTPUT_PARQUET, len(df_agg))
    del df_agg, ids_cf
    gc.collect()

# Parsing

def parse_args():
    """
    Parse CLI arguments for lactose input CSV, CF IDs parquet, output parquet,
    and CSV chunk size.
    """
    parser = argparse.ArgumentParser(description="Aggregate and filter lactose records.")
    parser.add_argument("--input-ltts-csv", type=Path, required=True, help="Path to raw lactose CSV.")
    parser.add_argument("--input-cf-ids", type=Path, required=True, help="Path to CF IDs parquet.")
    parser.add_argument("--output-parquet", type=Path, required=True, help="Path to output parquet.")
    parser.add_argument("--chunksize", type=int, default=200_000, help="CSV chunk size.")
    return parser.parse_args()

# Main call

if __name__ == "__main__":
    args = parse_args()
    RAW_LTTS_CSV = args.input_ltts_csv
    CF_IDS_PARQUET = args.input_cf_ids
    OUTPUT_PARQUET = args.output_parquet
    ltts_main(chunksize=args.chunksize)