"""
Milk Electrical Conductivity (EC) pipeline.

This script:
- Loads raw EC measurements from CSV.
- Restricts to animals present in the functional check (CF) universe.
- Cleans/normalizes date and ID fields, fills EC from fallback column when needed.
- Removes non-positive/NaN values and applies IQR-based outlier filtering.
- Aggregates to one EC value per animal-day (stable 'first' after sorting).
- Saves a compact Parquet dataset.

Context:
- Part of the mammary diseases indicators workflow.
- EC is often used as a proxy for subclinical mastitis risk; here we prepare
  a consistent daily signal for downstream modeling.

Outputs:
- ce_agg.parquet: daily EC per animal after filtering.
"""

from libraries import pd, Path, log, gc
import argparse

# Global variables

VARIABILE = "Conducibilità elettrica"
RENAME_MAP = {"idAnimale": "id", "Conducibilità elettrica": "ec", "giorno": "day", "mese": "month", "anno": "year"}

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

def build_ec_agg(ids_cf: set[int], chunksize: int) -> pd.DataFrame:
    """
    Read EC CSV in chunks, apply base filters, fill EC values when needed,
    aggregate one record per animal-day inside each chunk, then re-aggregate globally.
    """
    usecols = ["idAnimale", "anno", "mese", "giorno", VARIABILE, "valoreMisura"]
    dtype = {"idAnimale": "int64", "anno": "int16", "mese": "int8", "giorno": "int8", VARIABILE: "float32", "valoreMisura": "float32"}
    parts = []
    total_rows = 0
    reader = pd.read_csv(RAW_CE_CSV, low_memory=False, usecols=lambda c: c in usecols, dtype=dtype, chunksize=chunksize)
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
        if "valoreMisura" in chunk.columns:
            chunk[VARIABILE] = chunk[VARIABILE].fillna(chunk["valoreMisura"])
        chunk = chunk[chunk[VARIABILE].notna() & (chunk[VARIABILE] > 0)]
        if len(chunk) == 0:
            del chunk
            gc.collect()
            continue
        chunk_agg = chunk.groupby(
            ["idAnimale", "anno", "mese", "giorno"], observed=True, as_index=False).agg({VARIABILE: "first"})
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

def ec_main(chunksize: int) -> None:
    """
    Load CF IDs, process EC data in chunks, apply quality filtering,
    aggregate one daily value per animal, and save the final parquet output.
    """
    ids_cf = set(pd.read_parquet(CF_IDS_PARQUET)["id"].astype("int64").unique())
    log.info("Unique IDs from functional check: %d", len(ids_cf))
    df_agg = build_ec_agg(ids_cf=ids_cf, chunksize=chunksize)
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
    Parse CLI arguments for EC input CSV, CF IDs parquet, output parquet,
    and CSV chunk size.
    """
    parser = argparse.ArgumentParser(description="Aggregate and filter milk electrical conductivity records.")
    parser.add_argument("--input-ec-csv", type=Path, required=True, help="Path to raw EC CSV.")
    parser.add_argument("--input-cf-ids", type=Path, required=True, help="Path to CF IDs parquet.")
    parser.add_argument("--output-parquet", type=Path, required=True, help="Path to output parquet.")
    parser.add_argument("--chunksize", type=int, default=200_000, help="CSV chunk size.")
    return parser.parse_args()

# Main call

if __name__ == "__main__":
    args = parse_args()
    RAW_CE_CSV = args.input_ec_csv
    CF_IDS_PARQUET = args.input_cf_ids
    OUTPUT_PARQUET = args.output_parquet
    ec_main(chunksize=args.chunksize)