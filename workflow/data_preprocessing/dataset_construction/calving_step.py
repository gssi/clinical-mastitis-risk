"""
Calving events pipeline.

This script:
- Loads raw calving records from CSV.
- Keeps only animals present in the functional check (CF) universe.
- Collapses raw rows to one calving event per animal-day.
- Keeps at most one calving per animal-month.
- Removes empty events with zero recorded births across all raw components.
- Logs per-animal calving counts before/after preprocessing for coherence checks.
- Saves a compact Parquet dataset.

Context:
- Part of the mammary diseases indicators workflow.
- Designed to preserve calving events even when multiple registrations exist
  on the same day for the same animal.

Outputs:
- parti_agg.parquet: cleaned binary calving events.
"""

from libraries import pd, Path, log, gc
import argparse
from collections import defaultdict

# Global variables

RENAME_MAP = {"idAnimale": "id", "giorno": "day", "mese": "month", "anno": "year", "data": "calving_date", "Parto": "calving", "raw_rows_same_day": "same_day_records"}
BIRTH_COLS = ["NumeroFemmineNateVive", "NumeroFemmineNateMorte", "NumeroMaschiNatiVivi", "NumeroMaschiNatiMorti"]

# Support functions

def series_from_counter(counter_dict: dict) -> pd.Series:
    """
    Convert a per-animal counter dictionary to a pandas Series.
    """
    if not counter_dict:
        return pd.Series(dtype="int64")
    return pd.Series(counter_dict, dtype="int64")

def log_count_dist(label: str, counts: pd.Series) -> None:
    """
    Log the distribution of per-animal event counts.
    """
    if len(counts) == 0:
        log.info("Parts distributions[%s] - min:0  median:0  max:0", label)
        return
    log.info("Parts distributions[%s] - min:%d  median:%d  max:%d", label, int(counts.min()), int(counts.median()), int(counts.max()))

def build_calving_events(ids_cf: set[int], chunksize: int) -> tuple[pd.DataFrame, pd.Series]:
    """
    Read raw calving CSV in chunks, keep CF animals and recent years,
    collapse to one event per animal-day, and return the daily events plus
    raw per-animal counts before final monthly deduplication.
    """
    usecols = ["idAnimale", "giorno", "mese", "anno", *BIRTH_COLS]
    dtype = {"idAnimale": "int64", "giorno": "int8", "mese": "int8", "anno": "int16", "NumeroFemmineNateVive": "float32", "NumeroFemmineNateMorte": "float32",
             "NumeroMaschiNatiVivi": "float32", "NumeroMaschiNatiMorti": "float32"}
    raw_counts = defaultdict(int)
    parts = []
    suspicious_same_day = 0
    reader = pd.read_csv(RAW_CSV, low_memory=False, usecols=lambda c: c in usecols, dtype=dtype, chunksize=chunksize)
    for i, chunk in enumerate(reader, start=1):
        log.info("Processing chunk %d...", i)
        chunk = chunk[chunk["anno"] > 2018]
        chunk = chunk.drop_duplicates()
        chunk = chunk[chunk["idAnimale"].isin(ids_cf)]
        if len(chunk) == 0:
            del chunk
            gc.collect()
            continue
        vc = chunk["idAnimale"].value_counts()
        for k, v in vc.items():
            raw_counts[int(k)] += int(v)
        chunk = chunk.sort_values(["idAnimale", "anno", "mese", "giorno"], kind="mergesort")
        group_cols = ["idAnimale", "giorno", "mese", "anno"]
        birth_sums = chunk.groupby(group_cols, observed=True, as_index=False)[BIRTH_COLS].sum()
        same_day_counts = chunk.groupby(group_cols, observed=True, as_index=False).size().rename(columns={"size": "raw_rows_same_day"})
        day_df = pd.merge(birth_sums, same_day_counts, on=group_cols, how="inner")
        day_df["total_birth_records"] = day_df[BIRTH_COLS].sum(axis=1)
        suspicious_same_day += int((day_df["raw_rows_same_day"] > 2).sum())
        day_df = day_df[day_df["total_birth_records"] > 0]
        if len(day_df) == 0:
            del chunk, vc, birth_sums, same_day_counts, day_df
            gc.collect()
            continue
        parts.append(day_df[["idAnimale", "giorno", "mese", "anno", "raw_rows_same_day"]])
        del chunk, vc, birth_sums, same_day_counts, day_df
        gc.collect()
    if suspicious_same_day:
        log.warning("%d animal-day events had more than 2 raw registrations; they were kept as one event per day.", suspicious_same_day)
    raw_counts = series_from_counter(raw_counts)
    if not parts:
        return pd.DataFrame(columns=["idAnimale", "giorno", "mese", "anno", "raw_rows_same_day"]), raw_counts
    log.info("Concatenating chunk-level daily calving events...")
    df = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    log.info("Re-aggregating globally by animal-day...")
    df = df.groupby(["idAnimale", "giorno", "mese", "anno"], observed=True, as_index=False).agg({"raw_rows_same_day": "sum"})
    return df, raw_counts

def calving_main(chunksize: int = 200_000) -> None:
    """
    Load CF IDs, process raw calving records in chunks, keep one event per animal-day,
    keep at most one event per animal-month, log coherence, and save the final parquet.
    """
    ids_cf = set(pd.read_parquet(CF_IDS_PARQUET)["id"].astype("int64").unique())
    log.info("Unique IDs from functional check: %d", len(ids_cf))
    df, parts_start = build_calving_events(ids_cf=ids_cf, chunksize=chunksize)
    if len(df) == 0:
        OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        empty = df.copy()
        empty["data"] = pd.Series(dtype="datetime64[ns]")
        empty["Parto"] = pd.Series(dtype="int8")
        empty = empty.rename(columns=RENAME_MAP)
        empty.to_parquet(OUTPUT_PARQUET, index=False)
        log.info("No calving events to save after filtering.")
        del df, empty, ids_cf, parts_start
        gc.collect()
        return
    df["mese_anno"] = df["anno"].astype(str) + "-" + df["mese"].astype(str).str.zfill(2)
    df = (df.sort_values(["idAnimale", "mese_anno", "giorno"], kind="mergesort").drop_duplicates(subset=["idAnimale", "mese_anno"], 
                                                                                                 keep="first").drop(columns="mese_anno"))
    df["data"] = pd.to_datetime({"year": df["anno"], "month": df["mese"], "day": df["giorno"]}, errors="coerce",)
    df["Parto"] = 1
    parts_end = df.groupby("idAnimale", observed=True).size()
    log_count_dist("raw", parts_start)
    log_count_dist("final", parts_end)
    diff = (parts_start - parts_end).fillna(0).astype(int)
    mismatch_ids = diff[diff != 0].index.tolist()
    if mismatch_ids:
        log.warning("%d animals have gained/lost calving events after preprocessing.", len(mismatch_ids))
        log.debug("IDs with mismatch: %s", mismatch_ids[:50])
    else:
        log.info("Calving events count is coherent for all animals.")
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df = df.rename(columns=RENAME_MAP)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    log.info("File saved -> %s (%d rows)", OUTPUT_PARQUET, len(df))
    del df, ids_cf, parts_start, parts_end
    gc.collect()

# Parsing

def parse_args():
    """
    Parse CLI arguments for calving input CSV, CF IDs parquet, output parquet,
    and CSV chunk size.
    """
    parser = argparse.ArgumentParser(description="Aggregate and filter calving events.")
    parser.add_argument("--input-calving-csv", type=Path, required=True, help="Path to raw calving CSV.")
    parser.add_argument("--input-cf-ids", type=Path, required=True, help="Path to CF IDs parquet.")
    parser.add_argument("--output-parquet", type=Path, required=True, help="Path to output parquet.")
    parser.add_argument("--chunksize", type=int, default=200_000, help="CSV chunk size.")
    return parser.parse_args()

# Main call

if __name__ == "__main__":
    args = parse_args()
    RAW_CSV = args.input_calving_csv
    CF_IDS_PARQUET = args.input_cf_ids
    OUTPUT_PARQUET = args.output_parquet
    calving_main(chunksize=args.chunksize)