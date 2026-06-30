"""
Chunked data pipeline for Control Function records.

This script:
- Reads raw CF CSV in chunks.
- Filters years and keeps only required columns.
- Builds global breed counts per animal.
- Aggregates one record per animal-day.
- Assigns one final breed per animal by global majority vote.
- Applies quality filters and saves final outputs.
"""

from libraries import pd, Path, log, gc
import argparse
from collections import defaultdict

# Global variables

RENAME_MAP = {"idAnimale": "id", "razza": "breed", "giorno": "day", "mese": "month", "anno": "year", "LatteAlle24Ore": "milk", "ProteineAlle24Ore": "protein",
              "GrassoAlle24Ore": "fat", "LinearScore": "scs", "DataControlloFunzionaleLatte": "cf_date"}

MAPPATURA_RAZZE = {"00": "meticcia", "0": "meticcia", "0.0": "meticcia", 
                   "01": "bruna", "1": "bruna", "1.0": "bruna",
                   "02": "frisona", "2": "frisona", "2.0": "frisona", 
                   "04": "pezzata", "4": "pezzata", "4.0": "pezzata"}

RAZZE_DI_INTERESSE = ["frisona", "pezzata", "meticcia", "bruna"]

# Support functions

def map_razza(series: pd.Series) -> pd.Series:
    """
    Map raw breed codes to canonical breed labels.
    """
    return series.map(MAPPATURA_RAZZE)

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

def build_breed_majority_and_daily_agg(chunksize: int) -> tuple[pd.Series, pd.DataFrame]:
    """
    Read the CSV in chunks, compute global breed counts per animal,
    and build a daily aggregated dataset.
    """
    usecols = ["idAnimale", "anno", "mese", "giorno", "codiceRazzaAIA", "LinearScore", "LatteAlle24Ore", "ProteineAlle24Ore", "GrassoAlle24Ore"]
    dtype = {"idAnimale": "int64", "anno": "int16", "mese": "int8", "giorno": "int8", "codiceRazzaAIA": "string", "LinearScore": "float32", "LatteAlle24Ore": "float32",
             "ProteineAlle24Ore": "float32", "GrassoAlle24Ore": "float32"}
    breed_counts = defaultdict(lambda: defaultdict(int))
    daily_parts = []
    total_rows = 0
    reader = pd.read_csv(RAW_CF_CSV, low_memory=False, usecols=usecols, dtype=dtype, chunksize=chunksize)
    for i, chunk in enumerate(reader, start=1):
        log.info("Processing chunk %d...", i)
        total_rows += len(chunk)
        chunk = chunk[chunk["anno"] > 2018]
        chunk = chunk.drop_duplicates()
        chunk["razza"] = map_razza(chunk["codiceRazzaAIA"])
        chunk = chunk.drop(columns=["codiceRazzaAIA"])
        # Global breed counts per animal
        breed_chunk = chunk.dropna(subset=["razza"])[["idAnimale", "razza"]]
        breed_chunk_counts = breed_chunk.value_counts().reset_index(name="n")
        for row in breed_chunk_counts.itertuples(index=False):
            breed_counts[row.idAnimale][row.razza] += int(row.n)
        # Daily aggregation within chunk
        group_cols = ["idAnimale", "anno", "mese", "giorno"]
        agg_dict = {"LinearScore": "first", "LatteAlle24Ore": "first", "ProteineAlle24Ore": "first", "GrassoAlle24Ore": "first"}
        chunk_daily = chunk.groupby(group_cols, as_index=False).agg(agg_dict)
        daily_parts.append(chunk_daily)
        del chunk, breed_chunk, breed_chunk_counts, chunk_daily
        gc.collect()
    log.info("Total raw rows processed: %s", f"{total_rows:,}")
    # Build final breed map
    breed_map = {}
    for animal_id, counts in breed_counts.items():
        breed_map[animal_id] = max(counts.items(), key=lambda x: x[1])[0]
    breed_map = pd.Series(breed_map, name="razza")
    # Merge all daily parts and re-aggregate globally
    log.info("Concatenating aggregated daily parts...")
    daily_df = pd.concat(daily_parts, ignore_index=True)
    del daily_parts
    gc.collect()
    log.info("Re-aggregating globally by animal-day...")
    daily_df = daily_df.groupby(["idAnimale", "anno", "mese", "giorno"], as_index=False).agg({"LinearScore": "first", "LatteAlle24Ore": "first", 
                                                                                              "ProteineAlle24Ore": "first", "GrassoAlle24Ore": "first"})
    return breed_map, daily_df

def cf_main(chunksize: int) -> None:
    """
    Run the chunked CF pipeline and save final outputs.
    """
    breed_map, cf_agg = build_breed_majority_and_daily_agg(chunksize=chunksize)
    log.info("Assigning final breed per animal...")
    cf_agg["razza"] = cf_agg["idAnimale"].map(breed_map)
    del breed_map
    gc.collect()
    cf_agg = cf_agg[cf_agg["razza"].isin(RAZZE_DI_INTERESSE)]
    log.info("Building CF datetime...")
    cf_agg["DataControlloFunzionaleLatte"] = pd.to_datetime({"year": cf_agg["anno"], "month": cf_agg["mese"], "day": cf_agg["giorno"]}, errors="coerce")
    log.info("Applying quality filters...")
    for v in ["LatteAlle24Ore", "GrassoAlle24Ore", "ProteineAlle24Ore", "LinearScore"]:
        if v != "LinearScore":
            before = len(cf_agg)
            cf_agg = cf_agg[cf_agg[v] > 0]
            log.info("Positivity filter on '%s': %d -> %d records", v, before, len(cf_agg))
        cf_agg = IQR_filtering(cf_agg, v)
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_IDS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    cf_agg = cf_agg.rename(columns=RENAME_MAP)
    cf_agg.to_parquet(OUTPUT_PARQUET, index=False)
    log.info("File saved -> %s (%d rows)", OUTPUT_PARQUET, len(cf_agg))
    unique_ids = cf_agg["id"].drop_duplicates()
    unique_ids.to_frame().to_parquet(OUTPUT_IDS_PARQUET, index=False)
    log.info("Unique IDs saved -> %s (%d ID)", OUTPUT_IDS_PARQUET, len(unique_ids))
    del unique_ids, cf_agg
    gc.collect()

# Parsing

def parse_args():
    """
    Parse CLI arguments for raw CF input, outputs, and chunk size.
    """
    parser = argparse.ArgumentParser(description="Aggregate and filter control function records.")
    parser.add_argument("--input-cf-csv", type=Path, required=True, help="Path to raw CF CSV.")
    parser.add_argument("--output-parquet", type=Path, required=True, help="Path to aggregated output parquet.")
    parser.add_argument("--output-ids-parquet", type=Path, required=True, help="Path to unique IDs output parquet.")
    parser.add_argument("--chunksize", type=int, default=200_000, help="CSV chunk size.")
    return parser.parse_args()

# Main call

if __name__ == "__main__":
    args = parse_args()
    RAW_CF_CSV = args.input_cf_csv
    OUTPUT_PARQUET = args.output_parquet
    OUTPUT_IDS_PARQUET = args.output_ids_parquet
    cf_main(chunksize=args.chunksize)