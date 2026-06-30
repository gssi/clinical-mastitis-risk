"""
Data pipeline for animal registry (Anagrafica).

This script:
- Loads animal registry records (anagrafica) from raw CSV.
- Keeps only animals present in the functional check dataset (cf_ids).
- Filters birth dates for validity and reasonable years.
- Resolves duplicates and keeps one birth date per animal.
- Saves a clean table with animal IDs and their birth dates.

Context:
- Part of the mammary diseases indicators workflow.
- Ensures consistency between functional check records and registry information.

Outputs:
- ana_agg.parquet: mapping of animal IDs to validated birth dates
"""

from libraries import pd, Path, log, gc
import argparse

# Golbal variables

RENAME_MAP = {"idAnimale": "id", "DataNascita": "birth_date"}

# Support functions

def build_birth_registry(ids_cf: set[int], chunksize: int) -> pd.DataFrame:
    """
    Read registry CSV in chunks, keep CF animals, parse and filter valid birth dates,
    reduce each chunk to one minimum birth date per animal, then re-aggregate globally.
    """
    usecols = ["idAnimale", "DataNascita"]
    dtype = {"idAnimale": "int64", "DataNascita": "string"}
    parts = []
    total_rows = 0
    reader = pd.read_csv(RAW_ANA_CSV, low_memory=False, usecols=lambda c: c in usecols, dtype=dtype, chunksize=chunksize)
    for i, chunk in enumerate(reader, start=1):
        log.info("Processing chunk %d...", i)
        total_rows += len(chunk)
        chunk = chunk[chunk["idAnimale"].isin(ids_cf)]
        if len(chunk) == 0:
            del chunk
            gc.collect()
            continue
        chunk["DataNascita"] = pd.to_datetime(chunk["DataNascita"], errors="coerce")
        chunk = chunk[chunk["DataNascita"].notna()]
        chunk = chunk[(chunk["DataNascita"].dt.year > 2017) & (chunk["DataNascita"].dt.year < 2023)]
        chunk = chunk.drop_duplicates()
        if len(chunk) == 0:
            del chunk
            gc.collect()
            continue
        parts.append(chunk.groupby("idAnimale", as_index=False)["DataNascita"].min())
        del chunk
        gc.collect()
    log.info("Total raw rows processed: %s", f"{total_rows:,}")
    if not parts:
        return pd.DataFrame(columns=["idAnimale", "DataNascita"])
    log.info("Concatenating chunk-level registry aggregates...")
    df = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    log.info("Re-aggregating globally by animal...")
    return df.groupby("idAnimale", as_index=False)["DataNascita"].min()

def ana_main(chunksize: int = 200_000) -> None:
    """
    Load CF IDs, process registry data in chunks, keep one valid birth date per animal,
    and save the final parquet output.
    """
    log.info("Starting elaboration...")
    ids_cf = set(pd.read_parquet(CF_IDS_PARQUET)["id"].astype("int64").unique())
    log.info("Unique IDs from functional check: %d", len(ids_cf))
    log.info("Loading anagraphic data from %s", RAW_ANA_CSV)
    nascita_agg = build_birth_registry(ids_cf=ids_cf, chunksize=chunksize)
    log.info("Valid animals: %d", len(nascita_agg))
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    nascita_agg = nascita_agg.rename(columns=RENAME_MAP)
    nascita_agg.to_parquet(OUTPUT_PARQUET, index=False)
    log.info("File saved -> %s (%d rows)", OUTPUT_PARQUET, len(nascita_agg))
    del nascita_agg, ids_cf
    gc.collect()

# Parsing

def parse_args():
    """
    Parse CLI arguments for registry input CSV, CF IDs parquet, output parquet,
    and CSV chunk size.
    """
    parser = argparse.ArgumentParser(description="Extract, filter, and validate animal registry data.")
    parser.add_argument("--input-ana-csv", type=Path, required=True, help="Path to raw registry CSV.")
    parser.add_argument("--input-cf-ids", type=Path, required=True, help="Path to CF IDs parquet.")
    parser.add_argument("--output-parquet", type=Path, required=True, help="Path to output parquet.")
    parser.add_argument("--chunksize", type=int, default=200_000, help="CSV chunk size.")
    return parser.parse_args()

# Main call

if __name__ == "__main__":
    args = parse_args()
    RAW_ANA_CSV = args.input_ana_csv
    CF_IDS_PARQUET = args.input_cf_ids
    OUTPUT_PARQUET = args.output_parquet
    ana_main(chunksize=args.chunksize)