"""
Merge monthly LEO CSV tables into a single dataset-specific CSV file.

This script combines multiple raw CSV files downloaded from LEO open-data
endpoints into one merged CSV table. It is intended for the raw-data ingestion
stage of the dairy-cattle modeling or prediction pipeline, before dataset
cleaning, integration, feature engineering, and temporal construction.

Main inputs:
    input_dir:
        Directory containing the raw monthly CSV files.
    output_dir:
        Directory where the merged CSV file is saved.
    code:
        Dataset code used to identify which files should be merged.
    time:
        One or more years to include. If multiple years are provided, the full
        inclusive range between the minimum and maximum year is used.
    output_file:
        Name of the merged output CSV file.
    log_level:
        Logging verbosity level.

Main process:
    The script validates the input directory, creates the output directory,
    expands the requested year range, searches for CSV files matching the
    selected dataset code and year, and appends them into one output CSV using
    chunked reading. Each appended row receives a `_source_file` column to keep
    traceability to the original monthly file.

Main outputs:
    A merged CSV file containing all matching rows from the selected years.
    Logging messages reporting discovered files, processed rows, skipped years,
    failed files, and final merge status.

The resulting merged table is used as a raw consolidated source for subsequent
processing of milk-recording, conductivity, lactose, calving, or demographic
data in the mammary pathology risk workflow.
"""


from __future__ import annotations
import argparse
import logging
import os
from pathlib import Path
from typing import Iterable
import pandas as pd

# Global variables

logger = logging.getLogger(__name__)

# Support functions

def normalize_time_list(time_list: Iterable[int | str]) -> list[int]:
    """Convert the input time list into a sorted list of years."""
    years = {int(t) for t in time_list}
    if not years:
        raise ValueError("time list is empty; at least one year is required.")
    if len(years) == 1:
        single_year = next(iter(years))
        del years
        return [single_year]
    start_year, end_year = min(years), max(years)
    del years
    return list(range(start_year, end_year + 1))

def sort_by_date_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Try to sort the dataframe by year, month, and day columns."""
    if df.empty:
        return df
    # Map lowercase names to original column names
    cols_lower = {col.lower(): col for col in df.columns}
    # Define candidate names for date-like columns
    year_candidates = ["year", "anno"]
    month_candidates = ["month", "mese"]
    day_candidates = ["day", "giorno"]
    year_col = next((cols_lower[name] for name in year_candidates if name in cols_lower), None)
    month_col = next((cols_lower[name] for name in month_candidates if name in cols_lower), None)
    day_col = next((cols_lower[name] for name in day_candidates if name in cols_lower), None)
    # Sort only if all date-like columns are available
    if year_col and month_col and day_col:
        logger.info("Sorting combined dataframe by columns: %s, %s, %s", year_col, month_col, day_col)
        try:
            df_sorted = df.sort_values(by=[year_col, month_col, day_col]).reset_index(drop=True)
            del cols_lower
            return df_sorted
        except Exception as exc:
            logger.warning("Failed to sort by (%s, %s, %s): %s. Returning unsorted dataframe.", year_col, month_col, day_col, exc)
            del cols_lower
            return df
    logger.warning("Could not find year/month/day (or anno/mese/giorno) columns for sorting. ""Columns available: %s. Returning unsorted dataframe.", list(df.columns))
    del cols_lower
    return df

def merge_tables(input_dir: str, output_dir: str, code: str, time: list[int] | list[str], output_file: str) -> None:
    """Merge matching CSV files into one output CSV using chunked streaming."""
    in_path = Path(input_dir).expanduser().resolve()
    out_path = Path(output_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    # Validate input directory and normalize years
    if not in_path.exists() or not in_path.is_dir():
        msg = f"Input directory does not exist or is not a directory: {in_path}"
        logger.error(msg)
        raise ValueError(msg)
    years = normalize_time_list(time)
    output_file_path = out_path / output_file
    logger.info("Starting MERGE (streaming) | input_dir=%s | output_dir=%s | code=%s | years=%s | output_file=%s", in_path, out_path, code, years, output_file)
    # Remove old output file if it already exists
    if output_file_path.exists():
        logger.warning("Output file %s already exists and will be overwritten.", output_file_path)
        output_file_path.unlink()
    first_chunk = True
    total_rows = 0
    total_files = 0
    # Process files year by year
    for year in years:
        year_str = str(year)
        year_files = sorted(file_name for file_name in os.listdir(in_path) if file_name.endswith(".csv") and f"{code}-{year_str}" in file_name)
        if not year_files:
            logger.warning("No CSV files found for code=%s and year=%s in %s", code, year, in_path)
            del year_str, year_files
            continue
        logger.info("Found %d CSV files for code=%s and year=%s", len(year_files), code, year)
        total_files += len(year_files)
        # Read each file in chunks and append it to the output CSV
        for file_name in year_files:
            file_path = in_path / file_name
            logger.info("Processing file: %s", file_path)
            try:
                for chunk in pd.read_csv(file_path, chunksize=100_000, dtype=str):
                    chunk["_source_file"] = str(file_path)
                    write_mode = "w" if first_chunk else "a"
                    write_header = first_chunk
                    chunk.to_csv(output_file_path, mode=write_mode, header=write_header, index=False)
                    first_chunk = False
                    total_rows += len(chunk)
                    del chunk, write_mode, write_header
            except Exception as exc:
                logger.warning("Failed to read/append CSV file %s: %s", file_path, exc)
            del file_path
        del year_str, year_files
    # Validate final merge result
    if total_files == 0:
        del years, in_path, out_path, output_file_path
        msg = "No CSV files processed: nothing to merge."
        logger.error(msg)
        raise RuntimeError(msg)
    logger.info("Streaming merge completed | files=%d | rows≈%d | output=%s", total_files, total_rows, output_file_path)
    # Free references that are no longer needed
    del years, in_path, out_path, output_file_path

# Parsing

def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Merge LEO CSV files for one or more years into a single table.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Define CLI arguments
    parser.add_argument("--input-dir", type=str, required=True, help="Directory where CSV files are searched.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory where the merged CSV will be saved.")
    parser.add_argument("--code", type=str, required=True, help="Dataset code (for example: ACF_1097).")
    parser.add_argument("--time", type=int, nargs="+", required=True, help=("One or more years. If you pass two values, they are used as an inclusive range [min, max]. "))
    parser.add_argument("--output-file", type=str, required=True, help="Name of the output CSV file.")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level.")
    return parser

# Main

def main() -> None:
    """Parse CLI arguments and run the merge workflow."""
    parser = build_arg_parser()
    args = parser.parse_args()
    # Configure logging and show input arguments
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logger.info("CLI input: %s", args)
    # Run the merge procedure
    try:
        merge_tables(input_dir=args.input_dir, output_dir=args.output_dir, code=args.code, time=args.time, output_file=args.output_file)
    except Exception as exc:
        logger.error("Execution failed: %s", exc)
        del parser, args
        raise SystemExit(1)
    logger.info("Merge completed successfully.")
    # Free references that are no longer needed
    del parser, args


if __name__ == "__main__":
    main()

# COMMAND LINES

# Electrical conductivity: python3 merge_tables.py --input-dir workspace/data/database/ce --output-dir workspace/data/database/ce/merged --code ACF_1097 --time 2019 2024 --output-file ce.csv
# Lactose: python3 merge_tables.py --input-dir workspace/data/database/latt --output-dir workspace/data/database/latt/merged --code ACF_1007 --time 2019 2024 --output-file latt.csv
# Functional check: python3 merge_tables.py --input-dir workspace/data/database/cf --output-dir workspace/data/database/cf/merged --code CFL --time 2019 2024 --output-file cf.csv
# Calving: python3 merge_tables.py --input-dir workspace/data/database/parti --output-dir workspace/data/database/parti/merged --code PA --time 2019 2024 --output-file parti.csv
# Demography: python3 merge_tables.py --input-dir workspace/data/database/anag --output-dir workspace/data/database/anag/merged --code ANA --time 2018 2023 --output-file anag.csv
