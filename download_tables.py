"""
Download and extract monthly LEO open-data CSV tables.

This script downloads zipped monthly CSV files from a LEO open-data endpoint
and extracts them into a local output directory. It is used to collect raw
zootechnical data sources required by the modeling or prediction pipeline, such
as milk electrical conductivity, lactose, functional control records, calving
records, and animal demographic data.

Main inputs:
    base_url:
        Base URL of the LEO dataset endpoint.
    code:
        Dataset code used in the monthly zip and CSV filenames.
    time:
        One or more years to download. If multiple years are provided, the
        script downloads the full inclusive range between the minimum and
        maximum year.
    output_dir:
        Local directory where extracted CSV files are saved.
    log_level:
        Logging verbosity level.

Main process:
    The script validates the input URL, creates the output directory, expands
    the requested year range, and iterates over all months for each selected
    year. For each year-month pair, it builds the expected zip URL, skips the
    download if the corresponding CSV already exists, downloads the zip file in
    memory, extracts its content, verifies that the expected CSV was created,
    and logs download status together with optional CPU and memory information.

Main outputs:
    Extracted monthly CSV files saved in the requested output directory.
    Logging messages describing skipped files, successful extractions, failed
    downloads, and execution summary.

The downloaded raw tables provide the input layer for subsequent dairy-cattle
data processing steps, including dataset construction, temporal modeling, and
mammary pathology risk prediction workflows.
"""


from __future__ import annotations
import argparse
import logging
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile
try:
    import psutil
except ImportError:
    psutil = None

# Global variables

logger = logging.getLogger(__name__)

# Support functions

def normalize_time_list(time_list: Iterable[int | str]) -> list[int]:
    """Convert the input time list into a list of years."""
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

def prepare_output_dir(output_dir: str) -> Path:
    """Create the output directory and return its resolved path."""
    out_path = Path(output_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path

def check_base_url(base_url: str) -> None:
    """Validate that the base URL starts with http or https."""
    if not base_url.startswith(("http://", "https://")):
        msg = f"Invalid base_url (must start with http/https): {base_url}"
        logger.error(msg)
        raise ValueError(msg)

def build_zip_url(base_url: str, code: str, year: int, month_str: str) -> str:
    """Build the zip file URL for the given dataset and month."""
    base = base_url.rstrip("/")
    return f"{base}/{year}/{month_str}/{code}-{year}-{month_str}.csv.zip"

def download_and_extract_zip(zip_url: str, output_dir: Path) -> None:
    """Download a zip file in memory and extract it into the output directory."""
    with urllib.request.urlopen(zip_url) as response:
        data = response.read()
    with ZipFile(BytesIO(data)) as zip_file:
        zip_file.extractall(output_dir)
    # Free memory used by in-memory objects
    del data
    del zip_file

def get_resource_usage() -> tuple[float | None, float | None]:
    """Return current process memory in MB and CPU usage percentage."""
    if psutil is None:
        return None, None
    proc = psutil.Process()
    mem_bytes = proc.memory_info().rss
    mem_mb = mem_bytes / (1024 * 1024)
    cpu_percent = proc.cpu_percent(interval=0.0)
    # Free the process handle reference
    del proc
    return mem_mb, cpu_percent

def configure_logging(log_level: str) -> None:
    """Configure the global logging settings."""
    logging.basicConfig(level=getattr(logging, log_level), format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

def download_tables(base_url: str, code: str, time: list[int] | list[str], output_dir: str) -> None:
    """Download and extract LEO zipped CSV tables for the selected years."""
    years = normalize_time_list(time)
    out_dir = prepare_output_dir(output_dir)
    # Validate inputs and start the download procedure
    logger.info(
        "Starting download_tables | base_url=%s | code=%s | years=%s | output_dir=%s", base_url, code, years, out_dir)
    check_base_url(base_url)
    total_attempts = 0
    total_success = 0
    # Iterate over years and months and skip files that already exist
    for year in years:
        for month in range(1, 13):
            month_str = f"{month:02d}"
            file_name = f"{code}-{year}-{month_str}.csv"
            csv_path = out_dir / file_name
            if csv_path.exists():
                logger.info("File already exists, skipping: %s", csv_path)
                del month_str, file_name, csv_path
                continue
            zip_url = build_zip_url(base_url, code, year, month_str)
            total_attempts += 1
            logger.info("Downloading year=%s month=%s | url=%s | target=%s", year, month_str, zip_url, csv_path)
            # Download the zip and verify that the expected CSV exists
            try:
                download_and_extract_zip(zip_url, out_dir)
                if csv_path.exists():
                    total_success += 1
                    mem_mb, cpu_percent = get_resource_usage()
                    logger.info("Extracted: %s | rss=%.2f MB | cpu=%.1f%%", csv_path, mem_mb if mem_mb is not None else -1.0,
                                 cpu_percent if cpu_percent is not None else -1.0)
                    del mem_mb, cpu_percent
                else:
                    logger.warning("Zip downloaded from %s but expected CSV not found: %s", zip_url, csv_path)
            except urllib.error.URLError as exc:
                logger.warning("Failed to download %s (year=%s, month=%s): %s", zip_url, year, month_str, exc)
            except Exception as exc:
                logger.warning("Unexpected error processing %s (year=%s, month=%s): %s", zip_url, year, month_str, exc)
            # Free temporary references created in the loop
            del month_str, file_name, csv_path, zip_url
    # Check the final result of the execution
    if total_attempts == 0:
        logger.warning("No download attempts were made (maybe all files already existed?).")
        del years, out_dir
        return
    if total_success == 0:
        del years, out_dir
        msg = "All download attempts failed: no new tables were saved."
        logger.error(msg)
        raise RuntimeError(msg)
    logger.info("Completed download_tables | attempts=%d | success=%d", total_attempts, total_success)
    # Free references that are no longer needed
    del years, out_dir

# Parsing

def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Download LEO datasets for one or more years.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Define CLI arguments
    parser.add_argument("--base-url", type=str, required=True, help="Base URL of the LEO endpoint.")
    parser.add_argument("--code", type=str, required=True, help="LEO dataset code (for example: ACF_1097).")
    parser.add_argument("--time", type=int, nargs="+", required=True, help=("One or more years. If you pass two values, they are interpreted as an inclusive range "
                                                                            "(start, end). Examples: --time 2025 | --time 2023 2025"))
    parser.add_argument("--output-dir", type=str, required=True, help="Destination folder for extracted CSV files.")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level.")
    return parser

# Main

def main() -> None:
    """Parse CLI arguments and run the download workflow."""
    parser = build_arg_parser()
    args = parser.parse_args()
    # Configure logging and read CLI inputs
    configure_logging(args.log_level)
    logging.info("CLI input received: %s", args)
    base_url: str = args.base_url
    code: str = args.code
    time_list: list[int] = args.time
    output_dir: str = args.output_dir
    # Free parser and args references that are no longer needed later
    del parser
    # Run the main download function
    try:
        download_tables(base_url=base_url, code=code, time=time_list, output_dir=output_dir)
    except Exception as exc:
        logging.error("Execution failed: %s", exc)
        del args, base_url, code, time_list, output_dir
        raise SystemExit(1)
    logging.info("Download completed successfully.")
    # Free final references
    del args, base_url, code, time_list, output_dir


if __name__ == "__main__":
    main()

## COMMAND LINES

# Electrical conductivity : python3 download_tables.py --base-url https://opendata.leo-italy.eu/leo-api/public/ACF --code ACF_1097 --time 2019 2024 --output-dir workspace/data/database/ce
# Lactose: python3 download_tables.py --base-url https://opendata.leo-italy.eu/leo-api/public/ACF --code ACF_1007 --time 2019 2024 --output-dir workspace/data/database/latt
# Functional check: python3 download_tables.py --base-url https://opendata.leo-italy.eu/leo-api/public/CFL --code CFL --time 2019 2024 --output-dir workspace/data/database/cf
# Calving: python3 download_tables.py --base-url https://opendata.leo-italy.eu/leo-api/public/PA --code PA --time 2019 2024 --output-dir workspace/data/database/parti
# Demography: python3 download_tables.py --base-url https://opendata.leo-italy.eu/leo-api/public/ANA --code ANA --time 2018 2023 --output-dir workspace/data/database/anag
