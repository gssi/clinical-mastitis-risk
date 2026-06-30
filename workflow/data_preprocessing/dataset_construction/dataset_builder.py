"""
Flexible merged dataset builder.

This script:
- Loads a base parquet dataset.
- Sequentially merges additional parquet datasets provided from CLI.
- Harmonizes join-key dtypes before each merge.
- Saves a single merged parquet dataset.

CLI pattern:
- --base /path/to/base.parquet
- --join name=/path/to/file.parquet
- --keys name:key1,key2,...
- --how name:left|inner|outer|right

Example:
- base = cf
- join ltts on [id, day, month, year] with left
- join ce on [id, day, month, year] with left
- join parti on [id, month, year] with outer
- join coppie_trat on [id, month, year] with outer
- join ana on [id] with inner
"""

from libraries import Path, log, pd, gc, np
import argparse

# Global variables

VALID_JOIN_TYPES = {"left", "right", "inner", "outer"}

# Support functions

def parse_named_path(values: list[str]) -> dict[str, Path]:
    """
    Parse repeated CLI entries formatted as name=path into a dictionary.
    """
    result = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Invalid --join entry '{item}'. Expected format: name=path")
        name, path = item.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValueError(f"Invalid --join entry '{item}'. Expected format: name=path")
        result[name] = Path(path)
    return result

def parse_named_keys(values: list[str]) -> dict[str, list[str]]:
    """
    Parse repeated CLI entries formatted as name:key1,key2,... into a dictionary.
    """
    result = {}
    for item in values or []:
        if ":" not in item:
            raise ValueError(f"Invalid --keys entry '{item}'. Expected format: name:key1,key2,...")
        name, keys = item.split(":", 1)
        name = name.strip()
        key_list = [k.strip() for k in keys.split(",") if k.strip()]
        if not name or not key_list:
            raise ValueError(f"Invalid --keys entry '{item}'. Expected format: name:key1,key2,...")
        result[name] = key_list
    return result

def parse_named_how(values: list[str]) -> dict[str, str]:
    """
    Parse repeated CLI entries formatted as name:join_type into a dictionary.
    """
    result = {}
    for item in values or []:
        if ":" not in item:
            raise ValueError(f"Invalid --how entry '{item}'. Expected format: name:join_type")
        name, how = item.split(":", 1)
        name, how = name.strip(), how.strip().lower()
        if not name or how not in VALID_JOIN_TYPES:
            raise ValueError(f"Invalid --how entry '{item}'. Join type must be one of: {sorted(VALID_JOIN_TYPES)}")
        result[name] = how
    return result

def harmonize_join_keys(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Harmonize join-key dtypes between two DataFrames before merging.
    """
    left = left.copy()
    right = right.copy()
    for key in keys:
        if key not in left.columns:
            raise KeyError(f"Key '{key}' is missing in left DataFrame.")
        if key not in right.columns:
            raise KeyError(f"Key '{key}' is missing in right DataFrame.")
        lt, rt = left[key].dtype, right[key].dtype
        lt_name, rt_name = lt.name, rt.name
        if any(t == "category" for t in [lt_name, rt_name]) or any(t in {"object", "string"} for t in [lt_name, rt_name]):
            left[key] = left[key].astype(str)
            right[key] = right[key].astype(str)
            continue
        if np.issubdtype(lt, np.number) and np.issubdtype(rt, np.number):
            common = np.promote_types(lt, rt)
            left[key] = left[key].astype(common)
            right[key] = right[key].astype(common)
            continue
        raise TypeError(f"Incompatible dtypes for key '{key}': {lt_name} vs {rt_name}")
    return left, right

def merge_two(left: pd.DataFrame, right: pd.DataFrame, keys: list[str], how: str, label: str) -> pd.DataFrame:
    """
    Harmonize join keys and merge two DataFrames.
    """
    log.info("%s - merging on %s with how='%s'", label, keys, how)
    left, right = harmonize_join_keys(left, right, keys)
    merged = pd.merge(left, right, on=keys, how=how)
    if merged.empty:
        raise ValueError(f"{label} produced an empty DataFrame. Check join keys and join type.")
    del left, right
    gc.collect()
    log.info("%s - result: %d rows, %d columns", label, len(merged), merged.shape[1])
    return merged

def cleanup_merged_dataset(df: pd.DataFrame, drop_columns: list[str] | None = None) -> pd.DataFrame:
    """
    Drop technical columns created during merging and remove perfect duplicates.
    """
    drop_columns = drop_columns or []
    before_rows = len(df)
    df = df.drop(columns=drop_columns, errors="ignore")
    after_drop = len(df)
    df = df.drop_duplicates()
    after_dedup = len(df)
    log.info("Cleanup completed - rows: %d -> %d after column cleanup -> %d after dedup", before_rows, after_drop, after_dedup)
    return df

def merge_main(base_path: Path, joins: dict[str, Path], keys_map: dict[str, list[str]], how_map: dict[str, str], 
               output_parquet: Path, drop_columns: list[str] | None = None) -> None:
    """
    Load the base dataset, sequentially merge the requested datasets, and save the final parquet.
    """
    log.info("START MERGING PHASE")
    log.info("Loading base dataset: %s", base_path)
    merged = pd.read_parquet(base_path)
    log.info("Base dataset loaded: %d rows, %d columns", len(merged), merged.shape[1])
    for name, path in joins.items():
        if name not in keys_map:
            raise ValueError(f"Missing --keys entry for join '{name}'")
        if name not in how_map:
            raise ValueError(f"Missing --how entry for join '{name}'")
        log.info("Loading join dataset '%s': %s", name, path)
        right = pd.read_parquet(path)
        merged = merge_two(merged, right, keys=keys_map[name], how=how_map[name], label=f"Join '{name}'")
        del right
        gc.collect()
    if drop_columns:
        merged = cleanup_merged_dataset(merged,drop_columns=drop_columns)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_parquet, index=False)
    log.info("Merged dataset saved -> %s (%d rows, %d columns)", output_parquet, len(merged), merged.shape[1])
    del merged
    gc.collect()
    log.info("### END MERGING PHASE ###")

# Parsing

def parse_args():
    """
    Parse CLI arguments for flexible sequential parquet merges.
    """
    parser = argparse.ArgumentParser(description="Flexible sequential parquet merger.")
    parser.add_argument("--base", type=Path, required=True, help="Path to the base parquet dataset.")
    parser.add_argument("--join", action="append", default=[], help="Join dataset in the format name=path. Repeatable.")
    parser.add_argument("--keys", action="append", default=[], help="Join keys in the format name:key1,key2,... Repeatable.")
    parser.add_argument("--how", action="append", default=[], help="Join type in the format name:left|inner|outer|right. Repeatable.")
    parser.add_argument("--drop-columns", type=str, default="Marca,day_x,day_y,day,same_day_records", help="Comma-separated columns to drop at the end.")
    parser.add_argument("--output-parquet", type=Path, required=True, help="Path to output parquet.")
    return parser.parse_args()

# Main call

if __name__ == "__main__":
    args = parse_args()
    joins = parse_named_path(args.join)
    keys_map = parse_named_keys(args.keys)
    how_map = parse_named_how(args.how)
    drop_columns = [c.strip() for c in args.drop_columns.split(",") if c.strip()]
    merge_main(base_path=args.base, joins=joins, keys_map=keys_map, how_map=how_map, output_parquet=args.output_parquet, drop_columns=drop_columns)