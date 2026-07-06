"""
Treatments pipeline.

This script:
- Loads raw treatment records from CSV.
- Keeps only rows for a target diagnosis.
- Parses dates, filters invalid/old years, and deduplicates to one treatment per animal-month.
- Joins eartag codes to internal animal IDs (via 'coppie' mapping).
- Restricts to animals present in the functional check (CF) universe.
- Saves compact Parquet outputs for downstream analyses.
"""

from libraries import pd, Path, log, gc, reduce, np
import argparse
from collections import defaultdict

# Global variables 

RENAME_MAP = {"CAPO_IDENTIFICATIVO": "Marca", "TIPODIAGNOSI_CODICE": "diagnosis", "TRAT_DT_INIZIO_parsed": "t_date","giorno": "day", "mese": "month", "anno": "year"}

# Support functions

def counts_per_animal_from_dict(d: dict) -> pd.Series:
    """
    Convert a per-animal counter dictionary to a pandas Series.
    """
    if not d:
        return pd.Series(dtype="int64")
    return pd.Series(d, dtype="int64")

def show_dist(label: str, s: pd.Series) -> None:
    """
    Log row and per-animal count distribution.
    """
    if len(s) == 0:
        log.info("%s - rows:0  n_ids:0  min:0  median:0  max:0", label)
        return
    log.info("%s - rows:%d  n_ids:%d  min:%d  median:%d  max:%d", label, int(s.sum()), int(len(s)), int(s.min()), int(s.median()), int(s.max()))

def unisci(lista_df, lista_chiavi, metodo):
    """
    Merge multiple DataFrames on shared keys after harmonizing key dtypes.
    """
    if not lista_df:
        return pd.DataFrame()
    for chiave in lista_chiavi:
        tipi = []
        for df in lista_df:
            if chiave not in df.columns:
                raise KeyError(f"Column '{chiave}' is not in all DataFrames.")
            tipi.append(df[chiave].dtype)
        tipo_names = [t.name for t in tipi]
        if any(t.name == "category" for t in tipi) or ("object" in tipo_names or "string" in tipo_names):
            for i, df in enumerate(lista_df):
                lista_df[i] = df.copy()
                lista_df[i][chiave] = df[chiave].astype(str)
            continue
        if all(np.issubdtype(t, np.number) for t in tipi):
            tipo_comune = tipi[0]
            for t in tipi[1:]:
                tipo_comune = np.promote_types(tipo_comune, t)
            for i, df in enumerate(lista_df):
                lista_df[i] = df.copy()
                lista_df[i][chiave] = df[chiave].astype(tipo_comune)
            continue
        raise TypeError(f"Not compatible types for '{chiave}': {tipo_names}")
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=lista_chiavi, how=metodo), lista_df)
    if merged_df.empty:
        raise ValueError("Join step created an empty dataframe: please, check.")
    return merged_df

def build_treatments_agg(diagnosis: str, keep_all_years: bool, chunksize: int):
    """
    Read treatments CSV in chunks, compute filtering statistics, and keep one earliest
    treatment per animal-month after global re-aggregation.
    """
    cols = ["CAPO_IDENTIFICATIVO", "TIPODIAGNOSI_CODICE", "TRAT_DT_INIZIO"]
    dtype = {"CAPO_IDENTIFICATIVO": "string", "TIPODIAGNOSI_CODICE": "string"}
    count0 = defaultdict(int)
    count1 = defaultdict(int)
    count2 = defaultdict(int)
    ids_parse = set()
    ids_old = set()
    rows_parse_fail = 0
    rows_old = 0
    diagnosis_rows = 0
    parts = []
    reader = pd.read_csv(RAW_CSV, low_memory=False, usecols=cols, dtype=dtype, chunksize=chunksize)
    for i, chunk in enumerate(reader, start=1):
        log.info("Processing chunk %d...", i)
        chunk = chunk[chunk["TIPODIAGNOSI_CODICE"] == diagnosis]
        diagnosis_rows += len(chunk)
        if len(chunk) == 0:
            del chunk
            gc.collect()
            continue
        chunk = chunk.dropna(subset=["TRAT_DT_INIZIO"])
        if len(chunk) == 0:
            del chunk
            gc.collect()
            continue
        vc0 = chunk["CAPO_IDENTIFICATIVO"].value_counts()
        for k, v in vc0.items():
            count0[k] += int(v)
        chunk["TRAT_DT_INIZIO_parsed"] = pd.to_datetime(chunk["TRAT_DT_INIZIO"], errors="coerce", dayfirst=True)
        parse_fail = chunk["TRAT_DT_INIZIO_parsed"].isna()
        if parse_fail.any():
            rows_parse_fail += int(parse_fail.sum())
            ids_parse.update(chunk.loc[parse_fail, "CAPO_IDENTIFICATIVO"].astype(str).tolist())
        chunk = chunk[~parse_fail]
        if len(chunk) == 0:
            del chunk, vc0, parse_fail
            gc.collect()
            continue
        vc1 = chunk["CAPO_IDENTIFICATIVO"].value_counts()
        for k, v in vc1.items():
            count1[k] += int(v)
        chunk["anno"] = chunk["TRAT_DT_INIZIO_parsed"].dt.year.astype("int16")
        if not keep_all_years:
            mask_old = chunk["anno"] <= 2019
            if mask_old.any():
                rows_old += int(mask_old.sum())
                ids_old.update(chunk.loc[mask_old, "CAPO_IDENTIFICATIVO"].astype(str).tolist())
            chunk = chunk[~mask_old]
            if len(chunk) == 0:
                del chunk, vc0, vc1, parse_fail, mask_old
                gc.collect()
                continue
        vc2 = chunk["CAPO_IDENTIFICATIVO"].value_counts()
        for k, v in vc2.items():
            count2[k] += int(v)
        chunk["giorno"] = chunk["TRAT_DT_INIZIO_parsed"].dt.day.astype("int8")
        chunk["mese"] = chunk["TRAT_DT_INIZIO_parsed"].dt.month.astype("int8")
        chunk = chunk.sort_values("TRAT_DT_INIZIO_parsed", kind="mergesort")
        chunk = chunk.drop_duplicates(subset=["CAPO_IDENTIFICATIVO", "anno", "mese"], keep="first")
        parts.append(chunk[["CAPO_IDENTIFICATIVO", "TIPODIAGNOSI_CODICE", "TRAT_DT_INIZIO", "TRAT_DT_INIZIO_parsed", "anno", "mese", "giorno"]])
        del chunk, vc0, vc1, vc2, parse_fail
        gc.collect()
    log.info("%d rows related to '%s' diagnosis uploaded", diagnosis_rows, diagnosis)
    c0 = counts_per_animal_from_dict(count0)
    c1 = counts_per_animal_from_dict(count1)
    c2 = counts_per_animal_from_dict(count2)
    show_dist("Step 0 (raw)", c0)
    show_dist("Step 1 (parsed)", c1)
    show_dist("Step 2 (year filter)", c2)
    if rows_parse_fail:
        log.info("%d rows with invalid dates removed (involved animals: %d)", rows_parse_fail, len(ids_parse))
    if not keep_all_years and rows_old:
        log.info("%d rows with year <= 2019 removed (involved animals: %d)", rows_old, len(ids_old))
    if keep_all_years:
        log.info("Year filter deactivated.")
    delta = (c0 - c2).fillna(0).astype(int)
    mismatch = delta[delta != 0]
    if mismatch.empty:
        log.info("Coherent treatments count for all animals.")
    else:
        log.warning("%d animals with gained/lost treatments after filtering.", len(mismatch))
        log.info("    - %d associated to invalid dates.", len(ids_parse & set(mismatch.index.astype(str))))
        if not keep_all_years:
            log.info("    - %d associated to year <= 2019.", len(ids_old & set(mismatch.index.astype(str))))
        log.debug("Example IDs: %s", list(mismatch.index[:20]))
    if not parts:
        return pd.DataFrame(columns=["CAPO_IDENTIFICATIVO", "TIPODIAGNOSI_CODICE", "TRAT_DT_INIZIO", "TRAT_DT_INIZIO_parsed", "anno", "mese", "giorno"])
    log.info("Concatenating chunk-level monthly selections...")
    df = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    df = df.sort_values("TRAT_DT_INIZIO_parsed", kind="mergesort")
    df = df.drop_duplicates(subset=["CAPO_IDENTIFICATIVO", "anno", "mese"], keep="first")
    log.info("After selecting the first date per animal-month: %d rows", len(df))
    return df

def treat_main(diagnosis: str, keep_all_years: bool = False, chunksize: int = 200_000) -> None:
    """
    Load treatments, keep a target diagnosis, parse/filter dates, deduplicate to one
    treatment per animal-month, join with eartag mapping, restrict to CF animals,
    and save final parquet outputs.
    """
    ids_cf = set(pd.read_parquet(CF_IDS_PARQUET)["id"].astype("int64").unique())
    log.info("Unique IDs from functional check: %d", len(ids_cf))

    df = build_treatments_agg(diagnosis=diagnosis, keep_all_years=keep_all_years, chunksize=chunksize)
    if len(df) == 0:
        OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        COPPIE_TRAT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        empty = df.rename(columns={"CAPO_IDENTIFICATIVO": "Marca"}).rename(columns=RENAME_MAP)
        empty.to_parquet(OUTPUT_PARQUET, index=False)
        empty.to_parquet(COPPIE_TRAT_PARQUET, index=False)
        log.info("No rows to save after filtering.")
        del df, empty, ids_cf
        gc.collect()
        return

    coppie_df = pd.read_parquet(COPPIE_PARQUET).rename(columns={"idAnimale": "id"})
    df = df.rename(columns={"CAPO_IDENTIFICATIVO": "Marca"})
    df_joined = unisci([coppie_df, df], ["Marca"], "inner")
    log.info("After join: %d rows, %d animals with known ID", len(df_joined), df_joined["id"].nunique())
    del coppie_df, df
    gc.collect()
    df_joined = df_joined[df_joined["id"].isin(ids_cf)]
    log.info("After filtering with CF IDs: %d rows, %d animals", len(df_joined), df_joined["id"].nunique())
    df_joined = df_joined.drop("TRAT_DT_INIZIO", axis=1)
    df_joined = df_joined.rename(columns=RENAME_MAP)
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    COPPIE_TRAT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_joined.to_parquet(OUTPUT_PARQUET, index=False)
    log.info("File saved -> %s (%d rows)", OUTPUT_PARQUET, len(df_joined))
    df_joined.to_parquet(COPPIE_TRAT_PARQUET, index=False)
    del df_joined, ids_cf
    gc.collect()

# Parsing

def parse_args():
    """
    Parse CLI arguments for treatments input, mapping files, outputs, diagnosis,
    optional year filtering, and CSV chunk size.
    """
    parser = argparse.ArgumentParser(description="Aggregate and filter treatment records.")
    parser.add_argument("--input-treatments-csv", type=Path, help="Path to raw treatments CSV.")
    parser.add_argument("--input-coppie", type=Path, required=True, help="Path to coppie parquet.")
    parser.add_argument("--input-cf-ids", type=Path, required=True, help="Path to CF IDs parquet.")
    parser.add_argument("--output-parquet", type=Path, required=True, help="Path to output parquet.")
    parser.add_argument("--output-coppie-parquet", type=Path, required=True, help="Path to output coppie parquet.")
    parser.add_argument("--diagnosis", type=str, required=True, help="Diagnosis code to keep.")
    parser.add_argument("--keep-all-years", action="store_true", help="Disable the year > 2019 filter.")
    parser.add_argument("--chunksize", type=int, default=200_000, help="CSV chunk size.")
    return parser.parse_args()

# Main call

if __name__ == "__main__":
    args = parse_args()
    RAW_CSV = args.input_treatments_csv
    COPPIE_PARQUET = args.input_coppie
    CF_IDS_PARQUET = args.input_cf_ids
    OUTPUT_PARQUET = args.output_parquet
    COPPIE_TRAT_PARQUET = args.output_coppie_parquet
    treat_main(diagnosis=args.diagnosis, keep_all_years=args.keep_all_years, chunksize=args.chunksize)
