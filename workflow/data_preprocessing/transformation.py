"""
Pre-processing and domain informed transformation (dit) for the unified dairy dataset.
"""

from libraries import pd,np,log,gc,Path
import argparse

# Support functions 

def ensure_dt(df:pd.DataFrame,cols:list[str])->pd.DataFrame:
    """Coerce selected columns to datetime."""
    for c in cols:
        if c in df.columns: df[c]=pd.to_datetime(df[c],errors="coerce")
    return df

def pick_reference_date(df:pd.DataFrame)->pd.Series:
    """Build reference_date with priority cf_date > calving_date > t_date."""
    ref=pd.Series(pd.NaT,index=df.index,dtype="datetime64[ns]")
    if "cf_date" in df.columns: ref=df["cf_date"]
    if "calving_date" in df.columns: ref=ref.combine_first(df["calving_date"])
    if "t_date" in df.columns: ref=ref.combine_first(df["t_date"])
    return ref

def breed_coherence(df:pd.DataFrame)->pd.DataFrame:
    """Impute breed per animal using first non-null after stable temporal ordering."""
    if "breed" not in df.columns: return df
    breed_per_id=df.loc[df["breed"].notna()].groupby("id",observed=True)["breed"].first()
    df["breed"]=df["breed"].fillna(df["id"].map(breed_per_id))
    return df

def pre_processing(input_path:Path,output_path:Path)->None:
    """
    Load merged data, coerce dates, compute robust age, filter implausible ages,
    fill calving with 0, enforce breed coherence, and save cleaned parquet.
    """
    log.info("Starting pre-processing from file: %s",input_path)
    df=pd.read_parquet(input_path)
    log.info("Input loaded: %d rows, %d columns",len(df),df.shape[1])
    if "id" not in df.columns: raise KeyError("Missing required column 'id'.")
    if "birth_date" not in df.columns: raise KeyError("Missing required column 'birth_date'.")
    df=ensure_dt(df,["birth_date","cf_date","calving_date","t_date"])
    ids_with_birth=df.loc[df["birth_date"].notna(),"id"].unique()
    before=len(df)
    df=df[df["id"].isin(ids_with_birth)].copy()
    log.info("Rows kept after requiring at least one birth_date per animal: %d -> %d",before,len(df))
    birth_per_id=df.loc[df["birth_date"].notna()].groupby("id",observed=True)["birth_date"].first()
    df["birth_date"]=df["birth_date"].fillna(df["id"].map(birth_per_id))
    ref=pick_reference_date(df)
    valid_age_inputs=ref.notna()&df["birth_date"].notna()

    # age definition
    df["age"]=pd.Series(np.nan,index=df.index,dtype="float32")
    age_days=(ref-df["birth_date"]).dt.days
    df.loc[valid_age_inputs,"age"]=np.floor(age_days[valid_age_inputs]/365.25).astype("float32")
    missing_ref=int((~valid_age_inputs).sum())
    if missing_ref: log.warning("Rows with missing reference_date/birth_date for age computation: %d",missing_ref)
    invalid_age=df["age"].isna()|(df["age"]<2)|(df["age"]>7)
    removed_invalid_age=int(invalid_age.sum())
    df=df[~invalid_age].copy()
    df["age"]=df["age"].astype("int16")
    log.info("Rows removed by age filter [2,7]: %d; remaining rows: %d",removed_invalid_age,len(df))

    if "calving" not in df.columns:
        log.warning("Column 'calving' missing. Creating default 0.")
        df["calving"]=0
    df["calving"]=pd.to_numeric(df["calving"],errors="coerce").fillna(0).astype("int8")
    sort_cols=[c for c in ["id","year","month","cf_date","calving_date","t_date"] if c in df.columns]
    if sort_cols: df=df.sort_values(sort_cols,kind="mergesort")
    df=breed_coherence(df)
    if "breed" not in df.columns:
        log.warning("Column 'breed' missing. Creating empty column.")
        df["breed"]=pd.NA
    missing_breed=int(df["breed"].isna().sum())
    if missing_breed: log.warning("Rows with missing breed after coherence step: %d",missing_breed)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    df.to_parquet(output_path,index=False)
    log.info("File saved: %s (%d rows, %d columns)",output_path,len(df),df.shape[1])
    del df,birth_per_id,ref,ids_with_birth,age_days
    gc.collect()
    log.info("Pre-processing completed.")

def dit(input_path:Path,output_path:Path,season_mode:str="both",lactation_mode:str="both")->None:
    """
    Load cleaned data, compute healthy/season/lactation features and disease label,
    then save transformed parquet.
    """
    log.info("Starting domain-informed transformation from file: %s",input_path)
    df=pd.read_parquet(input_path)
    log.info("Input loaded: %d rows, %d columns",len(df),df.shape[1])
    if "id" not in df.columns: raise KeyError("Missing required column 'id'.")
    if "month" not in df.columns: raise KeyError("Missing required column 'month'.")
    if "year" not in df.columns: raise KeyError("Missing required column 'year'.")
    df=ensure_dt(df,["birth_date","cf_date","calving_date","t_date"])
    sort_cols=[c for c in ["id","year","month","cf_date","calving_date","t_date"] if c in df.columns]
    if sort_cols: df=df.sort_values(sort_cols,kind="mergesort")
    if "breed" not in df.columns:
        log.warning("Column 'breed' missing. Creating empty column.")
        df["breed"]=pd.NA
    removed_missing_breed=int(df["breed"].isna().sum())
    if removed_missing_breed: log.warning("Rows removed due to missing breed: %d",removed_missing_breed)
    df=df[df["breed"].notna()].copy()
    if "diagnosis" not in df.columns:
        log.warning("Column 'diagnosis' missing. Creating empty diagnosis column.")
        df["diagnosis"]=pd.NA
    if "scs" not in df.columns:
        log.warning("Column 'scs' missing. Creating empty scs column.")
        df["scs"]=pd.NA
    df["scs"]=pd.to_numeric(df["scs"],errors="coerce")
    ids_with_diagnosis=set(df.loc[df["diagnosis"].notna(),"id"].unique())
    ids_with_high_scs=set(df.loc[df["scs"].notna()&(df["scs"]>=5),"id"].unique())
    unhealthy_ids=ids_with_diagnosis|ids_with_high_scs

    # healthy definition
    df["healthy"]=(~df["id"].isin(unhealthy_ids)).astype("int8")
    animal_health=df.groupby("id",observed=True)["healthy"].first()
    log.info("Healthy animals: %d | unhealthy animals: %d",int(animal_health.eq(1).sum()),int(animal_health.eq(0).sum()))
    month_num=pd.to_numeric(df["month"],errors="coerce")

    # season and cyclic month definition
    season_conditions=[month_num.isin([12,1,2]),month_num.isin([3,4,5]),month_num.isin([6,7,8]),month_num.isin([9,10,11])]
    season_labels=["winter","spring","summer","autumn"]
    if season_mode in {"cat","both"}: df["season"]=np.select(season_conditions,season_labels,default=pd.NA)
    if season_mode in {"cyclic","both"}:
        angle=2*np.pi*(month_num.astype("float32")-1)/12.0
        df["month_sin"]=np.sin(angle).astype("float32")
        df["month_cos"]=np.cos(angle).astype("float32")
    if "calving" not in df.columns:
        log.warning("Column 'calving' missing. Creating default 0.")
        df["calving"]=0
    df["calving"]=pd.to_numeric(df["calving"],errors="coerce").fillna(0).astype("int8")
    if "calving_date" not in df.columns:
        log.warning("Column 'calving_date' missing. Creating NaT.")
        df["calving_date"]=pd.NaT
    df["last_calving_date"]=df.groupby("id",observed=True)["calving_date"].ffill()
    df["record_date"]=pd.to_datetime(dict(year=pd.to_numeric(df["year"],errors="coerce"),month=month_num,day=1),errors="coerce")
    valid_months=df["record_date"].notna()&df["last_calving_date"].notna()
    months_since=pd.Series(np.nan,index=df.index,dtype="float32")
    months_since.loc[valid_months]=((df.loc[valid_months,"record_date"].dt.year-df.loc[valid_months,"last_calving_date"].dt.year)*12+
                                    (df.loc[valid_months,"record_date"].dt.month-df.loc[valid_months,"last_calving_date"].dt.month)).astype("float32")
    months_since[months_since<0]=np.nan

    # months_since_calving and lactation_phase definition
    df["months_since_calving"]=months_since
    if lactation_mode in {"phase","both"}:
        conds=[df["months_since_calving"]==0,df["months_since_calving"]==1,df["months_since_calving"].isin([2,3]),df["months_since_calving"].isin([4,5,6]),df["months_since_calving"]>=7]
        labels=["peripartum","early_lactation","peak","mid_lactation","late_lactation"]
        df["lactation_phase"]=np.select(conds,labels,default=pd.NA)
    if lactation_mode in {"phase","both"}:
        missing_lp=int(df["lactation_phase"].isna().sum())
        if missing_lp:
            log.warning("Rows removed due to missing lactation_phase: %d",missing_lp)
        df=df[df["lactation_phase"].notna()].copy()
    elif lactation_mode=="months":
        missing_msc=int(df["months_since_calving"].isna().sum())
        if missing_msc:
            log.warning("Rows removed due to missing months_since_calving: %d",missing_msc)
        df=df[df["months_since_calving"].notna()].copy()
    if "cf_date" not in df.columns:
        log.warning("Column 'cf_date' missing. Creating NaT.")
        df["cf_date"]=pd.NaT
    if "t_date" not in df.columns:
        log.warning("Column 't_date' missing. Creating NaT.")
        df["t_date"]=pd.NaT
    n_cf=int(df["cf_date"].notna().sum())
    n_t=int(df["t_date"].notna().sum())
    n_diag=int(df["diagnosis"].notna().sum())
    log.info("Rows with cf_date: %d | t_date: %d | diagnosis: %d",n_cf,n_t,n_diag)

    # target disease definition: disease = 1 for a valid cf_date row if the same animal has a treatment recorded within 30 days after that cf_date.
    window_days = 30
    df["disease"] = np.int8(0)
    cf = df.loc[df["cf_date"].notna(), ["id", "cf_date"]].copy()
    cf["row_idx"] = cf.index
    tr = df.loc[df["t_date"].notna(),["id", "t_date"]].drop_duplicates()
    if not cf.empty and not tr.empty:
        matched = cf.merge(tr, on="id", how="left")
        delta_days = (matched["t_date"] - matched["cf_date"]).dt.days
        positive_idx = matched.loc[
            delta_days.notna()
            & (delta_days > 0)
            & (delta_days < window_days),
            "row_idx"].unique()
        if positive_idx.size:
            df.loc[positive_idx, "disease"] = np.int8(1)
        del matched, delta_days, positive_idx
        gc.collect()
    log.info("Rows labeled as disease=1 using treatment within %d days after cf_date: %d", window_days,int(df["disease"].sum()))
    if "diagnosis" in df.columns:
        unmatched_diag = int((df["diagnosis"].notna() & df["disease"].eq(0)).sum())
        if unmatched_diag:
            log.info("Rows with diagnosis not matched to a treatment within %d days after cf_date: %d", window_days, unmatched_diag)
    del cf, tr
    gc.collect()

    removed_missing_cf=int(df["cf_date"].isna().sum())
    if removed_missing_cf:
        log.warning("Rows removed due to missing cf_date after disease recovery step: %d",removed_missing_cf)
    df=df[df["cf_date"].notna()].copy()
    helper_cols=[c for c in ["record_date","last_calving_date"] if c in df.columns]
    if helper_cols: df=df.drop(columns=helper_cols,errors="ignore")
    if lactation_mode=="months" and "lactation_phase" in df.columns: df=df.drop(columns=["lactation_phase"],errors="ignore")
    if season_mode=="cat": df=df.drop(columns=[c for c in ["month_sin","month_cos"] if c in df.columns],errors="ignore")
    if season_mode=="cyclic": df=df.drop(columns=["season"],errors="ignore")
    output_path.parent.mkdir(parents=True,exist_ok=True)
    df = df[df['lactation_phase'] != 'peripartum'].copy()  # Remove peripartum phase rows
    df.to_parquet(output_path,index=False)
    log.info("File saved: %s (%d rows, %d columns)",output_path,len(df),df.shape[1])
    del df,ids_with_diagnosis,ids_with_high_scs,unhealthy_ids,month_num,months_since,animal_health
    gc.collect()
    log.info("Domain-informed transformation completed.")

# Parsing

def parse_args():
    """Parse CLI arguments for pre_processing or dit execution."""
    p=argparse.ArgumentParser(description="Pre-processing and domain-informed transformations for dairy dataset.")
    p.add_argument("--step",choices=["pre_processing","dit"],required=True,help="Pipeline step to execute.")
    p.add_argument("--input-path",type=Path,required=True,help="Input parquet path.")
    p.add_argument("--output-path",type=Path,required=True,help="Output parquet path.")
    p.add_argument("--season-mode",choices=["cat","cyclic","both"],default="both",help="Season representation for dit.")
    p.add_argument("--lactation-mode",choices=["phase","months","both"],default="both",help="Lactation representation for dit.")
    return p.parse_args()

# Main call

if __name__=="__main__":
    args=parse_args()
    if args.step=="pre_processing": pre_processing(args.input_path,args.output_path)
    else: dit(args.input_path,args.output_path,season_mode=args.season_mode,lactation_mode=args.lactation_mode)