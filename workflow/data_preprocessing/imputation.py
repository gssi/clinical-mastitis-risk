"""
Hierarchical clinical imputation for dairy data with fit/transform persistence.
"""

from libraries import pd,np,Path,log,gc
import argparse,time,pickle
from collections import defaultdict
try:
    from scipy.stats import wasserstein_distance as _scipy_wasserstein_distance
except Exception:
    _scipy_wasserstein_distance=None

# Global variables

VARIABLES_COLS=["scs","milk","protein","fat","lactose","ec"]
RANGES={"scs":(0,9),"milk":(0,700),"protein":(0,6),"fat":(0,8),"lactose":(4,5.5),"ec":(500,2000)}
IHG=[["id","lactation_phase","age_class","season"],["id","lactation_phase","age_class"],["id","lactation_phase"]]
PHG=[["breed","lactation_phase","age_class","season"],["breed","lactation_phase","age_class"],["breed","lactation_phase"]]

# Support functions

def safe_numeric(s:pd.Series)->pd.Series:
    """Convert series to numeric safely."""
    return pd.to_numeric(s,errors="coerce")

def corr_matrix(df:pd.DataFrame,cols:list[str])->pd.DataFrame:
    """Pairwise Pearson correlation on selected columns."""
    return df[cols].apply(pd.to_numeric,errors="coerce").corr(method="pearson")

def corr_delta(pre:pd.DataFrame,post:pd.DataFrame,cols:list[str])->tuple[float,pd.DataFrame]:
    """Mean absolute change in pairwise correlations and full absolute delta-correlation matrix."""
    if not cols:return np.nan,pd.DataFrame()
    c1=corr_matrix(pre,cols);c2=corr_matrix(post,cols);d=(c1-c2).abs()
    tri=d.where(np.triu(np.ones_like(d,dtype=bool),1))
    return float(np.nanmean(tri.values)),d

def wasserstein_distance_1d(x:np.ndarray,y:np.ndarray)->float:
    """Exact 1D Wasserstein distance for empirical samples."""
    if _scipy_wasserstein_distance is not None:
        return float(_scipy_wasserstein_distance(x,y))
    x=np.sort(np.asarray(x,dtype=float))
    y=np.sort(np.asarray(y,dtype=float))
    if x.size==0 or y.size==0:return np.nan
    all_vals=np.concatenate([x,y])
    all_vals.sort()
    deltas=np.diff(all_vals)
    if deltas.size==0:return 0.0
    x_cdf=np.searchsorted(x,all_vals[:-1],side="right")/x.size
    y_cdf=np.searchsorted(y,all_vals[:-1],side="right")/y.size
    return float(np.sum(np.abs(x_cdf-y_cdf)*deltas))

def normalized_wasserstein(pre_s:pd.Series,post_s:pd.Series)->float:
    """
    Normalized Wasserstein distance between observed pre-imputation values
    and post-imputation values, scaled by the pre-imputation IQR.
    """
    x=safe_numeric(pre_s).dropna().to_numpy()
    y=safe_numeric(post_s).dropna().to_numpy()
    if x.size==0 or y.size==0:return np.nan
    q1,q3=np.quantile(x,[0.25,0.75])
    iqr=float(q3-q1)
    wd=wasserstein_distance_1d(x,y)
    if not np.isfinite(wd):return np.nan
    if not np.isfinite(iqr) or iqr<=0:
        return 0.0 if wd==0 else np.nan
    return float(wd/iqr)

def nwd_summary(pre:pd.DataFrame,post:pd.DataFrame,cols:list[str])->tuple[float,dict[str,float]]:
    """Per-feature normalized Wasserstein distances and their mean."""
    per_feat={}
    for c in cols:
        if c not in pre.columns or c not in post.columns:continue
        per_feat[c]=normalized_wasserstein(pre[c],post[c])
    vals=[v for v in per_feat.values() if pd.notna(v) and np.isfinite(v)]
    g=float(np.mean(vals)) if vals else np.nan
    return g,per_feat

def summary_rows(pre:pd.DataFrame,post:pd.DataFrame,cols:list[str],imputed_mask:dict|None=None)->pd.DataFrame:
    """Per-variable correlation-preservation and distribution-shift summary metrics."""
    rows=[]
    _,g_delta_mat=corr_delta(pre,post,cols)
    _,per_feat_nwd=nwd_summary(pre,post,cols)
    per_feat_corr={}
    if not g_delta_mat.empty:
        for c in cols:
            others=[x for x in cols if x!=c and x in g_delta_mat.columns]
            per_feat_corr[c]=float(g_delta_mat.loc[c,others].mean()) if others else np.nan
    for c in cols:
        if c not in pre.columns or c not in post.columns:continue
        pct_missing=100.0*pre[c].isna().mean()
        pct_imputed=100.0*(imputed_mask[c].mean()) if imputed_mask and c in imputed_mask else pct_missing
        rows.append({
            "feature":c,
            "pct_missing_pre":round(pct_missing,2),
            "pct_imputed":round(pct_imputed,2),
            "mean_abs_delta_corr":round(per_feat_corr.get(c,np.nan),4),
            "normalized_wasserstein":round(per_feat_nwd.get(c,np.nan),4) if pd.notna(per_feat_nwd.get(c,np.nan)) else np.nan})
    return pd.DataFrame(rows)

def subgroup_feature_summary(pre_df:pd.DataFrame,post_df:pd.DataFrame,cols:list[str],group_col:str,imputed_mask:dict|None=None)->dict[str,pd.DataFrame]:
    """Compute per-feature delta-correlation and normalized Wasserstein summary within each subgroup level."""
    if group_col not in post_df.columns:return {}
    out={}
    levels=post_df[group_col].astype("object").fillna("NA")
    for lvl,idx in levels.groupby(levels).groups.items():
        idx=pd.Index(idx)
        pre_s=pre_df.loc[idx]
        post_s=post_df.loc[idx]
        valid_cols=[c for c in cols if c in pre_s.columns and c in post_s.columns]
        rows=[]
        if not valid_cols:
            out[str(lvl)]=pd.DataFrame()
            continue
        _,delta_mat=corr_delta(pre_s,post_s,valid_cols)
        _,per_feat_nwd=nwd_summary(pre_s,post_s,valid_cols)
        per_feat_corr={}
        if not delta_mat.empty:
            for c in valid_cols:
                others=[x for x in valid_cols if x!=c and x in delta_mat.columns]
                per_feat_corr[c]=float(delta_mat.loc[c,others].mean()) if others else np.nan
        for c in valid_cols:
            pct_missing=100.0*pre_s[c].isna().mean()
            if imputed_mask and c in imputed_mask:
                grp_mask=imputed_mask[c].reindex(idx).fillna(False)
                pct_imputed=100.0*grp_mask.mean()
            else:
                pct_imputed=pct_missing
            rows.append({
                "feature":c,
                "n_rows":int(len(idx)),
                "pct_missing_pre":round(pct_missing,2),
                "pct_imputed":round(pct_imputed,2),
                "mean_abs_delta_corr":round(per_feat_corr.get(c,np.nan),4),
                "normalized_wasserstein":round(per_feat_nwd.get(c,np.nan),4) if pd.notna(per_feat_nwd.get(c,np.nan)) else np.nan,
            })
        out[str(lvl)]=pd.DataFrame(rows).sort_values(["mean_abs_delta_corr","normalized_wasserstein","pct_imputed"],ascending=[False,False,False])
    return out

def subgroup_global_summary(pre_df:pd.DataFrame,post_df:pd.DataFrame,cols:list[str],group_col:str)->pd.DataFrame:
    """Compute subgroup-level mean absolute delta-correlation and mean normalized Wasserstein across all features."""
    if group_col not in post_df.columns:return pd.DataFrame()
    rows=[]
    levels=post_df[group_col].astype("object").fillna("NA")
    for lvl,idx in levels.groupby(levels).groups.items():
        idx=pd.Index(idx)
        pre_s=pre_df.loc[idx]
        post_s=post_df.loc[idx]
        valid_cols=[c for c in cols if c in pre_s.columns and c in post_s.columns]
        if not valid_cols:continue
        mean_delta_corr,_=corr_delta(pre_s,post_s,valid_cols)
        mean_nwd,_=nwd_summary(pre_s,post_s,valid_cols)
        rows.append({"level":str(lvl), "n_rows":int(len(idx)), "mean_abs_delta_corr":round(mean_delta_corr,4) if np.isfinite(mean_delta_corr) else np.nan,
                     "mean_nwd":round(mean_nwd,4) if np.isfinite(mean_nwd) else np.nan})
    return pd.DataFrame(rows).sort_values(["mean_abs_delta_corr","mean_nwd","n_rows"],ascending=[False,False,False])

def fmt_metric(x:float,decimals:int=4)->str:
    """Format metric values safely."""
    return "NA" if pd.isna(x) or not np.isfinite(x) else f"{x:.{decimals}f}"

def write_imputation_report(pre_df:pd.DataFrame,post_df:pd.DataFrame,report_path:Path,cols:list[str],imputed_mask:dict|None=None,strata_cols:list[str]|None=None)->Path:
    """Write a text report with global and subgroup delta-correlation and distribution differences."""
    report_path.parent.mkdir(parents=True,exist_ok=True)
    valid_cols=[c for c in cols if c in pre_df.columns and c in post_df.columns]
    summary=summary_rows(pre_df,post_df,valid_cols,imputed_mask=imputed_mask)
    g_mean_delta_corr,_=corr_delta(pre_df,post_df,valid_cols)
    g_mean_nwd,_=nwd_summary(pre_df,post_df,valid_cols)
    strata_cols=strata_cols or ["breed","lactation_phase","season","age_class"]
    subgroup_global={s:subgroup_global_summary(pre_df,post_df,valid_cols,s) for s in strata_cols if s in post_df.columns}
    subgroup_feature={s:subgroup_feature_summary(pre_df,post_df,valid_cols,s,imputed_mask=imputed_mask) for s in strata_cols if s in post_df.columns}
    lines=[]
    lines.append("IMPUTATION REPORT\n")
    lines.append("Global dependence shift is summarized by the mean absolute change in pairwise correlations (Δcorr).")
    lines.append("Per-feature dependence shift is summarized by the mean absolute change in correlations with the other variables.")
    lines.append("Distribution shift is summarized by the normalized Wasserstein distance (nWD), computed between observed pre-imputation values and post-imputation values, and normalized by the pre-imputation IQR.\n")
    lines.append("1) Global per-feature summary")
    hdr=f"{'feature':<16}{'%miss_pre':>12}{'%imputed':>11}{'Δcorr':>10}{'nWD':>10}"
    lines.append(hdr);lines.append("-"*len(hdr))
    for _,r in summary.iterrows():
        lines.append(f"{r['feature']:<16}{r['pct_missing_pre']:>12.2f}{r['pct_imputed']:>11.2f}{r['mean_abs_delta_corr']:>10.4f}{fmt_metric(r['normalized_wasserstein']):>10}")
    lines.append("")
    lines.append(f"Global mean Δcorr: {g_mean_delta_corr:.4f}" if np.isfinite(g_mean_delta_corr) else "Global mean Δcorr: NA")
    lines.append(f"Global mean nWD: {g_mean_nwd:.4f}" if np.isfinite(g_mean_nwd) else "Global mean nWD: NA")
    lines.append("")
    lines.append("2) Stratified global summary")
    for s,df_s in subgroup_global.items():
        lines.append(f"\n- {s}")
        if df_s.empty:
            lines.append("No eligible strata.")
            continue
        hdr=f"{'level':<24}{'n_rows':>10}{'mean Δcorr':>16}{'mean nWD':>14}"
        lines.append(hdr);lines.append("-"*len(hdr))
        for _,r in df_s.iterrows():
            dc=fmt_metric(r["mean_abs_delta_corr"])
            nw=fmt_metric(r["mean_nwd"])
            lines.append(f"{str(r['level']):<24}{int(r['n_rows']):>10}{dc:>16}{nw:>14}")
    lines.append("")
    lines.append("3) Stratified per-feature detail")
    for s,levels_map in subgroup_feature.items():
        lines.append(f"\n- {s}")
        if not levels_map:
            lines.append("No eligible strata.")
            continue
        for lvl,df_lvl in levels_map.items():
            lines.append(f"  * level={lvl}")
            if df_lvl.empty:
                lines.append("    no valid features")
                continue
            hdr=f"    {'feature':<14}{'n_rows':>8}{'%miss_pre':>12}{'%imputed':>11}{'Δcorr':>10}{'nWD':>10}"
            lines.append(hdr);lines.append("    "+"-"*(len(hdr)-4))
            for _,r in df_lvl.iterrows():
                lines.append(f"    {r['feature']:<14}{int(r['n_rows']):>8}{r['pct_missing_pre']:>12.2f}{r['pct_imputed']:>11.2f}{r['mean_abs_delta_corr']:>10.4f}{fmt_metric(r['normalized_wasserstein']):>10}")
            lines.append("")
    report_path.write_text("\n".join(lines),encoding="utf-8")
    log.info("Imputation report written to: %s",report_path)
    return report_path

class HierarchicalImputer:
    """
    Hierarchical clinical imputer with fit/transform persistence.
    """
    def __init__(self,clinical_cols:list[str]|None=None,ihg:list[list[str]]|None=None,phg:list[list[str]]|None=None,min_group_n:int=5,global_fallback:bool=True,
                 clip_imputed:bool=True):
        self.clinical_cols=clinical_cols or VARIABLES_COLS
        self.ihg=ihg or IHG
        self.phg=phg or PHG
        self.min_group_n=min_group_n
        self.global_fallback=global_fallback
        self.clip_imputed=clip_imputed
        self.fitted_=False
        self.group_stats_={}
        self.global_medians_={}
        self.support_cols_created_=["age_class"]

    def prepare_df(self,df:pd.DataFrame)->pd.DataFrame:
        """Create support columns needed by the imputer."""
        out=df.copy()
        if "age" in out.columns:
            age_num=pd.to_numeric(out["age"],errors="coerce")
            out["age_class"]=pd.cut(age_num,bins=[1.5,3.5,5.5,7.5],labels=["2_3","4_5","6_7"])
        else:
            out["age_class"]=pd.NA
        return out

    def cleanup_df(self,df:pd.DataFrame)->pd.DataFrame:
        """Remove support columns created internally by the imputer."""
        return df.drop(columns=[c for c in self.support_cols_created_ if c in df.columns],errors="ignore")

    def valid_levels(self,df:pd.DataFrame,levels:list[list[str]])->list[list[str]]:
        """Keep only levels whose keys exist in the dataframe and remove duplicates."""
        seen=set();res=[]
        for lvl in levels:
            lvl=[c for c in lvl if c in df.columns]
            if not lvl:continue
            key=tuple(lvl)
            if key in seen:continue
            seen.add(key);res.append(lvl)
        return res

    def fit_group_stats(self,df:pd.DataFrame,col:str,keys:list[str])->pd.DataFrame:
        """Fit per-group count and median for one variable."""
        base=df.loc[df[col].notna(),keys+[col]].copy()
        if base.empty:return pd.DataFrame(columns=keys+["__count","__median"])
        stats=base.groupby(keys,dropna=True,observed=True)[col].agg(__count="count",__median="median").reset_index()
        return stats.loc[stats["__count"]>=self.min_group_n].reset_index(drop=True)

    def fit(self,df:pd.DataFrame):
        """
        Learn group medians from training data only.
        """
        log.info("Fitting hierarchical imputer...")
        t0=time.time()
        work=self.prepare_df(df)
        valid_ihg=self.valid_levels(work,self.ihg)
        valid_phg=self.valid_levels(work,self.phg)
        self.group_stats_={}
        self.global_medians_={}
        for c in self.clinical_cols:
            if c not in work.columns:
                log.warning("Variable '%s' missing in training data. Skipping.",c)
                continue
            work[c]=pd.to_numeric(work[c],errors="coerce")
            self.group_stats_[c]={"ihg":[],"phg":[]}
            for lvl in valid_ihg:
                stats=self.fit_group_stats(work,c,lvl)
                self.group_stats_[c]["ihg"].append({"keys":lvl,"stats":stats})
            for lvl in valid_phg:
                stats=self.fit_group_stats(work,c,lvl)
                self.group_stats_[c]["phg"].append({"keys":lvl,"stats":stats})
            self.global_medians_[c]=work[c].median(skipna=True) if self.global_fallback else np.nan
        self.fitted_=True
        del work
        gc.collect()
        log.info("Imputer fitted in %.2fs",time.time()-t0)
        return self

    def apply_fitted_levels(self,df:pd.DataFrame,s:pd.Series,levels_data:list[dict],col:str,usage_log:list[dict])->pd.Series:
        """Apply fitted group medians to missing rows."""
        for item in levels_data:
            miss=s.isna()
            if not miss.any():break
            keys=item["keys"];stats=item["stats"]
            if stats.empty:continue
            target=df.loc[miss,keys].copy()
            target["__idx"]=target.index
            merged=target.merge(stats,on=keys,how="left")
            fillable=merged["__median"].notna()
            n_fill=int(fillable.sum())
            if n_fill:
                idx=merged.loc[fillable,"__idx"].to_numpy()
                vals=merged.loc[fillable,"__median"].to_numpy()
                s.loc[idx]=vals
                usage_log.append({"level":" -> ".join(keys),"filled":n_fill,"min_n":self.min_group_n})
            del target,merged
            gc.collect()
        return s

    def transform(self,df:pd.DataFrame,return_metadata:bool=False):
        """
        Apply fitted imputation rules to new data without refitting.
        """
        if not self.fitted_:raise RuntimeError("Imputer is not fitted. Call fit() first.")
        log.info("Transforming dataset with fitted imputer...")
        t0=time.time()
        work=self.prepare_df(df)
        usage=defaultdict(list)
        imputed_mask={}
        for c in self.clinical_cols:
            if c not in work.columns or c not in self.group_stats_:
                continue
            work[c]=pd.to_numeric(work[c],errors="coerce")
            orig_na=work[c].isna()
            if not orig_na.any():
                imputed_mask[c]=pd.Series(False,index=work.index)
                continue
            s=work[c].copy()
            s=self.apply_fitted_levels(work,s,self.group_stats_[c]["ihg"],c,usage[c])
            if s.isna().any():s=self.apply_fitted_levels(work,s,self.group_stats_[c]["phg"],c,usage[c])
            if s.isna().any():s=self.apply_fitted_levels(work,s,self.group_stats_[c]["ihg"],c,usage[c])
            if self.global_fallback and s.isna().any():
                gmed=self.global_medians_.get(c,np.nan)
                if pd.notna(gmed):
                    n_fill=int(s.isna().sum())
                    s=s.fillna(gmed)
                    usage[c].append({"level":"GLOBAL","filled":n_fill,"min_n":0})
            imp=orig_na&s.notna()
            if self.clip_imputed and c in RANGES and imp.any():
                lo,hi=RANGES[c]
                before=((s.loc[imp]<lo)|(s.loc[imp]>hi)).sum()
                if int(before)>0:
                    s.loc[imp]=s.loc[imp].clip(lo,hi)
                    usage[c].append({"level":f"CLIP[{lo},{hi}]","filled":int(before),"min_n":0})
            still_na=int(s.isna().sum())
            if still_na:log.warning("Variable '%s' still has %d missing values post-transform.",c,still_na)
            work[c]=s
            imputed_mask[c]=imp
        out=self.cleanup_df(work)
        del work
        gc.collect()
        for var,levels in usage.items():
            log.info("'%s' transformed using %d steps: %s",var,len(levels),levels)
        log.info("Transform completed in %.2fs",time.time()-t0)
        meta={"imputed_mask":imputed_mask,"usage_log":dict(usage)}
        return (out,meta) if return_metadata else out

    def fit_transform(self,df:pd.DataFrame,return_metadata:bool=False):
        """
        Fit on df and immediately transform df.
        """
        self.fit(df)
        return self.transform(df,return_metadata=return_metadata)

    def save(self,path:Path)->Path:
        """
        Save fitted imputer to disk.
        """
        path.parent.mkdir(parents=True,exist_ok=True)
        with open(path,"wb") as f:pickle.dump(self,f)
        log.info("Imputer object saved to: %s",path)
        return path

    @classmethod
    def load(cls,path:Path):
        """
        Load a fitted imputer from disk.
        """
        with open(path,"rb") as f:obj=pickle.load(f)
        log.info("Imputer object loaded from: %s",path)
        return obj

def run_imputation(input_path:Path,output_path:Path,model_path:Path|None=None,mode:str="fit_transform",report:bool=False,report_path:Path|None=None,strata_cols:list[str]|None=None,min_n:int=5,global_fallback:bool=True,clip_imputed:bool=True)->None:
    """
    Full workflow for CLI usage: fit_transform on train or transform on new data.
    """
    df=pd.read_parquet(input_path)
    cols=[c for c in VARIABLES_COLS if c in df.columns]
    if not cols:raise ValueError("No clinical columns found in input dataframe.")
    log.info("Input dataset loaded: %d rows, %d columns",len(df),df.shape[1])
    log.info("Variables to impute found: %s",cols)
    if mode=="fit_transform":
        imputer=HierarchicalImputer(clinical_cols=cols,min_group_n=min_n,global_fallback=global_fallback,clip_imputed=clip_imputed)
        df_imp,meta=imputer.fit_transform(df,return_metadata=True)
        if model_path is not None:imputer.save(model_path)
        if report:
            rp=report_path if report_path is not None else output_path.with_name(output_path.stem+"_imputation_report.txt")
            report_strata=strata_cols or ["breed","lactation_phase","season","age_class"]
            df_report=df.copy()
            df_imp_report=df_imp.copy()
            if "age" in df_report.columns:
                age_num=pd.to_numeric(df_report["age"],errors="coerce")
                df_report["age_class"]=pd.cut(age_num,bins=[1.5,3.5,5.5,7.5],labels=["2_3","4_5","6_7"])
            if "age" in df_imp_report.columns:
                age_num_imp=pd.to_numeric(df_imp_report["age"],errors="coerce")
                df_imp_report["age_class"]=pd.cut(age_num_imp,bins=[1.5,3.5,5.5,7.5],labels=["2_3","4_5","6_7"])
            report_cols=[c for c in list(dict.fromkeys(cols+report_strata)) if c in df_report.columns and c in df_imp_report.columns]
            write_imputation_report(df_report[report_cols].copy(),df_imp_report[report_cols].copy(),rp,cols,imputed_mask=meta["imputed_mask"],strata_cols=report_strata)
            del df_report,df_imp_report
            gc.collect()
    elif mode=="transform":
        if model_path is None:raise ValueError("In transform mode, --model-path is required.")
        imputer=HierarchicalImputer.load(model_path)
        df_imp,meta=imputer.transform(df,return_metadata=True)
        for c in cols:
            if c in meta["imputed_mask"]:
                log.info("Variable '%s' imputed rows during transform: %d",c,int(meta["imputed_mask"][c].sum()))
    else:
        raise ValueError("mode must be 'fit_transform' or 'transform'.")
    output_path.parent.mkdir(parents=True,exist_ok=True)
    df_imp.to_parquet(output_path,index=False)
    log.info("Imputed dataset saved: %s (%d rows, %d columns)",output_path,len(df_imp),df_imp.shape[1])
    del df,df_imp,meta
    gc.collect()
    log.info("Imputation pipeline completed.")

# Parsing

def parse_args():
    """Parse CLI arguments for clinical imputation."""
    p=argparse.ArgumentParser(description="Hierarchical clinical imputation with fit/transform persistence.")
    p.add_argument("--mode",choices=["fit_transform","transform"],required=True,help="Run fit_transform or transform.")
    p.add_argument("--input-path",type=Path,required=True,help="Input parquet path.")
    p.add_argument("--output-path",type=Path,required=True,help="Output parquet path.")
    p.add_argument("--model-path",type=Path,default=None,help="Path to save/load fitted imputer object.")
    p.add_argument("--min-group-n",type=int,default=10,help="Minimum group support for median-based imputation.")
    p.add_argument("--no-global-fallback",action="store_true",help="Disable global median fallback.")
    p.add_argument("--no-clip-imputed",action="store_true",help="Disable clipping of imputed values to plausible ranges.")
    p.add_argument("--write-report",action="store_true",help="Write pre/post imputation report (fit_transform only).")
    p.add_argument("--report-path",type=Path,default=None,help="Optional text report path.")
    p.add_argument("--strata",type=str,default="breed,lactation_phase,season,age_class",help="Comma-separated strata for report.")
    return p.parse_args()

# Main call

if __name__=="__main__":
    args=parse_args()
    strata=[x.strip() for x in args.strata.split(",") if x.strip()]
    run_imputation(input_path=args.input_path, output_path=args.output_path, model_path=args.model_path, mode=args.mode, report=args.write_report,
                   report_path=args.report_path, strata_cols=strata, min_n=args.min_group_n, 
                   global_fallback=not args.no_global_fallback, clip_imputed=not args.no_clip_imputed)
