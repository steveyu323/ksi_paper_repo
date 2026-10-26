"""
The kinase / drug / phospho-site pipeline behind the virus figures.

These helpers are transcribed from the code cells of
``eval_6_prediction_against_kinome_virus_demo.ipynb`` and
``eval_6_prediction_against_kinome_virus_full.ipynb``, which define them
inline.  Behaviour is kept identical, including the quirks, so the tables
match the figures.

Two reproducibility hazards are worth knowing about, both caused by pandas'
default non-stable sort being applied to columns full of ties:

1. ``summary_df`` is built by sorting predictions on ``logits`` and keeping
   the top ten kinases per site.  Kinases tied on logit swap places, so
   re-running from the raw model pickles changes 28 of 96 rows.  The
   pipeline therefore enters at the cached ``250118_virus_against_kinome_hits.pkl``,
   which is the exact object the figures were built from.
2. ``drug_long`` is sorted on ``nsub`` before ``head(2)`` picks two drugs per
   kinase and substrate.  Most drugs tie at ``nsub == 1``.  This one has no
   cached escape hatch, and is why the drug-by-kinase figures cannot be
   reproduced cell-for-cell.  See table_05 and table_06 for how each figure
   is handled.
"""

from __future__ import annotations

import ast
import re

import pandas as pd


# --------------------------------------------------------------------------
# Parsing the packed "KINASE:logit:prob" and "KINASE:DrugA;DrugB" strings
# --------------------------------------------------------------------------

def _as_list(x):
    """Coerce a cell that may be a real list or a string repr of one."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            v = ast.literal_eval(s)
            return v if isinstance(v, list) else [v]
        except Exception:
            s = s.strip("[]")
            parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
            return [p for p in parts if p]
    return [x]


def _norm_kinase_name(name) -> str:
    """Upper-case and strip punctuation so kinase names join across sources."""
    if name is None:
        return ""
    s = str(name).strip().upper()
    s = re.sub(r"[\s\-_]+", "", s)
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


_prob_re = re.compile(
    r"^\s*(?P<kin>[^:]+)\s*:\s*(?P<logit>-?\d+(\.\d+)?)\s*:\s*(?P<prob>\d+(\.\d+)?)\s*$")
_PROB_RE = re.compile(r"^\s*([^:]+)\s*:\s*([-+]?\d*\.?\d+)\s*:\s*")


def _parse_prob_item(item):
    m = _prob_re.match(str(item))
    if not m:
        return None
    return m.group("kin").strip(), float(m.group("logit")), float(m.group("prob"))


def _has_any_drug(matched_item) -> bool:
    s = str(matched_item)
    if ":" not in s:
        return False
    _, rhs = s.split(":", 1)
    return bool(rhs.strip())


def parse_prob_list(prob_list):
    """['KIN:logit:prob', ...] -> {kinase: logit}."""
    d = {}
    for it in _as_list(prob_list):
        m = _prob_re.match(str(it))
        if m:
            d[m.group("kin").strip()] = float(m.group("logit"))
    return d


def extract_logit_map(prob_list):
    d = {}
    for item in _as_list(prob_list):
        m = _PROB_RE.match(str(item))
        if m:
            d[m.group(1).strip()] = float(m.group(2))
    return d


def parse_passing_drugs(passing_drugs):
    """['KIN:DrugA;DrugB', ...] -> [(kinase, drug), ...], skipping empties."""
    rows = []
    for it in _as_list(passing_drugs):
        s = str(it)
        if ":" not in s:
            continue
        kin, rhs = s.split(":", 1)
        kin, rhs = kin.strip(), rhs.strip()
        if not rhs:
            continue
        for drug in rhs.split(";"):
            drug = drug.strip()
            if drug:
                rows.append((kin, drug))
    return rows


# --------------------------------------------------------------------------
# Attaching drugs to predicted kinases, then filtering
# --------------------------------------------------------------------------

def add_matched_drug(hits_df, drug_df, pred_col="pred_kinase",
                     drug_kinase_col="kinase_name", drug_name_col="drug_name",
                     out_col="matched_drug"):
    """For each predicted kinase, list the drugs known to target it."""
    d = drug_df.copy()
    d["_kin_norm"] = d[drug_kinase_col].map(_norm_kinase_name)
    d = d[d["_kin_norm"].astype(bool)].copy()
    kin_to_drugs = (d.groupby("_kin_norm")[drug_name_col]
                    .apply(lambda s: sorted(pd.unique([x for x in s.dropna().astype(str)])))
                    .to_dict())

    def _map_row(pred_val):
        out = []
        for kin in _as_list(pred_val):
            kin_str = str(kin).strip()
            drugs = kin_to_drugs.get(_norm_kinase_name(kin_str), [])
            out.append(f"{kin_str}:{';'.join(drugs)}" if drugs else f"{kin_str}:")
        return out

    out_df = hits_df.copy()
    out_df[out_col] = out_df[pred_col].apply(_map_row)
    return out_df


def filter_hits_by_logit_and_drug(df, logit_min=2.0):
    """Keep sites with at least one confidently predicted, druggable kinase."""
    out = df.copy()
    out["_prob_list"] = out["pred_kinase_prob"].apply(_as_list)
    out["_drug_list"] = out["matched_drug"].apply(_as_list)

    def per_row(prob_list, drug_list):
        kin2score = {}
        for it in prob_list:
            parsed = _parse_prob_item(it)
            if parsed:
                kin, logit, prob = parsed
                kin2score[kin] = (logit, prob)
        passing_kin, passing_drug_entries = [], []
        for d in drug_list:
            d = str(d)
            kin = d.split(":", 1)[0].strip() if ":" in d else d.strip()
            score = kin2score.get(kin)
            if score is None:
                continue
            logit, _ = score
            if logit >= logit_min and _has_any_drug(d):
                passing_kin.append(kin)
                passing_drug_entries.append(d)
        return passing_kin, passing_drug_entries

    tmp = out.apply(lambda r: per_row(r["_prob_list"], r["_drug_list"]),
                    axis=1, result_type="expand")
    out["passing_kinases"] = tmp[0]
    out["passing_drugs"] = tmp[1]
    out = out[out["passing_drugs"].map(len) > 0].copy()
    out.drop(columns=["_prob_list", "_drug_list"], inplace=True)
    return out


def top_logit_row(row):
    """The single best-scoring druggable kinase for one site."""
    passing = _as_list(row["passing_kinases"])
    if not passing:
        return None
    logit_map = extract_logit_map(row["pred_kinase_prob"])
    scored = [(k, logit_map.get(k, float("-inf"))) for k in passing]
    top_kinase, top_logit = max(scored, key=lambda x: x[1])
    top_drug = None
    for d in _as_list(row["passing_drugs"]):
        if str(d).startswith(f"{top_kinase}:"):
            top_drug = d
            break
    return pd.Series({"top_kinase": top_kinase, "top_logit": top_logit,
                      "top_drug": top_drug})


# --------------------------------------------------------------------------
# Reshaping into the long frames the figures plot
# --------------------------------------------------------------------------

def make_psite_label(df):
    return df["wt_aa"].astype(str) + df["p_ind"].astype(str)


def build_logit_long(filtered_df):
    """One row per (substrate, phospho-site, kinase) with its logit."""
    df = filtered_df.copy()
    df["psite"] = make_psite_label(df)
    all_kinases = sorted({k for ks in df["top_kinase"].apply(_as_list) for k in ks})
    rows = []
    for _, r in df.iterrows():
        logit_map = parse_prob_list(r["pred_kinase_prob"])
        for kin in _as_list(r["top_kinase"]):
            if kin in logit_map:
                rows.append({"substrate": r["substrate"], "psite": r["psite"],
                             "kinase": kin, "logit": logit_map[kin]})
    long = pd.DataFrame(rows)
    if long.empty:
        return long
    long["kinase"] = pd.Categorical(long["kinase"], categories=all_kinases, ordered=True)
    return long


def complete_with_zeros(df, kinase_order):
    """Fill the (substrate, psite) x kinase grid, absent combinations at 0."""
    df = df.copy()
    idx = (df[["substrate", "psite"]].drop_duplicates().assign(key=1)
           .merge(pd.DataFrame({"kinase": kinase_order, "key": 1}), on="key", how="outer")
           .drop(columns="key"))
    out = idx.merge(df, on=["substrate", "psite", "kinase"], how="left")
    out["logit"] = out["logit"].fillna(0)
    return out


def melt_kinase_drug_hits(df):
    """Explode each site into (substrate, psite, kinase, drug, logit, prob)."""
    rows = []
    for _, r in df.iterrows():
        substrate = r["substrate"]
        psite = f"{r['wt_aa']}{int(r['p_ind'])}"
        kp = {}
        for item in str(r["pred_kinase_prob"]).split(","):
            try:
                k, logit, prob = item.split(":")
                kp[k] = {"logit": float(logit), "prob": float(prob)}
            except ValueError:
                continue
        md = {}
        for x in r["matched_drug"]:
            try:
                k, drugs = x.split(":", 1)
                md[k] = [d for d in drugs.split(";") if d]
            except Exception:
                pass
        for kinase, vals in kp.items():
            for drug in md.get(kinase, [""]):
                rows.append({"substrate": substrate, "psite": psite, "kinase": kinase,
                             "drug": drug, "logit": vals["logit"],
                             "prob": vals["prob"], "hit": 1})
    return pd.DataFrame(rows)


def fill_missing_hits(long_df):
    """Complete the kinase x drug grid, marking unobserved pairs as 0."""
    required = {"kinase", "drug", "hit"}
    missing = required - set(long_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = long_df.copy()
    kinases = pd.DataFrame({"kinase": df["kinase"].dropna().unique()})
    drugs = pd.DataFrame({"drug": df["drug"].dropna().unique()})
    full = kinases.merge(drugs, how="cross")
    observed = (df[["kinase", "drug", "hit"]]
                .groupby(["kinase", "drug"], as_index=False).agg(hit=("hit", "max")))
    out = full.merge(observed, on=["kinase", "drug"], how="left")
    out["hit"] = out["hit"].fillna(0).astype(int)
    return out


def complete_with_zeros_drug(df):
    """Fill the (substrate, psite) x drug grid, absent combinations at 0."""
    df = df.copy()
    drug_order = df["drug"].dropna().astype(str).unique().tolist()
    idx = (df[["substrate", "psite"]].drop_duplicates().assign(key=1)
           .merge(pd.DataFrame({"drug": drug_order, "key": 1}), on="key", how="outer")
           .drop(columns="key"))
    out = idx.merge(df, on=["substrate", "psite", "drug"], how="left")
    out["logit"] = out["logit"].fillna(0)
    out["drug"] = pd.Categorical(out["drug"], categories=drug_order, ordered=True)
    return out, drug_order


# --------------------------------------------------------------------------
# The shared front half of both notebooks
# --------------------------------------------------------------------------

SUBSTRATE_NAME_MAP = {
    "WSN": "influenza \n H1",
    "BF520_human": "HIV \n human",
    "BF520_rhesus": "HIV \n rhesus",
    "Perth": "H3N2",
    "PR8_M1": "influenza \n M1",
    "ZIKA_E": "ZIKA E",
    "NCAP_SARS2": " SARS2:Nucleocapsid",
    "ORF9B_SARS2": "SARS2:ORF9b",
}

# Structural proteins dropped before the drug analysis.
EXCLUDED_SUBSTRATES = ["SPIKE_SARS2", "VME1_SARS2", "R1AB_SARS2"]


def build_top_df(hits_pkl, drug_kinase_pkl, logit_min=3):
    """From the cached kinome hits to the per-site table both notebooks plot.

    Reproduces cells 32-44 of the demo notebook.  Entering here rather than at
    the raw model pickles is deliberate: see the module docstring.
    """
    summary = pd.read_pickle(hits_pkl)
    summary = summary.loc[~summary["substrate"].isin(EXCLUDED_SUBSTRATES), :]

    drug_kinase_uniprot = pd.read_pickle(drug_kinase_pkl)

    dms = summary.copy()
    dms.columns = ["substrate", "substrate_seq_15aa", "p_ind", "wt_aa", "pred_kinase_prob"]
    dms["pred_kinase"] = dms["pred_kinase_prob"].apply(
        lambda s: [elem.strip("'").split(":")[0] for elem in s.split(",")])

    dms = add_matched_drug(dms, drug_kinase_uniprot)
    filtered_df = filter_hits_by_logit_and_drug(dms, logit_min=logit_min)
    filtered_df["substrate"] = filtered_df["substrate"].apply(lambda s: SUBSTRATE_NAME_MAP[s])

    top_df = filtered_df.copy()
    top_df[["top_kinase", "top_logit", "top_drug"]] = top_df.apply(top_logit_row, axis=1)
    top_df = top_df.dropna(subset=["top_kinase"]).reset_index(drop=True)

    # Both notebooks then widen these two columns from the single best kinase
    # to the full predicted list, which is what the figures actually show.
    top_df["top_kinase"] = top_df["pred_kinase"]
    top_df["top_drug"] = top_df["matched_drug"]
    return top_df


def build_drug_long(top_df, kinase_filter):
    """The kinase x drug hit grid.

    ``kinase_filter`` is the kinase column the notebooks restrict to before
    picking two drugs per kinase and substrate.  The ``sort_values('nsub')``
    inside is the non-stable sort described in the module docstring.
    """
    drug_long = melt_kinase_drug_hits(top_df)
    drug_long_count = (drug_long.drop_duplicates(["kinase", "drug"])
                       .groupby(["drug"]).count().reset_index()
                       .iloc[:, :2].sort_values("substrate"))
    drug_long_count.columns = ["drug", "nsub"]
    drug_long = pd.merge(drug_long, drug_long_count)
    drug_long_pre = drug_long.copy()

    drug_long = (drug_long.drop_duplicates()
                 .sort_values("nsub")
                 .loc[drug_long["kinase"].isin(kinase_filter), :]
                 .groupby(["kinase", "substrate"]).head(2)
                 .sort_values("kinase"))
    return drug_long, drug_long_pre
