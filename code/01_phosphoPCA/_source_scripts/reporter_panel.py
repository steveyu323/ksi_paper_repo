"""Build the reporter-candidate comparison table from source files.

Replaces the hand-typed orig_prob / finetuned_prob / binding_score lists in
13_kinbert_phosphoPCA_link.ipynb.  Every value is looked up; nothing is pasted.

Rebuilds df_finetuned from the same CSVs the notebook uses (cells 1-13, 31-35)
rather than reading 260212_ksi_reduced_df_tp_wtape.pkl, which was written by a
newer pandas than the current environment can unpickle.

Sources
-------
../screening_analysis/blib_kinases_with_acc.csv       kinase name/seq/acc lookup
251203_ksi_reduced_df_tp.pkl                          the screening library + binding_max
exp_tape_bertbase_sep_v2_early_loss__ksi_full.csv     KinBERT, dblysate-trained
finetune_tape_v2_phosphopca_exp6__pairs_aug_ksi_full.csv  KinBERT, phosphoPCA-finetuned
pairs_aug_ksi_full.csv                                pair list mapping domain seq -> kinase name
"""

import numpy as np
import pandas as pd

BLIB_CSV = "../screening_analysis/blib_kinases_with_acc.csv"
SCREEN_PKL = "./251203_ksi_reduced_df_tp.pkl"
INFERENCE_PKL = "./251130_ksi_inference_scores.pkl"  # per-PAABD binding measurements
RESCORE_CSV = "./rescore_results.csv"  # KinBERT run on pairs absent from the screen

ORIG_MODEL = "exp_tape_bertbase_sep_v2_early_loss"
FINETUNED_MODEL = "finetune_tape_v2_phosphopca_exp6"

# bf16 inference reproduces the saved orig-model scores to ~1e-3, not exactly
RESCORE_TOL = 5e-3
ORIG_CSV = "./exp_tape_bertbase_sep_v2_early_loss__ksi_full.csv"
FINETUNED_CSV = "./finetune_tape_v2_phosphopca_exp6__pairs_aug_ksi_full.csv"
PAIRS_AUG_CSV = "./pairs_aug_ksi_full.csv"

# (display name, kinase, 15mer centred on the p-site, intended PAABD, provenance)
#
# The PAABD is the reporter domain the candidate is built for, and it is what the
# figure label claims.  Binding score is read for THAT PAABD specifically: a
# substrate can be an experimental hit against several PAABDs with quite
# different strengths, so "binding_max across all PAABDs" can silently belong to
# a different reporter than the label.
#
# novel candidates + their PAABD: 9_screening_tp_to_reporter_candidates.ipynb
# DB 15mers: screening_analysis/240416_SPARK_candidate_selection.ipynb
CANDIDATES = [
    ("TBK-SPARK-novel", "TBK1_HUMAN", "QGPSQTSTPTDVTAI", "fha", "novel"),
    ("IKK-SPARK-novel", "IKKB_HUMAN", "PHEEPSNSLMAIECR", "ww", "novel"),
    ("TBK-SPARK-1", "TBK1_HUMAN", "DHPYASLTLPDDAAA", "fha", "DB"),
    ("IKKB-SPARK-1", "IKKB_HUMAN", "NPQERTLTLVDTGIG", "fha", "DB"),
]

PAABD_COLS = ["SH2", "b14-3-3", "WW", "FHA"]

# the notebook's kinase-name fixups (cells 10, 33)
NAME_FIXUPS = {
    "JAK2_HUMAN": "JAK2_HUMAN_1",
    "KS6B1_HUMAN": "KS6B1_HUMAN_active",
    "PLK1_HUMAN": "PLK1_HUMAN_active",
}


def assign_paabd(dat):
    """PAABD motif-compatibility flags on a 15mer, p-site at index 7.

    Follows 9_screening_tp_to_reporter_candidates.ipynb, which is the notebook
    that actually selected the candidates.

    NOTE: 13_kinbert_phosphoPCA_link.ipynb defines WW more strictly as
    s[8] == 'P'.  The two notebooks disagree, and the looser rule here is the
    one candidate selection ran under.
    """
    s15 = dat["substrate_seq_15aa"]
    dat = dat.copy()
    dat["SH2"] = s15.apply(lambda s: s[7] == "Y")
    dat["b14-3-3"] = s15.apply(
        lambda s: (s[7] in "ST") and (s[4] == "R") and (s[9] == "P")
    )
    dat["WW"] = s15.apply(lambda s: (s[7] in "ST") and (s[8] in ["P", "L", "V", "I"]))
    dat["FHA"] = s15.apply(lambda s: (s[7] == "T") and (s[10] == "D"))
    return dat


def paabd_label(row):
    hits = [c for c in PAABD_COLS if bool(row[c])]
    return "/".join(hits) if hits else "none"


def _load_orig_scores(pairs_aug):
    """exp_tape_...ksi_full.csv is keyed by (domain seq, 15mer) only, so the
    kinase name has to come from the pair list."""
    sc = pd.read_csv(
        ORIG_CSV,
        usecols=["kinase_seq", "substrate_seq", "probs", "logits"],
    ).rename(
        columns={
            "kinase_seq": "kinase_domain_seq",
            "substrate_seq": "substrate_seq_15aa",
            "probs": "tape_prob",
            "logits": "tape_logit",
        }
    )
    sc = pd.merge(pairs_aug, sc, on=["kinase_domain_seq", "substrate_seq_15aa"])
    return sc[["kinase", "substrate_seq_15aa", "tape_prob", "tape_logit"]].drop_duplicates()


def _load_finetuned_scores():
    """finetune_tape_...ksi_full.csv already carries kinase_human_name."""
    sc = pd.read_csv(
        FINETUNED_CSV,
        usecols=["kinase_human_name", "substrate_seq_15aa", "prob", "logit"],
    ).rename(columns={"prob": "tape_finetuned_prob", "logit": "tape_finetuned_logit"})
    sc["kinase"] = sc["kinase_human_name"].replace(NAME_FIXUPS)
    return sc[
        ["kinase", "substrate_seq_15aa", "tape_finetuned_prob", "tape_finetuned_logit"]
    ].drop_duplicates()


def build_scored_table():
    """Reproduce the notebook's df_finetuned."""
    blib = pd.read_csv(BLIB_CSV, header=None)
    blib.columns = ["kinase_type", "kinase", "kinase_seq", "kinase_acc"]

    screen = pd.read_pickle(SCREEN_PKL).rename(columns={"substrate": "substrate_seq_15aa"})
    df = pd.merge(blib, screen, on="kinase")
    df = assign_paabd(df)

    pairs_aug = pd.read_csv(
        PAIRS_AUG_CSV, usecols=["kinase_human_name", "kinase_domain_seq", "substrate_seq_15aa"]
    ).drop_duplicates()
    pairs_aug["kinase"] = pairs_aug["kinase_human_name"].replace(NAME_FIXUPS)
    pairs_aug = pairs_aug[["kinase", "kinase_domain_seq", "substrate_seq_15aa"]]

    orig = _load_orig_scores(pairs_aug)
    finetuned = _load_finetuned_scores()

    df = pd.merge(df, orig, on=["kinase", "substrate_seq_15aa"])
    df = pd.merge(df, finetuned, on=["kinase", "substrate_seq_15aa"])
    return df


def load_rescored(path=RESCORE_CSV, tol=RESCORE_TOL):
    """Probabilities generated for pairs that are in no saved inference output.

    The request CSV carries purpose=validate rows whose scores are already
    known; those are checked here so a mismatched checkpoint cannot quietly
    supply the regenerated values.  Returns {(kinase, seq): {orig, finetuned}}.
    """
    try:
        rs = pd.read_csv(path)
    except FileNotFoundError:
        return {}

    val = rs[rs["purpose"] == "validate"].dropna(subset=["expected_prob"])
    bad = val[(val["prob"] - val["expected_prob"]).abs() > tol]
    if len(bad):
        raise ValueError(
            f"{path}: validation rows do not reproduce the saved inference output "
            f"within {tol}:\n"
            + bad[["name", "model", "expected_prob", "prob"]].to_string(index=False)
        )

    out = {}
    for (kin, seq), g in rs[rs["purpose"] == "regenerate"].groupby(
        ["kinase", "substrate_seq"]
    ):
        by_model = dict(zip(g["model"], g["prob"]))
        out[(kin, seq)] = {
            "orig": by_model.get(ORIG_MODEL, np.nan),
            "finetuned": by_model.get(FINETUNED_MODEL, np.nan),
        }
    return out


def build_panel(candidates=CANDIDATES):
    scored = build_scored_table()
    screen = pd.read_pickle(SCREEN_PKL)
    inference = pd.read_pickle(INFERENCE_PKL)
    rescored = load_rescored()

    # what the screen physically contained
    screened_substrates = set(screen["substrate"])
    screened_pairs = set(zip(screen["kinase"], screen["substrate"]))

    rows = []
    for name, kinase, seq, paabd, provenance in candidates:
        hit = scored.loc[
            (scored["kinase"] == kinase) & (scored["substrate_seq_15aa"] == seq)
        ]
        if len(hit) > 1:
            raise ValueError(f"{name}: {len(hit)} rows for ({kinase}, {seq}), expected <=1")

        rec = {
            "name": name,
            "kinase": kinase,
            "substrate_seq_15aa": seq,
            "paabd": paabd,
            "provenance": provenance,
            "substrate_in_screen": seq in screened_substrates,
            "pair_in_screen": (kinase, seq) in screened_pairs,
        }

        # binding, resolved per PAABD rather than as a max over all of them
        meas = inference.loc[
            (inference["kinase"] == kinase) & (inference["substrate"] == seq)
        ]
        at_label = meas.loc[meas["paabd"] == paabd, "mean"]
        rec["binding_score"] = float(at_label.iloc[0]) if len(at_label) else np.nan

        # Where the value sits within everything the screen measured for this
        # same kinase-PAABD pair.  A raw binding score is not comparable across
        # PAABDs: their distributions have very different spreads and tails, so
        # position in the pair's own distribution is the meaningful quantity.
        pop = inference.loc[
            (inference["kinase"] == kinase) & (inference["paabd"] == paabd), "mean"
        ].dropna()
        if len(pop) and pd.notna(rec["binding_score"]):
            v = rec["binding_score"]
            rec["ctx_n"] = int(len(pop))
            rec["ctx_rank"] = int((pop > v).sum() + 1)
            rec["ctx_pct"] = float((pop < v).mean() * 100)
            rec["ctx_median"] = float(pop.median())
            rec["ctx_p95"] = float(pop.quantile(0.95))
            rec["ctx_max"] = float(pop.max())
        else:
            for k in ("ctx_n", "ctx_rank", "ctx_pct", "ctx_median", "ctx_p95", "ctx_max"):
                rec[k] = np.nan
        if len(meas):
            best = meas.loc[meas["mean"].idxmax()]
            rec["binding_max_any"] = float(best["mean"])
            rec["binding_max_paabd"] = best["paabd"]
            rec["paabd_measured"] = ";".join(sorted(meas["paabd"]))
        else:
            rec["binding_max_any"] = np.nan
            rec["binding_max_paabd"] = None
            rec["paabd_measured"] = None

        if len(hit) == 1:
            r = hit.iloc[0]
            rec.update(
                orig_prob=float(r["tape_prob"]),
                finetuned_prob=float(r["tape_finetuned_prob"]),
                prob_source="saved inference",
                paabd_motif_ok=paabd_label(r),
                true_positive=bool(r["true_positive"]),
            )
        elif (kinase, seq) in rescored:
            # absent from every saved inference output, so scored on purpose
            rec.update(
                orig_prob=float(rescored[(kinase, seq)]["orig"]),
                finetuned_prob=float(rescored[(kinase, seq)]["finetuned"]),
                prob_source="rescored",
                paabd_motif_ok=None,
                true_positive=None,
            )
        else:
            # Never assayed -> no binding score exists, and neither KinBERT run
            # covers it because both were scored over the screening pair list only.
            rec.update(
                orig_prob=np.nan,
                finetuned_prob=np.nan,
                prob_source=None,
                paabd_motif_ok=None,
                true_positive=None,
            )
        rows.append(rec)

    cols = [
        "name", "kinase", "substrate_seq_15aa", "paabd", "provenance",
        "substrate_in_screen", "pair_in_screen", "orig_prob", "finetuned_prob",
        "prob_source", "binding_score",
        "ctx_n", "ctx_rank", "ctx_pct", "ctx_median", "ctx_p95", "ctx_max",
        "binding_max_any", "binding_max_paabd",
        "paabd_measured", "paabd_motif_ok", "true_positive",
    ]
    return pd.DataFrame(rows)[cols]


COLOR_ORIG = "#378ADD"
COLOR_FINETUNED = "#1D9E75"
COLOR_BINDING = "#D85A30"
COLOR_CEIL = "#7A2E12"  # darker shade of the binding hue, for its reference marks
INK = "#2B2B2B"
MUTED = "#8A8A8A"


def _xlabels(panel):
    out = []
    for _, r in panel.iterrows():
        kin = r["kinase"].replace("_HUMAN", "")
        mark = "†" if r["prob_source"] == "rescored" else ""
        out.append(f"{r['name']}{mark}\n({kin}, {r['paabd'].upper()}, {r['provenance']})")
    return out


def plot_panel(panel, outfile="kinase_prob_plot_v2.pdf"):
    """Single axes: probability on the left, binding score on the right.

    Three bars per candidate - original prob, finetuned prob (left axis), and
    binding score (right axis).  Candidates absent from the screening library
    get an empty outlined slot in place of a binding bar.

    The two y-scales are independent, so bar heights across the axes are not
    comparable; the right axis is tinted to keep the binding bar tied to its own
    scale.  plot_panel_split() draws the same data without a second scale.
    """
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    n = len(panel)
    x = np.arange(n)
    bar_w = 0.26

    fig, ax1 = plt.subplots(figsize=(9, 5.4))
    ax2 = ax1.twinx()

    # --- probabilities on the left axis ------------------------------------
    ax1.bar(x - bar_w, panel["orig_prob"], width=bar_w, color=COLOR_ORIG,
            edgecolor="white", linewidth=0.5, zorder=3)
    ax1.bar(x, panel["finetuned_prob"], width=bar_w, color=COLOR_FINETUNED,
            edgecolor="white", linewidth=0.5, zorder=3)
    ax1.axhline(0.5, color=MUTED, linestyle="--", linewidth=1, zorder=2)

    ax1.set_ylabel("Predicted probability", fontsize=9, color=INK)
    ax1.set_ylim(0, 1.15)
    ax1.set_yticks(np.arange(0, 1.01, 0.2))
    ax1.tick_params(axis="y", labelcolor=INK, labelsize=8)

    # --- binding on the right axis -----------------------------------------
    # The bar is the labelled PAABD's own measurement.  The reference marks show
    # the screen's p95 and maximum for that SAME kinase-PAABD pair, so the bar
    # is read as a position in its own distribution rather than as a magnitude.
    ax2.bar(x + bar_w, panel["binding_score"], width=bar_w, color=COLOR_BINDING,
            edgecolor="white", linewidth=0.5, zorder=3)

    half = bar_w * 0.62
    for i, r in panel.iterrows():
        if pd.isna(r["ctx_p95"]):
            continue
        ax2.hlines(r["ctx_max"], i + bar_w - half, i + bar_w + half,
                   color=COLOR_CEIL, linewidth=1.4, zorder=5)
        ax2.hlines(r["ctx_p95"], i + bar_w - half, i + bar_w + half,
                   color=COLOR_CEIL, linewidth=1.2, linestyles=(0, (1.5, 1.5)),
                   zorder=5)

    finite = pd.concat([panel["binding_score"], panel["ctx_max"]]).dropna()
    b_top = (finite.max() * 1.30) if len(finite) else 1.0
    ax2.set_ylim(0, b_top)
    ax2.set_ylabel("Binding score (phosphoPCA screen)", fontsize=9, color=COLOR_BINDING)
    ax2.tick_params(axis="y", labelcolor=COLOR_BINDING, labelsize=8)
    ax2.spines["right"].set_edgecolor(COLOR_BINDING)
    ax2.spines["top"].set_visible(False)
    ax1.spines["top"].set_visible(False)

    # --- empty slot where the screen has no measurement ---------------------
    for i, r in panel.iterrows():
        if pd.isna(r["binding_score"]) and pd.isna(r["binding_max_any"]):
            ax2.bar(i + bar_w, b_top * 0.72, width=bar_w, facecolor="none",
                    edgecolor=MUTED, linewidth=0.8, linestyle=(0, (2, 2)),
                    hatch="///", zorder=2)
            ax2.text(i + bar_w, b_top * 0.74, "n/a", ha="center", va="bottom",
                     fontsize=7.5, color=MUTED, style="italic")
        # rank within this kinase-PAABD pair, above the reference marks
        elif pd.notna(r["ctx_pct"]):
            # anchored left of the empty space to the bar's right, so it clears
            # the finetuned bar however low this pair's ceiling sits
            ax2.text(
                i + bar_w + half * 0.45,
                max(r["ctx_max"], r["binding_score"]) + b_top * 0.02,
                f"{int(r['ctx_rank'])}/{int(r['ctx_n'])}\n{r['ctx_pct']:.1f} pct",
                ha="left", va="bottom", fontsize=7, color=COLOR_CEIL,
                linespacing=1.35,
            )

    # --- x axis, grid, legend ----------------------------------------------
    ax1.set_xticks(x)
    ax1.set_xticklabels(_xlabels(panel), fontsize=8)
    ax1.set_xlim(-0.6, n - 0.4)
    ax1.grid(axis="y", color="lightgray", linewidth=0.5, zorder=0)
    ax1.set_axisbelow(True)

    handles = [
        mpatches.Patch(facecolor=COLOR_ORIG, label="Original prob (KinBERT, dblysate)"),
        mpatches.Patch(facecolor=COLOR_FINETUNED, label="Finetuned prob (phosphoPCA KSI)"),
        mpatches.Patch(facecolor=COLOR_BINDING,
                       label="Binding score at labelled PAABD  [right axis]"),
        mlines.Line2D([], [], color=COLOR_CEIL, linewidth=1.4,
                      label="screen max, same kinase-PAABD"),
        mlines.Line2D([], [], color=COLOR_CEIL, linewidth=1.2,
                      linestyle=(0, (1.5, 1.5)),
                      label="screen p95, same kinase-PAABD"),
        mpatches.Patch(facecolor="none", edgecolor=MUTED, hatch="///",
                       label="not in screening library (n/a)"),
    ]
    # above the axes: 5 entries collide with the bars anywhere inside
    ax1.legend(handles=handles, fontsize=7.5, frameon=False, ncol=2,
               loc="lower left", bbox_to_anchor=(0, 1.01))

    notes = ["Dashed grey line = 0.5 decision boundary; left and right scales are independent"]
    if (panel["prob_source"] == "rescored").any():
        notes.append(
            "† absent from every saved inference output; scored directly from the "
            "same two checkpoints (rescore_results.csv)"
        )
    ax1.text(0.5, -0.22, "\n".join(notes), transform=ax1.transAxes, ha="center",
             fontsize=7, color=MUTED)

    fig.savefig(outfile, bbox_inches="tight")
    return fig


def plot_context(panel, outfile="binding_context.pdf", inference_pkl=INFERENCE_PKL):
    """Supporting figure: where each candidate sits in its kinase-PAABD screen.

    One panel per candidate that has screen data.  The histogram is every
    substrate the screen measured for that same kinase-PAABD pair, so the
    candidate's bar in the main figure can be read as a tail position rather
    than as an absolute binding magnitude.
    """
    import matplotlib.pyplot as plt

    inference = pd.read_pickle(inference_pkl)
    have = panel[panel["ctx_n"].notna()].reset_index(drop=True)
    if not len(have):
        raise ValueError("no candidate has screen data to plot")

    fig, axes = plt.subplots(
        1, len(have), figsize=(4.6 * len(have), 3.4), squeeze=False
    )

    for ax, (_, r) in zip(axes[0], have.iterrows()):
        pop = inference.loc[
            (inference["kinase"] == r["kinase"]) & (inference["paabd"] == r["paabd"]),
            "mean",
        ].dropna()

        ax.hist(pop, bins=60, color="#C9CCD1", edgecolor="white", linewidth=0.3,
                zorder=2)
        ax.axvline(r["ctx_median"], color=MUTED, linewidth=1.0,
                   linestyle=(0, (4, 2)), zorder=3)
        ax.axvline(r["ctx_p95"], color=COLOR_CEIL, linewidth=1.2,
                   linestyle=(0, (1.5, 1.5)), zorder=3)
        ax.axvline(r["binding_score"], color=COLOR_BINDING, linewidth=1.8, zorder=4)

        top = ax.get_ylim()[1]
        pad = (pop.max() - pop.min()) * 0.015
        ax.text(r["ctx_median"] - pad, top * 0.55, "median ", fontsize=7,
                color=MUTED, ha="right", va="top")
        ax.text(r["ctx_p95"] + pad, top * 0.55, " p95", fontsize=7,
                color=COLOR_CEIL, ha="left", va="top")

        # corner block rather than a leader line: the candidate always lands far
        # out in the tail, where an arrow has to cross the whole distribution
        ax.text(
            0.97, 0.97,
            f"{r['name']}\n{r['binding_score']:.3f}\n"
            f"rank {int(r['ctx_rank'])} / {int(r['ctx_n'])}\n"
            f"{r['ctx_pct']:.1f}th pct",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color=COLOR_BINDING, linespacing=1.5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2),
        )

        kin = r["kinase"].replace("_HUMAN", "")
        ax.set_title(
            f"{kin} - {r['paabd'].upper()}   (n = {int(r['ctx_n'])} substrates)",
            fontsize=9, color=INK,
        )
        ax.set_xlabel("Binding score (phosphoPCA screen)", fontsize=8.5, color=INK)
        ax.set_ylabel("Substrates", fontsize=8.5, color=INK)
        ax.tick_params(labelsize=8, colors=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="lightgray", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    return fig


def plot_panel_split(panel, outfile="kinase_prob_plot_split.pdf"):
    """Two panels sharing x rather than two y-scales on one axes.

    Probability and binding score have unrelated units, so twinned axes make bar
    heights look comparable when they are not.  Kept as the alternative to
    plot_panel(); same data, no second y-scale.
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    n = len(panel)
    x = np.arange(n)
    bar_w = 0.34

    fig, (ax_p, ax_b) = plt.subplots(
        2, 1, figsize=(8.5, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12},
    )

    # --- probability panel -------------------------------------------------
    ax_p.bar(x - bar_w / 2, panel["orig_prob"], width=bar_w, color=COLOR_ORIG,
             edgecolor="white", linewidth=0.5, zorder=3, label="KinBERT (dblysate-trained)")
    ax_p.bar(x + bar_w / 2, panel["finetuned_prob"], width=bar_w, color=COLOR_FINETUNED,
             edgecolor="white", linewidth=0.5, zorder=3, label="KinBERT (phosphoPCA-finetuned)")
    ax_p.axhline(0.5, color=MUTED, linestyle="--", linewidth=1, zorder=2)
    ax_p.text(n - 0.45, 0.5, " 0.5", va="center", ha="left", fontsize=7, color=MUTED)
    ax_p.set_ylabel("Predicted probability", fontsize=9, color=INK)
    ax_p.set_ylim(0, 1.05)
    ax_p.set_yticks(np.arange(0, 1.01, 0.25))
    ax_p.legend(fontsize=8, frameon=False, loc="upper left", ncol=1)

    # --- binding panel -----------------------------------------------------
    # Cap bar behind: the best any PAABD reporter achieved, i.e. the ceiling on
    # this substrate regardless of which domain the reporter ends up using.
    # Solid bar in front: the labelled PAABD's own measurement, which is what a
    # reporter built on that domain actually gets.
    ax_b.bar(x, panel["binding_max_any"], width=bar_w,
             facecolor="none", edgecolor=COLOR_BINDING, linewidth=1.0,
             linestyle="--", zorder=2)
    ax_b.bar(x, panel["binding_score"], width=bar_w, color=COLOR_BINDING,
             edgecolor="white", linewidth=0.5, zorder=3)
    ax_b.set_ylabel("Binding score\n(phosphoPCA screen)", fontsize=9, color=INK)
    finite = pd.concat([panel["binding_score"], panel["binding_max_any"]]).dropna()
    ax_b.set_ylim(0, (finite.max() * 1.30) if len(finite) else 1)

    cap_handles = [
        mpatches.Patch(facecolor=COLOR_BINDING, edgecolor="white",
                       label="at labelled PAABD (plotted)"),
        mpatches.Patch(facecolor="none", edgecolor=COLOR_BINDING, linestyle="--",
                       label="best across PAABDs (cap)"),
    ]
    ax_b.legend(handles=cap_handles, fontsize=7.5, frameon=False, loc="upper right")

    # label the cap where it differs from the plotted value
    for i, r in panel.iterrows():
        if pd.notna(r["binding_max_any"]) and (
            r["binding_max_any"] - (r["binding_score"] if pd.notna(r["binding_score"]) else 0)
            > 0.02
        ):
            ax_b.text(i, r["binding_max_any"], f" {r['binding_max_paabd']}",
                      ha="center", va="bottom", fontsize=7, color=COLOR_BINDING)

    # --- mark what was never measured, from the data ------------------------
    for i, r in panel.iterrows():
        if pd.isna(r["binding_score"]) and pd.isna(r["binding_max_any"]):
            # shade only the binding panel: these candidates do have predictions
            ax_b.axvspan(i - 0.5, i + 0.5, color=MUTED, alpha=0.08, zorder=0)
            ax_b.text(i, ax_b.get_ylim()[1] * 0.45,
                      "not in\nscreening library", ha="center", va="center",
                      fontsize=7.5, color=MUTED, style="italic")
        if pd.isna(r["orig_prob"]) and pd.isna(r["finetuned_prob"]):
            ax_p.text(i, 0.30, "not scored", ha="center", va="center",
                      fontsize=7.5, color=MUTED, style="italic")

    for ax in (ax_p, ax_b):
        ax.grid(axis="y", color="lightgray", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8, colors=INK)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(_xlabels(panel), fontsize=8)
    ax_b.set_xlim(-0.6, n - 0.4)

    if (panel["prob_source"] == "rescored").any():
        fig.text(
            0.5, 0.02,
            "† not present in any saved inference output; scored directly from the "
            "same two checkpoints (see rescore_results.csv)",
            ha="center", fontsize=7, color=MUTED,
        )

    fig.savefig(outfile, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    panel = build_panel()
    pd.set_option("display.width", 250)
    print(panel.to_string(index=False))
    plot_panel(panel, "kinase_prob_plot_v2.pdf")
    plot_context(panel, "binding_context.pdf")
    print("\nwrote kinase_prob_plot_v2.pdf, binding_context.pdf")
