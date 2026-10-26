#!/usr/bin/env python3
"""Figure 06 (Biological insights) - selectivity clustering and dark kinases.

Panels, and where each comes from:

  A  screening-derived substrate-selectivity tree beside the kinase-domain
     phylogeny, plus the pairwise distance comparison between them
     (11_ksi_tree_comparison_v2.ipynb, 11_2_kinase_tree_analysis.ipynb)
  B  novel substrate insight for understudied ("dark") kinases.
     Verified 2026-10-25: `--motif-kinase-set dark` (the default) reproduces
     the six deposited IC matrices bit-identically, so the published Fig 3B
     panels come from this path. The `stlib` option is retained as an
     alternative analysis that the manuscript does not use.
       - positional information-content panels per kinase
         (12_dark_kinase_analysis.ipynb + information_content_calculator.py)
       - GO enrichment of each dark kinase's substrate set
         (12_2_dark_kinase_go_analysis.ipynb)

Panel A needs MAFFT on PATH (sequence tree) and ete3 (cross-tree distances);
either sub-step can be skipped with --panel.

Consumes
    config.REDUCED_TP_PKL                  stage 07
    config.BLIB_KINASES_SRC_CSV
    config.SUBSTRATE_BACKGROUND_PKL        15-mer -> UniProt accession
    config.KINOME_NEWICK                   Manning (2002) kinome tree
    config.SCREEN_NEWICK_RENAMED           leaf-order-aligned screening tree
    config.DARK_SUB_IDMAPPING_TSV, config.ST_BACKGROUND_IDMAPPING_TSV

Produces (in config.FIG_DIR)
    fig06A_screening_tree.newick, fig06A_kinase_seq_tree.newick
    fig06A_aligned_dendrograms.pdf
    fig06A_cross_tree_scatter.pdf, fig06A_cross_tree_results.csv
    fig06B_IC_heatmap_{aa,prop}_{kinase}.png, fig06B_IC_matrix_*.csv
    fig06B_go_{kinase}.png, fig06B_go_results.csv
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import statistics
import subprocess
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage, to_tree
from scipy.spatial.distance import squareform
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

log = logging.getLogger("fig06")


# ═══════════════════════════════════════════════════════════════════════════
# Panel A - substrate-selectivity tree vs kinase-domain phylogeny
# ═══════════════════════════════════════════════════════════════════════════
def load_tree_hits() -> pd.DataFrame:
    """Confident hits per kinase, for the substrate-set tree.

    Two filters beyond the standard hit call: the lfsr cut is tightened to
    0.001, and a substrate must have been measured against several kinases.
    Both exist because the tree compares substrate *sets*, so a substrate seen
    by only one or two kinases carries no information about which kinases
    resemble each other, and a marginal call would add noise to every Jaccard
    distance it enters.
    """
    screen = pd.read_pickle(config.REDUCED_TP_PKL)

    y = screen.loc[screen["paabd_all"] == "sh2", :]
    well_rep_y = y.groupby("substrate")["kinase"].count().loc[
        lambda x: x > config.TREE_MIN_KINASES_Y
    ].index
    # NOTE: this line is copied verbatim from the notebook, parentheses and
    # all, because it is what produced the deposited tree. Python binds `&`
    # tighter than `<`, so it evaluates as
    #     (isin(well_rep_y) & lfsr_best_hit) < 0.001
    # i.e. the boolean mask is ANDed with the lfsr column and the result
    # compared to the threshold, which keeps the complement of the intended
    # filter: 11,657 of 13,943 Y rows, rather than the 1,058 that
    # `isin(...) & (lfsr_best_hit < 0.001)` would keep. The S/T branch below
    # does have the parentheses, so the two libraries are filtered
    # differently. Left as-is; see _source_notebooks/NOTES.md.
    y = y.loc[y["substrate"].isin(well_rep_y) & y["lfsr_best_hit"] < config.TREE_LFSR_MAX, :]

    st = screen.loc[screen["paabd_all"] != "sh2", :]
    well_rep_st = st.groupby("substrate")["kinase"].count().loc[
        lambda x: x > config.TREE_MIN_KINASES_ST
    ].index
    st = st.loc[
        st["substrate"].isin(well_rep_st)
        & (st["lfsr_best_hit"] < config.TREE_LFSR_MAX)
        & (st["fp_sub"] == False),  # noqa: E712 - keep the notebook's comparison
        :,
    ]

    hits = pd.concat([st, y])
    hits = hits.copy()
    hits["kinase"] = hits["kinase"].str.replace("_HUMAN", "")
    log.info("tree input: %d rows, %d kinases, %d substrates",
             len(hits), hits["kinase"].nunique(), hits["substrate"].nunique())
    return hits


def kinase_set_tree(kinase_to_subs: dict, metric: str, linkage_method: str):
    """Pairwise substrate-set distance between kinases, and its linkage.

    Jaccard on the hit sets: two kinases are close when they hit the same
    substrates, whatever their sequences say.
    """
    kinases = list(kinase_to_subs.keys())
    universe = sorted(set().union(*[set(kinase_to_subs[k]) for k in kinases]))
    col_index = {s: j for j, s in enumerate(universe)}

    X = np.zeros((len(kinases), len(universe)), dtype=np.uint8)
    for i, k in enumerate(kinases):
        for s in kinase_to_subs[k]:
            X[i, col_index[s]] = 1
    Xw = X.astype(float)

    n = len(kinases)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = Xw[i] > 0, Xw[j] > 0
            if metric == "jaccard":
                union = np.logical_or(a, b).sum()
                sim = 0.0 if union == 0 else np.logical_and(a, b).sum() / union
            elif metric == "dice":
                denom = a.sum() + b.sum()
                sim = 0.0 if denom == 0 else (2 * np.logical_and(a, b).sum()) / denom
            elif metric == "overlap":
                denom = min(a.sum(), b.sum())
                sim = 0.0 if denom == 0 else np.logical_and(a, b).sum() / denom
            else:
                raise ValueError(f"Unknown metric: {metric}")
            D[i, j] = D[j, i] = 1.0 - sim

    dist_df = pd.DataFrame(D, index=kinases, columns=kinases)
    Z = linkage(D[np.triu_indices(n, k=1)], method=linkage_method)
    return dist_df, Z


def _quote_newick_label(name) -> str:
    name = str(name)
    needs_quote = any(c in name for c in " ():,;[]\t\n\r'")
    if "'" in name:
        needs_quote = True
        name = name.replace("'", "''")
    return f"'{name}'" if needs_quote else name


def export_newick_exact(dist_df, Z, out_newick: Path, precision: int = 10) -> str:
    """Newick string matching the linkage tree exactly (for iTOL / ete3)."""
    labels = list(dist_df.index)
    if list(dist_df.columns) != labels:
        raise ValueError("dist_df columns must match dist_df.index order.")

    root, _ = to_tree(Z, rd=True)

    def rec(node, parent_dist):
        node_dist = float(node.dist)
        bl = 0.0 if parent_dist is None else max(parent_dist - node_dist, 0.0)
        if node.is_leaf():
            nm = _quote_newick_label(labels[node.id])
            return nm if parent_dist is None else f"{nm}:{bl:.{precision}f}"
        left = rec(node.left, node_dist)
        right = rec(node.right, node_dist)
        subtree = f"({left},{right})"
        return subtree if parent_dist is None else f"{subtree}:{bl:.{precision}f}"

    newick = rec(root, None) + ";"
    out_newick.write_text(newick + "\n")
    log.info("wrote %s", out_newick.name)
    return newick


def export_sequence_tree_newick(kinase_to_seq: dict, out_newick: Path, method: str = "nj"):
    """Kinase-domain tree: MAFFT alignment -> identity distance -> NJ."""
    from Bio import AlignIO, Phylo
    from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor

    def _sanitize(name):
        bad = " \t\n\r()[]{}:,;\"'"
        return "".join("_" if c in bad else c for c in str(name))

    records = [(_sanitize(k), str(seq).strip().upper())
               for k, seq in kinase_to_seq.items() if seq]
    if len(records) < 2:
        raise ValueError("Need at least 2 kinase sequences.")

    fasta_path = out_newick.with_suffix(".fasta")
    with open(fasta_path, "w") as f:
        for name, seq in records:
            f.write(f">{name}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")

    if shutil.which("mafft") is None:
        raise RuntimeError("MAFFT not found. Install via: conda install -c bioconda mafft")

    res = subprocess.run(["mafft", "--auto", str(fasta_path)],
                         capture_output=True, text=True, check=True)
    aln_path = out_newick.with_suffix(".aln.fasta")
    aln_path.write_text(res.stdout)
    alignment = AlignIO.read(str(aln_path), "fasta")

    dm = DistanceCalculator("identity").get_distance(alignment)
    names = dm.names
    dist_df = pd.DataFrame(
        [[dm[i, j] for j in range(len(names))] for i in range(len(names))],
        index=names, columns=names,
    )

    constructor = DistanceTreeConstructor()
    tree = constructor.nj(dm) if method == "nj" else constructor.upgma(dm)
    Phylo.write(tree, str(out_newick), "newick")
    log.info("wrote %s", out_newick.name)
    return dist_df


def _node_leaf_indices(node) -> list:
    if node.is_leaf():
        return [node.id]
    return _node_leaf_indices(node.left) + _node_leaf_indices(node.right)


def _median_position(node, order_map) -> float:
    return np.median([order_map.get(l, 0) for l in _node_leaf_indices(node)])


def _swap_to_match(node, order_map) -> None:
    if node.is_leaf():
        return
    _swap_to_match(node.left, order_map)
    _swap_to_match(node.right, order_map)
    if _median_position(node.left, order_map) > _median_position(node.right, order_map):
        node.left, node.right = node.right, node.left


def _tree_to_linkage(root, n_leaves: int) -> np.ndarray:
    rows = []
    next_id = [n_leaves]

    def _rec(node):
        if node.is_leaf():
            return node.id
        left_id = _rec(node.left)
        right_id = _rec(node.right)
        new_id = next_id[0]
        next_id[0] += 1
        n = node.count if hasattr(node, "count") else len(_node_leaf_indices(node))
        rows.append([float(left_id), float(right_id), float(node.dist), float(n)])
        return new_id

    _rec(root)
    return np.array(rows)


def reorder_linkage_to_match(Z_target, Z_reference, labels_target, labels_reference):
    """Align the drawing order of the two trees without changing either tree.

    At every fork of the screening tree, the two subtrees are swapped if that
    brings its leaves closer to the order the sequence tree puts them in.
    Topology and branch lengths are untouched; only left/right order changes,
    which is what makes the two panels readable side by side.
    """
    import warnings

    if set(labels_target) != set(labels_reference):
        common = sorted(set(labels_target) & set(labels_reference))
        warnings.warn(
            f"Label mismatch: using {len(common)} common kinases "
            f"({len(labels_target)} in screening, {len(labels_reference)} in sequence)."
        )
        labels_target = [l for l in labels_target if l in common]
        labels_reference = [l for l in labels_reference if l in common]

    ref_order = [labels_reference[i] for i in leaves_list(Z_reference)]
    ref_pos = {label: pos for pos, label in enumerate(ref_order)}
    order_map = {i: ref_pos.get(labels_target[i], i) for i in range(len(labels_target))}

    root, _ = to_tree(Z_target, rd=True)
    _swap_to_match(root, order_map)
    Z_reordered = _tree_to_linkage(root, n_leaves=len(labels_target))
    reordered_labels = [labels_target[i] for i in leaves_list(Z_reordered)]
    return Z_reordered, reordered_labels


def panel_a_trees(fig_dir: Path) -> None:
    hits = load_tree_hits()
    screened_kinases = pd.read_pickle(config.REDUCED_TP_PKL)["kinase"]

    blib = pd.read_csv(config.BLIB_KINASES_SRC_CSV, header=None)
    blib.columns = ["kinase_type", "kinase", "kinase_seq", "kinase_acc"]
    blib = blib.loc[blib["kinase"].isin(screened_kinases), :].copy()
    blib["kinase"] = blib["kinase"].str.replace("_HUMAN", "")

    kinase_to_subs = (
        hits.loc[:, ["kinase", "substrate"]]
        .groupby("kinase")["substrate"]
        .apply(lambda s: set(s.dropna().astype(str)))
        .to_dict()
    )
    dist_screen, Z_screen = kinase_set_tree(
        kinase_to_subs, metric=config.TREE_METRIC, linkage_method=config.TREE_LINKAGE
    )
    labels_screen = list(dist_screen.index)
    export_newick_exact(dist_screen, Z_screen, fig_dir / "fig06A_screening_tree.newick")

    kinase_to_seq = blib.set_index("kinase")["kinase_seq"].to_dict()
    dist_seq = export_sequence_tree_newick(
        kinase_to_seq, fig_dir / "fig06A_kinase_seq_tree.newick", method="nj"
    )
    labels_seq = list(dist_seq.index)
    Z_seq = linkage(squareform(dist_seq.values, checks=False), method=config.TREE_LINKAGE)

    Z_aligned, labels_aligned = reorder_linkage_to_match(
        Z_screen, Z_seq, labels_screen, labels_seq
    )
    log.info("leaf order after alignment: %s", ", ".join(labels_aligned))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    dendrogram(Z_aligned, labels=labels_screen, orientation="right", ax=axes[0])
    axes[0].set_title("Screening tree (leaf-order aligned)", fontsize=11)
    dendrogram(Z_seq, labels=labels_seq, orientation="right", ax=axes[1])
    axes[1].set_title("Sequence tree", fontsize=11)
    fig.suptitle("Leaf-order-aligned dendrograms", fontsize=13, y=1.01)
    fig.savefig(fig_dir / "fig06A_aligned_dendrograms.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote fig06A_aligned_dendrograms.pdf")


# ── Cross-tree distance comparison (11_2) ────────────────────────────────────
def load_ete_tree(newick_path: Path):
    """Parse a Newick tree, tolerating the kinome tree's [bootstrap] labels."""
    from ete3 import Tree

    clean = re.sub(r"\[\d+\.\d+\]", "", Path(newick_path).read_text())
    for fmt in (0, 1, 3):
        try:
            return Tree(clean, format=fmt)
        except Exception:
            continue
    raise ValueError(f"Could not parse tree: {newick_path}")


def normalised_distances(newick_path: Path):
    """All pairwise leaf distances, scaled by the tree's maximum."""
    t = load_ete_tree(newick_path)
    leaves = [leaf.name for leaf in t.get_leaves()]

    raw = {}
    for a, b in combinations(leaves, 2):
        dist = t.get_distance(a, b)
        raw[(a, b)] = dist
        raw[(b, a)] = dist

    max_dist = max(raw.values())
    return {pair: val / max_dist for pair, val in raw.items()}, max_dist, leaves


def panel_a_cross_tree(fig_dir: Path) -> None:
    """Classify kinase pairs by whether sequence and selectivity agree."""
    import seaborn as sns
    from scipy import stats

    screen_dists, screen_max, screen_leaves = normalised_distances(config.SCREEN_NEWICK_RENAMED)
    log.info("screening tree: %d leaves, max raw distance %.4f", len(screen_leaves), screen_max)
    kinome_dists, kinome_max, kinome_leaves = normalised_distances(config.KINOME_NEWICK)
    log.info("kinome tree: %d leaves, max raw distance %.4f", len(kinome_leaves), kinome_max)

    canonical = [s for s in config.SCREEN_TO_KINOME if s in screen_leaves]
    excluded = [
        s for s in screen_leaves
        if s not in config.SCREEN_TO_KINOME and s not in config.TREE_VARIANT_LEAVES
    ]
    log.info("%d screening leaves have a kinome counterpart; excluded: %s",
             len(canonical), excluded)

    rows = []
    for a, b in combinations(sorted(canonical), 2):
        sd = screen_dists.get((a, b))
        kd = kinome_dists.get((config.SCREEN_TO_KINOME[a], config.SCREEN_TO_KINOME[b]))
        if sd is not None and kd is not None:
            rows.append({"kinase_a": a, "kinase_b": b, "screen_dist": sd, "kinome_dist": kd})
    df = pd.DataFrame(rows)
    log.info("mappable pairs: %d", len(df))

    # NOTE: the notebook's comments and variable names say "median", but the
    # code it ran (and the figure) use the mean of each distance vector. Kept.
    k_thresh = float(statistics.mean(df["kinome_dist"]))
    s_thresh = float(statistics.mean(df["screen_dist"]))
    log.info("thresholds: kinome %.4f, screen %.4f", k_thresh, s_thresh)

    def classify(row):
        kc = row["kinome_dist"] < k_thresh
        sc = row["screen_dist"] < s_thresh
        if kc and sc:
            return "Conservation"      # related and similarly selective
        if kc and not sc:
            return "Divergence"        # related but differently selective
        if not kc and sc:
            return "Convergence"       # unrelated but similarly selective
        return "Expected far"

    df["category"] = df.apply(classify, axis=1)
    log.info("category counts: %s", df["category"].value_counts().to_dict())

    sv, kv = df["screen_dist"].values, df["kinome_dist"].values
    r, r_p = stats.pearsonr(sv, kv)
    rho, rho_p = stats.spearmanr(sv, kv)
    tau, tau_p = stats.kendalltau(sv, kv)
    n_ceiling = int(np.sum(sv >= s_thresh))
    log.info("screen mean %.3f sd %.3f; kinome mean %.3f sd %.3f",
             sv.mean(), sv.std(ddof=1), kv.mean(), kv.std(ddof=1))
    log.info("%d/%d (%.0f%%) screen distances are at ceiling; "
             "rank metrics are the primary effect sizes",
             n_ceiling, len(df), 100 * n_ceiling / len(df))
    log.info("Pearson r %.4f (p %.2e), Spearman rho %.4f (p %.2e), "
             "Kendall tau %.4f (p %.2e)", r, r_p, rho, rho_p, tau, tau_p)

    cat_order = ["Conservation", "Divergence", "Convergence", "Expected far"]
    palette = {
        "Conservation": "#2166ac",
        "Divergence": "#d73027",
        "Convergence": "#1a9850",
        "Expected far": "#aaaaaa",
    }
    fig, ax = plt.subplots(figsize=(7, 6))
    for cat in cat_order:
        sub = df[df["category"] == cat]
        ax.scatter(sub["screen_dist"], sub["kinome_dist"], c=palette[cat],
                   label=f"{cat} (n={len(sub)})", s=40, alpha=0.75,
                   edgecolors="white", linewidths=0.4, zorder=3)
    ax.axvline(s_thresh, color="#555555", lw=0.9, ls="--", alpha=0.6)
    ax.axhline(k_thresh, color="#555555", lw=0.9, ls="--", alpha=0.6)
    ax.text(0.03, 0.97,
            f"Pearson r = {r:.3f}\nSpearman ρ = {rho:.3f}\nKendall τ = {tau:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", alpha=0.85))
    ax.set_xlabel("Normalised screening-tree distance", fontsize=11)
    ax.set_ylabel("Normalised kinome-tree distance", fontsize=11)
    ax.set_title("Kinome sequence distance vs. substrate selectivity distance", fontsize=12)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlim(-0.02, 1.08)
    sns.despine(ax=ax)
    fig.savefig(fig_dir / "fig06A_cross_tree_scatter.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote fig06A_cross_tree_scatter.pdf")

    out = (
        df[["kinase_a", "kinase_b", "screen_dist", "kinome_dist", "category"]]
        .sort_values(["category", "screen_dist"])
        .reset_index(drop=True)
    )
    out.to_csv(fig_dir / "fig06A_cross_tree_results.csv", index=False)
    log.info("wrote fig06A_cross_tree_results.csv (%d rows)", len(out))


# ═══════════════════════════════════════════════════════════════════════════
# Panel B - dark-kinase substrate motifs and GO enrichment
# ═══════════════════════════════════════════════════════════════════════════
def promiscuity_filter(screen: pd.DataFrame, max_score: float,
                       min_kinases: int | None) -> pd.Index:
    """Substrates that are selective enough to carry positional information.

    A substrate positive for a large share of the kinases it was measured
    against reports something other than kinase specificity, and would
    otherwise dominate every position of the motif.
    """
    total = screen.groupby(["substrate"]).count().reset_index().iloc[:, [0, 1]]
    pos = (
        screen.loc[screen["positive"], :]
        .groupby(["substrate"]).count().reset_index().iloc[:, [0, 1]]
    )
    stats = pd.merge(pos, total, on="substrate")
    stats.columns = ["substrate", "pos_kinase", "tot_kinase"]
    stats["promiscuous_score"] = stats["pos_kinase"] / stats["tot_kinase"]
    keep = stats["promiscuous_score"] < max_score
    if min_kinases is not None:
        keep &= stats["tot_kinase"] > min_kinases
    return stats.loc[keep, "substrate"]


def panel_b_motifs(fig_dir: Path, kinase_set: str) -> None:
    from plotnine import theme

    from information_content_calculator import make_ic_panel1_plot_and_df

    screen = pd.read_pickle(config.REDUCED_TP_PKL)
    # S/T substrates only: the Y library has its own reporter and its own
    # background, and mixing them would blur the positional statistics.
    screen_st = screen.loc[screen["substrate"].apply(lambda s: s[7] != "Y"), :]

    # The background is every S/T substrate screened, taken before the
    # promiscuity filter, so that the motif is scored against the library as
    # designed rather than against the surviving subset.
    background = screen_st[["substrate"]].drop_duplicates()

    if kinase_set == "dark":
        kinases = config.DARK_KINASES
        keep = promiscuity_filter(screen_st, config.MOTIF_PROMISCUITY_MAX,
                                  config.MOTIF_MIN_KINASES)
        filtered = screen_st.loc[screen_st["substrate"].isin(keep), :]
    else:
        # 11_ksi_motif_analysis-stlib.ipynb: a looser promiscuity cut, plus two
        # extra filters that remove the substrates the 14-3-3 reporter reads
        # through its own R-3 motif rather than through the kinase.
        kinases = config.STLIB_MOTIF_KINASES
        keep = promiscuity_filter(screen_st, config.STLIB_PROMISCUITY_MAX, None)
        filtered = screen_st.loc[screen_st["substrate"].isin(keep), :]
        filtered = filtered.loc[filtered["paabd_all"] != "sh2", :]
        filtered = filtered.loc[filtered["substrate"].apply(lambda s: s[4] != "R"), :]

    hits = filtered.loc[filtered["kinase"].isin(kinases) & filtered["true_positive"], :]
    hits = hits.rename(columns={"substrate": "substrate_seq_15aa"})
    log.info("%s motif set: %d hits over %d kinases, %d background substrates",
             kinase_set, len(hits), hits["kinase"].nunique(), len(background))

    background_sequences = background["substrate"].values
    for kinase in [k for k in kinases if (hits["kinase"] == k).any()]:
        kinase_hits = hits.loc[hits["kinase"] == kinase, "substrate_seq_15aa"].values
        log.info("  %s: %d hits", kinase, len(kinase_hits))

        # block_plus1 masks the +1 proline: the WW reporter requires it, so it
        # is a property of the assay rather than of the kinase.
        p_aa, df_aa = make_ic_panel1_plot_and_df(
            kinase_hits, background_sequences, kinase_name=kinase,
            level="amino_acid", exclude_phospho_site=True, block_plus1=True,
        )
        p_prop, df_prop = make_ic_panel1_plot_and_df(
            kinase_hits, background_sequences, kinase_name=kinase,
            level="property", exclude_phospho_site=True, block_plus1=True,
        )

        p_aa = p_aa + theme(figure_size=(12, 8))
        p_aa.save(fig_dir / f"fig06B_IC_heatmap_aa_{kinase}.png", dpi=300, verbose=False)
        p_prop.save(fig_dir / f"fig06B_IC_heatmap_prop_{kinase}.png", dpi=300, verbose=False)
        df_aa.to_csv(fig_dir / f"fig06B_IC_matrix_aa_{kinase}.csv")
        df_prop.to_csv(fig_dir / f"fig06B_IC_matrix_prop_{kinase}.csv")
    log.info("wrote IC heatmaps and matrices to %s", fig_dir)


# ── GO enrichment ────────────────────────────────────────────────────────────
GO_ID_RE = re.compile(r"(GO:\d{7})")
GO_NAME_RE = re.compile(r"(.+?)\s*\[(GO:\d{7})\]")


def _split_terms(x) -> list:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if not isinstance(x, str):
        x = str(x)
    return [p.strip() for p in x.strip().split(";") if p.strip()]


def _merge_terms(series) -> str | float:
    """Union of a protein's GO terms across its rows, in first-seen order."""
    seen, out = set(), []
    for v in series.dropna():
        for term in _split_terms(v):
            if term not in seen:
                seen.add(term)
                out.append(term)
    return "; ".join(out) if out else np.nan


def _count_unique_terms(x) -> int:
    terms = []
    for v in x.dropna():
        terms.extend(_split_terms(v))
    return len(set(terms))


def _collapse_go(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (15-mer, kinase), with GO terms pooled over isoforms.

    A 15-mer can map to several accessions (isoforms, mouse orthologue); their
    annotations are unioned so the term set does not depend on which accession
    the mapping happened to pick.
    """
    merged = df.groupby("substrate_y", as_index=False)[config.GO_COLS].agg(_merge_terms)
    counts = (
        df.groupby("substrate_y", as_index=False)[config.GO_COLS]
        .agg(_count_unique_terms)
        .rename(columns={c: f"{c}__n_terms" for c in config.GO_COLS})
    )
    merged = merged.merge(counts, on="substrate_y", how="left")
    out = df.drop(columns=config.GO_COLS, errors="ignore").merge(
        merged, on="substrate_y", how="left"
    )
    return out.drop_duplicates().groupby(["substrate_seq_15aa", "kinase"]).head(1)


def build_go_map(df: pd.DataFrame) -> dict:
    """accession -> set of GO ids."""
    go_map = {}
    for _, r in df.iterrows():
        gos = set()
        for col in config.GO_COLS:
            val = r.get(col, "")
            if isinstance(val, str):
                gos |= set(GO_ID_RE.findall(val))
        go_map[r["substrate_x"]] = gos
    return go_map


def _go_id_name_pairs(val) -> list:
    if not isinstance(val, str):
        return []
    pairs = []
    for part in re.split(r";", val):
        m = GO_NAME_RE.search(part)
        if m:
            pairs.append((m.group(2), m.group(1).strip()))
    return pairs


def build_go_label_maps(df: pd.DataFrame, names: dict, types: dict) -> None:
    """GO id -> term name, and GO id -> which of the three GO branches."""
    for _, r in df.iterrows():
        for col in config.GO_COLS:
            for go_id, go_name in _go_id_name_pairs(r.get(col, "")):
                names.setdefault(go_id, go_name)
                types.setdefault(go_id, col)


def go_enrichment(hit_go: dict, bg_go: dict,
                  min_bg: int = config.GO_MIN_BG,
                  max_bg: int = config.GO_MAX_BG) -> pd.DataFrame:
    """One-sided Fisher test per GO term, hits against the rest of the library.

    The background excludes this kinase's own hit proteins, so the test asks
    whether the hits are enriched relative to the substrates that were screened
    and not hit. Terms on more than max_bg background proteins are too generic
    to be informative.
    """
    bg_subs = set(bg_go.keys()) - set(hit_go.keys())
    hit_subs = set(hit_go.keys())
    total_hit, total_bg = len(hit_subs), len(bg_subs)

    bg_counts = Counter(go for s in bg_subs for go in bg_go.get(s, []))

    rows = []
    for go_id, bg_n in bg_counts.items():
        if (bg_n < min_bg) or (bg_n > max_bg):
            continue
        hit_n = sum(go_id in hit_go.get(s, set()) for s in hit_subs)
        if hit_n == 0:
            continue
        odds, p = fisher_exact(
            [[hit_n, total_hit - hit_n], [bg_n, total_bg - bg_n]], alternative="greater"
        )
        rows.append({
            "GO_ID": go_id, "hit_with_GO": hit_n, "total_hits": total_hit,
            "bg_with_GO": bg_n, "total_bg": total_bg,
            "odds_ratio": odds, "p_value": p,
        })
    return pd.DataFrame(rows).sort_values("p_value")


def bh_fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    p = np.asarray(pvals)
    n = len(p)
    idx = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        q_val = p[idx[i]] * n / (i + 1)
        prev = min(prev, q_val)
        q[idx[i]] = prev
    return q


def plot_go_enrichment(df: pd.DataFrame, out_path: Path, kinase_name: str,
                       top_n: int = config.GO_PANEL_TOP_N, figsize=(12, 11)):
    """Faceted dot plot: -log10 p per GO term, one facet per GO branch."""
    import textwrap

    from plotnine import (aes, coord_cartesian, element_text, facet_wrap, geom_point,
                          geom_vline, ggplot, labs, scale_color_manual,
                          scale_size_continuous, scale_y_discrete, theme, theme_bw)

    df = df.copy()
    df["GO_category"] = (
        df["GO_type"].astype(str)
        .str.replace("Gene Ontology ", "", regex=False)
        .str.replace(r"[()]", "", regex=True)
        .str.strip().str.lower()
    )
    df["GO_name_short"] = df["GO_name"].apply(lambda x: "\n".join(textwrap.wrap(str(x), 30)))
    too_long = df["GO_name"].astype(str).str.len() > 50
    df.loc[too_long, "GO_name_short"] = df.loc[too_long, "GO_name_short"] + "..."
    df["gene_ratio"] = df["hit_with_GO"] / df["total_hits"]
    df["neg_log10_pval"] = -np.log10(df["p_value"].astype(float))

    categories = ["biological process", "cellular component", "molecular function"]
    df_plot = (
        df.sort_values("p_value", ascending=True)
        .groupby("GO_category", group_keys=False).head(top_n).copy()
    )
    df_plot = df_plot[df_plot["GO_category"].isin(categories)].copy()
    df_plot["GO_category"] = pd.Categorical(
        df_plot["GO_category"], categories=categories, ordered=True
    )

    def _order_within_facet(g):
        g = g.sort_values("p_value", ascending=False).copy()
        g["GO_name_short_ordered"] = pd.Categorical(
            g["GO_name_short"], categories=g["GO_name_short"], ordered=True
        )
        return g

    df_plot = df_plot.groupby("GO_category", group_keys=False).apply(_order_within_facet)
    df_plot["log_odds"] = np.log10(df_plot["odds_ratio"])

    color_map = {
        "biological process": "#E64B35",
        "cellular component": "#4DBBD5",
        "molecular function": "#00A087",
    }
    max_x = float(df_plot["neg_log10_pval"].max()) + 0.5

    p = (
        ggplot(df_plot, aes(x="neg_log10_pval", y="GO_name_short_ordered",
                            color="GO_category", size="log_odds"))
        + geom_vline(xintercept=-np.log10(0.05), linetype="dashed", alpha=0.7)
        + geom_point(alpha=0.8, stroke=0.5)
        + facet_wrap("~GO_category", ncol=1, scales="free")
        + scale_color_manual(values=color_map)
        + scale_size_continuous(range=(2, 14))
        + coord_cartesian(xlim=(-0.5, max_x))
        + scale_y_discrete(limits=lambda l: list(reversed(l)))
        + labs(title="GO Enrichment Analysis", subtitle=kinase_name,
               x="-log₁₀(p-value)", y="",
               color="Category", size="Gene Ratio (log odds)")
        + theme_bw()
        + theme(
            figure_size=figsize,
            title=element_text(size=16, weight="bold"),
            axis_title_x=element_text(size=11, weight="bold"),
            axis_text_y=element_text(size=9.5),
            axis_text_x=element_text(size=10),
            strip_text=element_text(size=12, weight="bold"),
            legend_title=element_text(size=11, weight="bold"),
            legend_text=element_text(size=10),
        )
    )
    p.save(out_path, dpi=300, verbose=False)
    log.info("wrote %s", out_path.name)
    return p


def panel_b_go(fig_dir: Path) -> None:
    screen = pd.read_pickle(config.REDUCED_TP_PKL)
    background = pd.read_pickle(config.SUBSTRATE_BACKGROUND_PKL)
    # NOTE: unlike the motif panels, the GO analysis keeps the whole hit table
    # (the notebook's S/T-only filter is commented out). The dark kinases are
    # S/T kinases, so their hits are S/T either way; the difference is in the
    # background, which here spans both libraries.
    mapped = pd.merge(background, screen,
                      left_on="substrate_seq_15aa", right_on="substrate")

    keep = promiscuity_filter(screen, config.MOTIF_PROMISCUITY_MAX, config.MOTIF_MIN_KINASES)
    hits = mapped.loc[
        mapped["kinase"].isin(config.DARK_KINASES) & mapped["true_positive"], :
    ]
    hits = hits.loc[hits["substrate_seq_15aa"].isin(keep), :]
    log.info("GO input: %d dark-kinase hit rows over %d accessions",
             len(hits), hits["substrate_x"].nunique())

    bg_ann_raw = pd.read_csv(config.ST_BACKGROUND_IDMAPPING_TSV, sep="\t")

    def annotate(frame):
        df = pd.merge(frame, bg_ann_raw, left_on="substrate_x", right_on="From")
        # Human and mouse only: the UniProt mapping also returns distant
        # orthologues, whose annotation is not comparable.
        df = df.loc[df["Organism"].isin(config.GO_ORGANISMS), :]
        return _collapse_go(df)

    bg_ann = annotate(mapped)
    sub_ann = annotate(hits)
    log.info("annotated: %d background rows, %d hit rows", len(bg_ann), len(sub_ann))

    hit_go = build_go_map(sub_ann)
    bg_go = build_go_map(bg_ann)
    log.info("background: %d 15-mers / %d accessions; hits: %d 15-mers / %d accessions",
             mapped["substrate_seq_15aa"].nunique(), bg_ann["substrate_x"].nunique(),
             hits["substrate_seq_15aa"].nunique(), sub_ann["substrate_x"].nunique())

    go_names: dict = {}
    go_types: dict = {}
    build_go_label_maps(sub_ann, go_names, go_types)
    build_go_label_maps(bg_ann, go_names, go_types)

    panels = []
    # NOTE: the per-kinase accession sets come from the un-annotated hit table,
    # as in the notebook, not from the collapsed one: where a 15-mer maps to
    # several accessions the collapse keeps one row per (15-mer, kinase), but
    # all of those accessions still count towards this kinase's hit set if they
    # carry annotation. Using the collapsed table instead changes the Fisher
    # denominators slightly.
    for kinase in hits["kinase"].unique():
        subs_k = set(hits.loc[hits["kinase"] == kinase, "substrate_x"])
        hit_go_k = {s: hit_go[s] for s in subs_k if s in hit_go}

        enrich = go_enrichment(hit_go_k, bg_go)
        enrich["q_value"] = bh_fdr(enrich["p_value"].values)
        enrich["GO_name"] = enrich["GO_ID"].map(go_names)
        enrich["GO_type"] = enrich["GO_ID"].map(go_types)
        enrich = enrich[["GO_ID", "GO_name", "GO_type", "hit_with_GO", "total_hits",
                         "bg_with_GO", "total_bg", "odds_ratio", "p_value", "q_value"]]
        log.info("%s: %d GO terms tested", kinase, len(enrich))

        hit_min = config.GO_PANEL_HIT_MIN.get(kinase)
        if hit_min is None:
            log.info("  no display filter configured for %s; not plotted", kinase)
            continue
        shown = enrich.loc[
            (enrich["bg_with_GO"] > config.GO_PANEL_BG_MIN)
            & (enrich["bg_with_GO"] < config.GO_PANEL_BG_MAX)
            & (enrich["hit_with_GO"] > hit_min)
            & (enrich["p_value"] < config.GO_PANEL_P_MAX), :
        ].groupby("GO_type").head(config.GO_PANEL_TOP_N).sort_values(
            ["GO_type", "p_value"], ascending=True
        ).copy()
        short = kinase.replace("_HUMAN", "")
        shown["kinase"] = short
        plot_go_enrichment(shown, fig_dir / f"fig06B_go_{short.lower()}.png", short)
        panels.append(shown)

    if panels:
        combined = pd.concat(panels)
        combined.to_csv(fig_dir / "fig06B_go_results.csv")
        log.info("wrote fig06B_go_results.csv (%d rows)", len(combined))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--panel", default="all",
                    choices=["a-trees", "a-cross", "b-motifs", "b-go", "all"],
                    help="which panel or sub-panel to build (default: all)")
    ap.add_argument("--motif-kinase-set", default="dark", choices=["dark", "stlib"],
                    help="dark: the four understudied kinases (12_dark_kinase_analysis); "
                         "stlib: the wider S/T panel with the looser filters of "
                         "11_ksi_motif_analysis-stlib")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("plotnine").setLevel(logging.WARNING)

    fig_dir = config.FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.panel in ("a-trees", "all"):
        log.info("=== panel A: trees ===")
        panel_a_trees(fig_dir)
    if args.panel in ("a-cross", "all"):
        log.info("=== panel A: cross-tree distances ===")
        panel_a_cross_tree(fig_dir)
    if args.panel in ("b-motifs", "all"):
        log.info("=== panel B: information content ===")
        panel_b_motifs(fig_dir, args.motif_kinase_set)
    if args.panel in ("b-go", "all"):
        log.info("=== panel B: GO enrichment ===")
        panel_b_go(fig_dir)


if __name__ == "__main__":
    main()
