"""Fig 05 panel B - mock Boltzmann illustration (module).

Mock illustration: how abundance and activity free energies propagate through a
three-state Boltzmann partition into barcode log2FC (the quantity the decoupling
model in 5_enrichment_analysis.ipynb splits into g_k and b(s,k)).

All numbers are simulated for illustration only.

States per kinase (Boltzmann weights, relative to the active state):
    active on substrate      1
    folded, not productive   exp(dG_act / RT)
    unfolded                 exp((dG_act + dG_abund) / RT)
Positive dG = unfavorable.

Picks: 3 kinases x 3 substrates. S1 and S2 are chosen so that K2 (unstable) on the
good substrate S1 and K1 (stable) on the weaker substrate S2 land on exactly the
same readout, log2FC = 0 - the same barcode count from two different ΔG mixes.

Imported by fig05_math.py; also runnable on its own. The mock screen is drawn
at import time from a fixed seed, so the figure is reproducible.

Produces  <fig dir>/mock_boltzmann_to_logfc.{png,pdf}
"""
import argparse
import logging
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from matplotlib.lines import Line2D

log = logging.getLogger("fig05B")

rng = np.random.default_rng(7)
RT = (273 + 30) * 0.001987  # kcal/mol, yeast at 30 C

# ---------------- mock screen ----------------
n_kin, n_sub = 500, 60
dG_abund = rng.normal(-0.8, 1.0, n_kin)                 # per kinase
dG_act = rng.normal(0.6, 1.2, (n_kin, n_sub))           # per kinase x substrate


def partition(dg_ab, dg_ac):
    w_active = 1.0
    w_folded = np.exp(dg_ac / RT)
    w_unfold = np.exp((dg_ac + dg_ab) / RT)
    Z = w_active + w_folded + w_unfold
    return w_unfold / Z, w_folded / Z, w_active / Z     # P_U, P_F, P_active


def p_fold(dg_ab):
    return 1.0 / (1.0 + np.exp(dg_ab / RT))


def act_for(p_target, dg_ab):
    """Activity energy that puts a kinase of given stability at a target active fraction."""
    return RT * np.log((1.0 / p_target - 1.0) / (1.0 + np.exp(dg_ab / RT)))


# growth-based readout: log2FC linear in the active fraction (as in MoCHI's linear link),
# with the intercept set so the library median log2FC = 0 (what size factors do)
W_LFC, NOISE = 5.0, 0.25
_, _, P_all = partition(dG_abund[:, None], dG_act)
P_ZERO = float(np.median(P_all))                        # active fraction that reads out as log2FC = 0
C_LFC = -W_LFC * P_ZERO


def to_lfc(p, noise=True):
    eps = rng.normal(0, NOISE, np.shape(p)) if noise else 0.0
    return W_LFC * p + C_LFC + eps


lfc_all = to_lfc(P_all)
X_all = np.repeat(dG_abund[:, None], n_sub, axis=1)     # abundance coordinate of every pair

# ---------------- picks: 3 kinases x 3 substrates ----------------
KIN = {"K1": -2.2, "K2": 1.0, "K3": -0.6}               # stable / unstable / middling
SUB = {"S1": act_for(P_ZERO, KIN["K2"]),                # good: carries unstable K2 to log2FC 0
       "S2": act_for(P_ZERO, KIN["K1"]),                # weaker: carries stable K1 to log2FC 0
       "S3": act_for(P_ZERO, KIN["K1"]) + 1.4}          # poor
KNAME = {"K1": "K1 stable", "K2": "K2 unstable", "K3": "K3 middling"}
SNAME = {"S1": "S1 good", "S2": "S2 weaker", "S3": "S3 poor"}
SMARK = {"S1": "o", "S2": "s", "S3": "^"}
KFACE = {"K1": "#0b0b0b", "K2": "white", "K3": "#8a8984"}
ZERO_PAIR = [("K1", "S2"), ("K2", "S1")]                # the two that read out identically

pairs = []
for k, gab in KIN.items():
    for s, gac in SUB.items():
        pu, pf, pa = partition(gab, gac)
        pairs.append(dict(k=k, s=s, gab=gab, gac=gac, pu=pu, pf=pf, pa=pa,
                          lfc=to_lfc(pa, noise=False)))
P = {(p["k"], p["s"]): p for p in pairs}

# ---------------- plane and fields ----------------
XLIM, YLIM = (-3.4, 2.2), (-2.4, 4.2)
NG = 400
gx, gy = np.meshgrid(np.linspace(*XLIM, NG), np.linspace(*YLIM, NG))
P_grid = partition(gx, gy)[2]
LFC_grid = to_lfc(P_grid, noise=False)

# ---------------- style ----------------
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
HIST = "#cfcec9"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a3282a"])
STROKE = [pe.withStroke(linewidth=3, foreground="white")]
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False,
})


def mark(ax, x, y, p, s=85):
    """Kinase = fill, substrate = shape."""
    ax.scatter(x, y, s=s, marker=SMARK[p["s"]], facecolor=KFACE[p["k"]],
               edgecolor="white" if p["k"] == "K1" else INK,
               linewidth=1.6 if p["k"] == "K1" else 1.8, zorder=6, clip_on=False)


def plane_axes(ax):
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xlabel("ΔG_abundance (kcal/mol)   ← more stable")
    ax.set_ylabel("ΔG_activity (kcal/mol)   ← more productive")


def pick_lattice(ax, key, fmt, label_axes=True):
    """3x3 lattice of picks; each node labelled with its value."""
    for k, xv in KIN.items():
        ax.plot([xv, xv], [min(SUB.values()), max(SUB.values())], color=INK, lw=1.0, ls="--", zorder=5)
        if label_axes:
            ax.text(xv, YLIM[1] - 0.12, k, ha="center", va="top", fontsize=10, weight="bold",
                    color=INK, path_effects=STROKE, zorder=7)
    for s, yv in SUB.items():
        ax.plot([min(KIN.values()), max(KIN.values())], [yv, yv], color=INK, lw=1.0, ls="--", zorder=5)
        if label_axes:
            ax.text(XLIM[1] - 0.08, yv, s, ha="right", va="bottom", fontsize=10, weight="bold",
                    color=INK, path_effects=STROKE, zorder=7)
    for p in pairs:
        mark(ax, p["gab"], p["gac"], p)
        ax.annotate(fmt.format(round(p[key], 2) + 0.0), (p["gab"], p["gac"]), xytext=(9, -4), textcoords="offset points",
                    fontsize=9, color=INK, zorder=7, path_effects=STROKE)


def equal_readout_callout(ax, note):
    """Connect the two picks that give the same readout from different ΔG mixes."""
    (k1, s1), (k2, s2) = ZERO_PAIR
    p, q = P[(k1, s1)], P[(k2, s2)]
    ax.annotate("", xy=(q["gab"], q["gac"]), xytext=(p["gab"], p["gac"]),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.8,
                                shrinkA=9, shrinkB=9, connectionstyle="arc3,rad=-0.25"), zorder=7)
    ax.text((p["gab"] + q["gab"]) / 2, max(p["gac"], q["gac"]) + 0.75, note, ha="center", va="bottom",
            fontsize=9.5, weight="bold", color=INK, path_effects=STROKE, zorder=8)


def value_scale(ax, cmap, norm, values, key, label, ticks, lim):
    """Colour scale beside a field panel, with the screen-wide distribution drawn along it."""
    lo, hi = lim
    cax = ax.inset_axes([1.03, 0, 0.045, 1])
    hax = ax.inset_axes([1.08, 0, 0.26, 1], sharey=cax)
    cax.imshow(np.linspace(lo, hi, 256)[:, None], aspect="auto", origin="lower",
               extent=(0, 1, lo, hi), cmap=cmap, norm=norm)
    cax.set_xticks([])
    cax.tick_params(axis="y", left=False, labelleft=False)
    for sp in cax.spines.values():
        sp.set_visible(False)
    hax.hist(np.ravel(values), bins=np.linspace(lo, hi, 41), orientation="horizontal",
             color=HIST, edgecolor="white", linewidth=0.8)
    hax.set_ylim(lo, hi)
    hax.yaxis.tick_right()
    hax.set_yticks(ticks)
    hax.tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True)
    hax.spines["left"].set_visible(False)
    hax.spines["right"].set_visible(True)
    hax.set_xticks([])
    hax.set_xlabel("all pairs", fontsize=8.5)
    hax.set_ylabel(label, rotation=270, labelpad=16)
    hax.yaxis.set_label_position("right")
    xmax = hax.get_xlim()[1]
    slot = {}
    for p in pairs:
        cax.axhline(p[key], color=INK, lw=1.0)
        hax.axhline(p[key], color=INK, lw=0.7, ls=":")
        i = slot[round(p[key], 2)] = slot.get(round(p[key], 2), -1) + 1   # stagger ties
        mark(hax, xmax * (0.45 + 0.22 * i + 0.04 * list(KIN).index(p["k"])), p[key], p, s=45)



def build_panel(out_dir):
    """Draw the three-step mock panel and write it to out_dir."""
    fig = plt.figure(figsize=(21, 8.8))
    gs = fig.add_gridspec(1, 3, wspace=0.62, left=0.045, right=0.955, top=0.79, bottom=0.36)
    a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))

    # a. joint distribution of the two free energies
    a.scatter(X_all.ravel(), dG_act.ravel(), s=4, color=MUTED, alpha=0.18, edgecolor="none", zorder=1)
    plane_axes(a)
    top = a.inset_axes([0, 1.03, 1, 0.22], sharex=a)
    right = a.inset_axes([1.03, 0, 0.22, 1], sharey=a)
    top.hist(dG_abund, bins=np.linspace(*XLIM, 30), color=HIST, edgecolor="white")
    right.hist(dG_act.ravel(), bins=np.linspace(*YLIM, 35), orientation="horizontal", color=HIST, edgecolor="white")
    top.tick_params(labelbottom=False, labelsize=8)
    right.tick_params(labelleft=False, labelsize=8)
    top.set_ylabel("kinases", fontsize=8.5)
    right.set_xlabel("pairs", fontsize=8.5)
    for k, v in KIN.items():
        top.axvline(v, color=INK, lw=1.5)
        top.text(v, top.get_ylim()[1], f" {k}\n P_fold={p_fold(v):.2f}", fontsize=8.5, va="top", color=INK2)
    for s, v in SUB.items():
        right.axhline(v, color=INK, lw=1.5)
        right.text(right.get_xlim()[1], v, f" {s}", fontsize=8.5, va="center", color=INK2)
    pick_lattice(a, "gac", "", label_axes=True)
    top.set_title("a  Joint distribution: ΔG abundance × ΔG activity", loc="left", pad=8)

    # b. Boltzmann partition: P_active over the plane
    norm_p = Normalize(0, 1)
    b.contourf(gx, gy, P_grid, levels=np.linspace(0, 1, 41), cmap=SEQ, norm=norm_p, zorder=1)
    cs = b.contour(gx, gy, P_grid, levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors="white", linewidths=0.8, zorder=3)
    b.clabel(cs, fmt="%.1f", fontsize=8.5, colors=INK)
    plane_axes(b)
    pick_lattice(b, "pa", "{:.2f}")
    equal_readout_callout(b, f"same P_active = {P_ZERO:.2f}")
    value_scale(b, SEQ, norm_p, P_all, "pa", "P_active", [0, 0.25, 0.5, 0.75, 1], (0, 1))
    b.set_title("b  Boltzmann partition → fraction of kinase active", loc="left", pad=8)

    # c. growth selection: expected log2FC over the plane
    lfc_lo, lfc_hi = -1.5, 4.5
    norm_l = TwoSlopeNorm(vmin=-lfc_hi, vcenter=0, vmax=lfc_hi)  # symmetric colour, so small depletion stays pale
    c.contourf(gx, gy, LFC_grid, levels=np.linspace(lfc_lo, lfc_hi, 49), cmap=DIV, norm=norm_l, zorder=1)
    cs = c.contour(gx, gy, LFC_grid, levels=[0, 1, 2, 3, 4], colors=INK2,
                   linewidths=[1.8, 0.6, 0.6, 0.6, 0.6], zorder=3)
    c.clabel(cs, fmt="%+d", fontsize=8.5, colors=INK)
    plane_axes(c)
    pick_lattice(c, "lfc", "{:+.2f}")
    equal_readout_callout(c, "same log2FC = 0, different ΔG mix")
    value_scale(c, DIV, norm_l, lfc_all, "lfc", "barcode log2FC (MTX vs pre)", [-1, 0, 1, 2, 3, 4], (lfc_lo, lfc_hi))
    c.set_title("c  Growth selection → barcode log2FC", loc="left", pad=8)

    # ---------------- descriptions ----------------
    DESC = {
        a: ("Two free energies set each pair's position. ΔG abundance is a property of the kinase (x); "
            "ΔG activity is a property of the kinase–substrate pair (y). Grey dots are all mock pairs and "
            "the marginals show each axis alone. The lattice marks three kinases (K1 stable, K2 unstable, "
            "K3 middling) crossed with three substrates (S1 good, S2 weaker, S3 poor)."),
        b: ("Boltzmann weighting over unfolded, folded-inactive and active states gives the fraction of "
            "kinase acting on the substrate: P_active = 1 / (1 + e^(ΔG_act/RT)·(1 + e^(ΔG_abund/RT))). "
            "Contours bend because activity needs folded kinase, so the two axes trade off: stable K1 on "
            f"the weaker S2 and unstable K2 on the good S1 reach the same P_active ({P_ZERO:.2f})."),
        c: ("More active complex → faster growth in MTX → more barcode reads after selection "
            "(log2FC ≈ w·P_active + c; 0 = library median). Because the readout is one number per pair, "
            "K1·S2 and K2·S1 are indistinguishable at log2FC 0. Separating the abundance part (g_k) from "
            "the substrate part (b(s,k)) is what the decoupling model is for."),
    }
    for ax, text in DESC.items():
        fig.text(ax.get_position().x0, 0.25, textwrap.fill(text, 70), ha="left", va="top",
                 fontsize=10, color=INK2, linespacing=1.45)

    fig.legend(
        handles=[Line2D([], [], lw=0, marker="o", markersize=9, markerfacecolor=KFACE[k],
                        markeredgecolor="white" if k == "K1" else INK, label=f"{KNAME[k]} = {f}")
                 for k, f in zip(KIN, ["black fill", "open", "grey fill"])]
        + [Line2D([], [], lw=0, marker=SMARK[s], markersize=9, markerfacecolor=MUTED, markeredgecolor="white",
                  label=f"{SNAME[s]} = {n}") for s, n in zip(SUB, ["circle", "square", "triangle"])],
        loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.04), fontsize=9.5)
    fig.suptitle("Mock example: ΔG abundance × ΔG activity  →  Boltzmann partition  →  barcode log2FC",
                 x=0.01, ha="left", y=0.995, fontsize=14, weight="bold", color=INK)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"mock_boltzmann_to_logfc.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s/mock_boltzmann_to_logfc.{png,pdf}", out_dir)

    # The numbers quoted on the panel.
    log.info("log2FC = 0 at P_active = %.3f", P_ZERO)
    for k, v in KIN.items():
        log.info("%s: dG_abund=%+.2f  P_fold=%.2f", k, v, p_fold(v))
    for s, v in SUB.items():
        log.info("%s: dG_act=%+.2f", s, v)
    for p in pairs:
        log.info("%s-%s: P_U=%.2f P_F=%.2f P_active=%.2f log2FC=%+.2f",
                 p["k"], p["s"], p["pu"], p["pf"], p["pa"], p["lfc"])
    return out_dir / "mock_boltzmann_to_logfc.pdf"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: config.FIG_DIR)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out_dir = args.out_dir
    if out_dir is None:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        import config
        out_dir = config.FIG_DIR
    build_panel(out_dir)


if __name__ == "__main__":
    main()
