"""Generate every figure in the manuscript from the saved experiment outputs.

Nothing here recomputes a result; each panel reads results/*.json or *.csv.
Figures are greyscale-safe (World Scientific accepts black and white only for
photographs, and greyscale reproduces reliably for line art).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT, RESULTS                                   # noqa: E402

FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "grid.linewidth": 0.4,
    "grid.color": "0.85",
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

W = 5.0          # text width in inches for IJTAF


def _load(name):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _save(fig, name):
    out = FIGS / f"{name}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------


def fig_simulator():
    d = _load("exp01_marginal_selfsim")
    if not d:
        return
    fig, ax = plt.subplots(1, 2, figsize=(W, 1.9))
    c = pd.DataFrame(d["convergence"])
    ax[0].axhspan(-2, 2, color="0.9", zorder=0)
    ax[0].axhline(0, color="0.4", lw=0.6)
    ax[0].plot(c["n_jumps"], c["bias_in_se"], "o-", color="k", ms=3.2)
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"series length $N$")
    ax[0].set_ylabel(r"bias in Monte Carlo s.e.")
    ax[0].set_ylim(-4, 4)
    ax[0].grid(True, alpha=0.6)
    ax[0].set_title("(a) truncation bias", loc="left")

    r = pd.DataFrame(d["runs"])
    marks = {0.3: "o", 0.4: "s", 0.5: "^"}
    for a, g in r.groupby("alpha"):
        ax[1].plot(g["t"], g["atom_prob"], marks.get(a, "o") + "-", color="k",
                   ms=3.2, mfc="w" if a != 0.5 else "k",
                   label=rf"$\alpha={a}$")
        ax[1].fill_between(g["t"], g["atom_prob"] - 2 * g["atom_se"],
                           g["atom_prob"] + 2 * g["atom_se"], color="0.85", lw=0)
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"horizon $t$ (64-fold range)")
    ax[1].set_ylabel(r"$\Pr[L_1(t)=L_2(t)]$")
    ax[1].grid(True, alpha=0.6)
    ax[1].legend(frameon=False, loc="best")
    ax[1].set_title("(b) scale-freeness", loc="left")
    fig.tight_layout(pad=0.3)
    _save(fig, "fig_simulator")


def fig_atom():
    d = _load("exp02_atom")
    if not d:
        return
    s = pd.DataFrame(d["loading_sweep"])
    fig, ax = plt.subplots(1, 2, figsize=(W, 1.9))
    ax[0].errorbar(s["a"], s["mc"], yerr=2 * s["mc_se"], fmt="o", color="k",
                   ms=3.4, capsize=2, lw=0.8, label="exact simulation")
    ax[0].plot(s["a"], s["theory"], "-", color="0.45", lw=1.4,
               label="compensation formula")
    ax[0].set_xlabel(r"common-shock loading $a$")
    ax[0].set_ylabel(r"$\pi_{12}=\Pr[L_1=L_2]$")
    ax[0].grid(True, alpha=0.6)
    ax[0].legend(frameon=False, loc="lower right")
    ax[0].set_title("(a) atom vs loading", loc="left")

    z = s[s["a"] > 0]
    ax[1].axhspan(-2, 2, color="0.9", zorder=0)
    ax[1].axhline(0, color="0.4", lw=0.6)
    ax[1].plot(z["a"], z["z"], "o", color="k", ms=3.4)
    ax[1].set_xlabel(r"common-shock loading $a$")
    ax[1].set_ylabel("(simulation $-$ theory) / s.e.")
    ax[1].set_ylim(-4, 4)
    ax[1].grid(True, alpha=0.6)
    ax[1].set_title("(b) residuals", loc="left")
    fig.tight_layout(pad=0.3)
    _save(fig, "fig_atom")


def fig_potential():
    d = _load("exp03_potential")
    if not d:
        return
    fig, ax = plt.subplots(1, 2, figsize=(W, 1.9))
    ls = d["log_singularity"]
    ax[0].plot(ls["deltas"], ls["values"], "o-", color="k", ms=3.2)
    ax[0].set_xscale("log")
    ax[0].invert_xaxis()
    ax[0].set_xlabel(r"diagonal offset $\delta$")
    ax[0].set_ylabel(r"$u(1,1+\delta)$")
    ax[0].grid(True, alpha=0.6)
    ax[0].set_title("(a) logarithmic ridge", loc="left")

    cells = pd.DataFrame(d["cells"])
    off = cells[~cells["diagonal"]]["ratio_box"].dropna()
    db = cells[cells["diagonal"]]["ratio_box"].dropna()
    ds = cells[cells["diagonal"]]["ratio_split"].dropna()
    ax[1].axhline(1.0, color="0.4", lw=0.8)
    parts = [off.values, db.values, ds.values]
    bp = ax[1].boxplot(parts, widths=0.55, patch_artist=True,
                       tick_labels=["off-diag", "diag\n(box)", "diag\n(split)"])
    for b in bp["boxes"]:
        b.set(facecolor="0.88", edgecolor="k", linewidth=0.7)
    for k in ("whiskers", "caps", "medians"):
        for b in bp[k]:
            b.set(color="k", linewidth=0.7)
    ax[1].set_ylabel("closed form / simulation")
    ax[1].grid(True, axis="y", alpha=0.6)
    ax[1].set_title("(b) cell-by-cell agreement", loc="left")
    fig.tight_layout(pad=0.3)
    _save(fig, "fig_potential")


def fig_correlation():
    d = _load("exp05_correlation")
    if not d:
        return
    sh = d["sweep"]["shares"]
    order = ["rising", "falling", "hump", "dip", "flat", "other"]
    labels = [k for k in order if k in sh]
    vals = [sh[k] for k in labels]
    fig, ax = plt.subplots(figsize=(W * 0.62, 1.85))
    bars = ax.bar(range(len(labels)), vals, color="0.75", edgecolor="k", lw=0.7)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("share of parameter cells (%)")
    ax.set_ylim(0, max(vals) * 1.22)
    ax.grid(True, axis="y", alpha=0.6)
    ax.set_title(f"term-structure shape, {d['sweep']['n_combinations']:,} cells",
                 loc="left")
    fig.tight_layout(pad=0.3)
    _save(fig, "fig_correlation")


# ---------------------------------------------------------------------------
# Empirical figures
# ---------------------------------------------------------------------------


def fig_event_definition():
    p = RESULTS / "crypto_durations.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    order = ["fill", "order", "W0.1", "W1", "W5"]
    labs = ["fills", "orders", "0.1 s", "1 s", "5 s"]
    data = [df[df.series == s]["hill_plateau"].dropna().values for s in order]
    fig, ax = plt.subplots(figsize=(W * 0.66, 2.0))
    bp = ax.boxplot(data, widths=0.6, patch_artist=True, tick_labels=labs)
    for b in bp["boxes"]:
        b.set(facecolor="0.88", edgecolor="k", linewidth=0.7)
    for k in ("whiskers", "caps", "medians", "fliers"):
        for b in bp[k]:
            b.set(color="k", linewidth=0.7)
            if k == "fliers":
                b.set(marker="o", markersize=2, markerfacecolor="none")
    ax.axhline(1.0, color="k", ls="--", lw=0.9)
    ax.set_yscale("log")
    ax.text(0.03, 0.96, r"dashed line: $\alpha=1$; the model requires $\hat\alpha<1$",
            transform=ax.transAxes, fontsize=6.5, va="top")
    ax.set_ylabel(r"Hill tail index $\hat\alpha$")
    ax.set_xlabel("event definition (merge window)")
    ax.grid(True, axis="y", alpha=0.6)
    fig.tight_layout(pad=0.3)
    _save(fig, "fig_event_definition")


def fig_cross_section():
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.0))
    for k, (name, series, title) in enumerate([
            ("crypto_durations.csv", "order", "(a) Binance, order-level"),
            ("hose_durations.csv", "second", "(b) HOSE, second-level")]):
        p = RESULTS / name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        s = df[(df.series == series) & df.hill_plateau.notna()]
        ax[k].plot(s["median_gap_s"], s["hill_plateau"], "o", color="k", ms=3.2,
                   mfc="none")
        ax[k].axhline(1.0, color="k", ls="--", lw=0.9)
        x = np.log10(s["median_gap_s"].values)
        y = s["hill_plateau"].values
        if len(x) > 3:
            b, a0 = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 20)
            ax[k].plot(10 ** xx, a0 + b * xx, "-", color="0.45", lw=1.2)
            r = np.corrcoef(x, y)[0, 1]
            ax[k].text(0.04, 0.93, rf"$r={r:+.3f}$", transform=ax[k].transAxes,
                       fontsize=7, va="top")
        ax[k].set_xscale("log")
        ax[k].set_xlabel("median inter-trade gap (s)")
        ax[k].set_ylabel(r"Hill tail index $\hat\alpha$")
        ax[k].grid(True, alpha=0.6)
        ax[k].set_title(title, loc="left")
    fig.tight_layout(pad=0.3)
    _save(fig, "fig_cross_section")


def fig_deseason():
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.0))
    for k, (name, base, title) in enumerate([
            ("crypto_durations.csv", "order", "(a) Binance"),
            ("hose_durations.csv", "second", "(b) HOSE")]):
        p = RESULTS / name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        piv = df.pivot_table(index="symbol", columns="series",
                             values="hill_plateau")
        if base not in piv or "deseas" not in piv:
            continue
        sub = piv[[base, "deseas"]].dropna()
        for _, row in sub.iterrows():
            ax[k].plot([0, 1], [row[base], row["deseas"]], "-",
                       color="0.6", lw=0.7, zorder=1)
        ax[k].plot(np.zeros(len(sub)), sub[base], "o", color="k", ms=3,
                   mfc="none", zorder=2)
        ax[k].plot(np.ones(len(sub)), sub["deseas"], "o", color="k", ms=3,
                   zorder=2)
        ax[k].axhline(1.0, color="k", ls="--", lw=0.9)
        ax[k].set_xticks([0, 1])
        ax[k].set_xticklabels(["raw", "diurnally\nadjusted"])
        ax[k].set_xlim(-0.35, 1.35)
        ax[k].set_ylabel(r"Hill tail index $\hat\alpha$")
        ax[k].grid(True, axis="y", alpha=0.6)
        ax[k].set_title(title, loc="left")
    fig.tight_layout(pad=0.3)
    _save(fig, "fig_deseason")


def main():
    print("generating figures ...")
    for f in (fig_simulator, fig_atom, fig_potential, fig_correlation,
              fig_event_definition, fig_cross_section, fig_deseason):
        try:
            f()
        except Exception as exc:                                   # noqa: BLE001
            print(f"  [skip] {f.__name__}: {type(exc).__name__}: {exc}")
    print("done")


if __name__ == "__main__":
    main()
