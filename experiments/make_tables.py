"""Emit LaTeX macros and tables from the saved experiment outputs.

Writes paper/results_macros.tex (numbers) and paper/tab_*.tex (tables), which
main.tex inputs.  No numeric result is ever typed into the manuscript by hand:
re-running the experiments and this script regenerates every figure in the text.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, RESULTS                                   # noqa: E402

PAPER = ROOT / "paper"
PAPER.mkdir(exist_ok=True)
MACROS: dict[str, str] = {}


def mac(name, value, fmt="{}"):
    MACROS[name] = fmt.format(value)


def load(name):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def esc(s):
    return str(s).replace("_", r"\_").replace("&", r"\&")


def si(x, nd=0):
    """Thousands-separated integer for LaTeX."""
    return f"{int(round(x)):,}".replace(",", r"{,}")


def sci(x, sig=1):
    """Render a small number as LaTeX math, e.g. 6.7e-16 -> $6.7\\times10^{-16}$."""
    x = float(x)
    if x == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10.0 ** e
    mant = f"{m:.{sig}f}".rstrip("0").rstrip(".")
    if e == 0:
        return f"${mant}$"
    return rf"${mant}\times 10^{{{e}}}$"


# ---------------------------------------------------------------------------
# Theory
# ---------------------------------------------------------------------------


def do_exp01():
    d = load("exp01_marginal_selfsim")
    if not d:
        return
    c = d["config"]
    mac("simPaths", si(c["n_paths"]))
    mac("simJumps", si(c["n_jumps"]))
    mac("simTratio", f"{c['t_grid'][-1] / c['t_grid'][0]:.0f}")

    conv = pd.DataFrame(d["convergence"])
    mac("convMaxBiasSE", f"{conv['bias_in_se'].abs().max():.1f}")
    mac("convBoundAtChosen",
        sci(float(conv.loc[conv.n_jumps == c['n_jumps'], 'analytic_bound'].iloc[0])))
    mac("convMinKSp", f"{conv['ks_p'].min():.2f}")

    r = pd.DataFrame(d["runs"])
    mac("marginalMinKSp", f"{r[['ks_p_1', 'ks_p_2']].values.min():.3f}")
    mac("marginalMaxMeanErr",
        f"{np.abs(r['mean_1'] - r['mean_exact']).max():.4f}")
    mac("nMarginalTests", f"{2 * len(r)}")

    rows = []
    for a, g in r.groupby("alpha"):
        rows.append(dict(alpha=a,
                         corr_lo=g["corr"].min(), corr_hi=g["corr"].max(),
                         atom_lo=g["atom_prob"].min(), atom_hi=g["atom_prob"].max(),
                         se=g["atom_se"].mean()))
    ss = pd.DataFrame(d["self_similarity"])
    mac("selfsimMinProjP", f"{ss['proj_p_min'].min():.3f}")

    lines = [r"\begin{tabular}{@{}cccccc@{}}", r"\toprule",
             r"$\alpha$ & \multicolumn{2}{c}{$\mathrm{corr}(H_1/t,H_2/t)$} & "
             r"\multicolumn{2}{c}{$\Pr[L_1(t)=L_2(t)]$} & two-sample $p$ \\",
             r" & min & max & min & max & (joint) \\", r"\colrule"]
    for row, sp in zip(rows, ss.itertuples()):
        lines.append(f"{row['alpha']:.1f} & {row['corr_lo']:.4f} & {row['corr_hi']:.4f} & "
                     f"{row['atom_lo']:.4f} & {row['atom_hi']:.4f} & {sp.proj_p_min:.3f} \\\\")
    lines += [r"\botrule", r"\end{tabular}"]
    (PAPER / "tab_selfsim.tex").write_text("\n".join(lines))


def do_exp02():
    d = load("exp02_atom")
    if not d:
        return
    s = pd.DataFrame(d["loading_sweep"])
    mac("atomNPaths", si(d["config"]["n_paths"]))
    nz = s[s["a"] > 0]
    mac("atomMaxAbsZ", f"{nz['z'].abs().max():.2f}")
    mac("atomAtOne", f"{float(s.loc[s.a == 1, 'mc'].iloc[0]):.5f}")
    mac("atomAtOneSE", f"{float(s.loc[s.a == 1, 'mc_se'].iloc[0]):.5f}")
    mac("atomAtOneTheory", f"{float(s.loc[s.a == 1, 'theory'].iloc[0]):.5f}")
    h = d["horizon_sweep"]
    mac("atomHorizSpread", f"{h['spread']:.4f}")
    mac("atomHorizPooled", f"{h['pooled']:.4f}")
    mac("atomHorizChisq", f"{h['chi2']:.2f}")
    mac("atomHorizDf", f"{h['df']}")
    mac("atomHorizRatio", f"{h['ratio']:.0f}")
    q = pd.DataFrame(d["quadrature_convergence"])
    mac("atomQuadFinest", f"{q['pi'].iloc[-1]:.5f}")
    mac("atomQuadDrift", f"{abs(q['pi'].iloc[-1] - q['pi'].iloc[-2]):.5f}")

    lines = [r"\begin{tabular}{@{}cccccr@{}}", r"\toprule",
             r"$a$ & simulation & s.e. & Eq.~(\ref{eq:atomformula}) & difference & $z$ \\",
             r"\colrule"]
    for t in s.itertuples():
        z = "---" if t.a == 0 else f"{t.z:+.2f}"
        lines.append(f"{t.a:g} & {t.mc:.5f} & {t.mc_se:.5f} & {t.theory:.5f} & "
                     f"{t.diff:+.5f} & {z} \\\\")
    lines += [r"\botrule", r"\end{tabular}"]
    (PAPER / "tab_atom.tex").write_text("\n".join(lines))


def do_exp03():
    d = load("exp03_potential")
    if not d:
        return
    s = d["cell_summary"]
    mac("potOffMedian", f"{s['off_diagonal']['median']:.3f}")
    mac("potOffPlo", f"{s['off_diagonal']['p05']:.3f}")
    mac("potOffPhi", f"{s['off_diagonal']['p95']:.3f}")
    mac("potOffMAD", f"{100 * s['off_diagonal']['mean_abs_dev']:.1f}")
    mac("potDiagBoxMedian", f"{s['diagonal_box']['median']:.3f}")
    mac("potDiagBoxLo", f"{s['diagonal_box']['lo']:.3f}")
    mac("potDiagBoxHi", f"{s['diagonal_box']['hi']:.3f}")
    mac("potDiagSplitMedian", f"{s['diagonal_split']['median']:.3f}")
    mac("potDiagSplitLo", f"{s['diagonal_split']['lo']:.3f}")
    mac("potDiagSplitHi", f"{s['diagonal_split']['hi']:.3f}")
    mac("potNCells", f"{len(d['cells'])}")
    mac("potHomogMaxErr", sci(max(h['rel_err'] for h in d['homogeneity'])))
    ls = d["log_singularity"]
    mac("logSingIncrMean", f"{np.mean(ls['increments']):.4f}")
    mac("logSingIncrSd", f"{np.std(ls['increments']):.4f}")
    mac("logSingNDecades", f"{len(ls['increments'])}")
    mac("logSingFirst", f"{ls['values'][0]:.4f}")
    mac("logSingLast", f"{ls['values'][-1]:.4f}")


def do_exp04():
    d = load("exp04_evolution")
    if not d:
        return
    sb = d["size_bias_constant"]
    mac("sizeBiasZ", f"{max(abs(sb['z1']), abs(sb['z2'])):.2f}")
    mac("sizeBiasMean", f"{sb['mean_s1']:.5f}")
    k = pd.DataFrame(d["kummer"])
    mac("kummerMaxRel", sci(k['rel_hyp_vs_quad'].max()))
    mac("kummerMaxZ", f"{k['z'].abs().max():.2f}")
    s = d["summary"]
    mac("localRatioLo", f"{s['local_ratio_min']:.4f}")
    mac("localRatioHi", f"{s['local_ratio_max']:.4f}")
    mac("localMaxZ", f"{s['max_local_z']:.2f}")
    mac("tensorRatioLo", f"{s['tensor_ratio_min']:.3f}")
    mac("tensorRatioHi", f"{s['tensor_ratio_max']:.3f}")

    e = pd.DataFrame(d["evolution"])
    lines = [r"\begin{tabular}{@{}ccrrcrc@{}}", r"\toprule",
             r"$\kappa_1$ & $\kappa_2$ & $M'(t)$ & size-biased & ratio & "
             r"tensorized & ratio \\", r"\colrule"]
    for t in e.itertuples():
        lines.append(f"{t.kappa1:g} & {t.kappa2:g} & {t.lhs:+.5f} & {t.rhs:+.5f} & "
                     f"{t.local_ratio:.4f} & {t.rhs_tensor:+.5f} & "
                     f"{t.tensor_ratio:.3f} \\\\")
    lines += [r"\botrule", r"\end{tabular}"]
    (PAPER / "tab_evolution.tex").write_text("\n".join(lines))


def do_exp05():
    d = load("exp05_correlation")
    if not d:
        return
    sw = d["sweep"]
    mac("sweepN", si(sw["n_combinations"]))
    for k, v in sw["shares"].items():
        mac("share" + k.capitalize(), f"{v:.1f}")
    mac("slopeRuleAcc", f"{sw['slope_rule_accuracy_pct']:.1f}")
    mac("sweepNdeltas", f"{len(d['config']['deltas'])}")
    mac("sweepDeltaLo", f"{min(d['config']['deltas']):g}")
    mac("sweepDeltaHi", f"{max(d['config']['deltas']):g}")

    lines = [r"\begin{tabular}{@{}lr@{}}", r"\toprule",
             r"Term-structure shape & Share of cells (\%) \\", r"\colrule"]
    for k in ("rising", "falling", "hump", "dip", "flat", "other"):
        if k in sw["shares"]:
            lines.append(f"{k.capitalize()} & {sw['shares'][k]:.1f} \\\\")
    lines += [r"\botrule", r"\end{tabular}"]
    (PAPER / "tab_sweep.tex").write_text("\n".join(lines))


def do_exp06():
    d = load("exp06_rho_exact")
    if not d:
        return
    cm = pd.DataFrame(d["clock_mean"])
    mac("clockMeanMaxZ", f"{cm['z'].abs().max():.2f}")
    b = d["brownian_summary"]
    mac("rhoBrownMaxZ", f"{b['max_abs_z']:.2f}")
    mac("rhoBrownSpread", f"{b['max_spread_over_delta']:.5f}")
    c = d["config"]
    mac("rhoDeltaRatio", f"{max(c['deltas']) / min(c['deltas']):.0f}")
    mac("rhoNCases", f"{len(set(r['case'] for r in d['brownian']))}")
    o = d["ou_summary"]
    mac("rhoLinSmallKt", f"{o['min_ratio_small_kt']:.3f}")
    mac("rhoLinLargeKt", f"{o['max_ratio_large_kt']:.3f}")
    mac("rhoKtSmall", f"{o['kt_small']:g}")
    mac("rhoKtLarge", f"{o['kt_large']:g}")

    br = pd.DataFrame(d["brownian"])
    lines = [r"\begin{tabular}{@{}crrrr@{}}", r"\toprule",
             r"Case & $\Delta$ & simulated $\rho$ & s.e. & Eq.~(\ref{eq:rhoflat}) \\",
             r"\colrule"]
    for t in br.itertuples():
        lines.append(f"{t.case + 1} & {t.delta:g} & {t.empirical:.5f} & "
                     f"{t.se:.5f} & {t.closed_form:.5f} \\\\")
    lines += [r"\botrule", r"\end{tabular}"]
    (PAPER / "tab_rho.tex").write_text("\n".join(lines))

    lin = pd.DataFrame(d["ou_linearization"])
    piv = lin.pivot_table(index="kappa_t", columns="delta", values="ratio")
    cols = list(piv.columns)
    lines = [r"\begin{tabular}{@{}c" + "r" * len(cols) + r"@{}}", r"\toprule",
             r"$\kappa t$ & " +
             " & ".join(rf"$\Delta={c:g}$" for c in cols) + r" \\", r"\colrule"]
    for kt, row in piv.iterrows():
        lines.append(f"{kt:g} & " + " & ".join(f"{row[c]:.3f}" for c in cols) + r" \\")
    lines += [r"\botrule", r"\end{tabular}"]
    (PAPER / "tab_oulin.tex").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Empirics
# ---------------------------------------------------------------------------


def _series_summary(df, series_list):
    out = []
    for s in series_list:
        v = df[df.series == s]["hill_plateau"].dropna()
        if len(v) == 0:
            continue
        out.append(dict(series=s, n=len(v), median=v.median(),
                        frac_below=float((v < 1).mean())))
    return pd.DataFrame(out)


def do_crypto():
    p = RESULTS / "crypto_durations.csv"
    d = load("exp10_crypto_durations")
    if not p.exists() or not d:
        return
    df = pd.read_csv(p)
    o = df[df.series == "order"]
    mac("cryptoNsym", f"{o['symbol'].nunique()}")
    mac("cryptoVolLo", si(o["qvol_24h"].min()))
    mac("cryptoVolHi", si(o["qvol_24h"].max()))
    mac("cryptoVolDecades",
        f"{np.log10(o['qvol_24h'].max() / o['qvol_24h'].min()):.0f}")
    mac("cryptoTotalOrders", si(o["n"].sum()))
    mac("cryptoTotalFills", si(df[df.series == "fill"]["n"].sum()))
    mac("cryptoFillRatioLo", f"{o['fills_per_order'].min():.2f}")
    mac("cryptoFillRatioHi", f"{o['fills_per_order'].max():.2f}")

    labels = {"fill": "fill", "order": "order", "W0.1": "Wa", "W1": "Wb",
              "W5": "Wc", "W30": "Wd", "deseas": "deseas"}
    summ = _series_summary(df, list(labels))
    for t in summ.itertuples():
        key = labels[t.series]
        mac(f"cryptoAlpha{key.capitalize()}Med", f"{t.median:.2f}")
        mac(f"cryptoAlpha{key.capitalize()}Frac", f"{100 * t.frac_below:.0f}")

    # the decisive count: plausible power law AND alpha < 1
    surv = {}
    for s in ("fill", "order", "deseas"):
        sub = df[(df.series == s) & df.csn_p.notna()]
        surv[s] = int(((sub.csn_alpha < 1) & (sub.csn_p >= 0.10)).sum())
        mac(f"crypto{s.capitalize()}Survivors", str(surv[s]))
        mac(f"crypto{s.capitalize()}Tested", str(len(sub)))
        mac(f"crypto{s.capitalize()}CsnMedian", f"{sub.csn_alpha.median():.2f}")
        mac(f"crypto{s.capitalize()}CsnMin", f"{sub.csn_alpha.min():.2f}")
    mac("cryptoSurvivorsTotal", str(sum(surv.values())))

    # is it the liquid pairs that burst?
    from scipy.stats import pearsonr
    lv = np.log10(o["qvol_24h"].values)
    pr = pearsonr(lv, o["fills_per_order"].values)
    mac("cryptoBurstVolCorr", f"{pr[0]:+.3f}")
    mac("cryptoBurstVolP", "<0.001" if pr[1] < 1e-3 else f"{pr[1]:.3f}")
    zf = df[df.series == "fill"].set_index("symbol")["zero_share"]
    zz = zf.reindex(o["symbol"]).values
    pz = pearsonr(lv, zz)
    mac("cryptoZeroVolCorr", f"{pz[0]:+.3f}")
    mac("cryptoZeroVolP", "<0.001" if pz[1] < 1e-3 else f"{pz[1]:.3f}")
    mac("cryptoZeroShareLo", f"{100 * np.nanmin(zz):.0f}")
    mac("cryptoZeroShareHi", f"{100 * np.nanmax(zz):.0f}")
    mac("cryptoZeroShareMed", f"{100 * np.nanmedian(zz):.0f}")

    ds = df[df.series == "deseas"]
    mac("cryptoDiurnalMed", f"{ds['diurnal_range'].median():.1f}")
    mac("cryptoDiurnalHi", f"{ds['diurnal_range'].max():.0f}")
    rm = df.pivot_table(index="symbol", columns="series", values="rm_ratio")
    mac("cryptoRMrawLo", f"{rm['order'].min():.2f}")
    mac("cryptoRMrawHi", f"{rm['order'].max():.2f}")
    mac("cryptoRMadjLo", f"{rm['deseas'].min():.2f}")
    mac("cryptoRMadjHi", f"{rm['deseas'].max():.2f}")

    cs = d["cross_section"]
    for s in ("fill", "order", "deseas"):
        if s in cs:
            mac(f"cryptoCorr{s.capitalize()}",
                f"{cs[s]['corr_loggap_alpha']:+.3f}")
            mac(f"cryptoCorrP{s.capitalize()}", f"{cs[s]['p_loggap']:.3f}")

    # per-symbol table
    piv = df.pivot_table(index="symbol", columns="series", values="hill_plateau")
    meta = o.set_index("symbol")[["qvol_24h", "n", "median_gap_s",
                                  "fills_per_order"]]
    csn = df[df.series == "order"].set_index("symbol")[["csn_alpha", "csn_p"]]
    tab = meta.join(piv[["fill", "order", "deseas"]]).join(csn)
    tab = tab.sort_values("qvol_24h", ascending=False)
    lines = [r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
             r"Pair & 24h volume & orders & median & fills & "
             r"\multicolumn{3}{c}{Hill $\hat\alpha$} & CSN \\",
             r" & (USDT) & & gap (s) & /order & fill & order & adj. & $p$ \\",
             r"\colrule"]
    for sym, t in tab.iterrows():
        f = lambda v, n=2: "---" if pd.isna(v) else f"{v:.{n}f}"
        lines.append(
            f"{esc(sym.replace('USDT',''))} & {si(t.qvol_24h)} & {si(t['n'])} & "
            f"{f(t.median_gap_s, 3)} & {f(t.fills_per_order)} & {f(t.fill)} & "
            f"{f(t.order)} & {f(t.deseas)} & {f(t.csn_p)} \\\\")
    lines += [r"\botrule", r"\end{tabular}"]
    (PAPER / "tab_crypto.tex").write_text("\n".join(lines))

    # event-definition summary
    lines = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"Event definition & median $\hat\alpha$ & share $\hat\alpha<1$ (\%) "
             r"& pairs \\", r"\colrule"]
    pretty = {"fill": "individual fills", "order": "orders (aggregated fills)",
              "W0.1": r"orders merged within $0.1$\,s",
              "W1": r"orders merged within $1$\,s",
              "W5": r"orders merged within $5$\,s",
              "deseas": "orders, diurnally adjusted"}
    for t in summ.itertuples():
        if t.series in pretty:
            lines.append(f"{pretty[t.series]} & {t.median:.2f} & "
                         f"{100 * t.frac_below:.0f} & {t.n} \\\\")
    lines += [r"\botrule", r"\end{tabular}"]
    (PAPER / "tab_eventdef.tex").write_text("\n".join(lines))


def do_hose():
    p = RESULTS / "hose_durations.csv"
    d = load("exp11_hose_durations")
    if not p.exists() or not d:
        return
    df = pd.read_csv(p)
    s = df[df.series == "second"]
    mac("hoseNsym", f"{s['symbol'].nunique()}")
    mac("hoseTotalTicks", si(s["n_ticks"].sum()))
    mac("hoseTieLo", f"{100 * s['tie_share'].min():.0f}")
    mac("hoseTieHi", f"{100 * s['tie_share'].max():.0f}")
    mac("hoseGapLo", f"{s['median_gap_s'].min():.0f}")
    mac("hoseGapHi", f"{s['median_gap_s'].max():.0f}")
    cs = d["cross_section"]
    for k in ("print", "second", "deseas"):
        if k in cs:
            c = cs[k]
            mac(f"hose{k.capitalize()}Med", f"{c['alpha_median']:.2f}")
            mac(f"hose{k.capitalize()}Lo", f"{c['alpha_min']:.2f}")
            mac(f"hose{k.capitalize()}Hi", f"{c['alpha_max']:.2f}")
            mac(f"hose{k.capitalize()}Frac", f"{100 * c['frac_below_one']:.0f}")
            mac(f"hose{k.capitalize()}Corr", f"{c['corr_loggap_alpha']:+.3f}")
            mac(f"hose{k.capitalize()}CorrP", f"{c['p_loggap']:.3f}")
            mac(f"hose{k.capitalize()}N", f"{c['n_symbols']}")
    surv = df[(df.series == "second") & df.csn_p.notna()]
    mac("hoseSurvivors", str(int(((surv.csn_alpha < 1) & (surv.csn_p >= 0.10)).sum())))
    mac("hoseTested", str(len(surv)))

    tab = df[df.series == "second"].set_index("symbol")
    piv = df.pivot_table(index="symbol", columns="series", values="hill_plateau")
    tab = tab[["n_ticks", "tie_share", "median_gap_s", "rm_ratio", "csn_p"]] \
        .join(piv[["print", "second", "deseas"]])
    tab = tab.sort_values("median_gap_s")
    lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
             r"Symbol & prints & ties & median & \multicolumn{3}{c}{Hill $\hat\alpha$} "
             r"& CSN \\",
             r" & & (\%) & gap (s) & print & second & adj. & $p$ \\", r"\colrule"]
    for sym, t in tab.iterrows():
        f = lambda v, n=2: "---" if pd.isna(v) else f"{v:.{n}f}"
        lines.append(f"{esc(sym)} & {si(t.n_ticks)} & {100*t.tie_share:.0f} & "
                     f"{f(t.median_gap_s, 1)} & {f(t['print'])} & {f(t.second)} & "
                     f"{f(t.deseas)} & {f(t.csn_p)} \\\\")
    lines += [r"\botrule", r"\end{tabular}"]
    (PAPER / "tab_hose.tex").write_text("\n".join(lines))


def main():
    for f in (do_exp01, do_exp02, do_exp03, do_exp04, do_exp05, do_exp06,
              do_crypto, do_hose):
        try:
            f()
        except Exception as exc:                                   # noqa: BLE001
            print(f"[skip] {f.__name__}: {type(exc).__name__}: {exc}")
    lines = ["% Auto-generated by experiments/make_tables.py -- do not edit.",
             "% Every numeric result in the manuscript comes from here.", ""]
    for k in sorted(MACROS):
        lines.append(rf"\newcommand{{\{k}}}{{{MACROS[k]}}}")
    (PAPER / "results_macros.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(MACROS)} macros to paper/results_macros.tex")
    for f in sorted(PAPER.glob("tab_*.tex")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
