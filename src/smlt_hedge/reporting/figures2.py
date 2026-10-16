"""Figures 11-19 for the second research pass (same visual conventions as report.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from .report import DIVERGING, GRID, INK, INK2, SERIES, SURFACE, _save, _src

ORDER = ["SMLT", "DIRECT", "MARKET", "RATES", "GOLD", "CNY", "OIL", "SMLT_BOND_IDX"]


def _heat(ax, c: pd.DataFrame, title: str) -> None:
    c = c.reindex(index=[o for o in ORDER if o in c.index], columns=[o for o in ORDER if o in c.columns])
    ax.imshow(c.values, cmap=DIVERGING, norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
    for (i, j), v in np.ndenumerate(c.values):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color=INK)
    lab = [x.replace("SMLT_BOND_IDX", "SMLT bonds") for x in c.columns]
    ax.set_xticks(range(len(lab)), lab, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(lab)), lab, fontsize=7)
    ax.grid(False)
    ax.set_title(title, fontsize=9.5)


def fig11(ctx):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, k, t in zip(axes, ["full", "250d"], ["Full sample", "Trailing 250 days"]):
        _heat(ax, ctx.read(f"correlation_{k}.csv", "out", index_col=0), t)
    fig.suptitle("Daily-return correlations: SMLT and Phase-B hedge sleeves (normal conditions)", fontweight="bold")
    _src(axes[0], "DIRECT = SMLT futures, MARKET = MIX, RATES = RGBI fut, GOLD = GLDRUB, CNY = CNYRUB, OIL = Brent")
    _save(fig, ctx, "11_correlation_matrix_normal.png")


def fig12(ctx):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, k, t in zip(axes, ["market_stress", "credit_stress", "downside"],
                        ["Broad-market stress (IMOEX worst decile)", "Issuer-credit stress (regime flag)", "SMLT down days"]):
        _heat(ax, ctx.read(f"correlation_{k}.csv", "out", index_col=0), t)
    fig.suptitle("Correlations in stress: which relationships persist, break or reverse", fontweight="bold", y=1.03)
    _save(fig, ctx, "12_correlation_matrix_stress.png")


def fig13(ctx):
    d = pd.read_csv(ctx.out / "factor_decomposition.csv")
    d = d[~d["window"].str.contains("CREDIT_Z")]
    rl = ctx.read("rolling_factor_loadings.csv", "out", index_col=0, parse_dates=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    comps = ["shapley_MKT", "shapley_SECTOR", "shapley_RATES", "shapley_CREDIT", "unexplained"]
    labs = ["market", "sector ex-SMLT", "rates", "credit", "unexplained"]
    bottom = np.zeros(len(d))
    cols = SERIES[:4] + ["#c8c7c2"]
    for c, l, col in zip(comps, labs, cols):
        a.bar(d["window"], d[c] * 100, bottom=bottom * 100, color=col, label=l, edgecolor=SURFACE, linewidth=1)
        bottom += d[c].values
    a.set_ylabel("share of SMLT daily-return variance, %")
    a.set_title("Shapley variance decomposition by window")
    a.tick_params(axis="x", rotation=30, labelsize=7.5)
    a.legend(fontsize=7.5, ncol=3, loc="upper left")
    for i, f in enumerate(["MKT", "SECTOR", "RATES", "CREDIT"]):
        s = rl[f"beta_{f}_120d"].dropna()
        b.plot(s.index, s.values, color=SERIES[i], lw=1.3, label=f)
    b.axhline(0, color=INK2, lw=0.8)
    b.set_title("Rolling 120d factor loadings (trailing, not shifted)")
    b.legend(fontsize=7.5)
    _src(a, "Residual = variance NOT explained by these factors, not automatically issuer-specific risk (see METHODOLOGY)")
    _save(fig, ctx, "13_factor_exposure_decomposition.png")


def _frontier(ctx, xcol, ycol, title, fname, rule):
    s = pd.read_csv(ctx.out / "smart_candidates_summary.csv")
    s = s[(s["horizon"] == "6m") & (s["capital"] == ctx.C)]
    sel = pd.read_csv(ctx.out / "smart_selected_portfolios.csv")
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    br = s["worst_model_breach" if rule == "robust" else "p_sleeve_breach"]
    buckets = [(0, 0.05, SERIES[2], "breach <= 5%"), (0.05, 0.10, SERIES[0], "5-10%"), (0.10, 0.20, SERIES[3], "10-20%"),
               (0.20, 1.01, SERIES[7], "> 20%")]
    m = s["family"] == "MULTI"
    for lo, hi, col, lab in buckets:
        k = m & (br >= lo) & (br < hi)
        ax.scatter(s.loc[k, xcol] / 1e6, s.loc[k, ycol] * 100, s=12, color=col, alpha=0.6, label=f"multi-asset, {lab}", lw=0)
    d = s[s["family"] == "DIRECT_ONLY"].sort_values(xcol)
    ax.plot(d[xcol] / 1e6, d[ycol] * 100, color=INK, lw=1.6, marker="o", ms=3.5, label="direct SMLT futures only")
    for fam, mk, lab in (("ANY", "*", "selected (any family)"), ("DIRECT_ONLY", "D", "selected (direct only)")):
        r = sel[(sel["rule"] == rule) & (sel["selection_scope"] == fam) & (sel["horizon"] == "6m") & (sel["capital"] == ctx.C)]
        if len(r) and r["cand_id"].iloc[0] != "NONE":
            ax.scatter(r[xcol] / 1e6, r[ycol] * 100, s=180 if mk == "*" else 70, marker=mk, color=SERIES[1],
                       edgecolor=INK, zorder=5, label=lab)
    ax.set_xlabel(("worst-model " if rule == "robust" else "FHS ") + "ES95 of 6m P&L, RUB m (lower = better)")
    ax.set_ylabel(("worst-model " if rule == "robust" else "FHS ") + "upside capture, %")
    ax.set_title(title)
    ax.legend(fontsize=7.5, loc="lower right")
    _src(ax, "Each dot = exact min-ES LP solution for a sleeve subset x margin budget x upside floor; capital RUB 10m")
    _save(fig, ctx, fname)


def fig14(ctx):
    _frontier(ctx, "ES95", "upside_capture", "Static smart-portfolio frontier (6m, FHS): protection vs upside vs liquidity",
              "14_static_smart_portfolio_frontier.png", "static")


def fig15(ctx):
    _frontier(ctx, "worst_model_ES95", "worst_model_upside", "Robust frontier: worst case over HS / FHS / recent / historical windows",
              "15_robust_portfolio_frontier.png", "robust")


def fig16(ctx):
    d = ctx.read("adaptive_daily_ADAPTIVE_SMART.csv", "interim", index_col=0, parse_dates=True)
    hc = [c for c in d.columns if c.startswith("h_")]
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for i, c in enumerate(hc):
        a.plot(d.index, d[c], color=SERIES[i % 8], lw=1.3, label=c[2:])
    a.axhline(0, color=INK2, lw=0.8)
    a.set_ylabel("notional / RUB 10m (short < 0)")
    a.set_title("Adaptive smart portfolio: target sleeve notionals (walk-forward, causal)")
    a.legend(fontsize=7.5, ncol=6, loc="lower left")
    rcode = {"NORMAL": 0, "RATES_STRESS": 1, "MARKET_STRESS": 2, "CREDIT_STRESS": 3, "COMBINED_STRESS": 4}
    b.step(d.index, d["regime"].map(rcode), where="post", color=SERIES[6], lw=1.2)
    b.set_yticks(list(rcode.values()), list(rcode.keys()), fontsize=7)
    b.set_title("Regime (causal)", fontsize=9)
    _save(fig, ctx, "16_adaptive_weights_over_time.png")


def fig17(ctx):
    c = pd.read_csv(ctx.out / "risk_contributions.csv")
    c = c[c["horizon"] == "6m"]
    ports = [p for p in ["FUT_50", "STATIC_SMART_PORTFOLIO", "ROBUST_SMART_PORTFOLIO", "ADAPTIVE_SMART_PORTFOLIO"] if p in set(c["portfolio"])]
    comps = [x for x in ["SMLT_SPOT", "DIRECT", "MARKET", "RATES", "GOLD", "CNY", "OIL", "LIQUIDITY_INTEREST", "COSTS"] if x in set(c["component"])]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    w = 0.8 / max(len(ports), 1)
    for k, (metric, ttl) in enumerate([("es95_contrib", "Contribution to ES95 (RUB m; loss positive)"),
                                       ("mean", "Contribution to mean 6m P&L (carry + costs, RUB m)")]):
        ax = axes[k]
        for i, p in enumerate(ports):
            v = c[c["portfolio"] == p].set_index("component").reindex(comps)[metric].fillna(0) / 1e6
            ax.bar(np.arange(len(comps)) + i * w, v.values, width=w, color=SERIES[i], label=p.replace("_PORTFOLIO", ""),
                   edgecolor=SURFACE, linewidth=1)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(np.arange(len(comps)) + w * (len(ports) - 1) / 2, comps, rotation=35, ha="right", fontsize=7.5)
        ax.set_title(ttl, fontsize=9.5)
    axes[0].legend(fontsize=7.5)
    _src(axes[0], "Euler/tail attribution on FHS 6m paths: ES contribution = -mean component P&L in the 5% worst paths", y=-0.34)
    _save(fig, ctx, "17_risk_contributions.png")


def fig18(ctx):
    keys = [("CONTROL", "CONTROL (unhedged)", INK2), ("FUT_50", "futures 50%", SERIES[1]), ("FUT_100", "futures 100%", SERIES[2]),
            ("ROLLING_HSTAR", "rolling h*", SERIES[3]), ("FIXED_SMART_PRE_OOS", "fixed smart (pre-OOS)", SERIES[4]),
            ("ADAPTIVE_SMART", "adaptive smart", SERIES[0])]
    fig, a = plt.subplots(figsize=(10.5, 4.6))
    for k, lab, col in keys:
        f = ctx.interim / f"adaptive_daily_{k}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, index_col=0, parse_dates=True)
        s = d["pnl_total"].cumsum() / 1e6
        a.plot(s.index, s.values, color=col, lw=1.6 if k != "ADAPTIVE_SMART" else 2.2, label=lab)
        a.annotate(lab, xy=(s.index[-1], s.iloc[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7.5, color=INK)
    a.set_title("Walk-forward, constant RUB 10m exposure: cumulative P&L incl. costs and carry (RUB m)")
    a.legend(fontsize=8, loc="lower left")
    _src(a, "Weekly causal decisions; adaptive thresholds selected quarterly from prior out-of-sample data only")
    _save(fig, ctx, "18_adaptive_vs_fixed_cum_pnl.png")


def fig19(ctx):
    w = pd.read_csv(ctx.out / "liquidity_waterfall_results.csv")
    w = w[w["horizon"] == "6m"]
    strat = [s for s in ["FUT_75", "FUT_100", "FIXED_SMART_PRE_OOS", "ADAPTIVE_SMART"] if s in set(w["strategy"])]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(strat))
    for i, pol in enumerate(["WATERFALL", "IMMEDIATE_CUT"]):
        d = w[w["liquidity_policy"] == pol].set_index("strategy").reindex(strat)
        a.bar(x + i * 0.38, d["ES95_terminal"] / 1e6, width=0.38, color=SERIES[i], label=pol, edgecolor=SURFACE)
        b.bar(x + i * 0.38, d["shares_sold_value"] / 1e6, width=0.38, color=SERIES[i], label=pol, edgecolor=SURFACE)
    for ax, t, yl in ((a, "ES95 of historical 6m episode P&L", "RUB m"),
                      (b, "SMLT sold as last resort (DE_RISKING), sum over episodes", "RUB m")):
        ax.set_xticks(x + 0.19, strat, fontsize=8, rotation=15)
        ax.set_title(t, fontsize=9.5)
        ax.set_ylabel(yl)
        ax.legend(fontsize=8)
    _src(a, "Waterfall: buffer -> productive collateral -> least-valuable sleeve -> direct futures -> sell SMLT (flagged)")
    _save(fig, ctx, "19_liquidity_waterfall.png")


def all_figures(ctx) -> None:
    for f in (fig11, fig12, fig13, fig14, fig15, fig16, fig17, fig18, fig19):
        try:
            f(ctx)
        except Exception as e:  # noqa: BLE001
            ctx.log.setdefault("figure_errors", {})[f.__name__] = repr(e)
