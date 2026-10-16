"""Regime V2 figures (fig_regime_v2r1_*.png), drawn from outputs/regime_v2r1_* only; the parent's visual conventions."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..reporting.report import DIVERGING, INK2, SERIES, _src, plt

COL = {"SMLT": SERIES[0], "SMLT_FUT": SERIES[7], "EQ_FUNDED": SERIES[1], "EQ_HEDGE_FUT": "#f2a47f", "OFZ_FUNDED": SERIES[2],
       "RATES_HEDGE_FUT": "#8fd8bd", "GOLD_FUNDED": SERIES[3], "CNY_PERP": SERIES[4], "OIL_FUT": SERIES[5],
       "BTC_RUB_EXPERIMENTAL": SERIES[6], "MMF": "#b8b6b0", "CASH": "#6b6a66"}
ROUTES = ["SMLT", "SMLT_FUT", "EQ_FUNDED", "EQ_HEDGE_FUT", "OFZ_FUNDED", "RATES_HEDGE_FUT", "GOLD_FUNDED", "CNY_PERP", "OIL_FUT",
          "BTC_RUB_EXPERIMENTAL"]
SHORT = {"SMLT": "SMLT", "SMLT_FUT": "SMLT fut", "EQ_FUNDED": "equity fund", "EQ_HEDGE_FUT": "mini-MIX", "OFZ_FUNDED": "OFZ fund",
         "RATES_HEDGE_FUT": "RGBI fut", "GOLD_FUNDED": "gold fund", "CNY_PERP": "CNY perp", "OIL_FUT": "Brent fut",
         "BTC_RUB_EXPERIMENTAL": "BTC (exp.)"}
POL = {"ADAPTIVE_REGIME_PIF_V2R1": SERIES[0], "STATIC_INVESTED_PIF_V2R1": SERIES[1], "CAPITAL_PRESERVATION_UNCONSTRAINED": SERIES[2],
       "HEDGE_KEEP_SMLT_50": SERIES[3], "HEDGE_KEEP_SMLT_25": SERIES[4], "OPERATIONAL_MONEY_MARKET": "#6b6a66",
       "ALL_MONEY_MARKET_REFERENCE": "#b8b6b0", "SELL_100_TO_MMF": "#8c8b87", "SELL_50_TO_MMF": SERIES[5], "CONTROL_INITIAL_PORTFOLIO": SERIES[7],
       "FUT25_LEGACY": "#c9a3e6", "FUT50_LEGACY": SERIES[6], "V41_STATIC_PROXY_NONCAUSAL": "#d4b483"}


def _save(fig, ctx, name):
    assert name.startswith("fig_regime_v2r1_")
    fig.tight_layout()
    fig.savefig(ctx.fig / name, dpi=200)
    plt.close(fig)


def _read(ctx, name):
    p = ctx.out / name
    return pd.read_csv(p) if p.exists() else None


def _na(ctx, name, title):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, f"{title}\n(input file not produced in this run)", ha="center", va="center", color=INK2)
    _save(fig, ctx, name)


def _bars(ax, d: pd.DataFrame, x, prefix: str):
    pos = np.zeros(len(d))
    neg = np.zeros(len(d))
    for r in ROUTES:
        c = f"{prefix}{r}"
        if c not in d:
            continue
        v = pd.to_numeric(d[c], errors="coerce").fillna(0).values * 100
        ax.bar(x, np.where(v > 0, v, 0), bottom=pos, color=COL[r], width=0.7, label=SHORT[r] if (np.abs(v) > 1e-9).any() else None)
        ax.bar(x, np.where(v < 0, v, 0), bottom=neg, color=COL[r], width=0.7)
        pos += np.where(v > 0, v, 0)
        neg += np.where(v < 0, v, 0)
    ax.axhline(0, color=INK2, lw=0.6)


def f01_state_vector(ctx):
    p = ctx.out / "regime_v2r1_current_decision.json"
    d = _read(ctx, "regime_v2r1_state_daily.csv")
    if not p.exists() or d is None:
        return _na(ctx, "fig_regime_v2r1_01_state_vector.png", "current state vector")
    obj = json.loads(p.read_text())
    d["date"] = pd.to_datetime(d["date"])
    d = d[d["date"] >= d["date"].max() - pd.Timedelta(days=400)]
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.2))
    panels = [("ISSUER_CREDIT_RISK", "credit_spread_pp", "Samolet credit spread, pp"), ("MARKET_RISK", "imoex_ret20", "IMOEX 20-session return"),
              ("RATES_STATE", "key_rate_pct", "key rate, %"), ("RUB_STATE", "rub_logret_window", "USD/RUB 20-session log return")]
    conf = obj["STATE_VECTOR"]["confirmed"]
    for ax, (axis, col, lab) in zip(axs.ravel(), panels):
        ax.plot(d["date"], d[col], color=SERIES[0], lw=1)
        if axis == "RUB_STATE":
            ax.plot(d["date"], d["rub_threshold"], color=INK2, lw=0.7, ls="--")
            ax.plot(d["date"], -d["rub_threshold"], color=INK2, lw=0.7, ls="--")
        ax.set_title(f"{axis}: {conf[axis]} (confirmed)", fontsize=9)
        ax.set_ylabel(lab, fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"Observable state vector at {obj['DATE']} (every input uses data up to that close)", fontsize=10)
    _src(axs[1, 0], "Source: outputs/regime_v2r1_state_daily.csv, regime_v2r1_current_decision.json", y=-0.22)
    _save(fig, ctx, "fig_regime_v2r1_01_state_vector.png")


def _frontier_ax(ax, f, title):
    for src_, mk in (("LP", "o"), ("REFINEMENT", "s"), ("FLOOR_ANCHOR_MIN_RISK", "D"), ("MIN_RISK_ANCHOR", "D"), ("OPERATIONAL_SAFE_ANCHOR", "P"),
                     ("CURRENT_BOOK", "X")):
        g = f[f["source"] == src_]
        if not len(g):
            continue
        ok = g["ADMISSION"] == "PASS"
        ax.scatter(g.loc[ok, "engine_ES95_worst_nav0"] * 100, g.loc[ok, "expected_net_excess_nav0"] * 100, marker=mk, color=SERIES[2], s=28,
                   label=f"{src_} PASS")
        ax.scatter(g.loc[~ok, "engine_ES95_worst_nav0"] * 100, g.loc[~ok, "expected_net_excess_nav0"] * 100, marker=mk, color=SERIES[7], s=28,
                   facecolors="none", label=f"{src_} FAIL")
    sel = f[f["selected_target"].astype(bool) & (f["source"] != "CURRENT_BOOK")]
    if len(sel):
        ax.scatter(sel["engine_ES95_worst_nav0"] * 100, sel["expected_net_excess_nav0"] * 100, marker="*", s=220, color=SERIES[3],
                   edgecolor="k", zorder=5, label="selected target")
    b = float(f["risk_budget_nav0"].iloc[0]) * 100
    ax.axvline(b, color=INK2, ls="--", lw=0.8)
    ax.set_xlabel("full-engine ES95 over 126 sessions, worst model, % NAV0")
    ax.set_ylabel("expected net excess over the money market, % NAV0")
    ax.set_title(title, fontsize=9)


def f02_frontier(ctx):
    f = _read(ctx, "regime_v2r1_frontier_15.csv")
    if f is None:
        return _na(ctx, "fig_regime_v2r1_02_frontier_15.png", "15% frontier")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _frontier_ax(ax, f[f["variant"] == "HEDGED_INVESTED_PIF"], "HEDGED_INVESTED_PIF at the 15% budget: every candidate re-admitted by the full engine")
    ax.legend(fontsize=6.5, ncol=2, loc="lower right")
    _src(ax, "Source: outputs/regime_v2r1_frontier_15.csv, regime_v2r1_frontier_refinement.csv")
    _save(fig, ctx, "fig_regime_v2r1_02_frontier_15.png")


def f03_mandates(ctx):
    s = _read(ctx, "regime_v2r1_cutoff_decisions.csv")
    if s is None:
        return _na(ctx, "fig_regime_v2r1_03_mandates.png", "capital preservation vs invested mandate")
    d = s[s["variant"].isin(["CAPITAL_PRESERVATION_UNCONSTRAINED", "HEDGED_INVESTED_PIF"])].set_index("variant").loc[
        ["CAPITAL_PRESERVATION_UNCONSTRAINED", "HEDGED_INVESTED_PIF"]].reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(d))
    _bars(ax, d, x, "w_tgt_")
    for i, r in d.iterrows():
        ax.text(i, ax.get_ylim()[1] * 0.92, f"{r['action']}\nexp. net excess {r['expected_net_target_nav0'] * 100:+.2f}%\nES95 {r['es95_target_nav0'] * 100:.1f}%",
                ha="center", va="top", fontsize=8, color=INK2)
    ax.set_xticks(x, ["A. capital preservation\n(0-100% money market)", "B. hedged invested PIF\n(>= 50% funded non-MMF)"], fontsize=8)
    ax.set_ylabel("% NAV (money market and reserve = the rest)")
    ax.set_title("Two mandates at the cutoff, same return model, risk model and engine", fontsize=10)
    ax.legend(fontsize=7, ncol=3, loc="lower center")
    _src(ax, "Source: outputs/regime_v2r1_cutoff_decisions.csv")
    _save(fig, ctx, "fig_regime_v2r1_03_mandates.png")


def f04_allocation(ctx):
    t = _read(ctx, "regime_v2r1_current_target.csv")
    if t is None:
        return _na(ctx, "fig_regime_v2r1_04_current_allocation.png", "current allocation")
    t = t[t["variant"] == "HEDGED_INVESTED_PIF"]
    t = t[t["instrument"].isin(ROUTES)].copy()
    t["target_weight_nav"] = pd.to_numeric(t["target_weight_nav"], errors="coerce").fillna(0)
    fig, ax = plt.subplots(figsize=(10, 5))
    order = ["FUNDED_ALLOCATION", "DIRECTIONAL_MACRO_DERIVATIVE", "PROTECTIVE_HEDGE_OVERLAY"]
    x = np.arange(len(order))
    for k, sl in enumerate(order):
        g = t[t["sleeve"] == sl]
        pos = neg = 0.0
        for _, r in g.iterrows():
            v = r["target_weight_nav"] * 100
            if abs(v) < 1e-9:
                continue
            ax.bar(k, v, bottom=pos if v > 0 else neg, color=COL[r["instrument"]], width=0.6)
            ax.text(k + 0.33, (pos if v > 0 else neg) + v / 2, f"{SHORT[r['instrument']]} {v:+.1f}%", fontsize=7, va="center")
            if v > 0:
                pos += v
            else:
                neg += v
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(x, ["funded allocation", "directional macro derivatives", "protective hedge overlays"], fontsize=8)
    ax.set_xlim(-0.5, 2.9)
    ax.set_ylabel("% NAV (derivatives: signed notional)")
    ax.set_title("HEDGED_INVESTED_PIF target at the cutoff by sleeve", fontsize=10)
    _src(ax, "Source: outputs/regime_v2r1_current_target.csv")
    _save(fig, ctx, "fig_regime_v2r1_04_current_allocation.png")


def f05_state_allocations(ctx):
    lib = _read(ctx, "regime_v2r1_state_target_library.csv")
    if lib is None:
        return _na(ctx, "fig_regime_v2r1_05_state_allocations.png", "state-dependent allocations")
    obj = json.loads((ctx.out / "regime_v2r1_current_decision.json").read_text())
    rub = obj["STATE_VECTOR"]["confirmed"]["RUB_STATE"]
    d = lib[lib["rub_state"] == rub].copy()
    d["lab"] = d["issuer_state"].str[:5] + "/" + d["market_state"].str[:4] + "/" + d["rates_state"].str[:4]
    cols = [f"w_tgt_{r}" for r in ROUTES if f"w_tgt_{r}" in d]
    M = d[cols].fillna(0).values * 100
    fig, ax = plt.subplots(figsize=(10, 7))
    lim = max(1.0, float(np.abs(M).max()))
    im = ax.imshow(M, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_yticks(range(len(d)), d["lab"], fontsize=6.5)
    ax.set_xticks(range(len(cols)), [SHORT[c[6:]] for c in cols], rotation=40, fontsize=7, ha="right")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if abs(M[i, j]) > 0.05:
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=5.5)
    fig.colorbar(im, ax=ax, label="% NAV")
    ax.set_title(f"Target from the actual book under each state (issuer / market / rates; RUB state {rub})", fontsize=9)
    ax.grid(False)
    _src(ax, "Source: outputs/regime_v2r1_state_target_library.csv (SCENARIO_ONLY except the actual state)", y=-0.2)
    _save(fig, ctx, "fig_regime_v2r1_05_state_allocations.png")


def f06_correlations(ctx):
    c = _read(ctx, "regime_v2r1_state_correlation.csv")
    if c is None:
        return _na(ctx, "fig_regime_v2r1_06_correlations.png", "correlations by state")
    sets = [("MARKET_RISK=NORMAL", "market NORMAL"), ("MARKET_RISK=STRESS", "market STRESS")]
    cur = c[c["state_set"] == "CURRENT_JOINT_STATE_MODEL"]
    sets.append((cur["state"].iloc[0] if len(cur) else "ALL_ORIGINS", "current joint state" if len(cur) else "global (current state model not available)"))
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (st, title) in zip(axs, sets):
        g = c[c["state"] == st]
        R = [r for r in ROUTES if r in set(g["route_i"])]
        Mx = np.full((len(R), len(R)), np.nan)
        for _, r in g.iterrows():
            i, j = R.index(r["route_i"]), R.index(r["route_j"])
            Mx[i, j] = Mx[j, i] = r["corr_shrunk"]
        ax.imshow(Mx, cmap=DIVERGING, vmin=-1, vmax=1)
        for i in range(len(R)):
            for j in range(len(R)):
                if np.isfinite(Mx[i, j]):
                    ax.text(j, i, f"{Mx[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
        ax.set_xticks(range(len(R)), [SHORT[r] for r in R], rotation=60, fontsize=6.5, ha="right")
        ax.set_yticks(range(len(R)), [SHORT[r] for r in R], fontsize=6.5)
        n = int(g["n_obs"].iloc[0]) if len(g) else 0
        ax.set_title(f"{title}\n(next-session returns after origins in the state, n = {n}; shrunk to global)", fontsize=8)
        ax.grid(False)
    _src(axs[0], "Source: outputs/regime_v2r1_state_correlation.csv", y=-0.35)
    _save(fig, ctx, "fig_regime_v2r1_06_correlations.png")


def f07_hedge_ratios(ctx):
    h = _read(ctx, "regime_v2r1_hedge_ratios.csv")
    if h is None:
        return _na(ctx, "fig_regime_v2r1_07_hedge_ratios.png", "hedge ratios")
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, hd in zip(axs, ["SMLT_FUT", "EQ_HEDGE_FUT", "RATES_HEDGE_FUT"]):
        g = h[h["hedge"] == hd]
        x = np.arange(len(g))
        ax.bar(x, g["h_star"], color=COL[hd], width=0.6)
        if len(g):
            r = g.iloc[0]
            ax.axhspan(r["rolling_h_min"], r["rolling_h_max"], color=INK2, alpha=0.08, label="60-session rolling range, last 250")
        for xi, (_, r) in zip(x, g.iterrows()):
            ax.text(xi, r["h_star"], f"{r['h_star']:.2f}\nN*={r.get('N_star_contracts', np.nan):.0f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x, g["sample"], rotation=35, fontsize=7, ha="right")
        ax.set_title(f"SMLT vs {SHORT[hd]}: h* = Cov/Var", fontsize=9)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.legend(fontsize=6.5)
    _src(axs[0], "Source: outputs/regime_v2r1_hedge_ratios.csv", y=-0.3)
    _save(fig, ctx, "fig_regime_v2r1_07_hedge_ratios.png")


def _alloc(ctx):
    a = _read(ctx, "regime_v2r1_allocation_path.csv")
    if a is not None:
        a["date"] = pd.to_datetime(a["date"])
    return a


def f08_smlt_exposure(ctx):
    a = _alloc(ctx)
    if a is None:
        return _na(ctx, "fig_regime_v2r1_08_smlt_exposure.png", "SMLT exposure through time")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for n in ("ADAPTIVE_REGIME_PIF_V2R1", "STATIC_INVESTED_PIF_V2R1", "CAPITAL_PRESERVATION_UNCONSTRAINED", "HEDGE_KEEP_SMLT_50", "HEDGE_KEEP_SMLT_25"):
        g = a[a["comparator"] == n]
        if len(g):
            ax.plot(g["date"], g["smlt_net_issuer"] * 100, color=POL[n], lw=1.2, label=f"{n} (net)")
            if n.startswith("HEDGE"):
                ax.plot(g["date"], g["SMLT"] * 100, color=POL[n], lw=0.7, ls=":", label=f"{n} (shares)")
    ax.set_ylabel("% NAV")
    ax.set_title("SMLT issuer exposure through time (shares + futures), one ledger per policy", fontsize=10)
    ax.legend(fontsize=7)
    _src(ax, "Source: outputs/regime_v2r1_allocation_path.csv")
    _save(fig, ctx, "fig_regime_v2r1_08_smlt_exposure.png")


def f09_allocation_time(ctx):
    a = _alloc(ctx)
    if a is None:
        return _na(ctx, "fig_regime_v2r1_09_allocation_time.png", "allocation through time")
    fig, axs = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, n in zip(axs, ("ADAPTIVE_REGIME_PIF_V2R1", "STATIC_INVESTED_PIF_V2R1")):
        g = a[a["comparator"] == n]
        fr = [r for r in ("SMLT", "EQ_FUNDED", "OFZ_FUNDED", "GOLD_FUNDED", "BTC_RUB_EXPERIMENTAL") if r in g]
        if len(g):
            ax.stackplot(g["date"], *[g[r].clip(lower=0) * 100 for r in fr], colors=[COL[r] for r in fr], labels=[SHORT[r] for r in fr])
            for r in ("CNY_PERP", "OIL_FUT"):
                if r in g and g[r].abs().max() > 0:
                    ax.plot(g["date"], g[r] * 100, color=COL[r], lw=1.2, label=f"{SHORT[r]} (notional)")
        ax.set_ylabel("% NAV")
        ax.set_title(f"{n}: funded allocation (stacked) and directional sleeves", fontsize=9)
        ax.legend(fontsize=7, ncol=4, loc="upper left")
    _src(axs[1], "Source: outputs/regime_v2r1_allocation_path.csv", y=-0.2)
    _save(fig, ctx, "fig_regime_v2r1_09_allocation_time.png")


def f10_overlays(ctx):
    a = _alloc(ctx)
    if a is None:
        return _na(ctx, "fig_regime_v2r1_10_overlays_time.png", "hedge overlays through time")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for n, ls in (("ADAPTIVE_REGIME_PIF_V2R1", "-"), ("HEDGE_KEEP_SMLT_50", "--"), ("HEDGE_KEEP_SMLT_25", ":")):
        g = a[a["comparator"] == n]
        for r in ("SMLT_FUT", "EQ_HEDGE_FUT", "RATES_HEDGE_FUT"):
            if len(g) and r in g and g[r].abs().max() > 0:
                ax.plot(g["date"], g[r] * 100, color=COL[r], ls=ls, lw=1.1, label=f"{n}: {SHORT[r]}")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_ylabel("% NAV (signed notional)")
    ax.set_title("Protective hedge overlays through time", fontsize=10)
    ax.legend(fontsize=7)
    _src(ax, "Source: outputs/regime_v2r1_allocation_path.csv")
    _save(fig, ctx, "fig_regime_v2r1_10_overlays_time.png")


def _nav(ctx):
    n = _read(ctx, "regime_v2r1_nav_paths.csv")
    if n is not None:
        n["date"] = pd.to_datetime(n["date"])
    return n


def f11_nav(ctx):
    n = _nav(ctx)
    if n is None:
        return _na(ctx, "fig_regime_v2r1_11_nav.png", "one-ledger NAV")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for c in n.columns[1:]:
        if "|" in c:
            continue
        ax.plot(n["date"], n[c] / 1e6, color=POL.get(c, INK2), lw=1.6 if c.startswith(("ADAPTIVE", "STATIC", "OPERATIONAL")) else 0.9, label=c)
    ax.set_ylabel("NAV, RUB m")
    ax.set_title("One self-financing ledger per policy, 2023-01-03 .. cutoff (RETROSPECTIVE, one history)", fontsize=10)
    ax.legend(fontsize=6.5, ncol=2)
    _src(ax, "Source: outputs/regime_v2r1_nav_paths.csv")
    _save(fig, ctx, "fig_regime_v2r1_11_nav.png")


def f12_drawdown(ctx):
    n = _nav(ctx)
    if n is None:
        return _na(ctx, "fig_regime_v2r1_12_drawdown.png", "drawdowns")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for c in n.columns[1:]:
        if "|" in c:
            continue
        v = n[c]
        ax.plot(n["date"], -(1 - v / v.cummax()) * 100, color=POL.get(c, INK2), lw=1.1, label=c)
    ax.set_ylabel("drawdown, %")
    ax.set_title("Drawdown of each ledger", fontsize=10)
    ax.legend(fontsize=6.5, ncol=2)
    _src(ax, "Source: outputs/regime_v2r1_nav_paths.csv")
    _save(fig, ctx, "fig_regime_v2r1_12_drawdown.png")


def f13_risk_contribution(ctx):
    r = _read(ctx, "regime_v2r1_risk_contribution.csv")
    if r is None:
        return _na(ctx, "fig_regime_v2r1_13_risk_contribution.png", "risk contribution")
    books = [("HEDGED_INVESTED_PIF", "CURRENT_BOOK"), ("HEDGED_INVESTED_PIF", "TARGET"), ("CAPITAL_PRESERVATION_UNCONSTRAINED", "TARGET"),
             ("STATIC_INVESTED_PIF_V2R1", "TARGET")]
    fig, ax = plt.subplots(figsize=(11, 5))
    for k, (v, b) in enumerate(books):
        g = r[(r["variant"] == v) & (r["book"] == b) & r["sleeve"].isin(ROUTES)]
        pos = neg = 0.0
        for _, q in g.iterrows():
            val = q["es_contribution_nav0"] * 100
            if abs(val) < 1e-6:
                continue
            ax.bar(k, val, bottom=pos if val > 0 else neg, color=COL[q["sleeve"]], width=0.6, label=SHORT[q["sleeve"]] if k == 1 else None)
            if val > 0:
                pos += val
            else:
                neg += val
        if len(g):
            ax.text(k, pos + 0.3, f"ES95 {g['book_ES95_static_nav0'].iloc[0] * 100:.1f}%", ha="center", fontsize=7.5)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(len(books)), [f"{v}\n{b}" for v, b in books], fontsize=7)
    ax.set_ylabel("Euler contribution to ES95, % NAV0 (worst model, static horizon)")
    ax.set_title("Where the tail risk comes from", fontsize=10)
    ax.legend(fontsize=7, ncol=3)
    _src(ax, "Source: outputs/regime_v2r1_risk_contribution.csv")
    _save(fig, ctx, "fig_regime_v2r1_13_risk_contribution.png")


def f14_forward(ctx):
    f = _read(ctx, "regime_v2r1_state_forward_stats.csv")
    if f is None:
        return _na(ctx, "fig_regime_v2r1_14_forward_state.png", "forward returns by state")
    f = f[(f["set"] == "MARGINAL") & f["is_current_state"].astype(bool)]
    routes = ["SMLT", "EQ_FUNDED", "OFZ_FUNDED", "GOLD_FUNDED", "CNY_PERP", "OIL_FUT", "BTC_RUB_EXPERIMENTAL"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    for ax, H in zip(axs, (21, 63, 126)):
        g = f[(f["horizon"] == H) & f["route"].isin(routes)]
        labs, st, al, adm = [], [], [], []
        for _, r in g.iterrows():
            labs.append(f"{SHORT[r['route']]}\n{r['condition'].split('=')[0][:6]}={r['condition'].split('=')[1][:6]}")
            st.append(r["mean_fwd_excess"] * 100)
            al.append(r["all_origin_mean"] * 100)
            adm.append(bool(r["admissible"]))
        x = np.arange(len(labs))
        ax.bar(x - 0.2, st, width=0.4, color=[SERIES[2] if a else SERIES[0] for a in adm], label="after origins in the current state")
        ax.bar(x + 0.2, al, width=0.4, color="#b8b6b0", label="all origins")
        ax.set_xticks(x, labs, fontsize=5.5, rotation=90)
        ax.set_title(f"forward {H} sessions (green = admissible differential)", fontsize=8)
        ax.axhline(0, color=INK2, lw=0.6)
    axs[0].set_ylabel("mean forward excess, %")
    axs[0].legend(fontsize=6.5)
    _src(axs[0], "Source: outputs/regime_v2r1_state_forward_stats.csv (RETROSPECTIVE; overlapping windows)", y=-0.55)
    _save(fig, ctx, "fig_regime_v2r1_14_forward_state.png")


def f15_scorecard(ctx):
    s = _read(ctx, "regime_v2r1_scenario_scorecard.csv")
    if s is None:
        return _na(ctx, "fig_regime_v2r1_15_scenario_scorecard.png", "regime scorecard")
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(s))
    axs[0].bar(x - 0.2, s["own_path_pnl_current_book_nav0"] * 100, width=0.4, color=SERIES[7], label="current book, no trade")
    axs[0].bar(x + 0.2, s["own_path_pnl_target_nav0"] * 100, width=0.4, color=SERIES[0], label="scenario target")
    axs[0].set_xticks(x, s["scenario"].str.replace("_", " ").str.slice(0, 26), rotation=40, fontsize=6.5, ha="right")
    axs[0].set_ylabel("P&L on the scenario's deterministic path, % NAV")
    axs[0].axhline(0, color=INK2, lw=0.6)
    axs[0].legend(fontsize=7)
    axs[0].set_title("Deterministic path of each regime (no probabilities)", fontsize=9)
    axs[1].bar(x - 0.2, s["expected_net_target_nav0"] * 100, width=0.4, color=SERIES[2], label="expected net excess (model)")
    axs[1].bar(x + 0.2, -s["es95_target_nav0"] * 100, width=0.4, color=SERIES[1], label="- ES95 (engine, worst model)")
    axs[1].set_xticks(x, s["scenario"].str.replace("_", " ").str.slice(0, 26), rotation=40, fontsize=6.5, ha="right")
    axs[1].axhline(0, color=INK2, lw=0.6)
    axs[1].legend(fontsize=7)
    axs[1].set_title("Expected excess and tail risk of each scenario target", fontsize=9)
    _src(axs[0], "Source: outputs/regime_v2r1_scenario_scorecard.csv", y=-0.45)
    _save(fig, ctx, "fig_regime_v2r1_15_scenario_scorecard.png")


def f16_liquidity(ctx):
    q = _read(ctx, "regime_v2r1_liquidity_path.csv")
    if q is None:
        return _na(ctx, "fig_regime_v2r1_16_liquidity.png", "liquidity, settlement and margin")
    q["date"] = pd.to_datetime(q["date"])
    fig, axs = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for n in ("ADAPTIVE_REGIME_PIF_V2R1", "STATIC_INVESTED_PIF_V2R1", "HEDGE_KEEP_SMLT_50"):
        g = q[q["comparator"] == n]
        if len(g):
            axs[0].plot(g["date"], g["available_nav"] * 100, color=POL[n], lw=1, label=n)
            axs[1].plot(g["date"], g["posted_initial_margin"] / g["nav"] * 100, color=POL[n], lw=1, label=f"{n}: initial margin")
            axs[1].plot(g["date"], g["pending_settlement_net"] / g["nav"] * 100, color=POL[n], lw=0.7, ls=":", label=f"{n}: pending settlement")
    axs[0].axhline(5, color=SERIES[7], ls="--", lw=0.8, label="5% hard minimum")
    axs[0].set_ylabel("immediately available, % NAV")
    axs[1].set_ylabel("% NAV")
    axs[0].legend(fontsize=7)
    axs[1].legend(fontsize=6.5, ncol=2)
    axs[0].set_title("Liquidity, settlement and margin on the replay ledgers", fontsize=10)
    _src(axs[1], "Source: outputs/regime_v2r1_liquidity_path.csv", y=-0.2)
    _save(fig, ctx, "fig_regime_v2r1_16_liquidity.png")


def f17_btc(ctx):
    f = _read(ctx, "regime_v2r1_frontier_15.csv")
    n = _nav(ctx)
    if f is None:
        return _na(ctx, "fig_regime_v2r1_17_btc_incremental.png", "BTC incremental value")
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for v, c in (("HEDGED_INVESTED_PIF", SERIES[0]), ("HEDGED_INVESTED_PIF|WITHOUT_BTC", SERIES[1])):
        g = f[(f["variant"] == v) & (f["ADMISSION"] == "PASS") & (f["source"] != "CURRENT_BOOK")]
        axs[0].scatter(g["engine_ES95_worst_nav0"] * 100, g["expected_net_excess_nav0"] * 100, color=c, s=24, label=v)
    axs[0].set_xlabel("engine ES95, % NAV0")
    axs[0].set_ylabel("expected net excess, % NAV0")
    axs[0].set_title("Admitted 15% frontier with and without BTC_RUB_EXPERIMENTAL", fontsize=9)
    axs[0].legend(fontsize=7)
    wf = _read(ctx, "regime_v2r1_walkforward.csv")
    a = _alloc(ctx)
    if a is not None:
        for v, c in (("ADAPTIVE_REGIME_PIF_V2R1", SERIES[0]), ("ADAPTIVE_REGIME_PIF_V2R1|WITHOUT_BTC", SERIES[1])):
            g = a[a["comparator"] == v]
            if len(g) and "BTC_RUB_EXPERIMENTAL" in g:
                axs[1].plot(g["date"], g["BTC_RUB_EXPERIMENTAL"] * 100, color=c, lw=1, label=f"{v}: BTC weight")
    if wf is not None:
        txt = []
        for v in ("ADAPTIVE_REGIME_PIF_V2R1", "ADAPTIVE_REGIME_PIF_V2R1|WITHOUT_BTC"):
            r = wf[wf["comparator"] == v]
            if len(r):
                txt.append(f"{v}: net excess {r['net_excess_over_money_market'].iloc[0] * 100:+.1f}%")
        axs[1].text(0.02, 0.95, "\n".join(txt), transform=axs[1].transAxes, fontsize=7.5, va="top")
    axs[1].set_ylabel("% NAV")
    axs[1].set_title("Replay: BTC weight of the adaptive ledger", fontsize=9)
    axs[1].legend(fontsize=7, loc="lower right")
    _src(axs[0], "Source: outputs/regime_v2r1_frontier_15.csv, regime_v2r1_allocation_path.csv, regime_v2r1_walkforward.csv")
    _save(fig, ctx, "fig_regime_v2r1_17_btc_incremental.png")


def f18_by_regime(ctx):
    lib = _read(ctx, "regime_v2r1_state_target_library.csv")
    if lib is None:
        return _na(ctx, "fig_regime_v2r1_18_allocation_by_regime.png", "allocation by regime")
    routes = ["OFZ_FUNDED", "EQ_FUNDED", "GOLD_FUNDED", "CNY_PERP"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (col, states) in zip(axs, (("market_state", ["NORMAL", "STRESS"]), ("rates_state", ["EASING", "NEUTRAL", "TIGHTENING"]),
                                        ("rub_state", ["RUB_STRENGTHENING", "RUB_STABLE", "RUB_WEAKENING"]))):
        x = np.arange(len(states))
        for k, r in enumerate(routes):
            c = f"w_tgt_{r}"
            v = [pd.to_numeric(lib.loc[lib[col] == s, c], errors="coerce").fillna(0).mean() * 100 if c in lib else 0.0 for s in states]
            ax.bar(x + (k - 1.5) * 0.2, v, width=0.2, color=COL[r], label=SHORT[r])
        ax.set_xticks(x, states, fontsize=7)
        ax.set_title(f"mean target weight by {col.replace('_', ' ')}\n(averaged over the other axes of the library)", fontsize=8)
    axs[0].set_ylabel("% NAV")
    axs[0].legend(fontsize=7)
    _src(axs[0], "Source: outputs/regime_v2r1_state_target_library.csv (SCENARIO_ONLY)", y=-0.2)
    _save(fig, ctx, "fig_regime_v2r1_18_allocation_by_regime.png")


ALL = [f01_state_vector, f02_frontier, f03_mandates, f04_allocation, f05_state_allocations, f06_correlations, f07_hedge_ratios,
       f08_smlt_exposure, f09_allocation_time, f10_overlays, f11_nav, f12_drawdown, f13_risk_contribution, f14_forward, f15_scorecard,
       f16_liquidity, f17_btc, f18_by_regime]
