"""Redesign figures (fig_adaptive_*.png), generated from outputs/adaptive_* only; legacy visual conventions."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..reporting.report import INK2, SERIES, _src, plt

COL = {"SMLT": SERIES[0], "SMLT_FUT": SERIES[7], "EQ_FUNDED": SERIES[1], "EQ_HEDGE_FUT": "#f2a47f", "OFZ_FUNDED": SERIES[2],
       "RATES_HEDGE_FUT": "#8fd8bd", "GOLD_FUNDED": SERIES[3], "CNY_PERP": SERIES[4], "OIL_FUT": SERIES[5],
       "BTC_RUB_EXPERIMENTAL": SERIES[6], "MMF": "#b8b6b0", "CASH": "#6b6a66"}
ROUTES = ["SMLT", "SMLT_FUT", "EQ_FUNDED", "EQ_HEDGE_FUT", "OFZ_FUNDED", "RATES_HEDGE_FUT", "GOLD_FUNDED", "CNY_PERP", "OIL_FUT",
          "BTC_RUB_EXPERIMENTAL"]


def _save(fig, ctx, name):
    assert name.startswith("fig_adaptive_")
    fig.tight_layout()
    fig.savefig(ctx.fig / name, dpi=200)
    plt.close(fig)


def _bars(ax, d, x, cols, prefix):
    pos = np.zeros(len(d))
    neg = np.zeros(len(d))
    for r in cols:
        c = f"{prefix}{r}"
        if c not in d:
            continue
        v = d[c].fillna(0).values * 100
        ax.bar(x, np.where(v > 0, v, 0), bottom=pos, color=COL[r], width=0.7, label=r if (np.abs(v) > 1e-9).any() else None)
        ax.bar(x, np.where(v < 0, v, 0), bottom=neg, color=COL[r], width=0.7)
        pos += np.where(v > 0, v, 0)
        neg += np.where(v < 0, v, 0)
    ax.axhline(0, color=INK2, lw=0.6)


def f01_balance_sheet(ctx):
    obj = json.loads((ctx.out / "adaptive_decision_object.json").read_text())
    d = pd.read_csv(ctx.out / "adaptive_current_portfolio.csv")
    nav = obj["CURRENT_NAV"]
    smlt = float(d.loc[d["instrument"] == "SMLT", "current_weight_nav"].iloc[0]) * nav
    mmf = float(d.loc[d["instrument"] == "MMF", "current_weight_nav"].iloc[0]) * nav
    items = [("SMLT shares", smlt, COL["SMLT"]), ("immediate RUB cash", obj["IMMEDIATE_CASH"], COL["CASH"]),
             ("money-market fund units", mmf, COL["MMF"]), ("posted margin", obj["POSTED_MARGIN"], SERIES[7]),
             ("pending settlement", obj["PENDING_SETTLEMENT"], SERIES[3])]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(items))
    ax.bar(x, [v / 1e6 for _, v, _ in items], color=[c for _, _, c in items], width=0.6)
    for xi, (_, v, _) in zip(x, items):
        ax.text(xi, v / 1e6 + 0.15, f"{v / 1e6:.1f}m\n{v / nav:.0%}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, [n for n, _, _ in items], fontsize=8)
    ax.set_ylim(0, max(v for _, v, _ in items) / 1e6 * 1.25)
    ax.set_ylabel("RUB m")
    adm = "admitted" if obj["CURRENT_BOOK_ADMISSION"]["admitted"] else "NOT admitted"
    ax.set_title(f"Fund balance sheet at {obj['DATE']}: NAV {nav / 1e6:.1f}m; current book {adm} at the 15% budget", fontsize=10)
    _src(ax, "Source: outputs/adaptive_decision_object.json, adaptive_current_portfolio.csv", y=-0.16)
    _save(fig, ctx, "fig_adaptive_01_balance_sheet.png")


def f02_frontier15(ctx):
    d = pd.read_csv(ctx.out / "adaptive_frontier_15.csv")
    d = d[(d["universe"] == "MAIN") & (d["mandate"] == "BROAD_PIF_NO_HIDDEN_LEVERAGE") & d["engine_ES95_worst_nav0"].notna()]
    fig, ax = plt.subplots(figsize=(9, 5))
    sty = {1.0: (SERIES[0], "o", 70), 0.5: (SERIES[1], "x", 50), 0.0: (SERIES[2], "o", 28)}
    for k in (1.0, 0.5, 0.0):
        g = d[d["kappa"] == k].sort_values("engine_ES95_worst_nav0")
        if g.empty:
            continue
        c, mk, sz = sty[k]
        ok = g["on_frontier"].astype(bool)
        y = (g["robust_excess_exec_nav0"] if k > 0 else g["nominal_excess_exec_nav0"]) * 100
        role = "canonical, worst case" if k == 1.0 else ("worst case" if k > 0 else "sensitivity, centre of the view set")
        ax.plot(g["engine_ES95_worst_nav0"] * 100, y, lw=1.0, color=c, alpha=0.5)
        if ok.any():
            ax.scatter(g.loc[ok, "engine_ES95_worst_nav0"] * 100, y[ok], color=c, marker=mk, s=sz, label=f"kappa {k:g} ({role}): admitted")
        if (~ok).any():
            ax.scatter(g.loc[~ok, "engine_ES95_worst_nav0"] * 100, y[~ok], facecolors="none", edgecolors=c, s=sz,
                       label=f"kappa {k:g}: rejected by the engine")
    ax.axvline(15, color=INK2, lw=0.8, ls="--")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlabel("full-engine worst-model ES95 (6m), % NAV0")
    ax.set_ylabel("net excess over the money market (6m), % NAV0")
    ax.set_title("15% risk budget: robust frontier, engine-admitted points", fontsize=10)
    ax.legend(fontsize=7, loc="center right")
    _src(ax, "Source: outputs/adaptive_frontier_15.csv. kappa 1 and 0.5: every ES target returns the money-market portfolio (one point).", y=-0.14)
    _save(fig, ctx, "fig_adaptive_02_frontier_15.png")


def f03_budgets(ctx):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True)
    for ax, b in zip(axs, (10, 15, 20)):
        d = pd.read_csv(ctx.out / f"adaptive_frontier_{b}.csv")
        d = d[(d["universe"] == "MAIN") & (d["mandate"] == "BROAD_PIF_NO_HIDDEN_LEVERAGE") & d["on_frontier"].astype(bool)]
        for k, c, mk in ((1.0, SERIES[0], "o"), (0.5, SERIES[1], "x"), (0.0, SERIES[2], "o")):
            g = d[d["kappa"] == k].sort_values("engine_ES95_worst_nav0")
            if len(g):
                ax.plot(g["engine_ES95_worst_nav0"] * 100, g["nominal_excess_exec_nav0"] * 100, marker=mk, ms=5 if k else 3, lw=1.2,
                        color=c, label=f"kappa {k:g}")
        ax.axvline(b, color=INK2, lw=0.8, ls="--")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(f"budget {b}% (its own frontier)", fontsize=9)
        ax.set_xlabel("engine ES95 (6m), % NAV0")
    axs[0].set_ylabel("centre net excess over the money market (6m), % NAV0")
    axs[0].legend(fontsize=7)
    fig.text(0.01, 0.01, "Source: outputs/adaptive_frontier_{10,15,20}.csv (admitted points only; budgets are never pooled)", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(ctx.fig / "fig_adaptive_03_budget_comparison.png", dpi=200)
    plt.close(fig)


def f04_current_allocation(ctx):
    obj = json.loads((ctx.out / "adaptive_decision_object.json").read_text())
    cur = pd.read_csv(ctx.out / "adaptive_current_portfolio.csv").set_index("instrument")["current_weight_nav"].to_dict()
    rows = [("current book", cur), ("canonical target (kappa 1)", obj["TARGET_WEIGHTS_NAV"] or {}),
            ("sensitivity (kappa 0)", obj["SENSITIVITY_NOMINAL_KAPPA0"]["target_weights"] or {})]
    d = pd.DataFrame([{"name": n, **{f"w_{r}": float(w.get(r, 0.0) or 0.0) for r in ROUTES}} for n, w in rows])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    _bars(ax, d, np.arange(len(d)), ROUTES, "w_")
    for i, (_, w) in enumerate(rows):
        if sum(abs(float(w.get(r, 0.0) or 0.0)) for r in ROUTES) < 1e-9:
            ax.text(i, 1.5, "no risky route:\nmoney market + reserve", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(np.arange(len(d)), d["name"])
    ax.set_ylabel("% NAV (remainder: money market + reserve)")
    st = obj["OBSERVABLE_RISK_STATE"]["confirmed"]
    ax.set_title(f"{obj['DATE']}: action {obj['SELECTED_ACTION']} in the confirmed state "
                 f"{st['ISSUER_CREDIT_RISK']} / {st['MARKET_RISK']} / {st['RATES_STATE']}", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    _src(ax, "Source: outputs/adaptive_decision_object.json", y=-0.12)
    _save(fig, ctx, "fig_adaptive_04_current_state_allocation.png")


def f05_state_allocations(ctx):
    d = pd.read_csv(ctx.out / "adaptive_policy_library.csv")
    fig, axs = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    for ax, k in zip(axs, (1.0, 0.0)):
        g = d[d["kappa"] == k].reset_index(drop=True)
        _bars(ax, g, np.arange(len(g)), ROUTES, "w_tgt_")
        ax.set_ylabel("% NAV")
        acts = ", ".join(sorted(g["action_from_actual_book"].unique()))
        ax.set_title(f"kappa {k:g} ({'canonical' if k == 1.0 else 'sensitivity'}): target from the actual book in each confirmed state "
                     f"(actions: {acts})", fontsize=9)
        wc = [c for c in g if c.startswith("w_tgt_")]
        if float(g[wc].abs().fillna(0).values.sum()) < 1e-9:
            ax.text(0.5, 0.5, "no risky route in any state: money market + reserve", transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color=INK2)
            ax.set_ylim(0, 10)
        else:
            ax.legend(fontsize=7, ncol=5)
        ax.set_xticks(np.arange(len(g)), [f"{a[:5]}|{b[:4]}|{c[:4]}" for a, b, c in zip(g["issuer_state"], g["market_state"], g["rates_state"])], rotation=70, fontsize=6.5)
    _src(axs[1], "Source: outputs/adaptive_policy_library.csv (issuer | market | rates; SEVERE rows are hypothetical)", y=-0.42)
    _save(fig, ctx, "fig_adaptive_05_state_allocations.png")


def f06_allocation_path(ctx):
    d = pd.read_csv(ctx.out / "adaptive_allocation_path.csv", parse_dates=["date"])
    names = [n for n in ("ADAPTIVE_PIF_V1", "ADAPTIVE_PIF_V1|kappa0") if n in set(d["comparator"])]
    fig, axs = plt.subplots(len(names), 1, figsize=(11, 3.6 * len(names)), sharex=True, squeeze=False)
    for ax, n in zip(axs[:, 0], names):
        g = d[d["comparator"] == n].set_index("date")
        cols = [c for c in ROUTES + ["MMF"] if c in g]
        pos = g[cols].clip(lower=0) * 100
        ax.stackplot(pos.index, pos.T.values, colors=[COL[c] for c in cols], labels=cols, alpha=0.9)
        neg = g[[c for c in cols if (g[c] < 0).any()]].clip(upper=0) * 100
        if len(neg.columns):
            ax.stackplot(neg.index, neg.T.values, colors=[COL[c] for c in neg.columns], alpha=0.9)
        ax.set_ylabel("% NAV")
        ax.set_title(f"{n}: actual one-ledger allocation path")
    axs[0, 0].legend(fontsize=6.5, ncol=6, loc="upper right")
    _src(axs[-1, 0], "Source: outputs/adaptive_allocation_path.csv (remainder = immediate cash, margin, pending settlements)", y=-0.2)
    _save(fig, ctx, "fig_adaptive_06_allocation_path.png")


def _nav(ctx):
    return pd.read_csv(ctx.out / "adaptive_nav_paths.csv", parse_dates=["date"]).set_index("date")


def f07_nav(ctx):
    d = _nav(ctx) / 1e6
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for i, c in enumerate(d.columns):
        ax.plot(d.index, d[c], lw=1.6 if c.startswith("ADAPTIVE") else 1.0, color=SERIES[i % 8], ls="-" if i < 8 else "--", label=c)
    ax.set_ylabel("NAV, RUB m")
    ax.set_title(f"One-ledger retrospective replay {d.index[0].date()} -> {d.index[-1].date()} (one history; not market out-of-sample)", fontsize=10)
    ax.legend(fontsize=6.5, ncol=2)
    _src(ax, "Source: outputs/adaptive_nav_paths.csv", y=-0.1)
    _save(fig, ctx, "fig_adaptive_07_nav_paths.png")


def f08_drawdown(ctx):
    d = _nav(ctx)
    dd = (d / d.cummax() - 1) * 100
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i, c in enumerate(dd.columns):
        ax.plot(dd.index, dd[c], lw=1.4 if c.startswith("ADAPTIVE") else 0.9, color=SERIES[i % 8], ls="-" if i < 8 else "--", label=c)
    ax.set_ylabel("drawdown, %")
    ax.set_title("Drawdown paths (mark-to-market NAV)")
    ax.legend(fontsize=6.5, ncol=2)
    _src(ax, "Source: outputs/adaptive_nav_paths.csv", y=-0.12)
    _save(fig, ctx, "fig_adaptive_08_drawdown.png")


def f09_smlt_exposure(ctx):
    d = pd.read_csv(ctx.out / "adaptive_allocation_path.csv", parse_dates=["date"])
    st = pd.read_csv(ctx.out / "adaptive_state_confirmed_daily.csv", parse_dates=["date"]).set_index("date")
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i, (n, g) in enumerate(d.groupby("comparator")):
        g = g.set_index("date")
        ax.plot(g.index, (g.get("SMLT", 0) + g.get("SMLT_FUT", 0)) * 100, lw=1.3, color=SERIES[i % 8], label=n)
    s = st["confirmed_ISSUER_CREDIT_RISK"].reindex(d["date"].unique())
    for lab, c in (("DISTRESSED", "#e34948"), ("DETERIORATING", "#eda100")):
        m = (s == lab).values
        ax.fill_between(s.index, 0, 1, where=m, transform=ax.get_xaxis_transform(), color=c, alpha=0.10, lw=0, label=f"confirmed {lab}")
    ax.set_ylabel("net SMLT issuer exposure, % NAV")
    ax.set_title("SMLT exposure through time (shares + futures) and confirmed issuer state")
    ax.legend(fontsize=6.5, ncol=2)
    _src(ax, "Source: outputs/adaptive_allocation_path.csv, adaptive_state_confirmed_daily.csv", y=-0.12)
    _save(fig, ctx, "fig_adaptive_09_smlt_exposure.png")


def f10_risk_contribution(ctx):
    d = pd.read_csv(ctx.out / "adaptive_risk_contribution.csv")
    d = d[d["book"] != "V41_ADVERSE_TARGET_WEIGHTS"]            # margin-squeeze test book: kept in the CSV, off the chart scale
    fig, ax = plt.subplots(figsize=(10, 4.6))
    books = list(dict.fromkeys(d["book"]))
    x = np.arange(len(books))
    bot_p, bot_n = np.zeros(len(books)), np.zeros(len(books))
    for r in [c for c in ROUTES + ["MMF_AND_RESERVE"] if c in set(d["sleeve"])]:
        v = np.array([float(d[(d["book"] == b) & (d["sleeve"] == r)]["es_contribution_nav0"].sum()) for b in books]) * 100
        ax.bar(x, np.where(v > 0, v, 0), bottom=bot_p, color=COL.get(r, "#b8b6b0"), width=0.6, label=r if (np.abs(v) > 1e-9).any() else None)
        ax.bar(x, np.where(v < 0, v, 0), bottom=bot_n, color=COL.get(r, "#b8b6b0"), width=0.6)
        bot_p += np.where(v > 0, v, 0)
        bot_n += np.where(v < 0, v, 0)
    tot = np.array([float(d[d["book"] == b]["book_ES95_static_nav0"].iloc[0]) for b in books]) * 100
    ax.scatter(x, tot, color=INK2, marker="D", s=22, zorder=5, label="book ES95 (sum)")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(x, books, rotation=15, fontsize=7)
    ax.set_ylabel("contribution to ES95 (6m), % NAV0")
    ax.set_title("Sleeve contributions to the tail loss (static horizon P&L on the worst model's tail paths)", fontsize=10)
    ax.legend(fontsize=7, ncol=3)
    _src(ax, "Source: outputs/adaptive_risk_contribution.csv (negative bars reduce the tail loss: money-market carry, hedges)", y=-0.24)
    _save(fig, ctx, "fig_adaptive_10_risk_contribution.png")


def f11_liquidity(ctx):
    d = pd.read_csv(ctx.out / "adaptive_liquidity_path.csv", parse_dates=["date"])
    names = list(dict.fromkeys(d["comparator"]))[:4]
    fig, axs = plt.subplots(len(names), 1, figsize=(11, 2.6 * len(names)), sharex=True, squeeze=False)
    for ax, n in zip(axs[:, 0], names):
        g = d[d["comparator"] == n].set_index("date")
        pay = g["pending_settlement_net"].clip(upper=0)          # cash already committed to unsettled purchases
        ax.stackplot(g.index, (g["immediate_cash"] - g["posted_initial_margin"] + pay).clip(lower=0) / g["nav"] * 100,
                     g["posted_initial_margin"] / g["nav"] * 100, g["mmf_value"] / g["nav"] * 100, g["pending_settlement_net"].clip(lower=0) / g["nav"] * 100,
                     colors=[COL["CASH"], SERIES[7], COL["MMF"], SERIES[3]],
                     labels=["immediately available cash", "posted margin", "MMF units", "pending receivables"])
        ax.axhline(5, color="#e34948", lw=0.8, ls="--")
        ax.set_ylabel("% NAV")
        ax.set_title(f"{n}: cash / margin / MMF / settlement (5% hard minimum dashed; breach days: {int(g['breach'].sum())})", fontsize=8.5)
    axs[0, 0].legend(fontsize=6.5, ncol=4)
    _src(axs[-1, 0], "Source: outputs/adaptive_liquidity_path.csv", y=-0.3)
    _save(fig, ctx, "fig_adaptive_11_liquidity_path.png")


def f12_margin_squeeze(ctx):
    d = pd.read_csv(ctx.out / "adaptive_margin_stress.csv")
    g = d[(d["section"] == "DETERMINISTIC_PATH_STRESS") & (d["scenario"] == "S02_SMLT_RALLY_THEN_CREDIT_GAP")]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, col, ttl in ((axs[0], "pnl_nav0", "terminal P&L, % NAV0"), (axs[1], "min_available_nav", "minimum available liquidity, % NAV")):
        p = g.pivot(index="book", columns="margin_multiplier", values=col) * 100
        p.plot.barh(ax=ax, color=[SERIES[0], SERIES[7]], width=0.7)
        ax.set_title(ttl, fontsize=9)
        ax.set_ylabel("")
        ax.legend(title="margin x", fontsize=7)
    axs[1].axvline(5, color="#e34948", lw=0.8, ls="--")
    fig.suptitle("Margin-squeeze survival: SMLT +70% rally, then -60% gap (deterministic path, no probability)", fontsize=10)
    _src(axs[0], "Source: outputs/adaptive_margin_stress.csv", y=-0.14)
    _save(fig, ctx, "fig_adaptive_12_margin_squeeze.png")


def f13_forward_state(ctx):
    d = pd.read_csv(ctx.out / "adaptive_state_forward_stats.csv")
    g = d[(d["horizon"] == 126) & (d["dimension"] == "ISSUER_CREDIT_RISK") & d["mean_fwd_excess"].notna()]
    routes = ["SMLT", "EQ_FUNDED", "OFZ_FUNDED", "GOLD_FUNDED", "BTC_RUB_EXPERIMENTAL"]
    states = [s for s in ("IMPROVING", "STABLE", "DETERIORATING", "DISTRESSED") if s in set(g["state"])]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    w = 0.8 / len(routes)
    for i, r in enumerate(routes):
        h = g[g["route"] == r].set_index("state").reindex(states)
        x = np.arange(len(states)) + i * w
        ax.bar(x, h["mean_fwd_excess"] * 100, width=w, color=COL[r], label=r)
        ax.errorbar(x, h["mean_fwd_excess"] * 100, yerr=[(h["mean_fwd_excess"] - h["ci90_lo_episode_bootstrap"]).clip(lower=0) * 100,
                                                          (h["ci90_hi_episode_bootstrap"] - h["mean_fwd_excess"]).clip(lower=0) * 100],
                    fmt="none", ecolor=INK2, lw=0.8, capsize=2)
        ax.hlines(h["all_origin_mean"] * 100, x - w / 2, x + w / 2, colors="black", lw=1.0, linestyles=":")
    n = g[g["route"] == "SMLT"].set_index("state").reindex(states)["n_nonoverlapping_windows"]
    ax.set_xticks(np.arange(len(states)) + 0.4 - w / 2, [f"{s}\n({int(v) if v == v else 0} non-overlapping windows)" for s, v in zip(states, n)])
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_ylabel("126-session forward excess over the money market, %")
    ax.set_title("Forward excess AFTER the issuer-credit state at the origin\n(bars: state mean; whiskers: 90% episode bootstrap; dotted: all-origin mean)", fontsize=10)
    ax.legend(fontsize=7, ncol=5)
    _src(ax, "Source: outputs/adaptive_state_forward_stats.csv (overlapping windows are not independent)", y=-0.2)
    _save(fig, ctx, "fig_adaptive_13_forward_after_state.png")


def f14_btc(ctx):
    d = pd.read_csv(ctx.out / "adaptive_btc_incremental_value.csv")
    g = d[d["section"] == "FRONTIER"]
    order = [u for u in ("WITHOUT_BTC", "MAIN", "MAIN_WITH_BTC_2019_2021_PRIOR_SENSITIVITY") if u in set(g["universe"])]
    cols = {"WITHOUT_BTC": "#b8b6b0", "MAIN": SERIES[6], "MAIN_WITH_BTC_2019_2021_PRIOR_SENSITIVITY": SERIES[3]}
    names = {"WITHOUT_BTC": "without BTC", "MAIN": "with BTC (canonical view set)",
             "MAIN_WITH_BTC_2019_2021_PRIOR_SENSITIVITY": "with BTC + excluded 2019-2021 prior (labelled sensitivity)"}
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, col, ttl in ((axs[0], "best_nominal_excess_nav0", "best admitted point: centre net excess, % NAV0"),
                         (axs[1], "max_btc_weight_on_frontier", "largest BTC weight on the admitted frontier, % NAV")):
        p = g.pivot_table(index=["budget", "kappa"], columns="universe", values=col)[order] * 100
        p.columns = [names[c] for c in p.columns]
        p.index = [f"{int(round(b * 100))}% | k{k:g}" for b, k in p.index]
        p.plot.bar(ax=ax, color=[cols[u] for u in order], width=0.8)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("risk budget | kappa")
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.legend(fontsize=6.5)
    fig.suptitle("BTC_RUB_EXPERIMENTAL: frontier with and without BTC (same paths, seeds and bounds)", fontsize=10)
    fig.text(0.01, 0.01, "Source: outputs/adaptive_btc_incremental_value.csv", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(ctx.fig / "fig_adaptive_14_btc_incremental.png", dpi=200)
    plt.close(fig)


ALL = [f01_balance_sheet, f02_frontier15, f03_budgets, f04_current_allocation, f05_state_allocations, f06_allocation_path, f07_nav,
       f08_drawdown, f09_smlt_exposure, f10_risk_contribution, f11_liquidity, f12_margin_squeeze, f13_forward_state, f14_btc]
