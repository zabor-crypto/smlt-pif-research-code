"""B5/B6/B11: static and robust multi-asset "smart" overlays under explicit capital accounting.

Candidates
- MULTI: for every non-empty subset of the Phase-B sleeves, every margin budget and every upside-capture floor,
  the min-ES95 Rockafellar-Uryasev LP is solved on FHS horizon scenarios (costs, margin <= budget*C, gross cap).
- DIRECT_ONLY: short SMLT futures notional grid 0..100% of shares (same accounting), the mandatory benchmark.
Each distinct candidate is re-evaluated path-by-path with the self-financing engine on FOUR scenario models:
  HS_FULL, FHS_FULL, HS_RECENT250 (block bootstrap, risk-neutral drift) and HIST_WINDOWS (every overlapping
  historical horizon window, drift-neutralised; deterministic, no sampling).
Selection is LEXICOGRAPHIC with pre-declared constraints (configs/project.yaml: smart_portfolio.selection):
  STATIC : min ES95(FHS) s.t. breach(FHS) <= b_max and upside(FHS) >= u_min
  ROBUST : min max_m ES95_m s.t. max_m breach_m <= b_max and min_m upside_m >= u_min   (min-max over models)
No weighted score is used anywhere.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from ..optimization.overlay import Problem, Sleeve, min_es
from ..scenarios.bootstrap import SleevePos, evaluate_portfolio, horizon_returns
from ..strategies.collateral import CASH_ONLY, Collateral

MODELS = ["HS_FULL", "FHS_FULL", "HS_RECENT250", "HIST_WINDOWS"]


def hist_windows(Xn: pd.DataFrame, cols: list[str], hz: int) -> dict:
    """Every overlapping horizon window of (drift-neutralised) historical joint returns as a deterministic path set."""
    X = Xn[cols].values
    n = len(X) - hz + 1
    idx = np.arange(n)[:, None] + np.arange(hz)[None, :]
    return {c: X[:, j][idx] for j, c in enumerate(cols)} | {"_idx": idx, "_dates": Xn.index}


def scenario_models(ctx, pl, specs, hz: int) -> tuple[dict, dict]:
    cols = ["SMLT"] + [s.series for s in specs]
    B = pl._bootstrap_inputs(ctx, cols, futures_like=tuple(s.series for s in specs))
    n_eval = ctx.cfg["smart_portfolio"]["eval_paths"]
    models = {v: pl._paths_for(ctx, B, v, hz, n_paths=n_eval) for v in ("HS_FULL", "FHS_FULL", "HS_RECENT250")}
    models["HIST_WINDOWS"] = hist_windows(B["variants"]["HS_FULL"], cols, hz)
    return models, B


def _problem(HR: pd.DataFrame, specs, tau: float, E: float, C: float, mmf_h: float, coll: Collateral,
             mb: float, u, gross: float, only: tuple | None = None) -> Problem:
    use = [s for s in specs if only is None or s.name in only]
    sl = [Sleeve(s.name, s.lo, s.hi, cost=s.horizon_cost(tau), margin=s.mr * coll.margin_multiplier * coll.capital_per_margin(),
                 group=s.name) for s in use]
    return Problem(r_s=HR["SMLT"].values, R=HR[[s.series for s in use]].values, sleeves=sl, k_liq=C / E,
                   m_liq=np.full(len(HR), mmf_h), alpha=0.95, margin_budget=mb * C / E, gross_cap=gross,
                   min_upside_capture=u)


def generate_candidates(ctx, specs, models, B, hz: int, C: float, coll: Collateral = CASH_ONLY) -> list[dict]:
    sp = ctx.cfg["smart_portfolio"]
    E, tau = ctx.E, hz / 252
    HR = horizon_returns(models["FHS_FULL"], ["SMLT"] + [s.series for s in specs]).head(sp["lp_scenarios"])
    mmf_h = (1 + B["y"]) ** tau - 1
    names = [s.name for s in specs]
    seen, cands = set(), []

    def add(h: dict, family: str, meta: dict):
        key = (family, tuple(round(h.get(n, 0.0), 3) for n in names))
        if key in seen:
            return
        seen.add(key)
        cands.append({"family": family, "h": {n: float(h.get(n, 0.0)) for n in names}, **meta})

    g = sp["margin_budget_grid"]
    mbs = np.round(np.arange(g["start"], g["stop"] + 1e-9, g["step"]), 4) if isinstance(g, dict) else g
    for k in range(1, len(names) + 1):
        for sub in combinations(names, k):
            for mb in mbs:
                for u in (sp["upside_floor_grid"] if k == len(names) else [None]):
                    sol = min_es(_problem(HR, specs, tau, E, C, mmf_h, coll, mb, u, sp["gross_cap"], only=sub))
                    if not sol.status.startswith("optimal"):
                        continue
                    add(dict(zip(sub, sol.h)), "MULTI", {"subset": "|".join(sub), "margin_budget": mb, "upside_floor": u,
                                                         "lp_es95": sol.es})
    direct = next(s for s in specs if s.name == "DIRECT")
    S0, F0 = B["S0"], B["F0"]
    for hs in np.arange(0.0, 1.0 + 1e-9, sp["direct_only_h_grid_step"]):
        notional_frac = round(hs * E / S0) * F0 / E            # exact contract notional (share-unit ratio hs)
        add({"DIRECT": -notional_frac}, "DIRECT_ONLY", {"subset": "DIRECT", "margin_budget": np.nan, "upside_floor": np.nan,
                                                       "lp_es95": np.nan, "share_hedge_ratio": hs})
    return cands


def evaluate_candidates(ctx, specs, models, B, cands, hz: int, C: float, coll: Collateral = CASH_ONLY) -> pd.DataFrame:
    E, tau = ctx.E, hz / 252
    rows = []
    for i, c in enumerate(cands):
        sl = [SleevePos(s.series, c["h"][s.name] * E, s.mr, s.cost_one_way, s.n_rolls(tau), "future")
              for s in specs if abs(c["h"][s.name]) > 1e-6]
        base = {"cand_id": f"{c['family']}_{hz}_{i:04d}", "family": c["family"], "horizon_days": hz, "capital": C,
                "collateral": coll.name, "subset": c["subset"], "margin_budget": c["margin_budget"],
                "upside_floor": c["upside_floor"], "lp_es95_pct": c["lp_es95"], "share_hedge_ratio": c.get("share_hedge_ratio", np.nan),
                "K_active": sum(1 for s in specs if abs(c["h"][s.name]) > 1e-4),
                **{f"w_{s.name}": c["h"][s.name] for s in specs}}
        for mname, Pth in models.items():
            m = evaluate_portfolio(Pth, E=E, C=C, sleeves=sl, mmf_daily=B["daily_y"], collateral=coll)
            rows.append({**base, "model": mname, **m})
    return pd.DataFrame(rows)


def summarise(ev: pd.DataFrame, control: pd.DataFrame) -> pd.DataFrame:
    """One row per candidate: FHS metrics + worst-case across models."""
    piv = ev.pivot_table(index="cand_id", columns="model", values=["ES95", "p_sleeve_breach", "upside_capture", "mean_pnl", "p05_pnl"],
                         aggfunc="first")
    s = ev[ev["model"] == "FHS_FULL"].set_index("cand_id").copy()
    s["worst_model_ES95"] = piv["ES95"].max(axis=1)
    s["worst_model_breach"] = piv["p_sleeve_breach"].max(axis=1)
    s["worst_model_upside"] = piv["upside_capture"].min(axis=1)
    s["worst_model_mean_pnl"] = piv["mean_pnl"].min(axis=1)
    for m in MODELS:
        s[f"ES95_{m}"] = piv["ES95"][m]
        s[f"breach_{m}"] = piv["p_sleeve_breach"][m]
    hist = ev[ev["model"] == "HIST_WINDOWS"].set_index("cand_id")
    s["worst_hist_window_pnl"] = hist["p05_pnl"].reindex(s.index)  # 5th pct of all historical windows
    ctrl = control.set_index("model")
    s["hedge_cost_fhs"] = float(ctrl.loc["FHS_FULL", "mean_pnl"]) - s["mean_pnl"]
    return s.reset_index()


def select(summary: pd.DataFrame, rule: str, cfg: dict, family: str | None = None) -> pd.Series | None:
    d = summary if family is None else summary[summary["family"] == family]
    if rule == "static":
        r = cfg["static"]
        ok = d[(d["p_sleeve_breach"] <= r["max_breach"]) & (d["upside_capture"] >= r["min_upside"])]
        key = "ES95"
    elif rule == "robust":
        r = cfg["robust"]
        ok = d[(d["worst_model_breach"] <= r["max_worst_breach"]) & (d["worst_model_upside"] >= r["min_worst_upside"])]
        key = "worst_model_ES95"
    else:
        raise ValueError(rule)
    if ok.empty:
        return None
    return ok.sort_values([key, "K_active", "initial_im"]).iloc[0]


def constrained_table(summary: pd.DataFrame) -> pd.DataFrame:
    """B5 list of constrained solutions for MULTI vs DIRECT_ONLY (same constraints)."""
    profiles = [("min ES", {}), ("min ES | upside>=40%", {"up": 0.4}), ("min ES | upside>=60%", {"up": 0.6}),
                ("min ES | breach<=5%", {"br": 0.05}), ("min ES | breach<=10%", {"br": 0.10}),
                ("min ES | breach<=20%", {"br": 0.20}), ("min ES | breach<=10% & upside>=40%", {"br": 0.10, "up": 0.4})]
    rows = []
    for fam in ("MULTI", "DIRECT_ONLY"):
        d = summary[summary["family"] == fam]
        for label, c in profiles:
            ok = d.copy()
            if "up" in c:
                ok = ok[ok["upside_capture"] >= c["up"]]
            if "br" in c:
                ok = ok[ok["p_sleeve_breach"] <= c["br"]]
            if ok.empty:
                rows.append({"profile": label, "family": fam, "feasible": False})
                continue
            r = ok.sort_values(["ES95", "K_active"]).iloc[0]
            rows.append({"profile": label, "family": fam, "feasible": True, **r.to_dict()})
    return pd.DataFrame(rows)


def complexity_multidim(summary: pd.DataFrame, cfg: dict, E: float) -> pd.DataFrame:
    """B11: best candidate per K under multi-dimensional constraints and Pareto test vs the best K=1 solution."""
    rows = []
    d = summary[summary["family"] == "MULTI"].copy()
    d = pd.concat([d, summary[summary["family"] == "DIRECT_ONLY"].assign(K_active=lambda x: x["K_active"].clip(lower=1))])
    r = cfg["static"]
    feas = d[(d["p_sleeve_breach"] <= r["max_breach"]) & (d["upside_capture"] >= r["min_upside"])]
    base = feas[feas["K_active"] <= 1].sort_values("ES95").iloc[0] if (feas["K_active"] <= 1).any() else None
    for K in range(1, 7):
        sub = feas[feas["K_active"] <= K]
        if sub.empty:
            continue
        b = sub.sort_values(["ES95", "K_active"]).iloc[0]
        # Pareto dominance vs best K=1 on (ES down, upside up, breach down, mean up, worst-model ES down)
        dom = None
        if base is not None:
            better_or_eq = (b["ES95"] <= base["ES95"] and b["upside_capture"] >= base["upside_capture"] - 1e-9 and
                            b["p_sleeve_breach"] <= base["p_sleeve_breach"] + 1e-9 and b["mean_pnl"] >= base["mean_pnl"] - 1e-9 and
                            b["worst_model_ES95"] <= base["worst_model_ES95"])
            strictly = (b["ES95"] < base["ES95"] - cfg["material_es_gain_pct_of_E"] * E)
            dom = bool(better_or_eq and strictly)
        rows.append({"K_max": K, "cand_id": b["cand_id"], "active": b["subset"] if b["family"] == "MULTI" else "DIRECT",
                     "K_active": int(b["K_active"]), "ES95_FHS": b["ES95"], "worst_model_ES95": b["worst_model_ES95"],
                     "upside_capture": b["upside_capture"], "breach_FHS": b["p_sleeve_breach"], "mean_pnl": b["mean_pnl"],
                     "initial_im": b["initial_im"], "transaction_costs": b["transaction_costs"],
                     "gross_notional": b["gross_derivatives_notional"],
                     "es_gain_vs_K1_rub": (base["ES95"] - b["ES95"]) if base is not None else np.nan,
                     "materially_pareto_dominates_K1": dom})
    return pd.DataFrame(rows)
