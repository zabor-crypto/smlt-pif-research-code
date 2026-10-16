"""Gate R3 outputs: return-view registry, robust frontiers per risk budget, zero-weight attribution, capital grid.

Frontier point = LP candidate at an ES target e (worst model, generation paths) -> executable integer book -> R1 engine
on independent admission paths. A point belongs to the frontier of budget b only if the engine admits it at b.
Budgets 10% / 15% / 20% are separate frontiers (never one band). kappa = 1 is canonical; 0.5 and 0 are sensitivities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import frontier as F
from . import scenarios as SC
from . import views as V


def objective(su: F.Setup, x: np.ndarray, kappa: float) -> dict:
    """Robust and nominal expected NET excess over the money market of a FIXED weight vector (per NAV0)."""
    iv = su.views
    beta = su.cfg["owner"]["liquidity_min_nav"]
    h = su.cfg["r1"]["liquidity"]["sweep_hysteresis_nav"]
    c = np.array([float(iv.loc[F.SERIES.get(r, r), "central"]) for r in su.routes])
    dL = np.array([float(iv.loc[F.SERIES.get(r, r), "d_long"]) for r in su.routes])
    dS = np.array([float(iv.loc[F.SERIES.get(r, r), "d_short"]) for r in su.routes])
    tc = np.array([su.cost[r] for r in su.routes])
    roll = np.array([2 * su.cost[r] * su.nroll.get(r, 0) for r in su.routes])
    drag = np.array([0.0 if r in F.FUNDED else (su.mr[r] + su.q99[r]) * su.mH for r in su.routes])
    dev = np.where(x > 0, dL * x, -dS * x)
    worst = np.sort(dev)[::-1][:su.gamma].sum()
    base = float(c @ x - (roll + drag) @ np.abs(x) - tc @ np.abs(x - su.x0) - (beta + h / 2) * su.mH)
    return {"nominal_excess": base, "robust_excess": base - kappa * float(worst)}


def views_at(ctx, cfg, P, t, state_views=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    fr = cfg["r3"]["frontier"]
    panel = SC.history_panel(P, t, fr["estimation_start"])
    priors = V.priors_pre2022(ctx, cfg)
    bp = priors_btc(ctx, cfg, P, priors)
    if bp:
        priors["BTC_RUB"] = bp
    reg, iv = V.registry(ctx, cfg, P, t, fr["horizon_sessions"] / 252, priors, panel, state_views)
    return reg, iv.set_index("route")


def views_with_excluded(ctx, cfg, P, t, reg: pd.DataFrame) -> pd.DataFrame:
    """Labelled sensitivity: the view set INCLUDING the excluded prior points (R3-A1)."""
    fr = cfg["r3"]["frontier"]
    panel = SC.history_panel(P, t, fr["estimation_start"])
    return V.intervals(reg, panel, fr["horizon_sessions"] / 252, include_excluded=True).set_index("route")


def priors_btc(ctx, cfg, P, priors) -> dict | None:
    """BTC_RUB pre-2022 prior = BTCUSDT (Binance, same clock) x CBR USD/RUB (prior fetch) minus RUONIA."""
    import re
    from pathlib import Path
    from .evidence import MANIFEST
    m = pd.read_csv(Path(ctx.root) / "data" / MANIFEST, dtype=str)
    rec = {}
    for rp in sorted(m[(m["dataset"] == "usd_official_prior") & (m["validation_status"] == "VALID")]["raw_path"]):
        t = (Path(ctx.root) / rp).read_bytes().decode("windows-1251")
        for d, v in re.findall(r'<Record Date="([0-9.]+)" Id="R01235"><Nominal>1</Nominal><Value>([0-9,]+)</Value>', t):
            rec[pd.to_datetime(d, format="%d.%m.%Y")] = float(v.replace(",", "."))
    ru = priors.get("_RUONIA")
    if not rec or ru is None:
        return None
    from ..data import crypto
    era = cfg["r3"]["prior_era"]
    usd = pd.Series(rec).sort_index()
    cal = pd.bdate_range(max(pd.Timestamp(era["start"]), pd.Timestamp("2019-01-02")), era["end"])
    bh = crypto.binance_hourly(ctx.root, cfg["cutoff"])
    btc = crypto.synced_close(bh, cal, cfg["r2"]["btc"]["bar_close_hour_utc"])
    from .fx import usd_known
    lvl = (btc * usd_known(usd, cal, 1)).dropna()
    r = lvl.pct_change().dropna()
    gap = pd.Series(r.index, index=r.index).diff().dt.days.fillna(1)
    rr = ru.reindex(ru.index.union(r.index)).ffill().reindex(r.index).shift(1)
    exc = (r - rr / 100 * gap / 365).dropna()
    yrs = len(exc) / 252
    return {"ann_excess": float(exc.mean() * 252), "se_ann": float(exc.std(ddof=1) * np.sqrt(252) / np.sqrt(yrs)),
            "from": str(exc.index.min().date()), "to": str(exc.index.max().date()), "days": len(exc),
            "note": "BTCUSDT (Binance, 15:00 UTC bar) x CBR USD/RUB; crypto trades weekends (aggregated)"}


def _prune(su, x, ex, cfg, kappa):
    """Post-admission parsimony: drop positions < 1% NAV0 only if the robust objective falls by <= 5 bp NAV0 and the
    pruned book is re-admitted by the engine."""
    pr = cfg["r3"]["parsimony"]["post_admission_removal"]
    small = [j for j, r in enumerate(su.routes) if 0 < abs(ex["w_exec"][r]) < pr["max_position_nav0"]]
    if not small:
        return None
    xp = x.copy()
    xp[small] = 0.0
    loss = objective(su, x, kappa)["robust_excess"] - objective(su, xp, kappa)["robust_excess"]
    if loss > pr["max_objective_loss_nav0"]:
        return {"removed": [su.routes[j] for j in small], "objective_loss": loss, "applied": False, "reason": "loss above tolerance"}
    exp = F.admit_candidate(su, xp)
    return {"removed": [su.routes[j] for j in small], "objective_loss": loss, "x": xp, "ex": exp, "applied": None}


def frontier_rows(su: F.Setup, kappas: list, grid: list, label: dict, cache: dict) -> list[dict]:
    cfg = su.cfg
    budgets = [cfg["r3"]["frontier"]["budgets"]["primary"], *cfg["r3"]["frontier"]["budgets"]["sensitivity"]]
    out = []
    for kappa in kappas:
        for e in grid:
            sol = F.solve(su, e, kappa)
            row = {**label, "kappa": kappa, "es_target_nav0": e, "lp_status": sol["status"]}
            if sol["x"] is None:
                out.append(row)
                continue
            key = (label["universe"], label["mandate"], label.get("capital", 0), tuple(np.round(sol["x"], 6)))
            if key not in cache:
                cache[key] = F.admit_candidate(su, sol["x"])
            ex = cache[key]
            pr = _prune(su, sol["x"], ex, cfg, kappa)
            if pr and pr.get("x") is not None:
                ok = all(F.verdict(pr["ex"], b, cfg).passed == F.verdict(ex, b, cfg).passed for b in budgets)
                if ok:
                    row["parsimony_removed"] = ";".join(pr["removed"])
                    row["parsimony_objective_loss_nav0"] = pr["objective_loss"]
                    x_use, ex = pr["x"], pr["ex"]
                else:
                    row["parsimony_removed"] = f"NOT_APPLIED ({';'.join(pr['removed'])}: re-admission changed)"
                    x_use = sol["x"]
            else:
                if pr:
                    row["parsimony_removed"] = f"NOT_APPLIED ({pr['reason']})"
                x_use = sol["x"]
            ob = objective(su, x_use, kappa)
            row.update({"lp_robust_excess_nav0": sol["robust_excess"], "lp_nominal_excess_nav0": sol["nominal_excess"],
                        "lp_cvar_worst_nav0": max(sol["cvar"].values()),
                        "robust_excess_exec_nav0": ob["robust_excess"], "nominal_excess_exec_nav0": ob["nominal_excess"],
                        **{f"w_lp_{r}": float(v) for r, v in zip(su.routes, sol["x"])},
                        **{f"w_exec_{r}": v for r, v in ex["w_exec"].items()},
                        **F.summarise(su, ex)})
            for b in budgets:
                v = F.verdict(ex, b, cfg)
                row[f"ADMISSION_{int(round(b * 100))}"] = "PASS" if v.passed else "FAIL"
                row[f"fail_reasons_{int(round(b * 100))}"] = " | ".join(v.reasons)
            out.append(row)
    return out


def zero_weight(su: F.Setup, top: dict, kappa: float, budget: float, fair: dict, cache: dict,
                nom_best: dict | None = None) -> pd.DataFrame:
    """Named cause for every route at the top admitted point of the canonical frontier (brief section J)."""
    cfg = su.cfg
    iv = su.views
    e = top["es_target_nav0"]
    nominal = F.solve(su, e, 0.0)
    rows = []
    nom_best = nom_best or {}
    for j, r in enumerate(su.routes):
        w = top.get(f"w_exec_{r}", 0.0)
        s = F.SERIES.get(r, r)
        c, dL, dS = float(iv.loc[s, "central"]), float(iv.loc[s, "d_long"]), float(iv.loc[s, "d_short"])
        b = cfg["r3"]["bounds"][r]
        tc = su.cost[r]
        roll = 2 * su.cost[r] * su.nroll.get(r, 0)
        drag = 0.0 if r in F.FUNDED else (su.mr[r] + su.q99[r]) * su.mH
        cost_h = tc + roll + drag
        mv_long = c - kappa * dL - cost_h if b["hi"] > 0 else np.nan
        mv_short = -c - kappa * dS - cost_h if b["lo"] < 0 else np.nan
        row = {"route": r, "exposure": V.EXPOSURE.get(s, ""), "w_exec": w, "allowed_long": b["hi"] > 0, "allowed_short": b["lo"] < 0,
               "view_central_h": c, "view_lo_h": c - dL, "view_hi_h": c + dS, "entry_cost_plus_roll_plus_carry_drag_h": cost_h,
               "marginal_worst_case_long_h": mv_long, "marginal_worst_case_short_h": mv_short,
               "breakeven_worst_case_excess_h": cost_h, "w_nominal_kappa0": float(nominal["x"][j]) if nominal["x"] is not None else np.nan,
               "w_kappa0_best_admitted_at_budget": nom_best.get(f"w_exec_{r}"),
               "fair_status": fair.get(r, "UNKNOWN")}
        if abs(w) > 1e-9:
            row["cause"] = ("ENTERS_POSITIVE_WORST_CASE_EXCESS" if max(np.nan_to_num(mv_long, nan=-1), np.nan_to_num(mv_short, nan=-1)) > 0
                            else "ENTERS_AS_RISK_REDUCER_OR_ROUTE")
        else:
            mv = np.nanmax([mv_long, mv_short])
            force = cfg["r3"]["zero_weight"]["forced_entry_nav0"]
            fs = F.solve(su, e, kappa, force={r: force})
            if fs["x"] is None:
                row.update({"forced_entry_lp": fs["status"], "cause": "INFEASIBLE_UNDER_CONSTRAINTS"})
            else:
                key = ("FORCED", r, tuple(np.round(fs["x"], 6)))
                if key not in cache:
                    cache[key] = F.admit_candidate(su, fs["x"])
                v = F.verdict(cache[key], budget, cfg)
                loss = top["robust_excess_exec_nav0"] - objective(su, fs["x"], kappa)["robust_excess"]
                row.update({"forced_entry_robust_objective_loss_nav0": loss, "forced_entry_admission": "PASS" if v.passed else "FAIL",
                            "forced_entry_fail_reasons": " | ".join(v.reasons)})
                if not (row["allowed_long"] or row["allowed_short"]):
                    row["cause"] = "UNIVERSE_OR_BOUND_EXCLUDED"
                elif mv <= 0:
                    row["cause"] = "WORST_CASE_EXCESS_BELOW_COST_AND_CARRY"
                elif not v.passed:
                    row["cause"] = "RISK_OR_LIQUIDITY_ADMISSION_BINDS"
                else:
                    row["cause"] = "DOMINATED_BY_OTHER_ROUTES_UNDER_ES_BUDGET"
            wk0 = row.get("w_kappa0_best_admitted_at_budget")
            if (wk0 is not None and wk0 == wk0 and abs(wk0) > 1e-6) or abs(row.get("w_nominal_kappa0") or 0.0) > 1e-6:
                row["note"] = "enters at the centre of the view set (kappa 0): the zero weight is caused by view uncertainty"
            else:
                row["note"] = "zero also at the centre of the view set (kappa 0): no positive centre net of costs, or dominated under the ES budget"
        row["conclusion_strength"] = "STRONG" if row["fair_status"] == "FAIR_TEST" else "CAVEATED (" + row["fair_status"] + ")"
        rows.append(row)
    return pd.DataFrame(rows)
