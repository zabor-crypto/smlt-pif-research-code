"""V3 Phase 0.1 / 0.2 / 0.5: independent TRAIN / VALIDATION / TEST scenario sets, robust candidate generation,
selection on VALIDATION only, integer contracts, and multi-seed confidence intervals on untouched TEST paths.

Scenario models (all drift-neutralised: spot -> money-market drift, futures -> 0; structural stationary SMLT basis):
  HS_FULL       circular block bootstrap of the common joint sample (2022-03-28 .. cutoff)
  FHS_FULL      filtered HS (EWMA vol rescaled to the cutoff, log space)
  HS_RECENT250  last 250 trading days
  REGIME_COND   only days whose CAUSAL regime flag equals the regime family at the cutoff (conditional dependence,
                re-neutralised drift; small sample -> documented)
  STRESS_HIST   (generation only) historical horizon windows in the worst tail / best rally tail (deterministic)
  HIST_WINDOWS  (evaluation, deterministic) every overlapping historical horizon window
Seeds: TRAIN, VALIDATION and every TEST replicate use different seeds (and every model its own offset), so the paths
that select a portfolio never evaluate it. Each seed is an independent resample of the SAME history: seed-to-seed
dispersion measures Monte-Carlo stability, NOT estimation error of the history (reported separately across models,
in the walk-forward and on historical windows).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from ..optimization.overlay import Problem, Sleeve, min_es, min_max_es
from ..scenarios.bootstrap import SleevePos, evaluate_portfolio, horizon_returns, paths, structural_futures
from ..strategies.collateral import CASH_ONLY, Collateral
from . import regimes as rg
from .smart import hist_windows

STOCH = ["HS_FULL", "FHS_FULL", "HS_RECENT250", "REGIME_COND"]
SEEDABLE = STOCH + ["REGIME_PURE"]      # REGIME_PURE: the state's own days only (no pooling); used by the library


# ------------------------------------------------------------------ scenario sets
class ScenarioFactory:
    """Builds seeded path sets for every model; nothing is cached across roles, so sets never overlap."""

    def __init__(self, ctx, pl, specs, extra_cols: tuple = (), regime_family: str | None = None,
                 regime_mask: pd.Series | None = None, label: str = "", pool_n0: float | None = None,
                 state_cols: tuple = ()):
        self.ctx, self.pl, self.specs = ctx, pl, specs
        cols = ["SMLT"] + [s.series for s in specs] + [c for c in extra_cols if c not in [s.series for s in specs]]
        self.cols = list(dict.fromkeys(cols))
        self.B = pl._bootstrap_inputs(ctx, self.cols, futures_like=tuple(s.series for s in specs))
        v = ctx.cfg["v3"]
        self.seed0 = ctx.cfg["simulation"]["seed"]
        self.off = v["seeds"]
        self.variants = dict(self.B["variants"])
        # REGIME_COND: causal regime at the cutoff (or an explicit mask supplied by the regime library)
        if regime_mask is None:
            P, R, V = pl._panel(ctx)
            cs = ctx.read("credit_state.csv", index_col=0, parse_dates=True)
            reg = rg.classify(R, cs, ctx.cfg["smart_portfolio"]["regimes"])
            fam = regime_family or str(reg["regime"].loc[:ctx.cutoff].iloc[-1])
            regime_mask = _family_mask(reg, fam)
            self.regime_family = fam
        else:
            self.regime_family = label or "CUSTOM"
        X = self.B["X"]
        mask = regime_mask.reindex(X.index).fillna(False).astype(bool)
        sub = X[mask.values]
        self.regime_n = int(len(sub))
        self.variants["REGIME_PURE"] = (pl._neutral_drift(sub, self.B["fut_cols"], self.B["daily_y"]) if len(sub) > 40
                                        else self.variants["HS_FULL"])
        self.pool_lambda = 0.0
        if pool_n0 and len(sub) > 0:
            # small-sample protection: mix the regime days with the full sample, weight lambda = n0/(n0+n) on the
            # full sample, by repeating the regime block (deterministic; documented in REGIME_MODEL.md)
            lam = pool_n0 / (pool_n0 + len(sub))
            reps = max(1, int(round((1 - lam) / lam * len(X) / len(sub))))
            sub = pd.concat([sub] * reps + [X])
            self.pool_lambda = float(len(X) / len(sub))
        self.variants["REGIME_COND"] = pl._neutral_drift(sub, self.B["fut_cols"], self.B["daily_y"]) if len(sub) > 40 else self.variants["HS_FULL"]
        # non-traded STATE variables (index, Samolet bonds, RGBI) sampled on the same rows for path-state detection;
        # demeaned; they never change the traded columns (block indices depend only on the sample length)
        self.state_cols = [c for c in state_cols if c not in self.cols]
        if self.state_cols:
            P_, R_, V_ = pl._panel(ctx)
            St = R_[self.state_cols].reindex(X.index).fillna(0.0)
            St = St - St.mean()
            for k in list(self.variants):
                self.variants[k] = self.variants[k].copy()
                self.variants[k][self.state_cols] = St.loc[self.variants[k].index].values
            self.cols = self.cols + self.state_cols

    def seed(self, role: str, model: str, k: int = 0) -> int:
        base = {"TRAIN": self.off["train"], "VAL": self.off["validation"], "TEST": self.off["test_base"] + k}[role]
        return int(self.seed0 + base + self.off["model_offset"] * (SEEDABLE.index(model) + 1) * 1000)

    def paths(self, model: str, hz: int, n: int, role: str, k: int = 0) -> dict:
        Xv = self.variants[model]
        sim = self.ctx.cfg["simulation"]
        P = paths(Xv, self.cols, hz, n, sim["block_length"], self.seed(role, model, k))
        if "SMLT_FUT" in self.cols:
            P["SMLT_FUT"] = structural_futures(P, self.B["e_res"].reindex(Xv.index).values, self.B["b"], self.B["e0"])
        return P

    def hist(self, hz: int) -> dict:
        return hist_windows(self.B["variants"]["HS_FULL"], self.cols, hz)

    def stress_hist(self, hz: int) -> pd.DataFrame:
        """Horizon returns of the deterministic STRESS_HIST set: worst tail + best rally tail of history."""
        v = self.ctx.cfg["v3"]
        HR = horizon_returns(self.hist(hz), self.cols)
        lo, hi = HR["SMLT"].quantile(v["stress_hist_tail"]), HR["SMLT"].quantile(1 - v["stress_hist_rally"])
        return HR[(HR["SMLT"] <= lo) | (HR["SMLT"] >= hi)].reset_index(drop=True)


def _family_mask(reg: pd.DataFrame, fam: str) -> pd.Series:
    if fam == "NORMAL":
        return reg["regime"].eq("NORMAL")
    if fam == "COMBINED_STRESS":
        return reg["COMBINED_STRESS"].astype(bool)
    return reg[fam].astype(bool) if fam in reg else reg["regime"].eq(fam)


# ------------------------------------------------------------------ candidate generation (TRAIN only)
def _problem(HR: pd.DataFrame, specs, tau, E, C, mmf_h, coll: Collateral, mb, u, gross, only=None) -> Problem:
    use = [s for s in specs if only is None or s.name in only]
    sl = [Sleeve(s.name, s.lo, s.hi, cost=s.horizon_cost(tau), margin=s.mr * coll.margin_multiplier * coll.capital_per_margin(),
                 group=s.name) for s in use]
    return Problem(r_s=HR["SMLT"].values, R=HR[[s.series for s in use]].values, sleeves=sl, k_liq=C / E,
                   m_liq=np.full(len(HR), mmf_h), alpha=0.95, margin_budget=mb * C / E, gross_cap=gross,
                   min_upside_capture=u)


def generate(ctx, fac: ScenarioFactory, hz: int, C: float, coll: Collateral = CASH_ONLY,
             models: list[str] | None = None, minmax_models: list[str] | None = None, max_K: int | None = None,
             margin_cap: float | None = None, n_lp: int | None = None, upside_floors: list | None = None) -> list[dict]:
    v, sp = dict(ctx.cfg["v3"]), ctx.cfg["smart_portfolio"]
    if max_K is not None:
        v["max_K_enumerated"] = max_K
    if n_lp is not None:
        v["n_train_lp"] = n_lp
    specs, E, tau = fac.specs, ctx.E, hz / 252
    models = models or v["generating_models"]
    mmf_h = (1 + fac.B["y"]) ** tau - 1
    names = [s.name for s in specs]
    HRs = {}
    mm_models = minmax_models or list(STOCH)
    for mdl in list(dict.fromkeys(models + (mm_models if "MINMAX" in models else []))):
        if mdl in SEEDABLE:
            HRs[mdl] = horizon_returns(fac.paths(mdl, hz, v["n_train_lp"], "TRAIN"), fac.cols)
        elif mdl == "STRESS_HIST":
            HRs[mdl] = fac.stress_hist(hz)
    subsets = [c for k in range(1, min(v["max_K_enumerated"], len(names)) + 1) for c in combinations(names, k)]
    if len(names) > v["max_K_enumerated"]:
        subsets.append(tuple(names))
    lo, hi = v["margin_budget_range"]
    if margin_cap is not None:
        hi = min(hi, margin_cap)
    floors = sp["upside_floor_grid"] if upside_floors is None else upside_floors
    seen, cands = set(), []

    def add(h: dict, family: str, meta: dict):
        key = tuple(round(h.get(n, 0.0), 3) for n in names)
        if key in seen:
            return
        seen.add(key)
        cands.append({"family": family, "h": {n: float(h.get(n, 0.0)) for n in names}, **meta})

    for mdl in models:
        step = v["primary_margin_step"] if mdl == "FHS_FULL" else v["secondary_margin_step"]
        for sub in subsets:
            for mb in np.round(np.arange(lo, hi + 1e-9, step), 4):
                for u in (floors if len(sub) == len(names) else [None]):
                    if mdl == "MINMAX":
                        probs = [_problem(HRs[m].head(v["minmax_n_lp"]), specs, tau, E, C, mmf_h, coll, mb, u, sp["gross_cap"],
                                          only=sub) for m in mm_models if m in HRs]
                        sol = min_max_es(probs)
                    else:
                        sol = min_es(_problem(HRs[mdl], specs, tau, E, C, mmf_h, coll, mb, u, sp["gross_cap"], only=sub))
                    if not sol.status.startswith("optimal"):
                        continue
                    add(dict(zip(sub, sol.h)), "MULTI", {"gen_model": mdl, "subset": "|".join(sub), "margin_budget": mb,
                                                         "upside_floor": u, "train_obj_es": sol.es})
    B = fac.B
    for hs in np.arange(0.0, 1.0 + 1e-9, sp["direct_only_h_grid_step"]):
        nf = round(hs * E / B["S0"]) * B["F0"] / E
        add({"DIRECT": -nf}, "DIRECT_ONLY", {"gen_model": "GRID", "subset": "DIRECT", "margin_budget": np.nan,
                                             "upside_floor": np.nan, "train_obj_es": np.nan, "share_hedge_ratio": hs})
    return cands


# ------------------------------------------------------------------ evaluation
def sleeves_for(specs, h: dict, E: float, tau: float) -> list[SleevePos]:
    return [SleevePos(s.series, h[s.name] * E, s.mr, s.cost_one_way, s.n_rolls(tau), "future")
            for s in specs if abs(h.get(s.name, 0.0)) > 1e-9]


KEEP = ["ES95", "ES99", "VaR95", "mean_pnl", "std_pnl", "p95_max_dd", "mean_max_dd", "upside_capture", "downside_capture",
        "p_sleeve_breach", "initial_im", "peak_collateral_p95", "transaction_costs", "min_pnl", "p05_pnl", "min_free_p05",
        "collateral_income_mean", "gross_derivatives_notional", "n_active_sleeves"]


def evaluate(ctx, fac: ScenarioFactory, cands: list[dict], hz: int, C: float, role: str, n: int, k: int = 0,
             coll: Collateral = CASH_ONLY, models: list[str] | None = None, include_hist: bool = True,
             derisk: dict | None = None) -> pd.DataFrame:
    """Metrics of every candidate on every model of one scenario role (VAL or TEST seed k)."""
    E, tau = ctx.E, hz / 252
    models = models or ctx.cfg["v3"]["admissible_models"]
    sets = {m: fac.paths(m, hz, n, role, k) for m in models}
    if include_hist:
        sets["HIST_WINDOWS"] = fac.hist(hz)
    rows = []
    S0 = fac.B["S0"]
    dcost = (0.1 + ctx.cfg["costs"]["equity_commission_bps"] * 1e-4 * S0) / S0
    for c in cands:
        sl = sleeves_for(fac.specs, c["h"], E, tau)
        for m, P in sets.items():
            out = evaluate_portfolio(P, E=E, C=C, sleeves=sl, mmf_daily=fac.B["daily_y"], collateral=coll,
                                     derisk=c.get("derisk", 0.0), derisk_cost=dcost)
            rows.append({"cand_id": c["cand_id"], "model": m, "role": role, "seed_k": k, **{x: out[x] for x in KEEP}})
    return pd.DataFrame(rows)


def summarise(ev: pd.DataFrame, primary: str = "FHS_FULL", admissible: list[str] | None = None) -> pd.DataFrame:
    """One row per candidate: primary-model metrics + worst case over the admissible model set (+ HIST_WINDOWS)."""
    adm = (admissible or STOCH) + ["HIST_WINDOWS"]
    e = ev[ev["model"].isin(adm)]
    piv = e.pivot_table(index="cand_id", columns="model", values=["ES95", "p_sleeve_breach", "upside_capture", "mean_pnl"],
                        aggfunc="first")
    s = ev[ev["model"] == primary].set_index("cand_id")[KEEP].copy()
    s["worst_model_ES95"] = piv["ES95"].max(axis=1)
    s["worst_model_name"] = piv["ES95"].idxmax(axis=1)
    s["worst_model_breach"] = piv["p_sleeve_breach"].max(axis=1)
    s["worst_model_upside"] = piv["upside_capture"].min(axis=1)
    for m in adm:
        if m in piv["ES95"]:
            s[f"ES95_{m}"] = piv["ES95"][m]
            s[f"breach_{m}"] = piv["p_sleeve_breach"][m]
    return s.reset_index()


def select(summary: pd.DataFrame, cands: pd.DataFrame, rule: str, cfg: dict, scope: str = "ANY") -> pd.Series | None:
    d = summary.merge(cands[["cand_id", "family", "K_active"]], on="cand_id")
    if scope != "ANY":
        d = d[d["family"] == scope]
    if rule == "static":
        r = cfg["static"]
        ok = d[(d["p_sleeve_breach"] <= r["max_breach"]) & (d["upside_capture"] >= r["min_upside"])]
        key = "ES95"
    else:
        r = cfg["robust"]
        ok = d[(d["worst_model_breach"] <= r["max_worst_breach"]) & (d["worst_model_upside"] >= r["min_worst_upside"])]
        key = "worst_model_ES95"
    if ok.empty:
        return None
    return ok.sort_values([key, "K_active", "initial_im", "cand_id"]).iloc[0]


# ------------------------------------------------------------------ integer contracts (0.5)
def round_contracts(specs, h: dict, E: float) -> tuple[dict, list[dict]]:
    """Nearest-integer MOEX contracts per sleeve; returns the realised notional fractions and a report."""
    out, rep = {}, []
    for s in specs:
        x = h.get(s.name, 0.0)
        q = s.contracts(x, E)
        real = q * s.contract_notional / E
        out[s.name] = real
        if abs(x) > 1e-9 or q:
            rep.append({"sleeve": s.name, "continuous_notional_frac": x, "continuous_notional_rub": x * E,
                        "contracts": q, "contract_notional_rub": s.contract_notional, "realised_notional_rub": q * s.contract_notional,
                        "rounding_error_rub": q * s.contract_notional - x * E, "initial_margin_rub": abs(q) * s.contract_notional * s.mr})
    return out, rep


# ------------------------------------------------------------------ multi-seed statistics
def delta_ci(test: pd.DataFrame, a: str, b: str, metric: str, level: float = 0.95) -> dict:
    """Paired difference a - b of `metric` per (model, seed) on identical TEST paths; CI across seeds per model."""
    from scipy import stats
    t = test.pivot_table(index=["model", "seed_k"], columns="cand_id", values=metric, aggfunc="first")
    if a not in t or b not in t:
        return {}
    d = (t[a] - t[b]).dropna()
    out = {}
    for m, g in d.groupby(level=0):
        x = g.values
        n = len(x)
        if n > 1:
            se = x.std(ddof=1) / np.sqrt(n)
            q = stats.t.ppf(0.5 + level / 2, n - 1)
            lo_, hi_ = x.mean() - q * se, x.mean() + q * se
        else:
            lo_ = hi_ = x.mean()
        out[m] = {"n_seeds": n, "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if n > 1 else 0.0,
                  "ci_lo": float(lo_), "ci_hi": float(hi_), "min": float(x.min()), "max": float(x.max()),
                  "share_negative": float((x < 0).mean()), "share_positive": float((x > 0).mean())}
    return out
