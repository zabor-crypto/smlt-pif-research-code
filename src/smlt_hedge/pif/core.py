"""V4 shared machinery: fund scenario sets (TRAIN / VALIDATION / TEST), frontier candidates on TRAIN, evaluation of
candidate NAV portfolios with the fund engine on VALIDATION / TEST, the pre-declared selection rule, integer rounding."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from .. import pipeline as pl
from ..portfolio import validation as va
from ..scenarios.bootstrap import horizon_returns
from ..strategies.collateral import CASH_ONLY
from . import fund
from . import optimize as opt
from . import universe as uni

STATE_COLS = ("IMOEX", "SMLT_BOND_IDX", "RGBI")
MODELS = ["HS_FULL", "FHS_FULL", "HS_RECENT250", "REGIME_COND"]


def factory(ctx, assets, regime_mask=None, label="", pool=True) -> va.ScenarioFactory:
    """V3 ScenarioFactory re-used: derivative series are futures-like (zero drift), funded series (SMLT, BTC spot)
    drift at the money-market rate; state variables ride along on the same rows."""
    derivs = [SimpleNamespace(series=a.series) for a in assets if a.kind == "DERIVATIVE"]
    funded = tuple(a.series for a in assets if a.kind == "FUNDED" and a.series != "SMLT")
    kw = {"regime_mask": regime_mask, "label": label, "pool_n0": ctx.cfg["v3"]["shrink_n0"]} if regime_mask is not None else {}
    return va.ScenarioFactory(ctx, pl, derivs, extra_cols=funded, state_cols=STATE_COLS, **kw)


def lp_models(fac, assets, hz: int, n: int, role: str, models=None, k: int = 0) -> list[opt.Model]:
    y = fac.B["y"]
    m = (1 + y) ** (hz / 252) - 1
    out = []
    for mdl in models or MODELS:
        P = fac.paths(mdl, hz, n, role, k)
        HR = horizon_returns(P, list(dict.fromkeys(["SMLT"] + [a.series for a in assets])))
        out.append(opt.Model(R=HR[[a.series for a in assets]].values, rs=HR["SMLT"].values, m=m))
    return out


def make_lp(ctx, assets, E, C, hz, fixed=None, turnover_max=np.inf) -> opt.FundLP:
    v4 = ctx.cfg["v4"]
    return opt.FundLP(assets, uni.initial_weights(assets, E, C), hz / 252, v4["liquidity_buffer_nav"], v4["max_long_exposure_nav"],
                      v4["max_derivative_gross_nav"], turnover_max, fixed)


def frontier(ctx, lp, models, tag: dict) -> list[dict]:
    v4 = ctx.cfg["v4"]
    rows = []
    for b in v4["es_budget_grid"]:
        r = opt.solve(lp, models, "max_participation", budget=b)
        if r["status"] == "optimal":
            rows.append({**tag, "objective": "max_participation", "es_budget": b, "part_floor": np.nan, **_pack(lp, r)})
    for f in v4["participation_floor_grid"]:
        r = opt.solve(lp, models, "min_worst_es", floor=f)
        if r["status"] == "optimal":
            rows.append({**tag, "objective": "min_worst_es", "es_budget": np.nan, "part_floor": f, **_pack(lp, r)})
    return rows


def _pack(lp, r) -> dict:
    return {"train_worst_es_nav": r["worst_es"], "train_min_participation": r["min_participation"],
            "train_mean_nav": float(np.mean(r["mean_by_model"])), "margin_nav": r["margin_used"],
            "funded_risky_nav": r["funded_risky"], "deriv_gross_nav": r["deriv_gross"],
            **{f"x_{a.name}": float(v) for a, v in zip(lp.assets, r["x"])}}


def xvec(row, assets) -> np.ndarray:
    v = [row.get(f"x_{a.name}", 0.0) for a in assets]
    return np.array([0.0 if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x) for x in v])


def static_policy(x: np.ndarray, name: str = "static") -> fund.FundPolicy:
    return fund.FundPolicy(name, x[None, :], rebalance=False)


def evaluate(ctx, fac, assets, cands: list[tuple[str, fund.FundPolicy, str | None]], hz, E, C, role, n, k=0, models=None,
             codes_fn=None, coll=CASH_ONLY) -> pd.DataFrame:
    """cands: (id, policy, code_kind) where code_kind in (None, 'portfolio'); returns one row per candidate x model."""
    rows = []
    x0 = uni.initial_weights(assets, E, C)
    md = (1 + fac.B["y"]) ** (1 / 252) - 1
    for mdl in models or MODELS:
        P = fac.paths(mdl, hz, n, role, k)
        codes = codes_fn(P) if codes_fn else None
        for cid, pol, ck in cands:
            st = np.zeros((n, hz + 1), int) if ck is None else codes[ck][:, :hz + 1]
            out = fund.run(P, md, assets, pol, st, E=E, C=C, x0_initial=x0, coll=coll,
                           shock=None if ck is None else codes["shock"][:, :hz])
            rows.append({"cand_id": cid, "model": mdl, "role": role, "seed_k": k,
                         **{kk: v for kk, v in out.items() if not kk.startswith("_")}})
    return pd.DataFrame(rows)


def summarise(ev: pd.DataFrame) -> pd.DataFrame:
    g = ev.groupby("cand_id")
    return pd.DataFrame({"worst_ES95": g["ES95"].max(), "worst_participation": g["participation"].min(),
                         "worst_breach": g["p_liquidity_breach"].max(), "mean_pnl_avg": g["mean_pnl"].mean(),
                         "worst_mdd_p95": g["p95_mdd_nav"].max(), "turnover": g["turnover_nav"].mean(),
                         "FHS_ES95": g.apply(lambda d: float(d.loc[d["model"] == "FHS_FULL", "ES95"].iloc[0]) if (d["model"] == "FHS_FULL").any() else np.nan),
                         "FHS_participation": g.apply(lambda d: float(d.loc[d["model"] == "FHS_FULL", "participation"].iloc[0])
                                                      if (d["model"] == "FHS_FULL").any() else np.nan)}).reset_index()


def select(summ: pd.DataFrame, nav0: float, es_budget: float, max_breach: float) -> pd.Series | None:
    """Pre-declared rule: max worst-model participation s.t. worst-model ES95 <= budget * NAV0 and worst breach <= limit;
    ties -> lower turnover, then cand_id."""
    ok = summ[(summ["worst_ES95"] <= es_budget * nav0) & (summ["worst_breach"] <= max_breach)]
    if ok.empty:
        return None
    return ok.sort_values(["worst_participation", "turnover", "cand_id"], ascending=[False, True, True]).iloc[0]


def integer_portfolio(ctx, assets, x: np.ndarray, nav0: float) -> tuple[np.ndarray, list[dict]]:
    """SMLT in whole shares (lot 1), derivatives in whole MOEX contracts; experimental BTC spot left continuous."""
    xi, rep = x.copy(), []
    for j, a in enumerate(assets):
        if abs(x[j]) < 1e-9 or a.contract_notional <= 0:
            continue
        q = int(np.round(x[j] * nav0 / a.contract_notional))
        xi[j] = q * a.contract_notional / nav0
        rep.append({"asset": a.name, "kind": a.kind, "continuous_nav_weight": x[j], "continuous_rub": x[j] * nav0,
                    "units": q, "unit": "shares" if a.kind == "FUNDED" else "contracts", "unit_value_rub": a.contract_notional,
                    "realised_rub": q * a.contract_notional, "rounding_error_rub": q * a.contract_notional - x[j] * nav0,
                    "initial_margin_rub": abs(q) * a.contract_notional * a.mr})
    return xi, rep
