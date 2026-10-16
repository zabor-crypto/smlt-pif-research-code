"""Gate R6 outputs: one-ledger chronological replay of every comparator and sensitivity; incremental-value test.

Every run is an independent deterministic ledger (fixed seeds), so runs are executed in parallel processes and
assembled in a fixed order. Evidence class of everything here: RETROSPECTIVE_WALK_FORWARD on one market history —
never market out-of-sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAIN_M = "BROAD_PIF_NO_HIDDEN_LEVERAGE"
V41_W = {"SMLT": 0.2844, "EQ_HEDGE_FUT": -0.1262}


def run_specs(cfg: dict) -> list[dict]:
    s = cfg["r6"]["sensitivities"]
    out = [
        {"name": "CONTROL_INITIAL_PORTFOLIO", "type": "fixed", "target": {"shares_frac": 1.0}},
        {"name": "SELL_50_TO_MONEY_MARKET", "type": "fixed", "target": {"shares_frac": 0.5}},
        {"name": "SELL_100_TO_MONEY_MARKET", "type": "fixed", "target": {"shares_frac": 0.0}},
        {"name": "FUT25_LEGACY", "type": "fixed", "target": {"shares_frac": 1.0, "hedge_frac": 0.25}},
        {"name": "FUT50_LEGACY", "type": "fixed", "target": {"shares_frac": 1.0, "hedge_frac": 0.50}},
        {"name": "V41_STATIC_PIF_WEIGHTS_PROXY_NONCAUSAL", "type": "fixed", "target": {"weights": V41_W}, "reset": True},
    ]
    for kappa in s["kappa"]:
        tag = "" if kappa == s["kappa"][0] else f"|kappa{kappa:g}"
        for kind, nm in (("STATIC_FRONTIER", "STATIC_DIVERSIFIED_V2"), ("ADAPTIVE", "ADAPTIVE_PIF_V1")):
            base = {"type": "policy", "kind": kind, "kappa": kappa, "delay": 1, "cost_mult": 1.0, "mandate": MAIN_M, "universe": "MAIN"}
            out.append({**base, "name": nm + tag})
            for d in s["delay_sessions"][1:]:
                out.append({**base, "name": f"{nm}{tag}|delay{d}", "delay": d})
            for c in s["cost_multipliers"][1:]:
                out.append({**base, "name": f"{nm}{tag}|cost_x{c:g}", "cost_mult": c})
            out.append({**base, "name": f"{nm}{tag}|LONG_ISSUER_FLOOR_0", "mandate": "LONG_ISSUER_FLOOR_0"})
            out.append({**base, "name": f"{nm}{tag}|WITHOUT_BTC", "universe": "WITHOUT_BTC"})
            if kind == "ADAPTIVE":                         # R4-A1: the originally committed level rule, labelled sensitivity
                for rule in s.get("state_view_rules", [])[1:]:
                    out.append({**base, "name": f"{nm}{tag}|{rule}", "state_view_rule": rule})
    return out


_G = {}


def _init(root: str):
    import warnings
    warnings.simplefilter("error", RuntimeWarning)
    from .. import pipeline as pl
    from . import r1, r3
    from . import routes as RT
    from . import views as V
    ctx = pl.Ctx(root)
    cfg = r1.load_cfg(ctx)
    P = RT.build_panel(ctx, cfg)
    pri = V.priors_pre2022(ctx, cfg)
    bp = r3.priors_btc(ctx, cfg, P, pri)
    if bp:
        pri["BTC_RUB"] = bp
    import os
    import sys
    src = os.path.realpath(os.path.join(root, "src"))
    bad = [m.__name__ for m in list(sys.modules.values()) if getattr(m, "__name__", "").startswith("smlt_hedge")
           and getattr(m, "__file__", None) and not os.path.realpath(m.__file__).startswith(src)]
    if bad:
        raise RuntimeError(f"worker import guard: modules outside {src}: {bad}")
    _G.update({"ctx": ctx, "cfg": cfg, "P": P, "pri": pri, "worlds": {}})


def _world(universe: str, cost_mult: float):
    from . import replay as RP
    key = (universe, cost_mult)
    if key not in _G["worlds"]:
        _G["worlds"][key] = RP.world(_G["ctx"], _G["cfg"], _G["P"], universe, _G["cfg"]["r6"]["start"], cost_mult)
    return _G["worlds"][key]


def run_one(spec: dict) -> dict:
    from . import replay as RP
    ctx, cfg, P = _G["ctx"], _G["cfg"], _G["P"]
    nav0 = float(cfg["r6"]["nav0_rub"])
    if spec["type"] == "fixed":
        W = _world("MAIN", 1.0)
        out = RP.run_fixed(ctx, cfg, P, W, spec["target"], reset_at_recal=spec.get("reset", False))
    else:
        W = _world(spec["universe"], spec["cost_mult"])
        out = RP.run_policy(ctx, cfg, P, W, spec["kind"], universe=spec["universe"], mandate=spec["mandate"], kappa=spec["kappa"],
                            delay=spec["delay"], cost_mult=spec["cost_mult"], priors=_G["pri"], name=spec["name"],
                            state_view_rule=spec.get("state_view_rule"))
    mmf = W.px["MMF"]
    lg = RP.ledger_frame(W, out["res"])
    keep = ["nav", "cash", "pending", "im", "available", "breach", "unresolved", "vm", "roll_cost"] + [c for c in lg if c[:2] in ("q_", "v_", "n_", "c_")]
    return {"name": spec["name"], "spec": {k: v for k, v in spec.items() if k != "target"},
            "metrics": RP.metrics(W, out["res"], spec["name"], mmf, nav0),
            "attribution": RP.attribution(W, out["res"], spec["name"], mmf, nav0),
            "decisions": out["decisions"], "ledger": lg[keep]}


def run_all(root: str, specs: list[dict], processes: int) -> list[dict]:
    import multiprocessing as mp
    if processes <= 1:
        _init(root)
        return [run_one(s) for s in specs]
    ctxm = mp.get_context("spawn")
    with ctxm.Pool(processes, initializer=_init, initargs=(root,)) as pool:
        return pool.map(run_one, specs, chunksize=1)


# ---------------------------------------------------------------------------------------------- post-processing
def money_market_row(W, nav0: float) -> dict:
    mm = W.px["MMF"].reindex(W.dates)
    g = mm / float(mm.iloc[0])
    r = g.pct_change().dropna()
    yrs = len(g) / 252
    return {"comparator": "ALL_MONEY_MARKET", "start": str(W.dates[0].date()), "end": str(W.dates[-1].date()), "sessions": len(g),
            "terminal_nav": float(nav0 * g.iloc[-1]), "total_return": float(g.iloc[-1] - 1), "money_market_total_return": float(g.iloc[-1] - 1),
            "net_excess_over_money_market": 0.0, "cagr": float(g.iloc[-1] ** (1 / yrs) - 1), "vol_ann": float(r.std(ddof=1) * np.sqrt(252)),
            "max_drawdown": float(((g.cummax() - g) / g.cummax()).max()), "worst_1d": float(r.min()),
            "evidence_class": "REFERENCE (NAV0 in LQDT units from the start; no costs, no liquidity reserve)"}


def blocks(ledgers: dict, mmf: pd.Series, a: str, b: str, months: list) -> pd.DataFrame:
    """Semi-annual block excess returns (over the money market) of ledgers a and b and their difference."""
    na, nb = ledgers[a]["nav"], ledgers[b]["nav"]
    mm = mmf.reindex(na.index)
    edges = [na.index[0]] + [d for i, d in enumerate(na.index[1:], 1) if d.month in months and na.index[i - 1].month != d.month] + [na.index[-1]]
    rows = []
    for s, e in zip(edges[:-1], edges[1:]):
        ra, rb, rm = na.loc[e] / na.loc[s] - 1, nb.loc[e] / nb.loc[s] - 1, mm.loc[e] / mm.loc[s] - 1
        rows.append({"block_start": str(s.date()), "block_end": str(e.date()), "sessions": int(na.loc[s:e].shape[0] - 1),
                     f"excess_{a}": ra - rm, f"excess_{b}": rb - rm, "difference": ra - rb})
    return pd.DataFrame(rows)


def incremental(bl: pd.DataFrame, seed: int, n_boot: int = 10000) -> dict:
    d = bl["difference"].values
    rng = np.random.default_rng(seed)
    if len(d) == 0 or np.allclose(d, 0.0):
        return {"n_blocks": int(len(d)), "mean_block_difference": 0.0, "sign_share_positive": 0.0, "ci90_lo": 0.0, "ci90_hi": 0.0,
                "max_single_block_share_of_gain": None, "identical_ledgers": True}
    bs = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    pos = d[d > 0].sum()
    return {"n_blocks": int(len(d)), "mean_block_difference": float(d.mean()), "sign_share_positive": float((d > 0).mean()),
            "ci90_lo": float(np.quantile(bs, 0.05)), "ci90_hi": float(np.quantile(bs, 0.95)),
            "max_single_block_share_of_gain": float(d.max() / pos) if pos > 0 else None, "identical_ledgers": False}


def verdict(base: dict, sens: list[dict], constraints_ok: bool) -> str:
    """Pre-declared rule (configs r6.incremental_value)."""
    if base.get("identical_ledgers"):
        return "ADAPTIVE_INCREMENTAL_VALUE_NOT_ESTABLISHED (the adaptive and static ledgers are identical)"
    ok = (base["sign_share_positive"] >= 0.8 and base["ci90_lo"] > 0 and (base["max_single_block_share_of_gain"] or 1.0) <= 0.5
          and all(s["mean_block_difference"] > 0 and s["ci90_lo"] > 0 for s in sens) and constraints_ok)
    if ok:
        return "ESTABLISHED"
    if base["ci90_lo"] <= 0 <= base["ci90_hi"]:
        return "ADAPTIVE_INCREMENTAL_VALUE_NOT_ESTABLISHED (block interval includes zero)"
    return "ADAPTIVE_INCREMENTAL_VALUE_NOT_ESTABLISHED"


def rebalance_log(name: str, lg: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    qcols = [c for c in lg if c.startswith("q_")]
    exec_days = set(pd.to_datetime(decisions["execution_date"])) if len(decisions) else set()
    rows = []
    dq = lg[qcols].diff().fillna(0.0)
    for d, r in dq.iterrows():
        for c in qcols:
            if abs(r[c]) > 1e-9:
                inst = c[2:]
                unit = (lg.loc[d, f"v_{inst}"] / lg.loc[d, c]) if f"v_{inst}" in lg and lg.loc[d, c] else (
                    (lg.loc[d, f"n_{inst}"] / lg.loc[d, c]) if f"n_{inst}" in lg and lg.loc[d, c] else np.nan)
                reason = ("POLICY_TARGET" if d in exec_days and inst != "MMF" else
                          "CASH_RULE_SWEEP_OR_RESTORE" if inst == "MMF" else "RESTORATION_OR_FORCED_REDUCTION")
                rows.append({"policy": name, "date": str(d.date()), "instrument": inst, "quantity_change": float(r[c]),
                             "quantity_after": float(lg.loc[d, c]), "is_integer": bool(abs(lg.loc[d, c] - round(lg.loc[d, c])) < 1e-9),
                             "unit_value_rub": float(unit) if unit == unit else None, "reason": reason})
    return pd.DataFrame(rows)
