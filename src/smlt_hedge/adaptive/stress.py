"""Section S — deterministic path stresses (no probabilities). Each scenario is ONE hand-specified joint path over 126
sessions, run through the R1 engine from the ACTUAL book with next-session execution, so the report says whether the
fund survives the PATH (margin calls, forced reductions, liquidity breaches), not only the terminal P&L.
A known register event never triggers these jumps (HYPOTHETICAL_FUTURE_EVENT_STRESS only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import admission as A
from . import engine as E
from . import frontier as F
from . import market as M
from . import orders as O
from . import scenarios as SC
from .r1 import reserve_min


def path(su: F.Setup, spec: dict, groups: dict) -> SC.PathSet:
    ps = su.gen[0][1]
    H = ps.R.shape[1]
    mm = ps.mmf_daily.copy() + spec.get("mm_shift_ann", 0.0) / 252
    R = np.zeros((1, H, len(ps.routes)))
    for k, r in enumerate(ps.routes):                        # base: zero excess
        R[0, :, k] = mm if r in SC.FUNDED_ROUTES else 0.0
    for g, d0, d1, tot in spec["legs"]:
        n = d1 - d0 + 1
        step = (1 + tot) ** (1 / n) - 1
        for r in groups[g]:
            if r in ps.routes:
                k = ps.routes.index(r)
                R[0, d0 - 1:d1, k] = (1 + R[0, d0 - 1:d1, k]) * (1 + step) - 1
    # STRESS_ONLY_ZERO_AUXILIARY: hand-specified NET shocks, no additional swap/carry forecast.
    swap = np.zeros((1, H)) if "CNY_PERP" in ps.routes else None
    return SC.PathSet("STRESS", R, mm, ps.routes, {}, cny_swap_frac=swap)


def to_market(su: F.Setup, ps: SC.PathSet, spec: dict, mult: float):
    idx = {r: k for k, r in enumerate(ps.routes)}
    H = ps.R.shape[1]
    Rf = np.stack([ps.R[:, :, idx[f.name]] if f.kind != E.MMF else np.zeros((1, H)) for f in su.fspecs], axis=2)
    Rd = np.stack([ps.R[:, :, idx[F.SERIES.get(d.name, d.name)]] for d in su.dspecs], axis=2)
    fin, din, mkt = M.forward_market(su.fspecs, su.dspecs, Rf, Rd, ps.R[:, :, idx["USDRUB"]], ps.mmf_daily,
                                     su.cfg["r1"]["liquidity"]["immediate_cash_rate"], cny_swap_frac=ps.cny_swap_frac)
    mh = spec.get("margin_hike")
    if mh:
        mkt.mr[mh["from_day"]:] = mkt.mr[mh["from_day"]:] * mh["multiplier"]
    if spec.get("extra_settlement_lag"):
        mkt.lag[:] = mkt.lag + spec["extra_settlement_lag"] * (mkt.lag > 0)
    return fin, din, mkt


def book_orders(su: F.Setup, weights: dict | None, delay: int):
    """Target weights (LP routes) -> executable orders from the actual book; None = no trade."""
    if weights is None:
        ex = F.executable(su, su.x0, delay)
        return [], ex
    x = np.array([weights.get(r, 0.0) for r in su.routes])
    ex = F.executable(su, x, delay)
    return ex["orders"], ex


def run(su: F.Setup, books: dict, cfg: dict) -> pd.DataFrame:
    st = cfg["stress"]
    rows = []
    for sname, spec in st["scenarios"].items():
        ps = path(su, spec, st["route_groups"])
        delay = spec.get("order_delay_sessions", cfg["owner"]["execution_delay_sessions"])
        for mult in st["margin_multipliers"]:
            fin, din, mkt = to_market(su, ps, spec, mult)
            for bname, w in books.items():
                orders, ex = book_orders(su, w, delay)
                prm = E.Params(beta=cfg["owner"]["liquidity_min_nav"], mult=mult, reserve_frac=ex["reserve_min"],
                               hysteresis=cfg["r1"]["liquidity"]["sweep_hysteresis_nav"], wf_rank=ex["wf_rank"],
                               restore_rank=ex["restore_rank"], log=True)
                res = E.Engine(fin, din, mkt, prm).run(su.book0, orders=orders)
                lg = pd.DataFrame(res["log"])
                nav = lg["nav"]
                first = spec["legs"][0]
                gap_day = max(d0 for _, d0, d1, tot in spec["legs"] if d0 == d1) if any(d0 == d1 for _, d0, d1, _ in spec["legs"]) else None
                qcols = [c for c in lg if c.startswith("q_") and c[2:] in su.din_names]
                hedge_before = {c[2:]: float(lg[c].iloc[gap_day - 1]) for c in qcols} if gap_day else {}
                hedge_entry = {c[2:]: float(lg[c].iloc[min(delay + 1, len(lg) - 1)]) for c in qcols}
                rows.append({"scenario": sname, "note": spec.get("note", ""), "book": bname, "margin_multiplier": mult,
                             "order_delay_sessions": delay, "nav0": float(nav.iloc[0]), "nav_end": float(nav.iloc[-1]),
                             "pnl_nav0": float(nav.iloc[-1] / nav.iloc[0] - 1),
                             "money_market_return_on_path": float(np.prod(1 + ps.mmf_daily) - 1),
                             "excess_over_money_market_nav0": float(nav.iloc[-1] / nav.iloc[0] - np.prod(1 + ps.mmf_daily)),
                             "min_nav_nav0": float(nav.min() / nav.iloc[0]),
                             "max_drawdown": float(((nav.cummax() - nav) / nav.cummax()).max()),
                             "min_available_nav": float((lg["available"] / lg["nav"]).min()),
                             "liquidity_breach_days": int(lg["breach"].sum()), "unresolved_deficit": bool(lg["unresolved"].any()),
                             "forced_contracts": float(res["forced_contracts"][0]), "forced_events": int(res["forced_events"][0]),
                             "peak_im_nav": float((lg["im"] / lg["nav"]).max()),
                             "gap_day": gap_day, "hedge_contracts_at_entry": str(hedge_entry),
                             "hedge_contracts_before_gap": str(hedge_before),
                             "hedge_preserved_until_gap": (all(abs(hedge_before.get(k, 0)) >= abs(v) - 1e-9 for k, v in hedge_entry.items() if v < 0)
                                                           if gap_day else None),
                             "survives_path": bool((not lg["unresolved"].any())),
                             "probability": "NOT_ASSIGNED", "kind": "HYPOTHETICAL_FUTURE_EVENT_STRESS"})
    return pd.DataFrame(rows)
