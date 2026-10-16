"""V4 asset universe: FUNDED assets (consume NAV) and DERIVATIVES (consume margin, VM liquidity and costs only).

All positions are fractions of fund NAV. The liquidity sleeve (money-market / LQDT-type) is the residual funded asset:
cash = 1 - sum(funded risky weights); posted initial margin is taken from cash and earns nothing (cash-only
collateral in the main case; collateral regimes are a sensitivity).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..portfolio import sleeves as sleeves_mod


@dataclass
class Asset:
    name: str
    kind: str              # FUNDED | DERIVATIVE
    series: str
    lo: float
    hi: float
    cost_one_way: float    # per unit of traded value / notional
    mr: float = 0.0        # initial-margin rate (derivatives)
    rolls_per_year: int = 0
    role: str = ""
    contract_notional: float = 0.0
    execution: str = ""

    def horizon_cost(self, tau: float) -> float:
        """Round-trip + roll costs over the horizon per unit of |position| (derivatives); funded: entry only."""
        if self.kind == "FUNDED":
            return self.cost_one_way
        rolls = int(np.ceil(self.rolls_per_year * tau - 1e-9))
        return self.cost_one_way * (2 + 2 * rolls)


def build(ctx, universe: str, nav0: float, smlt_cap: float) -> list[Asset]:
    v4 = ctx.cfg["v4"]
    names = v4["universes"][universe]
    S0 = float(ctx.read("prices.csv", index_col=0, parse_dates=True)["SMLT"].loc[ctx.cutoff])
    eq_cost = (0.1 + v4["equity_cost_bps"] * 1e-4 * S0) / S0
    dcfg = [d for d in v4["derivatives"] if d["name"] in names]
    specs = {s.name: s for s in sleeves_mod.build(ctx, extra=[{**d, "name": d["name"]} for d in dcfg])
             if s.name in {d["name"] for d in dcfg}} if dcfg else {}
    out = []
    for f in v4["funded"]:
        if f["name"] not in names:
            continue
        if f["name"] == "SMLT":
            out.append(Asset("SMLT", "FUNDED", "SMLT", 0.0, smlt_cap, eq_cost, role=f["role"], contract_notional=S0,
                             execution="MOEX TQBR shares (lot 1)"))
        else:
            out.append(Asset(f["name"], "FUNDED", f["series"], f["lo"], f["hi"], v4["btc_spot_cost_one_way"], role=f["role"],
                             execution="INSTRUCTOR_AUTHORIZED_EXPERIMENTAL (not verified as tradable for every real PIF)"))
    for d in dcfg:
        s = specs[d["name"]]
        out.append(Asset(d["name"], "DERIVATIVE", d["series"], d["lo"], d["hi"], s.cost_one_way, s.mr, d["rolls_per_year"],
                         role="MOEX derivative", contract_notional=s.contract_notional,
                         execution=f"MOEX FORTS {d['family']} (PIT terms)"))
    return out


def initial_weights(assets: list[Asset], E: float, C: float) -> np.ndarray:
    """Initial fund portfolio: SMLT = E / NAV0, everything else 0 (cash = C / NAV0 is the residual)."""
    w = np.zeros(len(assets))
    for i, a in enumerate(assets):
        if a.name == "SMLT":
            w[i] = E / (E + C)
    return w
