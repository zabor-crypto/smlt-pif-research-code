"""Phase-B sleeve specifications from configuration + exchange data (margin rate, costs, contract notional)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import load


@dataclass
class SleeveSpec:
    name: str
    series: str
    asset: str
    family: str
    lo: float
    hi: float
    rolls_per_year: int
    mr: float                  # NCC base IM rate at cutoff
    cost_one_way: float        # (exchange fee + broker fee + half tick) / contract notional, all point-in-time
    contract_notional: float   # RUB per contract at the cutoff SETTLE (point-in-time, V3 0.6)

    def horizon_cost(self, tau: float) -> float:
        rolls = int(np.ceil(self.rolls_per_year * tau - 1e-9))
        return self.cost_one_way * (2 + 2 * rolls)

    def n_rolls(self, tau: float) -> int:
        return int(np.ceil(self.rolls_per_year * tau - 1e-9))

    def contracts(self, h: float, E: float) -> int:
        """V3 0.5: integer contracts for a signed notional fraction h of the SMLT exposure E (nearest integer)."""
        return int(np.round(h * E / self.contract_notional)) if self.contract_notional > 0 else 0

    def rounded_h(self, h: float, E: float) -> float:
        return self.contracts(h, E) * self.contract_notional / E


FAMILY_SPEC = {"SMLT": "SMLT", "MIX": "MIX", "RGBI": "RGBI", "GLDRUBF": "GLDRUBF", "CNYRUBF": "CNYRUBF", "BR": "BR",
               "BTC": "BTC", "BTCUSDF": "BTCUSDF", "IBIT": "IBIT"}


def build(ctx, extra: list[dict] | None = None) -> list[SleeveSpec]:
    """Sleeves from configuration + POINT-IN-TIME exchange terms (data/specs.pit_terms): no post-cutoff settle,
    margin or price enters notional, cost or risk (V3 0.6)."""
    from ..data.specs import pit_terms
    cfg = list(ctx.cfg["smart_portfolio"]["sleeves"]) + list(extra or [])
    terms = getattr(ctx, "_pit_terms", None)
    if terms is None:
        terms = pit_terms(ctx.root, ctx.cutoff).set_index("family")
        ctx._pit_terms = terms
    lim = ctx.read("ncc_futures_limits.csv", parse_dates=["tradedate"])
    last = lim[lim["tradedate"] <= pd.Timestamp(ctx.cutoff)].sort_values("tradedate").groupby("assetcode").tail(1).set_index("assetcode")
    broker = ctx.cfg["costs"]["broker_fee_futures_rub_per_contract"]
    out = []
    for c in cfg:
        t = terms.loc[FAMILY_SPEC[c["family"]]]
        notional = float(t["notional_pit"])
        fee = float(t["exchange_fee_rub_pit"])
        broker_fee = fee if c["family"] != "SMLT" else broker   # broker tariff = 100% of exchange fee (assumption)
        cost = (fee + broker_fee + float(t["half_tick_rub_pit"])) / notional
        out.append(SleeveSpec(c["name"], c["series"], c["asset"], c["family"], c["lo"], c["hi"], c["rolls_per_year"],
                              float(last.loc[c["asset"], "mr1"]), cost, notional))
    return out
