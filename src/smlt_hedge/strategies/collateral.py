"""Generic collateral / margin regimes shared by every engine (bootstrap, contract simulator, scenarios)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Collateral:
    """Generic collateral / margin regime (no specific broker is assumed).

    yield_fraction  : share of posted initial margin that is posted as productive collateral (e.g. MMF units) and
                      therefore still earns the money-market yield; the rest is idle cash (earns 0).
    haircut         : haircut applied to productive collateral -> posting X of margin with it consumes X/(1-haircut)
                      of capital.
    margin_multiplier: broker add-on over the exchange (NCC) initial-margin rate.
    """
    name: str = "A_CASH_ONLY_CONSERVATIVE"
    yield_fraction: float = 0.0
    haircut: float = 0.0
    margin_multiplier: float = 1.0

    def capital_per_margin(self) -> float:
        return (1.0 - self.yield_fraction) + self.yield_fraction / (1.0 - self.haircut)

    def earning_per_margin(self) -> float:
        return self.yield_fraction / (1.0 - self.haircut)


CASH_ONLY = Collateral()


REGIMES = {
    "A_CASH_ONLY_CONSERVATIVE": Collateral("A_CASH_ONLY_CONSERVATIVE", 0.0, 0.0, 1.0),
    "B_PARTIALLY_PRODUCTIVE": Collateral("B_PARTIALLY_PRODUCTIVE", 0.5, 0.10, 1.0),
    "C_HIGH_COLLATERAL_EFFICIENCY": Collateral("C_HIGH_COLLATERAL_EFFICIENCY", 1.0, 0.05, 1.0),
}


def grid(yield_fractions=(0.0, 0.5, 1.0), multipliers=(1.0, 1.25, 1.5), haircut=0.10) -> list[Collateral]:
    out = []
    for y in yield_fractions:
        for mm in multipliers:
            out.append(Collateral(f"yf{y:.2f}_hc{haircut if y > 0 else 0:.2f}_mm{mm:.2f}", y, haircut if y > 0 else 0.0, mm))
    return out
