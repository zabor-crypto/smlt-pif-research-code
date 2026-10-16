"""Gate R3 — forward scenario paths for the candidate generator (LP) and the admission authority (engine).

Model set (risk only; means are set by the return-view registry, never by the sample mean):
  HS_BLOCK      contiguous 10-session blocks of the joint route-return panel, 2022-03-28 .. t
  FHS_EWMA      the same blocks of EWMA-standardised residuals (lambda 0.94, sigma_t from data <= t-1), rescaled by
                the EWMA volatility at the decision date t (captures the current risk state)
  HS_RECENT_1Y  contiguous blocks from the last 252 sessions only
Blocks are contiguous calendar sessions; no state filtering and no splicing of non-consecutive state days (D3-11).

Centring: each route gets a constant daily log shift so its mean funded or fixed-count derivative horizon return over the
path set equals the declared central view exactly (funded: R - R_MMF; futures + collateral: R_F). NO_EXCESS_VIEW
therefore means E[R - R_MMF] = 0 at the horizon, not zero total return (D2-02) and not a daily-centring residual (D2-03).
The money-market path is deterministic at the trailing 60-session LQDT yield at t (observable).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SIM_ROUTES = ["SMLT", "SMLT_FUT", "EQ_FUNDED", "EQ_HEDGE_FUT", "OFZ_FUNDED", "RATES_HEDGE_FUT", "GOLD_FUNDED",
              "CNY_PERP", "OIL_FUT_USD", "BTC_RUB_EXPERIMENTAL", "USDRUB"]
FUNDED_ROUTES = {"SMLT", "EQ_FUNDED", "OFZ_FUNDED", "GOLD_FUNDED", "BTC_RUB_EXPERIMENTAL"}
MODELS = ("HS_BLOCK", "FHS_EWMA", "HS_RECENT_1Y")


@dataclass
class PathSet:
    model: str
    R: np.ndarray            # (N, H, K) daily route returns (centred)
    mmf_daily: np.ndarray    # (H,)
    routes: list
    shift: dict              # applied daily log shift per route
    cny_swap_frac: np.ndarray | None = None  # raw auxiliary, SAME block chronology, never FHS-scaled
    indices: np.ndarray | None = None       # full panel row positions (diagnostic chronology)
    centring: list = field(default_factory=list)
    dspecs: list | None = None


def history_panel(P: dict, t: pd.Timestamp, start: str) -> pd.DataFrame:
    """Joint daily panel up to t with labelled proxy fills before a route's own history exists."""
    R, B = P["R"], P["B"]
    d = pd.DataFrame(index=P["cal"])
    for r in SIM_ROUTES:
        if r == "OIL_FUT_USD":
            d[r] = B["OIL_FUT_USD"]
        elif r == "USDRUB":
            d[r] = B["USDRUB_CBR"]
        else:
            d[r] = R[r]
    # labelled benchmark proxies before the route's own history (estimation only; trading respects availability)
    d["GOLD_FUNDED"] = d["GOLD_FUNDED"].fillna(B["GLDRUB_TOM"])
    actual_cny = d["CNY_PERP"].notna()
    d["CNY_PERP"] = d["CNY_PERP"].fillna(R["CNY_FUNDED"] - R["MMF"].fillna(0.0))
    cny = P["fut"]["CNY_PERP"].reindex(d.index)
    d["CNY_SWAP_FRAC"] = (cny["swap_pts"] / cny["Fn"].shift(1)).where(actual_cny, R["MMF"].fillna(0.0))
    # Before listing this is ASSUMPTION / ESTIMATION_PROXY_ONLY; execution availability is unchanged.
    mmf = R["MMF"].copy()
    ruo = B.get("RUONIA_ACCRUAL")
    if ruo is not None:
        mmf = mmf.fillna(ruo)
    d["MMF"] = mmf
    d = d.loc[start:t]
    d = d.iloc[1:] if d.index[0] == P["cal"][0] else d
    return d.dropna()


def mmf_forward_daily(panel: pd.DataFrame, H: int, window: int = 60) -> np.ndarray:
    y = float((1 + panel["MMF"].iloc[-window:]).prod() ** (1 / window) - 1)
    return np.full(H, y)


def block_indices(T: int, N: int, H: int, L: int, rng) -> np.ndarray:
    nb = int(np.ceil(H / L))
    starts = rng.integers(0, T - L + 1, size=(N, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(N, nb * L)[:, :H]
    return idx


def _blocks(X: np.ndarray, N: int, H: int, L: int, rng) -> np.ndarray:
    return X[block_indices(X.shape[0], N, H, L, rng)]


def ewma_sigma(X: np.ndarray, lam: float = 0.94, init: int = 20) -> np.ndarray:
    """sigma_t estimated from returns up to t-1 (row t uses rows < t); last row = sigma at the decision date."""
    T, K = X.shape
    s2 = np.empty((T + 1, K))
    s2[0] = X[:init].var(axis=0, ddof=1)
    for i in range(T):
        s2[i + 1] = lam * s2[i] + (1 - lam) * X[i] ** 2
    return np.sqrt(np.maximum(s2, 1e-16))


def simulate(panel: pd.DataFrame, model: str, N: int, H: int, seed: int, block: int = 10,
             panel_now: pd.DataFrame | None = None) -> PathSet:
    """`panel`: bootstrap sample (frozen at the last recalibration). `panel_now` (optional, data <= decision date):
    source of the CURRENT EWMA volatility and money-market rate, which are observable inputs updated at each decision."""
    routes = [c for c in SIM_ROUTES if c in panel]
    X = panel[routes].values.astype(float)
    now = panel if panel_now is None else panel_now
    rng = np.random.default_rng(seed)
    if model == "HS_BLOCK":
        idx = block_indices(len(X), N, H, block, rng)
        R = X[idx]
    elif model == "HS_RECENT_1Y":
        offset = max(len(X) - 252, 0)
        idx = block_indices(len(X) - offset, N, H, block, rng) + offset
        R = X[idx]
    elif model == "FHS_EWMA":
        s = ewma_sigma(X)
        Z = X / s[:-1]
        s_now = s[-1] if panel_now is None else ewma_sigma(now[routes].values.astype(float))[-1]
        idx = block_indices(len(X), N, H, block, rng)
        R = Z[idx] * s_now[None, None, :]
        R = np.maximum(R, -0.95)
    else:
        raise ValueError(model)
    swap = panel["CNY_SWAP_FRAC"].values.astype(float)[idx] if "CNY_SWAP_FRAC" in panel else None
    return PathSet(model, R, mmf_forward_daily(now, H), routes, {}, swap, idx)


def centre(ps: PathSet, target_excess: dict, dspecs: list | None = None) -> PathSet:
    """Funded centring unchanged. Derivatives: deterministic scalar solve in fixed-count engine units."""
    from . import derivative_units as U
    R = ps.R.copy()
    H = R.shape[1]
    mH = float(np.prod(1 + ps.mmf_daily) - 1)
    specs = dspecs if dspecs is not None else U.default_specs(ps)
    by_series = {U.SERIES.get(d.name, d.name): d for d in specs}
    shift, diagnostics = {}, []
    for k, r in enumerate(ps.routes):
        if r not in target_excess or r == "USDRUB":
            continue
        if r in by_series:
            R[:, :, k], a, diagnostic = U.centre_route(ps, by_series[r], k, target_excess[r])
            diagnostics.append(diagnostic)
        elif r in FUNDED_ROUTES:
            g = np.prod(1 + R[:, :, k], axis=1)
            tgt = 1 + target_excess[r] + mH
            a = (np.log(tgt) - np.log(g.mean())) / H
            R[:, :, k] = (1 + R[:, :, k]) * np.exp(a) - 1
        else:
            raise ValueError(f"missing derivative contract terms: {r}")
        shift[r] = a
    return PathSet(ps.model, R, ps.mmf_daily, ps.routes, shift, ps.cny_swap_frac, ps.indices, diagnostics, specs)


def horizon_excess(ps: PathSet, market=None) -> pd.DataFrame:
    """Funded product(1+r)-1; derivative Market-authoritative fixed-count VM / initial notional."""
    from . import derivative_units as U
    mH = float(np.prod(1 + ps.mmf_daily) - 1)
    specs = ps.dspecs if ps.dspecs is not None else U.default_specs(ps)
    values = {}
    if specs:
        m = market if market is not None else U.path_market(ps, specs)
        g = U.horizon(m)
        values = {U.SERIES.get(d.name, d.name): g[:, j] for j, d in enumerate(specs)}
    return pd.DataFrame({r: values[r] if r in values else np.prod(1 + ps.R[:, :, k], axis=1) - 1 -
                         (mH if r in FUNDED_ROUTES else 0.) for k, r in enumerate(ps.routes)})
