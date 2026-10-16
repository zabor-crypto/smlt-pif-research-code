"""Regime V2 — joint forward scenario models and state-conditional dependence diagnostics.

Every model resamples ACTUAL joint history: a block is 10 consecutive sessions of the whole cross-section (the same
dates for every asset). No asset is simulated on its own and no pairwise correlation is imposed.

  GLOBAL_FORWARD             blocks from the whole sample (parent HS_BLOCK)
  GLOBAL_FORWARD_FHS         blocks of EWMA-standardised returns rescaled to the volatility at the decision date
                             (parent FHS_EWMA)
  RECENT_FORWARD             blocks from the last 252 sessions (parent HS_RECENT_1Y)
  STATE_CONDITIONAL_FORWARD  a block may start only at session i whose PREVIOUS session i-1 is in the matched confirmed
                             state: the block is what actually followed an origin in that state. The return of the
                             state-defining day itself is never the first return of its own block. If the matched state
                             has fewer than 3 distinct episodes or fewer than 126 eligible block starts, the condition is
                             reduced along the declared hierarchy; if no level qualifies the model is NOT AVAILABLE and
                             admission rests on the three unconditional models (no fabricated conditional distribution).

Path means are set to the mechanical carry only (`scenarios.centre`), so the risk measure does not depend on estimated
premia or state signals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..adaptive import scenarios as SC
from . import states as S2
from .cfg import StopCondition

MODELS = ("GLOBAL_FORWARD", "GLOBAL_FORWARD_FHS", "RECENT_FORWARD", "STATE_CONDITIONAL_FORWARD")
PARENT_MODEL = {"GLOBAL_FORWARD": "HS_BLOCK", "GLOBAL_FORWARD_FHS": "FHS_EWMA", "RECENT_FORWARD": "HS_RECENT_1Y"}
STATE_MODEL = "STATE_CONDITIONAL_FORWARD"


def eligible_starts(index: pd.DatetimeIndex, conf: pd.DataFrame, cond: dict, L: int) -> tuple[np.ndarray, int]:
    """Row positions i (1 <= i <= T - L) of `index` whose previous row is in the state; number of distinct episodes that
    supply at least one eligible origin."""
    m = S2.cond_mask(conf.reindex(index).ffill(), cond)
    T = len(index)
    origin_ok = m.values[:-1]                                 # origin = row i-1
    pos = np.arange(1, T)
    ok = origin_ok & (pos <= T - L)
    starts = pos[ok]
    ids = S2.episode_ids(m).values[:-1][ok]
    return starts, int(len(np.unique(ids))) if len(ids) else 0


def state_model_choice(index: pd.DatetimeIndex, conf: pd.DataFrame, states_now: dict, cfg: dict) -> dict:
    """The first level of the declared hierarchy with enough evidence; `available` False if none."""
    c = cfg["v2"]["risk"]["STATE_CONDITIONAL_FORWARD"]
    L = int(cfg["v2"]["risk"]["block_length"])
    tried = []
    for axes in c["hierarchy"]:
        cond = {a: states_now[a] for a in axes}
        starts, n_ep = eligible_starts(index, conf, cond, L)
        ok = n_ep >= int(c["min_distinct_episodes"]) and len(starts) >= int(c["min_eligible_block_starts"])
        tried.append({"level": S2.cond_label(cond), "n_axes": len(axes), "n_episodes": n_ep, "n_eligible_block_starts": int(len(starts)),
                      "qualifies": bool(ok)})
        if ok:
            return {"available": True, "condition": cond, "level": S2.cond_label(cond), "starts": starts, "n_episodes": n_ep,
                    "n_eligible_block_starts": int(len(starts)), "tried": tried}
    return {"available": False, "condition": None, "level": "MODEL_NOT_AVAILABLE_FALLBACK_TO_GLOBAL_AND_RECENT", "starts": None,
            "n_episodes": 0, "n_eligible_block_starts": 0, "tried": tried}


def simulate_state(panel: pd.DataFrame, starts: np.ndarray, N: int, H: int, seed: int, L: int,
                   panel_now: pd.DataFrame | None = None) -> SC.PathSet:
    routes = [c for c in SC.SIM_ROUTES if c in panel]
    X = panel[routes].values.astype(float)
    now = panel if panel_now is None else panel_now
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(H / L))
    pick = starts[rng.integers(0, len(starts), size=(N, nb))]
    idx = (pick[:, :, None] + np.arange(L)[None, None, :]).reshape(N, nb * L)[:, :H]
    swap = panel["CNY_SWAP_FRAC"].values.astype(float)[idx] if "CNY_SWAP_FRAC" in panel else None
    return SC.PathSet(STATE_MODEL, X[idx], SC.mmf_forward_daily(now, H), routes, {}, swap, idx)


def simulate(model: str, panel: pd.DataFrame, N: int, H: int, seed: int, L: int, panel_now: pd.DataFrame | None,
             state_choice: dict | None) -> SC.PathSet | None:
    if model == STATE_MODEL:
        if not state_choice or not state_choice["available"]:
            return None
        return simulate_state(panel, state_choice["starts"], N, H, seed, L, panel_now)
    ps = SC.simulate(panel, PARENT_MODEL[model], N, H, seed, L, panel_now)
    ps.model = model
    return ps


# ---------------------------------------------------------------------------------------------- dependence diagnostics
def excess_panel(P: dict, routes: list, funded: set, start: str, t: pd.Timestamp) -> pd.DataFrame:
    """Daily excess returns of the LP routes (funded: minus the money market; derivatives: the futures return)."""
    R = P["R"].loc[start:t]
    mmf = R["MMF"].fillna(P["B"]["RUONIA_ACCRUAL"].loc[start:t])
    d = pd.DataFrame({r: (R[r] - mmf if r in funded else R[r]) for r in routes})
    return d.dropna().astype(float)


def _div(a: float, b: float) -> float:
    return float(a / b) if b > 0 and np.isfinite(b) else float("nan")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    sa, sb = float(a.std(ddof=1)), float(b.std(ddof=1))
    return _div(float(np.cov(a, b, ddof=1)[0, 1]), sa * sb)


def _pd_check(S: np.ndarray, what: str) -> None:
    try:
        np.linalg.cholesky(S)
    except np.linalg.LinAlgError as e:
        raise StopCondition(f"covariance not positive definite after the declared shrinkage: {what}") from e


def state_sets(conf: pd.DataFrame, states_now: dict, state_choice: dict | None) -> list[tuple]:
    sets = [("GLOBAL", "ALL_ORIGINS", None)]
    for a in S2.AXES:
        for v in S2.ORDER[a]:
            sets.append(("MARGINAL", f"{a}={v}", {a: v}))
    if state_choice and state_choice["available"]:
        sets.append(("CURRENT_JOINT_STATE_MODEL", state_choice["level"], state_choice["condition"]))
    return sets


def dependence(X: pd.DataFrame, conf: pd.DataFrame, states_now: dict, state_choice: dict | None, cfg: dict, P: dict | None = None) -> dict:
    """Covariance / correlation (raw and shrunk toward the global matrix) and tail co-movement by state.

    Observation for state s: the NEXT-session excess return r_{t+1} of every route, for origins t in s."""
    from sklearn.covariance import LedoitWolf
    c = cfg["v2"]["risk"]["covariance_diagnostics"]
    L = int(cfg["v2"]["risk"]["block_length"])
    routes = list(X.columns)
    p = len(routes)
    G = LedoitWolf().fit(X.values).covariance_
    _pd_check(G, "GLOBAL (Ledoit-Wolf)")
    cf = conf.reindex(X.index).ffill()
    q, Ht = float(c["tail_quantile"]), int(c["tail_horizon_sessions"])
    fwd = np.expm1(np.log1p(X)[::-1].rolling(Ht, min_periods=Ht).sum()[::-1].shift(-1))
    from ..adaptive.derivative_units import DERIVATIVES, historical_forward
    for route in routes:
        if route in DERIVATIVES:
            if P is None:
                raise ValueError("derivative tail horizon requires full historical contractual panel")
            # Reconstruct on the FULL session calendar before reindexing the complete-case daily X.
            fwd[route] = historical_forward(P["fut"][route].reindex(P["cal"]), Ht).reindex(X.index)
    cov_rows, cor_rows, tail_rows = [], [], []
    for kind, label, cond in state_sets(conf, states_now, state_choice):
        if cond is None:
            sel = np.ones(len(X), bool)
            sel[0] = False
            n_ep = 1
        else:
            m = S2.cond_mask(cf, cond)
            sel = np.r_[False, m.values[:-1]]                       # row i is used if origin i-1 is in the state
            n_ep = int(len(np.unique(S2.episode_ids(m).values[:-1][m.values[:-1]]))) if m.values[:-1].any() else 0
        n = int(sel.sum())
        base = {"state_set": kind, "state": label, "n_obs": n, "n_episodes": n_ep,
                "is_current": bool(cond is not None and all(states_now.get(a) == v for a, v in cond.items()))}
        if n < p + 2:
            cov_rows.append({**base, "status": "NOT_ENOUGH_OBSERVATIONS"})
            continue
        Xs = X.values[sel]
        Sraw = np.cov(Xs, rowvar=False, ddof=1)
        d = 0.0 if cond is None else p / (p + n / L)
        Sst = G if cond is None else (1 - d) * Sraw + d * G
        _pd_check(Sst, label)
        sd_r, sd_s = np.sqrt(np.diag(Sraw)), np.sqrt(np.diag(Sst))
        for i, a in enumerate(routes):
            for j, b in enumerate(routes):
                if j < i:
                    continue
                cov_rows.append({**base, "route_i": a, "route_j": b, "cov_ann_raw": float(Sraw[i, j] * 252),
                                 "cov_ann_shrunk": float(Sst[i, j] * 252), "shrink_weight_on_global": d, "status": "OK"})
                cor_rows.append({**base, "route_i": a, "route_j": b, "corr_raw": _div(Sraw[i, j], sd_r[i] * sd_r[j]),
                                 "corr_shrunk": _div(Sst[i, j], sd_s[i] * sd_s[j]), "shrink_weight_on_global": d})
        js = routes.index("SMLT")
        smlt = Xs[:, js]
        down = smlt < 0
        worst = smlt <= np.quantile(smlt, q)
        f = fwd.values[np.r_[sel[1:], False]] if cond is not None else fwd.values     # forward windows from the same origins
        f = f[~np.isnan(f).any(axis=1)]
        for i, a in enumerate(routes):
            row = {**base, "route": a, "vol_ann_shrunk": float(sd_s[i] * np.sqrt(252)),
                   "beta_of_SMLT_on_route": _div(Sst[js, i], Sst[i, i]) if i != js else 1.0,
                   "corr_with_SMLT_shrunk": _div(Sst[js, i], sd_s[js] * sd_s[i])}
            if i != js:
                row["downside_corr_with_SMLT"] = _corr(smlt[down], Xs[down, i])
                row["mean_return_on_SMLT_worst_decile_days"] = float(Xs[worst, i].mean())
                if len(f) >= 20:
                    ts = f[:, js] <= np.quantile(f[:, js], q)
                    tr = f[:, i] <= np.quantile(f[:, i], q)
                    row["tail_coexceedance_21d"] = float((ts & tr).sum() / max(ts.sum(), 1))
                    row["tail_coexceedance_21d_if_independent"] = q
                    row["n_forward_windows_overlapping"] = int(len(f))
            tail_rows.append(row)
    return {"covariance": pd.DataFrame(cov_rows), "correlation": pd.DataFrame(cor_rows), "tail": pd.DataFrame(tail_rows)}
