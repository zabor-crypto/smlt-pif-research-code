"""Regime V2 — multi-axis observable state vector and forward statistics by (joint) state.

Axes (each measured at close t with data through t; trailing windows only):
  ISSUER_CREDIT_RISK, MARKET_RISK, RATES_STATE   the parent rules, unchanged (adaptive.state)
  RUB_STATE                                       new: official USD/RUB known at the close;
                                                  r20 = log(USD_t / USD_{t-20});  s = trailing 250-session std of daily log
                                                  returns;  WEAKENING if r20 >= +k s sqrt(20), STRENGTHENING if r20 <= -k s
                                                  sqrt(20), else STABLE (k = 1.0, fixed before any output).
The policy uses CONFIRMED states (3 closes to deteriorate, 10 to relax — the parent rule, applied to every axis).

Forward statistics: for an origin t the compounded excess return over sessions t+1 .. t+H (never r_t). A state may be a
JOINT condition on several axes; its episodes are the contiguous runs of sessions on which the whole condition holds.
"""
from __future__ import annotations

import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..adaptive import events as EV
from ..adaptive import fx
from ..adaptive import policy as PL
from ..adaptive import state as ST
from ..adaptive.evidence import MANIFEST

AXES = ("ISSUER_CREDIT_RISK", "MARKET_RISK", "RATES_STATE", "RUB_STATE")
ORDER = {**ST.ORDER, "RUB_STATE": ["RUB_STRENGTHENING", "RUB_STABLE", "RUB_WEAKENING"]}


# ---------------------------------------------------------------------------------------------- RUB axis
def usd_records_long(root: Path) -> pd.Series:
    """Official USD/RUB by VALUE date, 2013 .. cutoff: prior-era fetch + the research-window fetch of the parent."""
    m = pd.read_csv(Path(root) / "data" / MANIFEST, dtype=str)
    m = m[m["dataset"].isin(["usd_official_prior", "usd_official"]) & (m["validation_status"] == "VALID")]
    rec = {}
    for rp in sorted(m["raw_path"]):
        t = (Path(root) / rp).read_bytes().decode("windows-1251")
        for d, v in re.findall(r'<Record Date="([0-9.]+)" Id="R01235"><Nominal>1</Nominal><Value>([0-9,]+)</Value>', t):
            rec[pd.Timestamp(pd.to_datetime(d, format="%d.%m.%Y"))] = float(v.replace(",", "."))
    return pd.Series(rec).sort_index()


def rub_features(usd: pd.Series, c: dict) -> pd.DataFrame:
    """`usd`: rate known at each close (index = sessions, may start before the research calendar for the warm-up)."""
    lr = np.log(usd).diff()
    n, w = int(c["return_window_sessions"]), int(c["vol_window_sessions"])
    r = np.log(usd) - np.log(usd).shift(n)
    s = lr.rolling(w, min_periods=int(c["vol_min_sessions"])).std()
    thr = float(c["threshold_sigma"]) * s * np.sqrt(n)
    return pd.DataFrame({"usdrub_known": usd, "rub_logret_window": r, "rub_daily_vol_trailing": s, "rub_threshold": thr})


def classify_rub(f: pd.DataFrame) -> pd.Series:
    r, thr = f["rub_logret_window"], f["rub_threshold"]
    out = np.where(r >= thr, "RUB_WEAKENING", np.where(r <= -thr, "RUB_STRENGTHENING", "RUB_STABLE"))
    out = np.where(r.isna() | thr.isna(), "RUB_STABLE", out)       # warm-up: no evidence -> STABLE
    return pd.Series(out, index=f.index, name="RUB_STATE")


def raw_states(ctx, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(features, raw states) on the research calendar; every row uses data up to its own close only."""
    f3 = ST.features(ctx, cfg)
    s3 = ST.classify(f3, cfg)
    cal = f3.index
    rec = usd_records_long(ctx.root)
    warm = pd.bdate_range(cal[0] - pd.Timedelta(days=560), cal[0] - pd.Timedelta(days=1))
    sessions = warm.append(cal)
    usd = fx.usd_known(rec, sessions, 1).dropna()
    fr = rub_features(usd, cfg["v2"]["state"]["rub"]).reindex(cal)
    rub = classify_rub(fr)
    return pd.concat([f3, fr], axis=1), pd.concat([s3, rub], axis=1)


def severe_series(ctx, cfg: dict, index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.Series]:
    from . import review_context as RC
    E = RC.corrected_register(ctx, cfg) if getattr(ctx, "_regime_v2r2_combined", False) else EV.register(ctx, cfg)
    return E, pd.Series([EV.state_at(E, d)["severe_restriction"] for d in index], index=index)


def confirmed(raw: pd.DataFrame, cfg: dict, severe: pd.Series | None = None) -> pd.DataFrame:
    c = cfg["v2"]["state"]["confirmation"]
    return pd.DataFrame({a: PL.confirm(raw[a], ORDER[a], c["deterioration_consecutive_closes"],
                                       c["rerisk_normalisation_consecutive_closes"], severe) for a in AXES})


# ---------------------------------------------------------------------------------------------- joint conditions
def cond_key(cond: dict) -> tuple:
    return tuple((a, cond[a]) for a in AXES if a in cond)


def cond_label(cond: dict) -> str:
    return " & ".join(f"{a}={v}" for a, v in cond_key(cond)) or "ALL_ORIGINS"


def cond_mask(conf: pd.DataFrame, cond: dict) -> pd.Series:
    m = pd.Series(True, index=conf.index)
    for a, v in cond.items():
        m &= conf[a] == v
    return m


def episode_ids(mask: pd.Series) -> pd.Series:
    """Episode id of each session on which the condition holds (contiguous runs); -1 elsewhere."""
    v = mask.values.astype(bool)
    start = v & ~np.r_[False, v[:-1]]
    ids = np.cumsum(start) - 1
    return pd.Series(np.where(v, ids, -1), index=mask.index)


def episode_table(conf: pd.DataFrame, start: str) -> pd.DataFrame:
    rows = []
    c = conf.loc[start:]
    for a in AXES:
        e = ST.episodes(c[a])
        for i, r in e.iterrows():
            rows.append({"axis": a, "episode_id": int(i), "state": r["state"], "start": str(pd.Timestamp(r["start"]).date()),
                         "end": str(pd.Timestamp(r["end"]).date()), "sessions": int(r["sessions"])})
    return pd.DataFrame(rows)


def forward_excess(R: pd.DataFrame, mmf: pd.Series, route: str, funded: bool, H: int, terms: pd.DataFrame | None = None) -> pd.Series:
    if funded:
        return ST.forward_returns(R[route], H) - ST.forward_returns(mmf, H)
    if terms is None:
        raise ValueError(f"derivative horizon requires observed Fp/Fn/pv: {route}")
    from ..adaptive.derivative_units import historical_forward
    return historical_forward(terms.reindex(R.index), H)


def _seed(base: int, *parts) -> np.random.Generator:
    return np.random.default_rng([int(base), zlib.crc32("|".join(str(p) for p in parts).encode())])


def cond_stats(x: pd.Series, mask: pd.Series, H: int, n_boot: int, seed: int, tag: str, min_ep: int, min_win: int) -> dict:
    """Forward statistics of `x` (forward excess by origin date) after origins where `mask` holds.

    n_episodes counts DISTINCT episodes that contribute at least one complete forward window; the interval and the
    standard error come from resampling whole episodes. The all-origin mean is treated as fixed (stated limitation).
    Admissible (may enter the expected return) only with >= min_ep episodes, >= min_win non-overlapping windows and an
    interval that excludes the all-origin mean. The value used is the differential shrunk by its own reliability."""
    x = x.reindex(mask.index)
    valid = x.notna()
    glob = x[valid]
    gmean = float(glob.mean()) if len(glob) else np.nan
    m = mask & valid
    out = {"n_origins": int(m.sum()), "n_episodes": 0, "n_nonoverlapping_windows": 0, "all_origin_mean": gmean,
           "n_all_origins": int(valid.sum()), "mean_fwd_excess": np.nan, "median_fwd_excess": np.nan, "vol_fwd_excess": np.nan,
           "differential": np.nan, "se_episode_bootstrap": np.nan, "ci90_lo": np.nan, "ci90_hi": np.nan,
           "ci90_excludes_all_origin_mean": False, "reliability": 0.0, "differential_used": 0.0, "admissible": False,
           "status": "NO_ORIGINS"}
    if not m.any():
        return out
    ids = episode_ids(mask)[m]
    xs = x[m]
    pos = np.flatnonzero(m.values)
    nonov, last = 0, -10**9
    for p_ in pos:
        if p_ - last >= H:
            nonov += 1
            last = p_
    uniq = np.unique(ids.values)
    sums = np.array([xs.values[ids.values == i].sum() for i in uniq])
    cnts = np.array([(ids.values == i).sum() for i in uniq], float)
    rng = _seed(seed, tag, H)
    pick = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boots = sums[pick].sum(axis=1) / cnts[pick].sum(axis=1)
    mean = float(xs.mean())
    se = float(boots.std(ddof=1)) if len(uniq) > 1 else np.nan
    lo, hi = float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))
    d = mean - gmean
    excl = bool((lo > gmean) or (hi < gmean))
    enough = len(uniq) >= min_ep and nonov >= min_win
    adm = bool(enough and excl and np.isfinite(se))
    rel = max(0.0, 1.0 - se * se / (d * d)) if adm and d != 0 else 0.0
    out.update({"n_episodes": int(len(uniq)), "n_nonoverlapping_windows": int(nonov), "mean_fwd_excess": mean,
                "median_fwd_excess": float(xs.median()), "vol_fwd_excess": float(xs.std(ddof=1)) if len(xs) > 1 else np.nan,
                "differential": d, "se_episode_bootstrap": se, "ci90_lo": lo, "ci90_hi": hi,
                "ci90_excludes_all_origin_mean": excl, "reliability": rel, "differential_used": rel * d, "admissible": adm,
                "status": ("ADMISSIBLE" if adm else "STATE_RETURN_VIEW_NOT_ADMISSIBLE")})
    if not adm:
        out["why_not"] = ("fewer than %d distinct episodes" % min_ep if len(uniq) < min_ep else
                          "fewer than %d non-overlapping windows" % min_win if nonov < min_win else
                          "episode interval includes the all-origin mean")
    return out


class ForwardLibrary:
    """Forward excess series of every route on data <= t_recal, and memoised joint-condition statistics."""

    def __init__(self, cfg: dict, P: dict, conf: pd.DataFrame, t_recal: pd.Timestamp, n_boot: int, funded: set,
                 routes: list, carry_by_origin: dict | None = None):
        sa = cfg["v2"]["returns"]["state_adjustment"]
        self.cfg, self.sa, self.t = cfg, sa, t_recal
        self.n_boot = int(n_boot)
        self.conf = conf.loc[cfg["v2"]["state"]["sample_start"]:t_recal]
        R = P["R"].loc[:t_recal]
        mmf = R["MMF"].fillna(P["B"]["RUONIA_ACCRUAL"].loc[:t_recal])
        self.R, self.mmf, self.funded, self.routes = R, mmf, funded, routes
        self.terms = {r: h.loc[:t_recal] for r, h in P.get("fut", {}).items()}
        self._x, self._memo = {}, {}
        self.carry_by_origin = carry_by_origin

    def x(self, route: str, H: int) -> pd.Series:
        k = (route, H)
        if k not in self._x:
            total = forward_excess(self.R, self.mmf, route, route in self.funded, H, self.terms.get(route)).reindex(self.conf.index)
            if self.carry_by_origin is None:
                self._x[k] = total  # legacy synthetic use; canonical V2R1 Model always provides carry
            else:
                table = self.carry_by_origin[int(H)].reindex(self.conf.index)
                known = table[route] if route in table else pd.Series(np.nan,index=self.conf.index)
                self._x[k] = total - known
        return self._x[k]

    def stats(self, route: str, cond: dict, H: int) -> dict:
        k = (route, cond_key(cond), H)
        if k not in self._memo:
            s = cond_stats(self.x(route, H), cond_mask(self.conf, cond), H, self.n_boot, self.sa["bootstrap_seed"],
                           f"{route}|{cond_label(cond)}", self.sa["min_distinct_episodes"], self.sa["min_nonoverlapping_windows"])
            self._memo[k] = {"basis": "RESIDUAL_AFTER_ORIGIN_KNOWN_CARRY" if self.carry_by_origin is not None else "TOTAL_EXCESS_LEGACY", "route": route, "condition": cond_label(cond), "n_axes": len(cond), "horizon": H, **s}
        return self._memo[k]

    def adjustment(self, route: str, states_now: dict) -> dict:
        """Hierarchy: all listed axes jointly -> drop the last axis -> ... -> first axis alone -> no adjustment."""
        axes = self.sa["axes_by_route"].get(route)
        H = int(self.sa["horizon_primary"])
        tried = []
        if self.carry_by_origin is not None and route not in self.carry_by_origin.get(H, pd.DataFrame()).columns:
            return {"route":route,"value":0.0,"level":"NO_CLEAN_CARRY_DECOMPOSITION","stats":None,"tried":tried}
        if not axes:
            return {"route": route, "value": 0.0, "level": "NOT_MAPPED", "stats": None, "tried": tried}
        for k in range(len(axes), 0, -1):
            cond = {a: states_now[a] for a in axes[:k]}
            s = self.stats(route, cond, H)
            tried.append(s)
            if s["admissible"]:
                return {"route": route, "value": float(s["differential_used"]), "level": cond_label(cond), "stats": s, "tried": tried}
        return {"route": route, "value": 0.0, "level": "STATE_RETURN_VIEW_NOT_ADMISSIBLE", "stats": None, "tried": tried}
