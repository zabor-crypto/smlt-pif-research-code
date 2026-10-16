"""Gate R4 — observable causal state and FORWARD state statistics.

Three small, explainable dimensions measured at each close t with data available through t (trailing windows only):
  ISSUER_CREDIT_RISK : Samolet bond market (MOEX): DISTRESSED if credit spread >= 10 pp or median clean price <= 80% par;
                       DETERIORATING if the 20-session change of the median YTM >= +5 pp; IMPROVING if <= -5 pp;
                       UNKNOWN if fewer than 3 traded bonds; else STABLE. Thresholds = V4.1 `v41.credit_state` (declared
                       before V4.1 results; reused unchanged). Credit spread = ISS Z-spread where published (late 2025
                       on), otherwise the G-spread proxy: median YTM - OFZ zero-coupon yield at the median bond duration
                       (MOEX G-curve, last weekly fixing <= t).
  MARKET_RISK        : STRESS if IMOEX 20-session return <= -8% or 20-session vol >= its trailing-500-session 80th
                       percentile (V2 `smart_portfolio.regimes`, unchanged); else NORMAL.
  RATES_STATE        : TIGHTENING if RGBI 20-session return <= -2.5% (V2 threshold) or a key-rate HIKE took effect in the
                       last 60 calendar days; EASING if a key-rate CUT took effect in the last 180 days (V4.1
                       `easing_lookback_days`) and not TIGHTENING; else NEUTRAL. Key rate: Bank of Russia primary series,
                       effective dates (announcement earlier -> conservative availability).
Forward statistics use R_{t+1..t+H} (contiguous future sessions) for origins t; state-day compression is never used.
The legacy contemporaneous r_t | state_t statistics are kept only as DESCRIPTIVE_STRESS_STATISTICS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DIMENSIONS = ("ISSUER_CREDIT_RISK", "MARKET_RISK", "RATES_STATE")
STATE_VIEW_RULES = ("STATE_SPECIFIC_DIFFERENTIAL", "ORIGINAL_R4_LEVEL")
ORDER = {"ISSUER_CREDIT_RISK": ["IMPROVING", "STABLE", "UNKNOWN", "DETERIORATING", "DISTRESSED"],
         "MARKET_RISK": ["NORMAL", "STRESS"], "RATES_STATE": ["EASING", "NEUTRAL", "TIGHTENING"]}


def features(ctx, cfg: dict) -> pd.DataFrame:
    px = pd.read_csv(ctx.proc / "prices.csv", index_col=0, parse_dates=True)
    rt = pd.read_csv(ctx.proc / "returns.csv", index_col=0, parse_dates=True)
    cal = px.index[px.index <= pd.Timestamp(cfg["cutoff"])]
    cs = pd.read_csv(ctx.proc / "credit_state.csv", index_col=0, parse_dates=True).reindex(cal)
    from ..regime_v2 import curve
    g = curve.read(ctx).sort_index()
    g = g.reindex(g.index.union(cal)).ffill().reindex(cal)
    tenors = np.array([float(c) for c in g.columns])
    dur_y = (cs["bond_duration_median_days"] / 365.0).ffill()
    gy = np.array([np.interp(d, tenors, row) if np.isfinite(d) and np.isfinite(row).all() else np.nan
                   for d, row in zip(dur_y.values, g.values)])
    ytm = cs["ytm_median_pct"].ffill()
    gspread = ytm - gy
    spread = cs["zspread_median_pp"].where(cs["zspread_median_pp"].notna(), gspread)
    kr = pd.read_csv(ctx.interim / "cbr_key_rate.csv", index_col=0, parse_dates=True)["key_rate_pct"].sort_index()
    kr = kr[kr.index <= pd.Timestamp(cfg["cutoff"])]
    chg = kr.diff()
    cuts, hikes = chg[chg < 0].index, chg[chg > 0].index

    def since(ev, t):
        ev = ev[ev <= t]
        return (t - ev[-1]).days if len(ev) else np.inf
    mkt = rt["IMOEX"].reindex(cal).fillna(0.0)
    m20 = (1 + mkt).rolling(20).apply(np.prod, raw=True) - 1
    v20 = mkt.rolling(20).std() * np.sqrt(252)
    vq = v20.rolling(500, min_periods=120).quantile(0.80)
    rg = rt["RGBI"].reindex(cal).fillna(0.0)
    r20 = (1 + rg).rolling(20).apply(np.prod, raw=True) - 1
    f = pd.DataFrame({"credit_spread_pp": spread, "spread_source": np.where(cs["zspread_median_pp"].notna(), "ZSPREAD", "GSPREAD_PROXY"),
                      "gspread_pp": gspread, "zspread_pp": cs["zspread_median_pp"], "price_pct_par": cs["price_median_pct_par"],
                      "ytm_median_pct": ytm, "ytm_change_20d_pp": ytm - ytm.shift(20), "n_bonds": cs["n_bonds"],
                      "imoex_ret20": m20, "imoex_vol20": v20, "imoex_vol_q80_trailing500": vq, "rgbi_ret20": r20,
                      "key_rate_pct": kr.reindex(kr.index.union(cal)).ffill().reindex(cal),
                      "days_since_cut": [since(cuts, t) for t in cal], "days_since_hike": [since(hikes, t) for t in cal]},
                     index=cal)
    return f


def classify(f: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = cfg["r4"]["thresholds"]
    unknown = f["n_bonds"].fillna(0) < c["min_bonds"]
    distressed = (f["credit_spread_pp"] >= c["distressed_spread_pp"]) | (f["price_pct_par"] <= c["distressed_price_pct_par"])
    issuer = np.select([unknown, distressed, f["ytm_change_20d_pp"] >= c["ytm_change_20d_pp"],
                        f["ytm_change_20d_pp"] <= -c["ytm_change_20d_pp"]],
                       ["UNKNOWN", "DISTRESSED", "DETERIORATING", "IMPROVING"], "STABLE")
    market = np.where((f["imoex_ret20"] <= c["market_ret20"]) | (f["imoex_vol20"] >= f["imoex_vol_q80_trailing500"]), "STRESS", "NORMAL")
    tight = (f["rgbi_ret20"] <= c["rates_rgbi_ret20"]) | (f["days_since_hike"] <= c["hike_lookback_days"])
    easing = (f["days_since_cut"] <= c["easing_lookback_days"]) & ~tight
    rates = np.select([tight, easing], ["TIGHTENING", "EASING"], "NEUTRAL")
    out = pd.DataFrame({"ISSUER_CREDIT_RISK": issuer, "MARKET_RISK": market, "RATES_STATE": rates}, index=f.index)
    warm = f["imoex_vol_q80_trailing500"].isna() | f["ytm_change_20d_pp"].isna() | f["rgbi_ret20"].isna()
    out.loc[warm, "MARKET_RISK"] = np.where(f.loc[warm, "imoex_ret20"] <= c["market_ret20"], "STRESS", "NORMAL")
    return out


def forward_returns(R: pd.Series, H: int) -> pd.Series:
    """Compounded return over the H sessions strictly AFTER each origin (NaN if incomplete or any day missing)."""
    lr = np.log1p(R.astype(float))
    s = lr[::-1].rolling(H, min_periods=H).sum()[::-1].shift(-1)
    return np.expm1(s)


def episodes(s: pd.Series) -> pd.DataFrame:
    run = (s != s.shift()).cumsum()
    g = s.groupby(run)
    return pd.DataFrame({"state": g.first(), "start": g.apply(lambda x: x.index[0]), "end": g.apply(lambda x: x.index[-1]),
                         "sessions": g.size()}).reset_index(drop=True)


def _es(x: np.ndarray, a: float = 0.95) -> float:
    x = np.sort(x)
    k = max(1, int(np.ceil((1 - a) * len(x))))
    return float(-x[:k].mean())


def forward_stats(states: pd.DataFrame, R: pd.DataFrame, mmf: pd.Series, routes: list, horizons: list, n0: float,
                  start: str, rng_seed: int, n_boot: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (dimension, state, route, horizon): forward EXCESS return statistics after origins in that state, with
    independent-episode counts, non-overlapping window counts, shrinkage toward the all-origin mean and an episode
    block-bootstrap interval. Funded routes: excess = R - R_MMF; derivative routes: the futures return is the excess."""
    rows, ep_rows = [], []
    rng = np.random.default_rng(rng_seed)
    fwd = {}
    for H in horizons:
        m_f = forward_returns(mmf, H)
        for r, kind in routes:
            x = forward_returns(R[r], H)
            fwd[(r, H)] = (x - m_f) if kind in ("FUNDED", "MMF") else x
    smlt_f = {H: fwd[("SMLT", H)] for H in horizons}
    for dim in DIMENSIONS:
        s = states[dim].loc[start:]
        eps = episodes(s)
        eps["episode_id"] = np.arange(len(eps))
        for _, e in eps.iterrows():
            ep_rows.append({"dimension": dim, **e.to_dict()})
        ep_of = pd.Series(np.repeat(eps["episode_id"].values, eps["sessions"].values), index=s.index)
        for H in horizons:
            for r, kind in routes:
                x_all = fwd[(r, H)].reindex(s.index)
                glob = x_all.dropna()
                gmean = float(glob.mean()) if len(glob) else np.nan
                for st in ORDER[dim]:
                    m = (s == st) & x_all.notna()
                    xs = x_all[m]
                    if xs.empty:
                        rows.append({"dimension": dim, "state": st, "route": r, "horizon": H, "n_origins": 0,
                                     "n_episodes": 0, "evidence": "NO_ORIGINS"})
                        continue
                    ids = ep_of[m]
                    n_ep = int(ids.nunique())
                    pos = np.flatnonzero(m.values)
                    nonov = 0
                    last = -10**9
                    for p_ in pos:
                        if p_ - last >= H:
                            nonov += 1
                            last = p_
                    shr = (nonov * xs.mean() + n0 * gmean) / (nonov + n0)
                    # episode block bootstrap of the mean (resample whole episodes)
                    grp = [xs[ids == i].values for i in ids.unique()]
                    boots = np.array([np.concatenate([grp[j] for j in rng.integers(0, len(grp), len(grp))]).mean()
                                      for _ in range(n_boot)]) if len(grp) else np.array([np.nan])
                    sm = smlt_f[H].reindex(xs.index)
                    q = sm.quantile(0.10)
                    tail = xs[sm <= q]
                    rows.append({"dimension": dim, "state": st, "route": r, "horizon": H, "n_origins": int(m.sum()),
                                 "n_episodes": n_ep, "n_nonoverlapping_windows": nonov, "mean_fwd_excess": float(xs.mean()),
                                 "median_fwd_excess": float(xs.median()), "vol_fwd_excess": float(xs.std(ddof=1)) if len(xs) > 1 else np.nan,
                                 "ES95_fwd_excess_loss": _es(xs.values), "all_origin_mean": gmean,
                                 "shrunk_mean": float(shr), "shrink_n0_windows": n0,
                                 "ci90_lo_episode_bootstrap": float(np.nanquantile(boots, 0.05)),
                                 "ci90_hi_episode_bootstrap": float(np.nanquantile(boots, 0.95)),
                                 "corr_with_smlt_fwd": float(xs.corr(sm)) if r != "SMLT" and len(xs) > 2 else np.nan,
                                 "mean_when_smlt_fwd_bottom_decile": float(tail.mean()) if len(tail) else np.nan,
                                 "evidence": "WEAK" if nonov < 3 else ("MODERATE" if nonov < 6 else "USABLE"),
                                 "overlapping_windows_are_not_independent": True})
    return pd.DataFrame(rows), pd.DataFrame(ep_rows)


def view_table(fs: pd.DataFrame, min_windows: int = 3) -> pd.DataFrame:
    """Which state statistics may become SCENARIO_VIEW points (amendment R4-A1).

    A state view must be STATE-SPECIFIC: the 90% episode-bootstrap interval of the state's forward mean has to exclude
    the ALL-ORIGIN mean of the same route and horizon (the state tells something the unconditional sample does not),
    with at least `min_windows` non-overlapping windows. Its value is the shrunk DIFFERENTIAL (state minus all origins),
    i.e. a deviation from the NO_EXCESS anchor - never the level, which would carry the post-2022 sample mean into the
    view set. The originally committed level rule (interval excludes zero; value = shrunk level) is kept as a labelled
    column and sensitivity. The all-origin mean is treated as fixed (same sample) - a stated limitation."""
    u = fs.copy()
    lo, hi, g = u["ci90_lo_episode_bootstrap"], u["ci90_hi_episode_bootstrap"], u["all_origin_mean"]
    enough = u["n_nonoverlapping_windows"].fillna(0) >= min_windows
    u["ci90_excludes_zero"] = (lo > 0) | (hi < 0)
    u["ci90_excludes_all_origin_mean"] = (lo > g) | (hi < g)
    u["state_differential_shrunk"] = u["shrunk_mean"] - g
    u["usable_under_original_level_rule"] = u["ci90_excludes_zero"] & enough
    u["usable_as_state_view"] = u["ci90_excludes_all_origin_mean"] & enough
    return u


def transitions(states: pd.DataFrame, start: str) -> pd.DataFrame:
    rows = []
    for dim in DIMENSIONS:
        s = states[dim].loc[start:]
        prev = s.shift()
        for a in ORDER[dim]:
            n_a = int((prev == a).sum())
            eps = episodes(s)
            dur = eps[eps["state"] == a]["sessions"]
            for b in ORDER[dim]:
                n = int(((prev == a) & (s == b)).sum())
                rows.append({"dimension": dim, "from": a, "to": b, "count": n, "prob_daily": n / n_a if n_a else np.nan,
                             "mean_spell_sessions_from": float(dur.mean()) if len(dur) else np.nan,
                             "n_spells_from": int(len(dur))})
    return pd.DataFrame(rows)


def descriptive_same_day(states: pd.DataFrame, R: pd.DataFrame, routes: list, start: str) -> pd.DataFrame:
    """DESCRIPTIVE_STRESS_STATISTICS: r_t | state_t (the legacy construction) — not a forecast."""
    rows = []
    for dim in DIMENSIONS:
        s = states[dim].loc[start:]
        for st in ORDER[dim]:
            m = s == st
            for r, _ in routes:
                x = R[r].reindex(s.index)[m].dropna()
                if len(x) < 2:
                    continue
                rows.append({"dimension": dim, "state": st, "route": r, "days": len(x), "mean_ann_same_day": float(x.mean() * 252),
                             "vol_ann_same_day": float(x.std(ddof=1) * np.sqrt(252)), "label": "DESCRIPTIVE_STRESS_STATISTICS"})
    return pd.DataFrame(rows)
