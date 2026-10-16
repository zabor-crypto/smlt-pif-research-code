"""Regime V2 — expected 126-session EXCESS return over the money market, per unit of position.

    expected excess = strategic prior + current mechanical carry + state adjustment

  strategic prior   prior era only (2013-01-01 .. 2021-12-31; never post-2022): non-overlapping 126-session simple
                    returns of the CARRY-STRIPPED level series, counted back from the last prior-era session;
                    location m = median; SE = bootstrap standard deviation of the median; shrinkage toward zero with
                    reliability = max(0, 1 - SE^2 / m^2). Fewer than 3 windows -> no own prior.
  carry             observable at the decision date (dividend yield, curve carry and roll-down, exchange swap, futures
                    curve, contractual basis), minus the money market where the route is funded.
  state adjustment  forward excess after origins in the current confirmed state minus the all-origin forward excess
                    (states.ForwardLibrary): >= 3 distinct episodes, >= 3 non-overlapping windows, interval excluding the
                    all-origin mean, shrunk by its own reliability; otherwise zero.

Stripping rule (no double count): a carry component is added at its CURRENT value only if the same component was
removed from the historical series. Equity: dividends removed (price index). Gold, BTC: no income; the money-market leg
is removed by using the price return. CNY: spot return (contains no funding differential). OFZ: coupons removed (clean
price index); the clean-price return still contains the historical roll-down, which the current carry also contains, so
a positive OFZ residual is not admitted (capped at zero). SMLT: market-implied (beta x broad equity). Oil: no archived
prior-era history -> zero residual prior.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..adaptive import fx
from ..adaptive import views as V
from ..adaptive.routes import adaptive_frames
from . import inputs
from . import states as S2

FUNDED = ("SMLT", "EQ_FUNDED", "OFZ_FUNDED", "GOLD_FUNDED", "BTC_RUB_EXPERIMENTAL")
SERIES = {"OIL_FUT": "OIL_FUT_USD"}                       # LP route -> scenario series
EXPOSURE_ROUTES = ("SMLT", "EQ_FUNDED", "OFZ_FUNDED", "GOLD_FUNDED", "CNY_PERP", "OIL_FUT", "BTC_RUB_EXPERIMENTAL")
ALL_ROUTES = ("SMLT", "SMLT_FUT", "EQ_FUNDED", "EQ_HEDGE_FUT", "OFZ_FUNDED", "RATES_HEDGE_FUT", "GOLD_FUNDED", "CNY_PERP",
              "OIL_FUT", "BTC_RUB_EXPERIMENTAL")


# ---------------------------------------------------------------------------------------------- strategic prior
def shrink(m: float, se: float) -> tuple[float, float]:
    """(reliability, shrunk value): positive-part signal-to-noise shrinkage toward zero excess. No free parameter."""
    if not np.isfinite(m) or not np.isfinite(se) or m == 0.0:
        return 0.0, 0.0
    rel = max(0.0, 1.0 - se * se / (m * m))
    return rel, rel * m


def windows(level: pd.Series, H: int, start: str, end: str) -> np.ndarray:
    """Non-overlapping H-session simple returns, counted back from the last observation in [start, end]."""
    s = level.loc[start:end].dropna()
    v = s.values.astype(float)
    out = []
    i = len(v) - 1
    while i - H >= 0:
        out.append(v[i] / v[i - H] - 1.0)
        i -= H
    return np.array(out[::-1])


def location(obs: np.ndarray, draws: int, seed: int, tag: str) -> tuple[float, float]:
    """(median, bootstrap standard deviation of the median) of the non-overlapping observations."""
    obs = np.asarray(obs, float)
    m = float(np.median(obs))
    rng = S2._seed(seed, "prior", tag)
    idx = rng.integers(0, len(obs), size=(int(draws), len(obs)))
    se = float(np.median(obs[idx], axis=1).std(ddof=1)) if len(obs) > 1 else np.nan
    return m, se


def _index_level(root: Path, sec: str) -> pd.Series:
    f = adaptive_frames(root, "index_history_long", sec)
    f["TRADEDATE"] = pd.to_datetime(f["TRADEDATE"])
    return f.drop_duplicates("TRADEDATE").set_index("TRADEDATE")["CLOSE"].astype(float).sort_index()


def _currency_level(root: Path, sec: str) -> pd.Series:
    f = adaptive_frames(root, "currency_history_prior", sec)
    f["TRADEDATE"] = pd.to_datetime(f["TRADEDATE"])
    f = f[f["TRADEDATE"].dt.dayofweek < 5].drop_duplicates("TRADEDATE").set_index("TRADEDATE")
    c = "CLOSE" if "CLOSE" in f else "WAPRICE"
    return f[c].astype(float).replace(0, np.nan).dropna().sort_index()


def btc_rub_prior_level(ctx, cfg: dict, sessions: pd.DatetimeIndex) -> pd.Series:
    """BTC/RUB = BTCUSDT (Binance hour bar closing 15:00 UTC) x official USD/RUB known at the close, on MOEX sessions of
    the prior era. 2017-08 .. 2018-12 from the V2 fetch, 2019-01 .. 2021-12 from the parent archive."""
    from ..data import crypto
    h = pd.concat([inputs.binance_hourly_prior(ctx.root), crypto.binance_hourly(ctx.root, cfg["cutoff"])])
    h = h[~h.index.duplicated(keep="first")].sort_index()
    btc = crypto.synced_close(h, sessions, cfg["r2"]["btc"]["bar_close_hour_utc"])
    usd = fx.usd_known(S2.usd_records_long(ctx.root), sessions, 1)
    return (btc * usd).dropna()


def prior_levels(ctx, cfg: dict) -> dict:
    era = cfg["v2"]["returns"]["prior_era"]
    imoex = _index_level(ctx.root, "IMOEX")
    out = {"EQ_FUNDED": imoex, "OFZ_FUNDED": _index_level(ctx.root, "RGBI"),
           "GOLD_FUNDED": _currency_level(ctx.root, "GLDRUB_TOM"), "CNY_PERP": _currency_level(ctx.root, "CNYRUB_TOM")}
    sessions = imoex.loc[era["start"]:era["end"]].index
    out["BTC_RUB_EXPERIMENTAL"] = btc_rub_prior_level(ctx, cfg, sessions)
    return out


def strategic_priors(ctx, cfg: dict, levels: dict | None = None) -> pd.DataFrame:
    """One row per exposure route: the prior-era evidence and the shrunk prior (horizon excess, fraction)."""
    rc = cfg["v2"]["returns"]
    pc, era, H = rc["prior"], rc["prior_era"], int(rc["horizon_sessions"])
    levels = prior_levels(ctx, cfg) if levels is None else levels
    rows = []
    for r in EXPOSURE_ROUTES:
        src = pc["sources"][r]
        row = {"route": r, "series": src["series"], "kind": src["kind"], "residual": src["residual"],
               "stripped_carry": src.get("stripped_carry", ""), "horizon_sessions": H}
        if r in levels:
            lv = levels[r].loc[era["start"]:era["end"]].dropna()
            obs = windows(lv, H, era["start"], era["end"])
            row.update({"history_from": str(lv.index[0].date()), "history_to": str(lv.index[-1].date()), "sessions": int(len(lv)),
                        "n_windows": int(len(obs))})
            if len(obs) >= int(pc["min_windows"]):
                m, se = location(obs, pc["bootstrap_draws"], pc["bootstrap_seed"], r)
                rel, val = shrink(m, se)
                capped = bool(src.get("cap_at_zero", False) and val > 0)
                row.update({"location_median": m, "mean_of_windows": float(obs.mean()), "se_bootstrap_median": se,
                            "min_window": float(obs.min()), "max_window": float(obs.max()), "reliability": rel,
                            "prior_before_cap": val, "prior": 0.0 if capped else val, "capped_at_zero": capped,
                            "status": "OWN_PRIOR_ERA_HISTORY" + ("_POSITIVE_RESIDUAL_NOT_ADMITTED" if capped else "")})
            else:
                row.update({"reliability": 0.0, "prior": 0.0, "status": "PRIOR_NOT_ADMISSIBLE_FEWER_THAN_%d_WINDOWS" % pc["min_windows"]})
        elif r == "SMLT":
            row.update({"n_windows": 2, "reliability": np.nan, "prior": np.nan, "status": "MARKET_IMPLIED_BETA_X_BROAD_EQUITY",
                        "note": "listed 2020-10-29: at most 2 complete 126-session windows before 2022, below the minimum of 3"})
        else:
            row.update({"n_windows": 0, "reliability": 0.0, "prior": 0.0, "status": "PRIOR_HISTORY_NOT_ARCHIVED_ZERO_RESIDUAL"})
        rows.append(row)
    return pd.DataFrame(rows).set_index("route")


def smlt_beta(P: dict, start: str, t_recal: pd.Timestamp) -> dict:
    """OLS slope of the SMLT daily return on the broad-equity total-return index, data <= t_recal (risk input only)."""
    d = pd.concat([P["R"]["SMLT"], P["B"]["MCFTR"]], axis=1).loc[start:t_recal].dropna()
    x, y = d.iloc[:, 1].values, d.iloc[:, 0].values
    vx = float(np.var(x, ddof=1))
    b = float(np.cov(y, x, ddof=1)[0, 1] / vx) if vx > 0 else 0.0
    return {"beta": b, "n": int(len(d)), "from": str(d.index[0].date()), "to": str(d.index[-1].date())}


# ---------------------------------------------------------------------------------------------- current carry
def money_market(panel_now: pd.DataFrame, H: int) -> dict:
    m_d = float((1 + panel_now["MMF"].iloc[-60:]).prod() ** (1 / 60) - 1)
    return {"m_daily": m_d, "m_ann": (1 + m_d) ** 252 - 1, "m_H": (1 + m_d) ** H - 1}


def _smlt_spot(ctx, t: pd.Timestamp) -> float:
    if not hasattr(ctx, "_v2_prices"):
        ctx._v2_prices = pd.read_csv(ctx.proc / "prices.csv", index_col=0, parse_dates=True)
    return float(ctx._v2_prices["SMLT"].loc[:t].iloc[-1])


def carry_at(ctx, cfg: dict, P: dict, t: pd.Timestamp, panel_now: pd.DataFrame) -> dict:
    """CURRENT mechanical carry per exposure at t (horizon excess over the money market), with its inputs."""
    H = int(cfg["v2"]["returns"]["horizon_sessions"])
    tau = H / 252
    mm = money_market(panel_now, H)
    dy = V._div_yield(P, t)
    oc = V._ofz_carry(ctx, t, mm["m_ann"], tau)
    sw = V._cny_swap(P, t)
    orl = V._oil_roll(ctx, t, tau)
    bas = V._smlt_basis(ctx, t, _smlt_spot(ctx, t))
    c = {"EQ_FUNDED": dy * tau - mm["m_H"], "OFZ_FUNDED": oc["excess"], "GOLD_FUNDED": -mm["m_H"],
         "CNY_PERP": (-sw * tau) if np.isfinite(sw) else 0.0, "OIL_FUT": float(orl["carry_horizon"]) if orl else 0.0,
         "BTC_RUB_EXPERIMENTAL": -mm["m_H"]}
    info = {"date": str(t.date()), **mm, "dividend_yield_trailing_12m": dy, "ofz_duration_y": oc["duration_y"],
            "ofz_yield_at_D": oc["yield_at_D"], "ofz_roll_down": oc["roll_down"], "ofz_yield_carry": oc["carry"],
            "cny_swap_ann": sw, "oil_curve": orl if orl else "FEWER_THAN_TWO_LISTED_CONTRACTS",
            "smlt_futures_implied_carry_ann": bas["b"], "smlt_futures_held": bas["held"],
            "smlt_basis_convergence_h": (mm["m_ann"] - bas["b"]) * tau}
    return {"carry": c, "info": info, "mm": mm}


def origin_carry(ctx, cfg: dict, P: dict, conf: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """D12: declared mechanical carry observed at each origin; future prices never enter its estimate.

    SMLT's market-implied beta carry is not issuer mechanical income, so its state signal is conservatively
    disabled rather than imposing an arbitrary issuer carry decomposition. Hedge means follow their exposure.
    """
    from copy import deepcopy
    from ..adaptive import scenarios as SC
    out = {}
    for H in [*cfg["v2"]["returns"]["monitoring_horizons"], cfg["v2"]["returns"]["horizon_sessions"]]:
        c = deepcopy(cfg)
        c["v2"]["returns"]["horizon_sessions"] = H
        rows = []
        origins = conf.index[conf.index >= pd.Timestamp(cfg["v2"]["state"]["sample_start"])]
        for t in origins:
            try:
                now = SC.history_panel(P,t,cfg["r3"]["frontier"]["estimation_start"])
                observation = carry_at(ctx,c,P,t,now)
                carry = dict(observation["carry"])
                if not isinstance(observation["info"]["oil_curve"], dict):
                    carry["OIL_FUT"] = np.nan
                if not np.isfinite(observation["info"]["cny_swap_ann"]):
                    carry["CNY_PERP"] = np.nan
                rows.append({"origin":t,**carry})
            except (IndexError,KeyError):
                rows.append({"origin":t})  # missing origin inputs: no invented carry / no state alpha
        table = pd.DataFrame(rows).set_index("origin")
        table["EQ_HEDGE_FUT"] = table["EQ_FUNDED"]
        table["RATES_HEDGE_FUT"] = table["OFZ_FUNDED"]
        out[int(H)] = table
    return out


# ---------------------------------------------------------------------------------------------- expected vector
def expected(cfg: dict, priors: pd.DataFrame, carry: dict, beta: float, adj: dict, mode: str) -> pd.DataFrame:
    """Per LP route: prior, carry, state adjustment and expected excess `mu` under `mode`; `centre_risk` = the carry
    component alone (the risk scenarios are centred on it, so that estimated premia never reduce measured risk)."""
    md = cfg["v2"]["returns"]["modes"][mode]
    ps, ss = float(md["prior_scale"]), float(md["state_scale"])
    c, info = carry["carry"], carry["info"]
    follow = cfg["v2"]["returns"]["prior"]["hedge_routes_follow_exposure"]
    eq_prior = float(priors.loc["EQ_FUNDED", "prior"])
    base = {}
    for r in EXPOSURE_ROUTES:
        if r == "SMLT":
            base[r] = {"prior": beta * eq_prior, "carry": beta * c["EQ_FUNDED"]}
        else:
            base[r] = {"prior": float(priors.loc[r, "prior"]), "carry": float(c[r])}
        base[r]["state_adj"] = float(adj.get(r, 0.0))
    rows = []
    for r in ALL_ROUTES:
        e = follow.get(r, r)
        b = dict(base[e])
        if r == "SMLT_FUT":
            b["carry"] += float(info["smlt_basis_convergence_h"])
        mu = ps * b["prior"] + b["carry"] + ss * b["state_adj"]
        rows.append({"route": r, "exposure_route": e, "prior": ps * b["prior"], "carry": b["carry"], "state_adj": ss * b["state_adj"],
                     "mu": mu, "centre_risk": b["carry"], "mode": mode})
    return pd.DataFrame(rows).set_index("route")


def views_frame(mu: pd.DataFrame) -> pd.DataFrame:
    """The interval-table layout of the parent (index = scenario series): central = mu, no uncertainty set."""
    d = pd.DataFrame({"central": mu["mu"], "d_long": 0.0, "d_short": 0.0, "centre_risk": mu["centre_risk"]})
    d.index = [SERIES.get(r, r) for r in mu.index]
    d.index.name = "route"
    return d


def uncertainty_table(cfg: dict, priors: pd.DataFrame, mu: pd.DataFrame, adj_detail: dict, beta: dict, panel: pd.DataFrame) -> pd.DataFrame:
    """Reported, never optimised against: standard error of each non-mechanical component and a 90% band for mu.
    SE(prior used) = reliability x SE(median); SE(state adjustment used) = reliability x episode SE; combined in
    quadrature (independent samples: prior era vs post-2022). The post-2022 volatility of the horizon excess is shown for
    scale only."""
    H = int(cfg["v2"]["returns"]["horizon_sessions"])
    follow = cfg["v2"]["returns"]["prior"]["hedge_routes_follow_exposure"]
    rows = []
    for r in mu.index:
        e = follow.get(r, r)
        if e == "SMLT":
            p = priors.loc["EQ_FUNDED"]
            se_p = abs(beta["beta"]) * float(p.get("reliability", 0.0) or 0.0) * float(p.get("se_bootstrap_median", np.nan))
        else:
            p = priors.loc[e]
            se_raw = float(p.get("se_bootstrap_median", np.nan)) if "se_bootstrap_median" in p else np.nan
            se_p = float(p.get("reliability", 0.0) or 0.0) * se_raw if np.isfinite(se_raw) else 0.0
            if bool(p.get("capped_at_zero", False)):
                se_p = 0.0
        a = adj_detail.get(e) or {}
        st = a.get("stats") or {}
        se_a = float(st.get("reliability", 0.0)) * float(st.get("se_episode_bootstrap", 0.0)) if st else 0.0
        se = float(np.sqrt(np.nan_to_num(se_p) ** 2 + se_a ** 2))
        s = SERIES.get(r, r)
        sig = np.nan
        if s in panel:
            x = panel[s] - (panel["MMF"] if r in FUNDED else 0.0)
            sig = float(x.std(ddof=1) * np.sqrt(H))
        rows.append({"route": r, "mu": float(mu.loc[r, "mu"]), "se_prior_used": float(np.nan_to_num(se_p)), "se_state_adj_used": se_a,
                     "se_mu": se, "mu_lo_90": float(mu.loc[r, "mu"]) - 1.645 * se, "mu_hi_90": float(mu.loc[r, "mu"]) + 1.645 * se,
                     "carry_is_observable_at_decision_date": True, "horizon_vol_post2022": sig,
                     "state_level_used": a.get("level", "NOT_MAPPED")})
    return pd.DataFrame(rows)
