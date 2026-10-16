"""Gate R3 — return-view registry (horizon EXCESS returns over the money market; never a sample mean).

For each route at decision date t, a set of view POINTS (all horizon excess, fraction of position value):
  NO_EXCESS_VIEW    E[R - R_MM] = 0 for the economic exposure; route mechanics that are CONTRACTUAL and observable at t
                    are added (SMLT futures: basis convergence (m - b_t) * tau, b_t = implied carry of the held contract).
  MECHANICAL_CARRY  unchanged prices / curves (random-walk prices, random-walk FX): income yield minus financing, plus
                    roll-down / roll yield where a curve is observable at t (OFZ: G-curve at the RGBI duration; oil: BR
                    front/next term structure; CNY: current exchange swap; equity: trailing IMOEX dividend yield).
  EMPIRICAL_PRIOR   pre-2022 era (2013-2021) mean excess over RUONIA, AUXILIARY ONLY (never pooled with post-2022);
                    only where the archived history exists.
  SCENARIO_VIEW     state-conditional forward statistics (R4) — added by the adaptive policy, not by the R3 frontier.
  EXPERIMENTAL_VIEW BTC_RUB_EXPERIMENTAL: NO_EXCESS and MECHANICAL only, labelled experimental (its 2019-2021 prior has
                    fewer than the required 1,260 prior-era sessions and is kept as EMPIRICAL_PRIOR_EXCLUDED: R3-A1).
Interval: [min(points) - s, max(points) + s], s = post-2022 standard error of the horizon mean excess
(sigma_ann * tau / sqrt(years in sample)). Central = midpoint of [min(points), max(points)] — the centre of the
uncertainty set, used for the NOMINAL frontier and to centre the risk scenarios; the ROBUST frontier uses the whole set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VIEW_TYPES = ("NO_EXCESS_VIEW", "MECHANICAL_CARRY", "EMPIRICAL_PRIOR", "SCENARIO_VIEW", "EXPERIMENTAL_VIEW")
EXPOSURE = {"SMLT": "SMLT_ISSUER_EQUITY", "SMLT_FUT": "SMLT_ISSUER_EQUITY", "EQ_FUNDED": "BROAD_RUSSIAN_EQUITY",
            "EQ_HEDGE_FUT": "BROAD_RUSSIAN_EQUITY", "OFZ_FUNDED": "OFZ_DURATION", "RATES_HEDGE_FUT": "OFZ_DURATION",
            "GOLD_FUNDED": "GOLD", "CNY_PERP": "CNY_RUB_FX", "OIL_FUT_USD": "OIL", "BTC_RUB_EXPERIMENTAL": "BTC"}


def _smlt_basis(ctx, t, S_t) -> dict:
    f = pd.read_csv(ctx.interim / "futures_SMLT_FUT.csv", index_col=0, parse_dates=True).loc[:t]
    r = f.iloc[-1]
    tau = max(float(r["days_to_expiry"]), 1.0) / 365.0
    b = float(np.log(float(r["F_held"]) / S_t) / tau)
    return {"b": b, "F": float(r["F_held"]), "S": S_t, "days_to_expiry": float(r["days_to_expiry"]), "held": r["held"]}


def _oil_roll(ctx, t, tau: float) -> dict | None:
    """Decision-date horizon term-structure carry VIEW in initial-notional engine units.
    Held/far rule unchanged; not a guaranteed realised roll P&L. Log slope is information only.
    """
    from ..data import load
    if not hasattr(ctx, "_adaptive_fut"):
        ctx._adaptive_fut = load.futures_contracts(ctx.root, ctx.cutoff)
    fc = ctx._adaptive_fut
    d = fc[(fc["ASSETCODE"] == "BR") & (fc["TRADEDATE"] == t) & fc["SETTLEPRICE"].notna() & (fc["LSTTRADE"] > t)]
    d = d.sort_values("LSTTRADE")
    held = pd.read_csv(ctx.interim / "futures_BR_FUT.csv", index_col=0, parse_dates=True)["held"].loc[:t].iloc[-1]
    if len(d) < 2 or held not in set(d["SECID"]):
        return None
    near = d[d["SECID"] == held].iloc[0]
    later = d[d["LSTTRADE"] > near["LSTTRADE"]]
    if later.empty:
        return None
    target = t + pd.Timedelta(days=int(round(tau * 365)))
    far = later.iloc[(later["LSTTRADE"] - target).abs().values.argmin()]
    f1, f2 = float(d["SETTLEPRICE"].iloc[0]), float(d["SETTLEPRICE"].iloc[1])
    dt = (d["LSTTRADE"].iloc[1] - d["LSTTRADE"].iloc[0]).days / 365.0
    return {"carry_horizon": 1 - float(far["SETTLEPRICE"]) / float(near["SETTLEPRICE"]),
            "log_slope_information_only": float(np.log(float(near["SETTLEPRICE"]) / float(far["SETTLEPRICE"]))),
            "ratio_diagnostic_only": float(near["SETTLEPRICE"]) / float(far["SETTLEPRICE"]) - 1, "held": held,
            "far": far["SECID"], "far_last_trade": str(far["LSTTRADE"].date()),
            "front_next_spread_ann_info_only": float(np.log(f1 / f2) / dt), "front": d["SECID"].iloc[0], "next": d["SECID"].iloc[1]}


def _cny_swap(P, t) -> float:
    s = P["fut"]["CNY_PERP"].loc[:t]
    return float((s["swap_pts"] / s["Fn"].shift(1)).tail(20).mean() * 252)


def _ofz_carry(ctx, t, m_ann, tau) -> dict:
    from ..regime_v2 import curve
    g = curve.read(ctx).loc[:t].iloc[-1]
    ten = np.array([float(c) for c in g.index])
    cs = pd.read_csv(ctx.proc / "credit_state.csv", index_col=0, parse_dates=True).loc[:t]
    D = float(cs["rgbi_duration_days"].dropna().iloc[-1]) / 365.0
    y = lambda x: float(np.interp(x, ten, g.values)) / 100.0          # noqa: E731
    carry = (y(D) - m_ann) * tau
    roll = D * (y(D) - y(max(D - tau, ten[0])))
    return {"duration_y": D, "yield_at_D": y(D), "roll_down": roll, "carry": carry, "excess": carry + roll}


def _div_yield(P, t) -> float:
    B = P["B"].loc[:t].tail(252)
    return float(np.log1p(B["MCFTR"]).sum() - np.log1p(B["IMOEX"]).sum())


def registry(ctx, cfg: dict, P: dict, t: pd.Timestamp, tau: float, priors: dict | None, panel: pd.DataFrame,
             state_views: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (view rows, per-route interval table)."""
    m_d = float((1 + panel["MMF"].iloc[-60:]).prod() ** (1 / 60) - 1)
    m_ann = (1 + m_d) ** 252 - 1
    mH = (1 + m_d) ** round(tau * 252) - 1
    S_t = float(pd.read_csv(ctx.proc / "prices.csv", index_col=0, parse_dates=True)["SMLT"].loc[:t].iloc[-1])
    rows = []

    def add(route, vtype, value, source, category, note="", valid_until=None, applicability="ALL_STATES"):
        rows.append({"route": route, "exposure": EXPOSURE[route], "view_type": vtype, "horizon_sessions": int(round(tau * 252)),
                     "basis": "EXCESS_OVER_MONEY_MARKET", "value_horizon": float(value), "source": source,
                     "category": category, "valid_from": str(t.date()),
                     "valid_until": valid_until or str((t + pd.DateOffset(months=6)).date()),
                     "state_applicability": applicability, "note": note})
    bas = _smlt_basis(ctx, t, S_t)
    route_carry_fut = (m_ann - bas["b"]) * tau
    # NO_EXCESS
    for r in EXPOSURE:
        extra = route_carry_fut if r == "SMLT_FUT" else 0.0
        add(r, "NO_EXCESS_VIEW" if r != "BTC_RUB_EXPERIMENTAL" else "EXPERIMENTAL_VIEW", extra,
            "definition: zero expected excess of the economic exposure" + (f"; + contractual basis convergence (m {m_ann:.4f} - b {bas['b']:.4f}) x tau, held {bas['held']}" if r == "SMLT_FUT" else ""),
            "NO_EXCESS_VIEW" if r != "BTC_RUB_EXPERIMENTAL" else "EXPERIMENTAL_VIEW")
    # MECHANICAL_CARRY (random-walk prices / curves unchanged)
    add("SMLT", "MECHANICAL_CARRY", -mH, "random-walk price, no dividend expected (distressed issuer): 0 income - money market", "MECHANICAL_CARRY")
    add("SMLT_FUT", "MECHANICAL_CARRY", -mH + route_carry_fut, "random-walk spot + contractual basis convergence", "MECHANICAL_CARRY")
    dy = _div_yield(P, t)
    add("EQ_FUNDED", "MECHANICAL_CARRY", dy * tau - mH, f"trailing 12m IMOEX dividend yield {dy:.4f} (log MCFTR - log IMOEX) - money market", "MECHANICAL_CARRY")
    add("EQ_HEDGE_FUT", "MECHANICAL_CARRY", dy * tau - mH, "futures + collateral = TR excess under random-walk price", "MECHANICAL_CARRY")
    oc = _ofz_carry(ctx, t, m_ann, tau)
    add("OFZ_FUNDED", "MECHANICAL_CARRY", oc["excess"], f"G-curve unchanged: (y(D) {oc['yield_at_D']:.4f} - m) x tau + roll-down {oc['roll_down']:.4f}, D {oc['duration_y']:.2f}y (RGBI index duration proxy)", "MECHANICAL_CARRY")
    add("RATES_HEDGE_FUT", "MECHANICAL_CARRY", oc["excess"], "RGBI futures + collateral ~ OFZ TR excess under unchanged curve", "MECHANICAL_CARRY")
    add("GOLD_FUNDED", "MECHANICAL_CARRY", -mH, "random-walk gold (USD) and USD/RUB: no income - money market", "MECHANICAL_CARRY")
    sw = _cny_swap(P, t)
    add("CNY_PERP", "MECHANICAL_CARRY", -sw * tau, f"random-walk CNY/RUB: minus current exchange swap {sw:.4f} p.a. (RUB-CNY funding differential)", "MECHANICAL_CARRY")
    orl = _oil_roll(ctx, t, tau)
    if orl is not None:
        add("OIL_FUT_USD", "MECHANICAL_CARRY", orl["carry_horizon"],
            f"decision-date horizon term-structure carry view: 1 - F {orl['far']} / F {orl['held']}; "
            f"normalized to initial notional; not guaranteed roll P&L; far expiry {orl['far_last_trade']}; "
            f"info only: front/next spread {orl['front']}->{orl['next']} annualised {orl['front_next_spread_ann_info_only']:.4f} (not used)",
            "MECHANICAL_CARRY")
    add("BTC_RUB_EXPERIMENTAL", "EXPERIMENTAL_VIEW", -mH, "EXPERIMENTAL: random-walk BTC/RUB: no income - money market", "EXPERIMENTAL_VIEW",
        note="mechanical carry of a non-yielding asset")
    # EMPIRICAL_PRIOR (pre-2022, auxiliary)
    for r, key in (("EQ_FUNDED", "MCFTR"), ("EQ_HEDGE_FUT", "MCFTR"), ("OFZ_FUNDED", "RGBITR"), ("RATES_HEDGE_FUT", "RGBITR"),
                   ("GOLD_FUNDED", "GLDRUB_TOM"), ("CNY_PERP", "CNYRUB_TOM"), ("BTC_RUB_EXPERIMENTAL", "BTC_RUB")):
        pr = (priors or {}).get(key)
        if pr is not None and np.isfinite(pr.get("ann_excess", np.nan)):
            ok = pr.get("days", 0) >= cfg["r3"]["prior_era"]["min_sessions"]      # R3-A1
            add(r, "EMPIRICAL_PRIOR" if ok else "EMPIRICAL_PRIOR_EXCLUDED", pr["ann_excess"] * tau,
                f"pre-2022 era {pr['from']}..{pr['to']} mean excess over RUONIA {pr['ann_excess']:.4f} p.a. (SE {pr['se_ann']:.4f}, "
                f"{pr.get('days', 0)} sessions); AUXILIARY prior, not pooled", "EMPIRICAL_PRIOR" if ok else "EXCLUDED_INSUFFICIENT_PRIOR_ERA",
                note=(pr.get("note", "") + ("" if ok else " | excluded from the view set: fewer than "
                                            f"{cfg['r3']['prior_era']['min_sessions']} prior-era sessions")).strip(" |"))
    for r, v in (state_views or {}).items():
        add(r, "SCENARIO_VIEW", v["value"], v["source"], "SCENARIO_VIEW", applicability=v["state"])
    V = pd.DataFrame(rows)
    return V, intervals(V, panel, tau, m_ann, mH)


def add_state_views(V: pd.DataFrame, state_views: dict, t: pd.Timestamp, tau: float) -> pd.DataFrame:
    """Append SCENARIO_VIEW points (state-conditional forward statistics) to recalibration-time view points."""
    rows = [{"route": r, "exposure": EXPOSURE[r], "view_type": "SCENARIO_VIEW", "horizon_sessions": int(round(tau * 252)),
             "basis": "EXCESS_OVER_MONEY_MARKET", "value_horizon": float(v["value"]), "source": v["source"],
             "category": "SCENARIO_VIEW", "valid_from": str(t.date()), "valid_until": str((t + pd.DateOffset(months=6)).date()),
             "state_applicability": v["state"], "note": ""} for r, v in (state_views or {}).items()]
    return pd.concat([V[V["view_type"] != "SCENARIO_VIEW"], pd.DataFrame(rows)], ignore_index=True) if rows else V


def intervals(V: pd.DataFrame, panel: pd.DataFrame, tau: float, m_ann: float | None = None, mH: float | None = None,
              include_excluded: bool = False) -> pd.DataFrame:
    """Per-route uncertainty set from the view points and the post-2022 standard error of the horizon mean.
    EMPIRICAL_PRIOR_EXCLUDED rows are ignored unless `include_excluded` (labelled sensitivity only)."""
    if not include_excluded:
        V = V[V["view_type"] != "EMPIRICAL_PRIOR_EXCLUDED"]
    if m_ann is None:
        m_d = float((1 + panel["MMF"].iloc[-60:]).prod() ** (1 / 60) - 1)
        m_ann = (1 + m_d) ** 252 - 1
        mH = (1 + m_d) ** round(tau * 252) - 1
    years = len(panel) / 252.0
    iv = []
    for r in EXPOSURE:
        col = r if r in panel else None
        x = panel[col] if col else None
        if x is None:
            continue
        exc = x - (panel["MMF"] if r in ("SMLT", "EQ_FUNDED", "OFZ_FUNDED", "GOLD_FUNDED", "BTC_RUB_EXPERIMENTAL") else 0.0)
        sig = float(exc.std(ddof=1) * np.sqrt(252))
        se = sig * tau / np.sqrt(max(years, 1.0))
        pts = V.loc[V["route"] == r, "value_horizon"].values
        lo_p, hi_p = float(pts.min()), float(pts.max())
        iv.append({"route": r, "exposure": EXPOSURE[r], "n_view_points": len(pts), "points_min": lo_p, "points_max": hi_p,
                   "central": 0.5 * (lo_p + hi_p), "se_horizon_post2022": se, "sigma_ann_post2022": sig,
                   "lo": lo_p - se, "hi": hi_p + se, "d_long": 0.5 * (hi_p - lo_p) + se, "d_short": 0.5 * (hi_p - lo_p) + se,
                   "sample_years": years, "money_market_ann": m_ann, "money_market_horizon": mH})
    return pd.DataFrame(iv)


def priors_pre2022(ctx, cfg: dict) -> dict:
    """AUXILIARY pre-2022 era priors: annualised mean excess over RUONIA and its standard error (daily data)."""
    import json
    import re
    from pathlib import Path
    from .evidence import MANIFEST
    from .routes import adaptive_frames
    m = pd.read_csv(Path(ctx.root) / "data" / MANIFEST, dtype=str)
    era = cfg["r3"]["prior_era"]
    out = {}
    # RUONIA (CBR html tables)
    ru = {}
    for rp in sorted(m[(m["dataset"] == "ruonia_prior") & (m["validation_status"] == "VALID")]["raw_path"]):
        t = (Path(ctx.root) / rp).read_bytes().decode("utf-8", errors="replace")
        for d, v in re.findall(r"<td>(\d{2}\.\d{2}\.\d{4})</td>\s*<td[^>]*>\s*([0-9]+,[0-9]+)\s*</td>", t):
            ru[pd.to_datetime(d, format="%d.%m.%Y")] = float(v.replace(",", "."))
    if not ru:
        return out
    ru = pd.Series(ru).sort_index()
    out["_RUONIA"] = ru

    def idx_series(sec):
        f = adaptive_frames(ctx.root, "index_history_long", sec)
        if f.empty:
            return None
        f["TRADEDATE"] = pd.to_datetime(f["TRADEDATE"])
        return f.drop_duplicates("TRADEDATE").set_index("TRADEDATE")["CLOSE"].astype(float).sort_index()

    def cur_series(sec):
        f = adaptive_frames(ctx.root, "currency_history_prior", sec)
        if f.empty:
            return None
        f["TRADEDATE"] = pd.to_datetime(f["TRADEDATE"])
        f = f[f["TRADEDATE"].dt.dayofweek < 5].drop_duplicates("TRADEDATE").set_index("TRADEDATE")
        c = "CLOSE" if "CLOSE" in f else "WAPRICE"
        return f[c].astype(float).replace(0, np.nan).dropna().sort_index()
    series = {"MCFTR": idx_series("MCFTR"), "RGBITR": idx_series("RGBITR"), "GLDRUB_TOM": cur_series("GLDRUB_TOM"),
              "CNYRUB_TOM": cur_series("CNYRUB_TOM")}
    for k, s in series.items():
        if s is None or len(s) < 500:
            continue
        s = s.loc[era["start"]:era["end"]]
        r = s.pct_change().dropna()
        gap = pd.Series(r.index, index=r.index).diff().dt.days.fillna(1)
        rr = ru.reindex(ru.index.union(r.index)).ffill().reindex(r.index).shift(1)
        exc = (r - rr / 100 * gap / 365).dropna()
        yrs = len(exc) / 252
        out[k] = {"ann_excess": float(exc.mean() * 252), "se_ann": float(exc.std(ddof=1) * np.sqrt(252) / np.sqrt(yrs)),
                  "from": str(exc.index.min().date()), "to": str(exc.index.max().date()), "days": len(exc),
                  "note": "spot only (CNY interest not included)" if k == "CNYRUB_TOM" else ""}
    return out
