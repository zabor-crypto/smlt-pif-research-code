"""End-to-end deterministic pipeline. Each stage reads only raw data / earlier stage outputs.

raw -> validation -> processed -> features -> screening -> strategies -> optimiser ->
walk-forward -> scenarios -> risk metrics -> reporting
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data import load, processed
from .features.candidates import candidate_table, rolling_hedge_ratio_panel
from .features.stats import rolling_beta, rolling_vol
from .risk.covariance import oos_compare, oos_summary
from .strategies.options import greeks, implied_vol
from .strategies.screening import screen
from .validation.checks import validate_manifest, validate_panel


class Ctx:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.cfg = yaml.safe_load((self.root / "configs" / "project.yaml").read_text())
        self.src = yaml.safe_load((self.root / "configs" / "data_sources.yaml").read_text())
        self.uni = yaml.safe_load((self.root / "configs" / "universe.yaml").read_text())
        self.out = self.root / "outputs"
        self.proc = self.root / "data" / "processed"
        self.interim = self.root / "data" / "interim"
        self.fig = self.root / "figures"
        for d in (self.out, self.proc, self.interim, self.fig):
            d.mkdir(parents=True, exist_ok=True)
        self.cutoff = self.cfg["decision_date"]
        self.E = float(self.cfg["capital"]["smlt_exposure_rub"])
        self.C = float(self.cfg["capital"]["hedge_capital_main_rub"])
        self.log: dict = {}

    def save(self, df: pd.DataFrame, name: str, where: str = "out", index: bool = True) -> Path:
        base = {"out": self.out, "proc": self.proc, "interim": self.interim}[where]
        p = base / name
        df.to_csv(p, index=index, float_format="%.10g")
        return p

    def read(self, name: str, where: str = "proc", **kw) -> pd.DataFrame:
        base = {"out": self.out, "proc": self.proc, "interim": self.interim}[where]
        return pd.read_csv(base / name, **kw)


# ============================================================ S1 raw integrity
def s1_raw(ctx: Ctx) -> None:
    man = load.manifest(ctx.root)
    chk = validate_manifest(man, ctx.root)
    ctx.save(chk, "raw_checksum_report.csv", index=False)
    ctx.log["raw_files"] = int(len(chk))
    ctx.log["raw_checksum_ok"] = int(chk["checksum_ok"].sum())
    assert chk["checksum_ok"].all(), "raw snapshot checksum mismatch"


# ============================================================ S2 processed panel
def s2_panel(ctx: Ctx) -> None:
    b = processed.build(ctx.root, ctx.cfg, ctx.src)
    P, R, V = b["prices"], b["returns"], b["value_rub"]
    ctx.save(P, "prices.csv", "proc")
    ctx.save(R, "returns.csv", "proc")
    ctx.save(V, "value_rub.csv", "proc")
    ctx.save(b["meta"], "series_meta.csv", "proc", index=False)
    (ctx.proc / "load_info.json").write_text(json.dumps(b["info"], indent=1, default=str))
    for name, tab in b["futures"].items():
        ctx.save(tab, f"futures_{name}.csv", "interim")
    cs = processed.credit_state(ctx.root, ctx.cutoff, b["calendar"])
    ctx.save(cs, "credit_state.csv", "proc")
    bonds = load.bonds(ctx.root, ctx.cutoff)
    ctx.save(bonds, "smlt_bonds_history.csv", "interim", index=False)
    z = load.zcyc(ctx.root, ctx.cutoff)
    if not z.empty:
        ctx.save(z, "gcurve_yearyields.csv", "proc")
    lim = load.rms_limits(ctx.root, ctx.cutoff)
    if not lim.empty:
        ctx.save(lim, "ncc_futures_limits.csv", "proc", index=False)
        # PIT daily IM-rate series per asset: last published value with tradedate <= t
        imr = lim.pivot_table(index="tradedate", columns="assetcode", values="mr1", aggfunc="last").sort_index()
        imr = imr.reindex(imr.index.union(b["calendar"])).ffill().reindex(b["calendar"])
        imr.index.name = "date"
        ctx.save(imr, "im_rate_mr1.csv", "proc")
    v = validate_panel(P, R, ctx.cutoff)
    ctx.save(v, "data_validation.csv", index=False)
    ctx.log["panel_series"] = int(R.shape[1])
    ctx.log["calendar"] = [str(b["calendar"].min().date()), str(b["calendar"].max().date()), int(len(b["calendar"]))]
    assert v["PASS_no_lookahead"].all()


def _panel(ctx: Ctx):
    P = ctx.read("prices.csv", index_col=0, parse_dates=True)
    R = ctx.read("returns.csv", index_col=0, parse_dates=True)
    V = ctx.read("value_rub.csv", index_col=0, parse_dates=True)
    cp = ctx.proc / "credit_state.csv"
    if cp.exists():
        # Samolet bond "index": constant-composition median clean-price return; level = cumulative product
        cs = pd.read_csv(cp, index_col=0, parse_dates=True)
        R["SMLT_BOND_IDX"] = cs["bond_price_ret_median"].reindex(R.index)
        lvl = (1 + R["SMLT_BOND_IDX"].fillna(0)).cumprod() * 100
        P["SMLT_BOND_IDX"] = lvl.where(cs["n_bonds"].reindex(R.index).fillna(0) > 0)
        bh = ctx.interim / "smlt_bonds_history.csv"
        if bh.exists():
            b = pd.read_csv(bh, parse_dates=["TRADEDATE"])
            V["SMLT_BOND_IDX"] = b.groupby("TRADEDATE")["VALUE"].sum().reindex(R.index)
    return P, R, V


# ============================================================ S3 candidates + dossier market stats
def s3_candidates(ctx: Ctx) -> None:
    P, R, V = _panel(ctx)
    cols = [c for c in R.columns if R[c].notna().sum() > 60]
    st = candidate_table(R[cols], target="SMLT", market="IMOEX", cutoff=ctx.cutoff)
    ctx.save(st, "candidate_stats.csv", index=False)
    hedges = [h for h in ["SMLT_FUT", "IMOEXF", "MIX_FUT", "MXI_FUT", "MCFTR", "MOEXRE", "PIKK", "LSRG", "ETLN",
                          "RGBI_FUT", "RGBITR", "GOLD_FUT", "GLDRUB_TOM", "CNYRUB_TOM", "BR_FUT"] if h in R]
    rh = rolling_hedge_ratio_panel(R, "SMLT", hedges)
    ctx.save(rh, "rolling_hedge_ratios.csv")
    # SMLT market dossier
    s, px = R["SMLT"], P["SMLT"]
    t = pd.Timestamp(ctx.cutoff)
    peak_all = px.loc[:t].max()
    d = {
        "close_2026_09_25": float(px.loc[t]),
        "close_2026_09_04": float(px.loc["2026-09-04"]) if pd.Timestamp("2026-09-04") in px.index else np.nan,
        "sept_2026_drawdown_from_sep_high": float(1 - px.loc[t] / px.loc["2026-09-01":t].max()),
        "drawdown_from_2022_high": float(1 - px.loc[t] / peak_all),
        "date_of_2022plus_high": str(px.loc[:t].idxmax().date()),
        "min_close_2026": float(px.loc["2026-01-01":t].min()),
        "date_min_close_2026": str(px.loc["2026-01-01":t].idxmin().date()),
        "ytd_2026_return": float(px.loc[t] / px.loc["2026-01-01":].iloc[0] - 1),
        "worst_day": float(s.min()), "worst_day_date": str(s.idxmin().date()),
        "best_day": float(s.max()), "best_day_date": str(s.idxmax().date()),
        "adv60_rub": float(V["SMLT"].loc[:t].tail(60).mean()),
        "adv20_rub": float(V["SMLT"].loc[:t].tail(20).mean()),
        "position_share_of_adv20": float(ctx.E / V["SMLT"].loc[:t].tail(20).mean()),
        "shares_in_position": float(ctx.E / px.loc[t]),
    }
    for w in (20, 60, 120, 250):
        d[f"vol_{w}d"] = float(rolling_vol(s, w).loc[t])
    for bench in ("IMOEX", "MOEXRE", "MCFTR"):
        if bench in R:
            for w in (120, 250):
                d[f"beta_{bench}_{w}d"] = float(rolling_beta(s, R[bench], w).loc[t])
    ctx.save(pd.Series(d, name="value").to_frame(), "smlt_market_dossier.csv")
    ctx.log["smlt_dossier"] = d


# ============================================================ S4 options board at decision date
def s4_options(ctx: Ctx) -> None:
    o = load.options_snapshot(ctx.root, ctx.cutoff)
    if o.empty:
        ctx.log["options"] = "DATA_NOT_AVAILABLE"
        return
    fut = ctx.read("futures_SMLT_FUT.csv", "interim", index_col=0, parse_dates=True)
    t = pd.Timestamp(ctx.cutoff)
    F = float(fut.loc[t, "F_held"])
    held = fut.loc[t, "held"]
    y = _mmf_yield(ctx)
    # expiry by weekly suffix, verified against SECNAME (e.g. SS1050CJ6A = SMLTP300926..., B = 071026, none = 211026)
    exp = {"A": "2026-09-30", "B": "2026-10-07", "": "2026-10-21"}
    rx = re.compile(r"^SS(\d+)([A-Z])([A-X])(\d)([A-Z]?)$")
    parsed = []
    for _, r in o.iterrows():
        mt = rx.match(r["SECID"])
        if not mt:
            continue
        parsed.append({"secid": r["SECID"], "strike": float(mt.group(1)),
                       "type": "C" if mt.group(3) <= "L" else "P",   # month letters A-L calls, M-X puts
                       "expiry": exp.get(mt.group(5)), "settle": float(r["SETTLEPRICE"]), "theor_price": r.get("THEOR_PRICE"),
                       "open_interest": r["OPENPOSITION"], "volume": r["VOLUME"], "num_trades": r["NUMTRADES"]})
    q = pd.DataFrame(parsed)
    q["tau_years"] = (pd.to_datetime(q["expiry"]) - t).dt.days / 365.0
    S0 = float(_panel(ctx)[0]["SMLT"].loc[t])
    rows = []
    for (e, tau), g in q.groupby(["expiry", "tau_years"]):
        # underlying identified from put-call parity: implied forward at the most-traded strike
        both = g.pivot_table(index="strike", columns="type", values="settle")
        oi = g.groupby("strike")["open_interest"].sum()
        k_ref = oi.loc[both.dropna().index].idxmax()
        F_impl = k_ref + (both.loc[k_ref, "C"] - both.loc[k_ref, "P"]) * np.exp(y * tau)
        for _, r in g.iterrows():
            iv = implied_vol(r["settle"], F_impl, r["strike"], tau, y, r["type"])
            gk = greeks(F_impl, r["strike"], tau, iv, y, r["type"]) if np.isfinite(iv) else {}
            rows.append({**r.to_dict(), "S0_official_close": S0, "future_front": held, "future_settle": F,
                         "parity_strike": k_ref, "F_implied_parity": F_impl, "F_over_S0": F_impl / S0,
                         "iv_from_settle": iv, **{f"greek_{k}": v for k, v in gk.items()}})
    df = pd.DataFrame(rows).sort_values(["expiry", "type", "strike"])
    ctx.save(df, "smlt_options_20260925.csv", index=False)
    ctx.log["options_rows"] = len(df)


def _listed_atm_iv(opt: pd.DataFrame | None, expiry: str = "2026-10-21") -> float:
    """IV of the longest listed expiry at its put-call-parity (most-traded) strike; nan if unavailable."""
    if opt is None or opt.empty:
        return float("nan")
    g = opt[(opt["expiry"] == expiry) & (opt["strike"] == opt["parity_strike"]) & opt["iv_from_settle"].notna()]
    return float(g["iv_from_settle"].mean()) if len(g) else float("nan")


def _mmf_yield(ctx: Ctx, series: str = "LQDT", days: int = 60) -> float:
    P, R, V = _panel(ctx)
    p = P[series].dropna()
    p = p.loc[:ctx.cutoff]
    a, b = p.iloc[-days - 1], p.iloc[-1]
    n = (p.index[-1] - p.index[-days - 1]).days
    return float((b / a) ** (365.0 / n) - 1.0)


# ============================================================ S5 screening
def s5_screen(ctx: Ctx) -> None:
    P, R, V = _panel(ctx)
    st = pd.read_csv(ctx.out / "candidate_stats.csv")
    t = pd.Timestamp(ctx.cutoff)
    fut_oi, notional = {}, {}
    snap = _forts_snapshot(ctx)
    for name, (asset, kind) in processed.FUT_ASSETS.items():
        f = ctx.interim / f"futures_{name}.csv"
        if not f.exists():
            continue
        tab = pd.read_csv(f, index_col=0, parse_dates=True)
        if kind == "dated" and t in tab.index:
            fut_oi[asset] = float(tab.loc[t, "oi_held"])
        elif kind == "perpetual" and t in tab.index:
            fut_oi[asset] = float(tab.loc[t, "OPENPOSITION"])
        if asset in snap:
            notional[asset] = snap[asset]
    options = ctx.out / "smlt_options_20260925.csv"
    special = {}
    if options.exists():
        o = pd.read_csv(options)
        max_oi = o["open_interest"].max()
        need = ctx.E / float(P["SMLT"].loc[t])
        special["SMLT_OPT"] = {"G1": "PASS", "G3": "PASS", "G4": "FAIL" if max_oi < 10 * need else "PASS",
                               "G5": "PASS", "limited": True,
                               "reasons": [f"listed expiries only to {o['expiry'].max()} (< 6m horizon)",
                                           f"strikes listed: {sorted(o['strike'].unique().tolist())}",
                                           f"max OI per series {int(max_oi)} vs ~{int(need)} contracts needed"]}
    bonds = load.bonds(ctx.root, ctx.cutoff)
    bt = bonds[bonds["TRADEDATE"] == t]
    n_b, val_b = int(bt["SECID"].nunique()), float(bt["VALUE"].fillna(0).sum())
    bond_reason = [f"{n_b} Samolet bond issues traded on {ctx.cutoff} (RUB {val_b/1e6:.1f}m traded value)"]
    special["SMLT_BONDS"] = {"G1": "PASS" if n_b else "FAIL", "reasons": bond_reason}
    special["SMLT_BOND_SHORT"] = {"G1": "PASS" if n_b else "FAIL", "reasons": bond_reason}
    # V3/V4: MOEX BTC futures have a short execution history; the economic history is the validated BTC proxy
    corr = "n/a"
    if {"BTC_PROXY", "BTC_FUT"} <= set(R.columns):
        ov = R[["BTC_PROXY", "BTC_FUT"]].dropna()
        corr = f"{ov.corr().iloc[0, 1]:.3f} ({len(ov)} overlapping days)" if len(ov) > 20 else "n/a"
    special["BTC_MOEX_FUT"] = {"proxy_history": f"economic history via BTC_ECONOMIC_PROXY (daily corr with MOEX BTC futures {corr})"}
    rf = snap.get("_RUONIA_OI")
    special["RUONIA_FUT"] = {"G1": "PASS", "G3": "N/A", "G4": "FAIL", "G5": "N/A",
                             "reasons": [f"RUONIA futures open interest on {ctx.cutoff}: {rf}"]}
    uni = screen(ctx.uni["candidates"], ctx.uni["gates"], st, P, R, V, fut_oi, notional, ctx.cutoff, ctx.E, special,
                 hedge_capital_rub=ctx.C)
    # role-correct wording
    uni["label"] = np.where(uni["role"].eq("DIVERSIFIER") & uni["G5_relevance"].eq("NO_HEDGE_EVIDENCE"),
                            "DIVERSIFIER (not a hedge: no stress evidence)", uni["role"])
    ctx.save(uni, "instrument_universe.csv", index=False)
    ctx.log["universe_classification"] = uni["classification"].value_counts().to_dict()


def _forts_snapshot(ctx: Ctx) -> dict:
    """Contract notional (RUB) per asset AT THE CUTOFF (V3 0.6: settle and point value from the decision-date
    exchange history; the post-cutoff board snapshot supplies static terms only)."""
    from .data.specs import pit_terms
    t = pit_terms(ctx.root, ctx.cutoff)
    t = t[t["status"] == "OK"]
    out = {r["family"]: float(r["notional_pit"]) for _, r in t.iterrows()}
    fc = load.futures_contracts(ctx.root, ctx.cutoff)
    ru = fc[(fc["ASSETCODE"] == "RUONIA") & (fc["TRADEDATE"] == pd.Timestamp(ctx.cutoff))]
    out["_RUONIA_OI"] = ru.set_index("SECID")["OPENPOSITION"].to_dict()   # OI on the decision date (PIT)
    return out


# ============================================================ S6 covariance comparison
def s6_covariance(ctx: Ctx) -> None:
    P, R, V = _panel(ctx)
    cols = [c for c in ["SMLT", "SMLT_FUT", "IMOEXF", "MIX_FUT", "RGBI_FUT", "GLDRUB_TOM", "CNYRUB_TOM"] if c in R]
    X = R[cols].dropna()
    res = []
    for w in ctx.cfg["risk"]["covariance_windows"]:
        r = oos_compare(X, window=w, hold=5, fixed_col="SMLT", lam=ctx.cfg["risk"]["ewma_lambda"])
        s = oos_summary(r)
        s["window"] = w
        res.append(s.reset_index())
    out = pd.concat(res, ignore_index=True)
    out["columns"] = "|".join(cols)
    out["first_date"] = str(X.index.min().date())
    ctx.save(out, "covariance_oos.csv", index=False)
    ctx.log["covariance_best"] = out.sort_values("mean_qlike").iloc[0][["estimator", "window"]].to_dict()


# ============================================================ S7 walk-forward (daily + episodes)
from .backtest.walk_forward import WFConfig, daily_walk_forward, fixed, rolling_hstar, run_episodes  # noqa: E402
from .risk.metrics import es_hist, he_pnl, he_var, summary  # noqa: E402
from .strategies.futures import FuturesCosts  # noqa: E402
from .strategies.state_policy import state_policy, state_vector  # noqa: E402


def _history_frame(ctx: Ctx) -> pd.DataFrame:
    """Decision-time information set used by policies: returns + PIT state variables."""
    P, R, V = _panel(ctx)
    H = R[["SMLT", "SMLT_FUT"]].copy()
    H["SMLT_PX"] = P["SMLT"]
    cs = ctx.read("credit_state.csv", index_col=0, parse_dates=True)
    H["ZSPREAD"] = cs["zspread_median_pp"].reindex(H.index)
    gp = ctx.proc / "gcurve_yearyields.csv"
    if gp.exists():
        g = pd.read_csv(gp, index_col=0, parse_dates=True)
        col = [c for c in g.columns if abs(float(c) - 1.0) < 1e-9]
        if col:
            H["G1Y"] = g[col[0]].reindex(H.index.union(g.index)).ffill().reindex(H.index)
    return H


def _costs(ctx: Ctx) -> FuturesCosts:
    c = ctx.cfg["costs"]
    return FuturesCosts(exchange_fee_rub=0.05, broker_fee_rub=c["broker_fee_futures_rub_per_contract"],
                        half_spread_rub=c["futures_half_spread_ticks"] * 1.0)


def _policies() -> dict:
    return {"FIXED_25": fixed(0.25), "FIXED_50": fixed(0.50), "FIXED_75": fixed(0.75), "FIXED_100": fixed(1.0),
            "ROLLING_HSTAR_120": rolling_hstar(120), "STATE_DEPENDENT": state_policy()}


def s7_walk_forward(ctx: Ctx) -> None:
    P, R, V = _panel(ctx)
    H = _history_frame(ctx)
    fut = ctx.read("futures_SMLT_FUT.csv", "interim", index_col=0, parse_dates=True)
    mmf = R["LQDT"].fillna(0.0)
    costs = _costs(ctx)
    # per-contract RUB cost is fixed by the tick (1 RUB) and fees -> as a fraction of notional it depends on the price at t
    cost_rate = (costs.per_contract() / fut["F_held"]).reindex(R.index).ffill()
    wf = ctx.cfg["walk_forward"]
    E, C = ctx.E, ctx.C
    # ---------- daily constant-exposure walk-forward, threshold grid
    series, rows = {}, []
    ctrl = None
    for name, pol in {"CONTROL": fixed(0.0), **_policies()}.items():
        for thr in (wf["thresholds_grid"] if name in ("ROLLING_HSTAR_120", "STATE_DEPENDENT") else [0.0]):
            cfg = WFConfig(window=wf["estimation_window"], rebalance_every=wf["rebalance_every_days"],
                           threshold=thr, cost_rate=cost_rate, roll_cost_rate=2 * cost_rate)
            d = daily_walk_forward(H.join(R[[]]), fut["is_roll"], pol, cfg, E, C, mmf, wf["oos_start"])
            key = f"{name}|thr={thr:.2f}"
            series[key] = d
            if name == "CONTROL":
                ctrl = d
    # nested (walk-forward) threshold selection: at each quarter start choose the threshold with the lowest
    # realised daily ES95 over all PRIOR out-of-sample days; apply it to the next quarter.
    selected = {}
    for pol in ("ROLLING_HSTAR_120", "STATE_DEPENDENT"):
        keys = [k for k in series if k.startswith(pol + "|")]
        idx = series[keys[0]].index
        q_starts = pd.date_range(idx.min(), idx.max(), freq="QS")
        pieces, chosen = [], []
        best = keys[0]
        for i, qs in enumerate(q_starts):
            qe = q_starts[i + 1] if i + 1 < len(q_starts) else idx.max() + pd.Timedelta(days=1)
            prior = idx[idx < qs]
            if len(prior) >= 60:
                scores = {k: es_hist(series[k].loc[prior, "pnl_total"].values, 0.95)
                               + 1e-9 * series[k].loc[prior, "trade"].abs().sum() for k in keys}
                best = min(scores, key=scores.get)
            chosen.append({"policy": pol, "quarter_start": qs.date(), "selected": best})
            pieces.append(series[best].loc[(idx >= qs) & (idx < qe)])
        series[f"{pol}|thr=WF_SELECTED"] = pd.concat(pieces)
        selected[pol] = chosen
    ctx.save(pd.DataFrame([c for v in selected.values() for c in v]), "threshold_selection_walkforward.csv", index=False)
    # ---------- metrics (A: per RUB 10m SMLT; B: per RUB 20m total capital)
    unhedged = ctrl["pnl_spot"] + ctrl["carry"]
    for key, d in series.items():
        pnl = d["pnl_total"].reindex(ctrl.index).dropna()
        for base_name, base in (("A_per_SMLT_10m", E), ("B_per_total_capital", E + C)):
            m = summary(pnl, base_rub=base, bench_pnl_rub=unhedged)
            m.update({"strategy": key, "basis": base_name, "HE_VAR": he_var(pnl.values, unhedged.reindex(pnl.index).values),
                      "HE_PNL": he_pnl(d["pnl_hedge"].values, d["pnl_spot"].values),
                      "turnover_notional_multiple": float(d["trade"].abs().sum()), "n_trades": int((d["trade"] != 0).sum()),
                      "total_costs_rub": float(d["costs"].sum()), "avg_hedge_ratio": float(d["h_held"].mean()),
                      "start": str(pnl.index.min().date()), "end": str(pnl.index.max().date())})
            rows.append(m)
    wfm = pd.DataFrame(rows)
    ctx.save(wfm, "walkforward_daily_metrics.csv", index=False)
    cum = pd.DataFrame({k: v["pnl_total"] for k, v in series.items()}).cumsum()
    ctx.save(cum, "walkforward_cum_pnl.csv", "interim")
    # ---------- episodes with full futures mechanics
    imr = ctx.read("im_rate_mr1.csv", index_col=0, parse_dates=True)["SMLT"]
    ep_rows = []
    lar_paths = {}
    for Cgrid in ctx.cfg["capital"]["hedge_capital_grid_rub"]:
        for name, pol in {"CONTROL": fixed(0.0), **_policies()}.items():
            if name == "FIXED_25" and Cgrid != C:
                continue
            for hz_name, hz in ctx.cfg["horizons_trading_days"].items():
                ep, sims = run_episodes(P["SMLT"], fut, H, pol, horizon=hz, exposure=E, capital=float(Cgrid), mmf=mmf,
                                        im_rate=imr, costs=costs, threshold=0.10 if name in ("ROLLING_HSTAR_120", "STATE_DEPENDENT") else 0.0,
                                        window=wf["estimation_window"], rebalance_every=wf["rebalance_every_days"],
                                        step=21, first_start="2022-09-01",
                                        margin_budget_frac=ctx.cfg["optimization"]["margin_budget_fraction_of_hedge_capital"],
                                        label=name)
                ep["horizon"] = hz_name
                ep_rows.append(ep)
                if Cgrid == C and hz_name == "6m" and name in ("FIXED_100", "FIXED_50"):
                    worst = ep.sort_values("max_cum_vm_drain").iloc[-1]["start"]
                    lar_paths[f"{name}_{hz_name}"] = sims[worst]
    eps = pd.concat(ep_rows, ignore_index=True)
    ctx.save(eps, "episodes.csv", index=False)
    for k, sim in lar_paths.items():
        ctx.save(sim, f"lar_worst_episode_{k}.csv", "interim")
    ctx.log["episodes"] = int(len(eps))


# ============================================================ S8 bootstrap at decision date
from .optimization.overlay import Problem, Sleeve, cardinality_path, min_es, min_variance, max_sharpe, portfolio_pnl  # noqa: E402
from .scenarios.bootstrap import evaluate_static, horizon_returns, paths, structural_futures  # noqa: E402


def _fhs(R: pd.DataFrame, cols: list[str], lam: float, cutoff: str, groups: dict | None = None) -> pd.DataFrame:
    """Filtered historical simulation input: z_t = r_t / sigma_{t-1} (EWMA, causal), rescaled by sigma at cutoff.

    Series in the same group (e.g. SMLT spot and SMLT futures) share ONE scaling path (the group leader's), so FHS
    does not distort the hedge relationship between two claims on the same underlying.
    """
    groups = groups or {}
    leader = {m: g for g, members in groups.items() for m in members}
    X = R[cols].dropna()
    L = np.log1p(X.clip(lower=-0.999))            # LOG space: rescaled returns stay > -100% (no negative prices)
    out = {}
    for c in cols:
        base = leader.get(c, c)
        v = L[base].pow(2).ewm(alpha=1 - lam, adjust=False).mean()
        sig_prev = np.sqrt(v.shift(1)).bfill()
        out[c] = np.expm1(L[c] / sig_prev * float(np.sqrt(v.iloc[-1])))
    return pd.DataFrame(out, index=X.index)


def _basis_residual(ctx: Ctx, index: pd.DatetimeIndex) -> tuple[pd.Series, float, float]:
    """Stationary SMLT futures basis residual e_t = ln(F/S) - c_t * d/365, c_t = trailing 60d median annualised basis."""
    P, R, V = _panel(ctx)
    fut = ctx.read("futures_SMLT_FUT.csv", "interim", index_col=0, parse_dates=True)
    lb = np.log(fut["F_held"] / P["SMLT"].reindex(fut.index))
    d = fut["days_to_expiry"].clip(lower=1)
    ann = lb * 365.0 / d
    c = ann.rolling(60, min_periods=20).median().shift(1)   # PIT carry estimate
    e = (lb - c * d / 365.0)
    t = pd.Timestamp(ctx.cutoff)
    e_now = float(e.loc[:t].dropna().iloc[-1])
    b_now = float(ann.loc[t])
    return e.reindex(index).ffill().fillna(0.0), b_now, e_now


def _bootstrap_inputs(ctx: Ctx, cols: list[str], futures_like: tuple = ()) -> dict:
    """Shared scenario inputs (cached per column set): sample, HS/FHS/RECENT variants with risk-neutral drift,
    stationary basis residual, IM rate and money-market yield. futures_like = extra columns executed through
    futures (zero drift, margin instead of capital)."""
    key = (tuple(cols), tuple(futures_like))
    cache = getattr(ctx, "_boot_cache", {})
    if key in cache:
        return cache[key]
    P, R, V = _panel(ctx)
    sim = ctx.cfg["simulation"]
    X = R[cols].dropna()
    X = X[X.index >= pd.Timestamp(sim["bootstrap_start"])]
    y = _mmf_yield(ctx)
    daily_y = (1 + y) ** (1 / 252) - 1
    fut_cols = [c for c in cols if c.endswith("_FUT") or c in ("IMOEXF", "CNYRUBF", "GLDRUBF") or c in futures_like]
    variants = {"HS_FULL": X,
                "FHS_FULL": _fhs(X, cols, ctx.cfg["risk"]["ewma_lambda"], ctx.cutoff, groups={"SMLT": ["SMLT", "SMLT_FUT"]}),
                "HS_RECENT250": X.tail(250)}
    variants = {k: _neutral_drift(v, fut_cols, daily_y) for k, v in variants.items()}
    e_res, b_now, e_now = _basis_residual(ctx, X.index)
    imr_now = ctx.read("im_rate_mr1.csv", index_col=0, parse_dates=True).loc[:ctx.cutoff].iloc[-1]
    S0, F0, _pv = _fut_ctx(ctx)
    out = {"X": X, "variants": variants, "e_res": e_res, "b": b_now, "e0": e_now, "y": y, "daily_y": daily_y, "S0": S0, "F0": F0,
           "mr": float(imr_now.get("SMLT", 0.5)), "imr_now": imr_now, "fut_cols": fut_cols, "cols": cols}
    cache[key] = out
    ctx._boot_cache = cache
    return out


def _paths_for(ctx: Ctx, B: dict, variant: str, hz: int, n_paths: int | None = None) -> dict:
    sim = ctx.cfg["simulation"]
    Xv = B["variants"][variant]
    Pth = paths(Xv, B["cols"], hz, n_paths or sim["n_paths"], sim["block_length"], sim["seed"])
    Pth["SMLT_FUT"] = structural_futures(Pth, B["e_res"].reindex(Xv.index).values, B["b"], B["e0"])
    return Pth


def _neutral_drift(X: pd.DataFrame, fut_cols: list[str], spot_daily: float) -> pd.DataFrame:
    """Keep dispersion/dependence/tails, remove the sample mean; spot -> money-market drift, futures -> 0."""
    Y = X - X.mean()
    for c in Y.columns:
        if c not in fut_cols:
            Y[c] = Y[c] + spot_daily
    return Y


def _sleeve_specs(ctx: Ctx, uni: pd.DataFrame, S0: float) -> list[dict]:
    """Economically distinct sleeves for the optimiser, one representative per economic factor."""
    imr = ctx.read("im_rate_mr1.csv", index_col=0, parse_dates=True).loc[:ctx.cutoff].iloc[-1]
    cls = uni.set_index("id")["classification"].to_dict()
    per = _costs(ctx).per_contract() / S0
    specs = [
        {"name": "SMLT_FUT", "series": "SMLT_FUT", "lo": -1.0, "hi": 0.0, "margin": float(imr.get("SMLT", 0.5)),
         "cost_6m": 6 * per, "cost_12m": 10 * per, "group": "DIRECT", "cls": cls.get("SMLT_FUT")},
        {"name": "INDEX", "series": "MIX_FUT", "lo": -1.5, "hi": 0.0, "margin": float(imr.get("MIX", 0.12)),
         "cost_6m": 0.0006, "cost_12m": 0.0010, "group": "INDEX", "cls": cls.get("MIX_FUT")},
        {"name": "SECTOR_PIKK", "series": "PIKK", "lo": -0.5, "hi": 0.0, "margin": 0.5,
         "cost_6m": 0.0010 + 0.05 / 2, "cost_12m": 0.0010 + 0.05, "group": "SECTOR", "cls": cls.get("PIKK_SHORT")},
        {"name": "RATES_RGBI", "series": "RGBI_FUT", "lo": -1.0, "hi": 1.0, "margin": float(imr.get("RGBI", 0.11)),
         "cost_6m": 0.0005, "cost_12m": 0.0008, "group": "RATES", "cls": cls.get("RGBI_FUT")},
        {"name": "GOLD", "series": "GLDRUB_TOM", "lo": 0.0, "hi": 0.5, "margin": 0.0,
         "cost_6m": 0.0030, "cost_12m": 0.0030, "group": "DIVERSIFIER", "cls": cls.get("GLDRUBF"), "capital": 1.0,
         "route": "returns: GLDRUB_TOM spot; execution: GLDRUBF perpetual (EXECUTABLE); modelled as fully funded"},
        {"name": "CNY", "series": "CNYRUB_TOM", "lo": 0.0, "hi": 0.5, "margin": 0.0,
         "cost_6m": 0.0010, "cost_12m": 0.0010, "group": "DIVERSIFIER", "cls": cls.get("CNY_FUT"), "capital": 1.0,
         "route": "returns: CNYRUB_TOM spot; execution: CNY futures (EXECUTABLE); modelled as fully funded"},
        {"name": "OIL_BR", "series": "BR_FUT", "lo": -0.5, "hi": 0.5, "margin": float(imr.get("BR", 0.25)),
         "cost_6m": 0.0010, "cost_12m": 0.0015, "group": "DIVERSIFIER", "cls": cls.get("BR_FUT")},
    ]
    return specs


def s8_bootstrap(ctx: Ctx) -> None:
    P, R, V = _panel(ctx)
    uni = pd.read_csv(ctx.out / "instrument_universe.csv")
    S0 = float(P["SMLT"].loc[ctx.cutoff])
    sim = ctx.cfg["simulation"]
    y = _mmf_yield(ctx)
    ctx.log["mmf_yield_LQDT_60d"] = y
    specs = [s for s in _sleeve_specs(ctx, uni, S0) if s["series"] in R and s["cls"] in ("EXECUTABLE", "LIMITED_EXECUTABILITY")]
    cols = ["SMLT", "SMLT_FUT"] + [s["series"] for s in specs if s["series"] != "SMLT_FUT"]
    B = _bootstrap_inputs(ctx, cols)
    X, variants, e_res, b_now, e_now, mr, daily_y0 = (B["X"], B["variants"], B["e_res"], B["b"], B["e0"], B["mr"],
                                                      B["daily_y"])
    ctx.log["bootstrap_sample"] = [str(X.index.min().date()), str(X.index.max().date()), int(len(X)), cols]
    ctx.log["basis_model"] = {"carry_ann_at_cutoff": b_now, "resid_at_cutoff": e_now,
                              "resid_std": float(e_res.std()), "resid_abs_p99": float(e_res.abs().quantile(0.99))}
    ctx.log["bootstrap_drift"] = "risk-neutral: spot mean = money-market daily yield, futures mean = 0"
    opt = pd.read_csv(ctx.out / "smlt_options_20260925.csv") if (ctx.out / "smlt_options_20260925.csv").exists() else None
    listed_iv = _listed_atm_iv(opt)
    rv250 = float(R["SMLT"].tail(250).std() * np.sqrt(252))
    ctx.log["listed_iv_median_oct21"] = listed_iv
    ctx.log["realised_vol_250"] = rv250
    fut_cost = _costs(ctx).per_contract() / S0
    derisk_cost = (0.1 + ctx.cfg["costs"]["equity_commission_bps"] * 1e-4 * S0) / S0  # half of a 0.2-RUB tick + commission
    ctx.log["fut_cost_one_way"] = fut_cost
    rows, cap_rows, cx_rows, pareto_rows = [], [], [], []
    daily_y = (1 + y) ** (1 / 252) - 1
    for vname, Xv in variants.items():
        for hz_name, hz in ctx.cfg["horizons_trading_days"].items():
            # structural (stationary) basis: futures path = spot path x carry decay x empirical basis residual
            Pth = _paths_for(ctx, B, vname, hz)
            tau = hz / 252
            n_rolls = int(np.ceil(tau / 0.25))  # quarterly SMLT contracts
            base_strats = _static_strategies(listed_iv, rv250, tau)
            for Cg in ctx.cfg["capital"]["hedge_capital_grid_rub"]:
                for name, kw in base_strats.items():
                    fn = _fut_notional(ctx, kw.get("h_fut", 0.0), kw.get("derisk", 0.0))
                    kw = {**kw, "fut_notional": fn} if fn else kw
                    need_im = fn * mr
                    feasible = need_im <= ctx.cfg["optimization"]["margin_budget_fraction_of_hedge_capital"] * Cg + 1e-6
                    if not feasible:
                        continue
                    m = evaluate_static(Pth, E=ctx.E, C=float(Cg), im_rate_fut=mr, mmf_daily=daily_y,
                                        fut_cost_one_way=fut_cost, n_rolls=n_rolls, derisk_cost=derisk_cost, **kw)
                    m.update({"variant": vname, "horizon": hz_name, "capital": Cg, "strategy": name,
                              "class": _klass(name), "executability": _exec(name)})
                    rows.append(m)
            # ---- optimiser on horizon scenarios (main capital + grid), cardinality and pareto
            HR = horizon_returns(Pth, cols)
            sub = HR.sample(n=min(4000, len(HR)), random_state=sim["seed"]).reset_index(drop=True)
            for Cg in ctx.cfg["capital"]["hedge_capital_grid_rub"]:
                prob = _problem(sub, specs, hz_name, ctx, float(Cg), y * tau)
                if vname == "FHS_FULL" or (vname == "HS_FULL" and Cg == ctx.C):
                    path = cardinality_path(prob, [1] + ctx.cfg["optimization"]["cardinality_grid"])
                    for K, sol in path.items():
                        if sol is None:
                            continue
                        lin = {s["series"]: v for s, v in zip(specs, sol.h) if s["name"] != "SMLT_FUT" and abs(v) > 1e-6}
                        hf = -float(sol.h[[s["name"] for s in specs].index("SMLT_FUT")]) if "SMLT_FUT" in [s["name"] for s in specs] else 0.0
                        ev = evaluate_static(Pth, E=ctx.E, C=float(Cg), h_fut=hf, fut_notional=hf * ctx.E, lin=lin, im_rate_fut=mr,
                                             im_rate_lin={s["series"]: s["margin"] for s in specs}, mmf_daily=daily_y,
                                             fut_cost_one_way=fut_cost, n_rolls=n_rolls,
                                             lin_cost_one_way={s["series"]: s[f"cost_{hz_name}"] / 2 for s in specs})
                        cx_rows.append({"variant": vname, "horizon": hz_name, "capital": Cg, "K": K,
                                        "active": "|".join(sol.active), "lp_ES95_pct": sol.es, "lp_cost_pct": sol.cost,
                                        "lp_margin_pct": sol.margin, **{f"w_{n}": float(v) for n, v in zip(sol.names, sol.h)},
                                        **{f"path_{k}": v for k, v in ev.items()}})
                    base = min_es(prob)
                    mv = min_variance(prob)
                    ms = max_sharpe(prob)
                    for tag, sol in (("MIN_ES", base), ("MIN_VAR", mv), ("MAX_SHARPE_DIAGNOSTIC", ms)):
                        cap_rows.append({"variant": vname, "horizon": hz_name, "capital": Cg, "objective": tag,
                                         **sol.as_dict()})
                if vname == "FHS_FULL" and Cg == ctx.C:
                    for u in (None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                        prob.min_upside_capture = u
                        sol = min_es(prob)
                        up = sub["SMLT"] > 0
                        pnl = portfolio_pnl(prob, sol.h)
                        pareto_rows.append({"horizon": hz_name, "min_upside_capture": u, **sol.as_dict(),
                                            "realised_upside_capture": float(pnl[up].mean() / prob.base()[up.values].mean())})
                    prob.min_upside_capture = None
    bs = pd.DataFrame(rows)
    ctx.save(bs, "bootstrap_strategy_metrics.csv", index=False)
    ctx.save(pd.DataFrame(cap_rows), "optimizer_solutions.csv", index=False)
    ctx.save(pd.DataFrame(cx_rows), "complexity_frontier.csv", index=False)
    ctx.save(pd.DataFrame(pareto_rows), "pareto_es_upside.csv", index=False)


def _fut_ctx(ctx: Ctx) -> tuple[float, float, float]:
    """(S0 official close, F0 front settle, point value) at the cutoff."""
    P = ctx.read("prices.csv", index_col=0, parse_dates=True)
    fut = ctx.read("futures_SMLT_FUT.csv", "interim", index_col=0, parse_dates=True)
    t = pd.Timestamp(ctx.cutoff)
    return float(P["SMLT"].loc[t]), float(fut.loc[t, "F_held"]), 1.0


def _fut_notional(ctx: Ctx, h: float, derisk: float = 0.0) -> float:
    """Exact contract notional for a share-unit hedge ratio (same rounding as the contract engine)."""
    if not h:
        return 0.0
    from .strategies.futures import contract_position
    S0, F0, pv = _fut_ctx(ctx)
    return contract_position(h, ctx.E * (1 - derisk) / S0, F0, pv)["notional_rub"]


def _static_strategies(listed_iv: float, rv: float, tau: float) -> dict:
    s = {"CONTROL": {}}
    for h in (0.25, 0.5, 0.75, 1.0):
        s[f"FUT_{int(h * 100)}"] = {"h_fut": h}
    for d in (0.25, 0.5, 0.75, 1.0):
        s[f"DERISK_SELL_{int(d * 100)}"] = {"derisk": d}
    for k in (0.9, 0.8):
        for src, iv in (("listedIV", listed_iv), ("realisedVol", rv)):
            if np.isfinite(iv):
                s[f"OTC_PUT_{int(k * 100)}_{src}"] = {"put": {"K": k, "T": tau, "iv": iv}}
    if np.isfinite(listed_iv):
        s["OTC_COLLAR_90_120_listedIV"] = {"put": {"K": 0.9, "T": tau, "iv": listed_iv, "short_call_K": 1.2}}
        s["OTC_COLLAR_80_130_listedIV"] = {"put": {"K": 0.8, "T": tau, "iv": listed_iv, "short_call_K": 1.3}}
    return s


def _klass(name: str) -> str:
    if name == "CONTROL":
        return "CONTROL"
    if name.startswith("FUT_"):
        return "HEDGE_DIRECT"
    if name.startswith("DERISK"):
        return "DE_RISKING"
    if "PUT" in name or "COLLAR" in name:
        return "CONVEX_INSURANCE"
    return "OTHER"


def _exec(name: str) -> str:
    if "OTC" in name:
        return "THEORETICAL_ONLY"
    return "EXECUTABLE"


def _problem(HR: pd.DataFrame, specs: list[dict], hz: str, ctx: Ctx, C: float, mmf_h: float) -> Problem:
    sl, cols = [], []
    for s in specs:
        cols.append(s["series"])
        # capital-using long sleeves (gold/CNY spot) consume hedge capital 1:1
        margin = s["margin"] if not s.get("capital") else 1.0
        sl.append(Sleeve(s["name"], s["lo"], s["hi"], cost=s[f"cost_{hz}"], margin=margin, group=s["group"]))
    R = HR[cols].values.copy()
    # spot sleeves bought with sleeve cash lose the money-market yield they would otherwise earn
    for j, s in enumerate(specs):
        if s.get("capital"):
            R[:, j] = R[:, j] - mmf_h
    budget = ctx.cfg["optimization"]["margin_budget_fraction_of_hedge_capital"] * C / ctx.E
    return Problem(r_s=HR["SMLT"].values, R=R, sleeves=sl, k_liq=C / ctx.E, m_liq=np.full(len(HR), mmf_h),
                   alpha=0.95, margin_budget=budget, gross_cap=ctx.cfg["optimization"]["max_gross_notional_multiple_of_smlt"])


# ============================================================ S9 deterministic scenarios
from .scenarios.deterministic import Market, Strategy, event_overlay, scenario_matrix  # noqa: E402


def market_at_cutoff(ctx: Ctx) -> tuple[Market, dict]:
    P, R, V = _panel(ctx)
    t = pd.Timestamp(ctx.cutoff)
    fut = ctx.read("futures_SMLT_FUT.csv", "interim", index_col=0, parse_dates=True)
    S0 = float(P["SMLT"].loc[t])
    F0 = float(fut.loc[t, "F_held"])
    dte = float(fut.loc[t, "days_to_expiry"])
    basis = float(np.log(F0 / S0) * 365.0 / dte)
    imr = ctx.read("im_rate_mr1.csv", index_col=0, parse_dates=True)
    lim = ctx.read("ncc_futures_limits.csv", parse_dates=["tradedate"])
    sm = lim[lim["assetcode"] == "SMLT"].set_index("tradedate").sort_index().loc[:t]
    mr1, mr2 = float(sm["mr1"].iloc[-1]), float(sm["mr2"].iloc[-1])
    relief = float(sm["mr1"].loc[t - pd.Timedelta(days=730):].median())
    mix = ctx.read("futures_MIX_FUT.csv", "interim", index_col=0, parse_dates=True)
    idx_basis = float(np.log(mix.loc[t, "F_held"] / (P["IMOEX"].loc[t] * 100)) * 365.0 / mix.loc[t, "days_to_expiry"]) \
        if "IMOEX" in P and t in mix.index else 0.0
    x = R[["SMLT", "IMOEX"]].dropna().tail(250)
    beta_idx = float(np.cov(x["IMOEX"], x["SMLT"], ddof=1)[0, 1] / x["SMLT"].var(ddof=1))
    rg = load.index(ctx.root, "RGBI", ctx.cutoff)[0]
    dur_days = float(rg["DURATION"].dropna().iloc[-1]) if "DURATION" in rg and rg["DURATION"].notna().any() else np.nan
    rg_y = float(rg["YIELD"].dropna().iloc[-1]) / 100 if "YIELD" in rg and rg["YIELD"].notna().any() else np.nan
    dmod = (dur_days / 365.0) / (1 + rg_y) if np.isfinite(dur_days) and np.isfinite(rg_y) else 5.0
    y = _mmf_yield(ctx)
    opt = pd.read_csv(ctx.out / "smlt_options_20260925.csv")
    listed_iv = _listed_atm_iv(opt)
    rv = float(R["SMLT"].tail(250).std() * np.sqrt(252))
    per = _costs(ctx).per_contract() / S0
    m = Market(S0=S0, basis_ann=basis, y_mmf=y, mr_base=mr1, mr_stress=mr2, mr_relief=relief, beta_idx=beta_idx,
               index_basis_ann=idx_basis, rgbi_dmod=dmod, rgbi_yield=rg_y if np.isfinite(rg_y) else 0.14,
               listed_iv=listed_iv, realised_vol_250=rv, fut_cost_rate=per, F0=F0, point_value=1.0)
    info = {"S0": S0, "F0": F0, "front_future": fut.loc[t, "held"], "days_to_expiry": dte, "basis_ann": basis,
            "mr1": mr1, "mr2": mr2, "mr1_median_2y": relief, "beta_index_on_smlt_250d": beta_idx,
            "index_basis_ann": idx_basis, "rgbi_duration_days": dur_days, "rgbi_yield": rg_y, "rgbi_dmod": dmod,
            "mmf_yield_LQDT_60d": y, "listed_iv_oct21_median": listed_iv, "realised_vol_250d": rv,
            "fut_cost_rate_one_way": per}
    return m, info


def s9_scenarios(ctx: Ctx) -> None:
    m, info = market_at_cutoff(ctx)
    ctx.save(pd.Series(info, name="value").to_frame(), "market_inputs_at_cutoff.csv")
    cx = pd.read_csv(ctx.out / "complexity_frontier.csv")
    strats = [Strategy("CONTROL", "CONTROL")]
    for h in (0.25, 0.5, 0.75, 1.0):
        strats.append(Strategy(f"FUT_{int(h * 100)}", "HEDGE_DIRECT", h_fut=h))
    strats += [
        Strategy("INDEX_HSTAR", "CROSS_HEDGE", h_idx=max(float(pd.read_csv(ctx.out / "candidate_stats.csv").set_index("series")
                                                          .loc["MIX_FUT", "hstar_250d"]), 0.0)),
        Strategy("FUT_75_PLUS_INDEX", "COMBO", h_fut=0.75, h_idx=0.2),
        Strategy("FUT_75_PLUS_RGBI_LONG", "COMBO", h_fut=0.75, w_rgbi=0.3),
        Strategy("OTC_PUT_90_listedIV", "CONVEX_INSURANCE", put_K=0.9, executability="THEORETICAL_ONLY"),
        Strategy("OTC_PUT_80_listedIV", "CONVEX_INSURANCE", put_K=0.8, executability="THEORETICAL_ONLY"),
        Strategy("OTC_PUT_SPREAD_90_60", "CONVEX_INSURANCE", put_K=0.9, put_low_K=0.6, executability="THEORETICAL_ONLY"),
        Strategy("OTC_COLLAR_90_120", "CONVEX_INSURANCE", put_K=0.9, call_K=1.2, executability="THEORETICAL_ONLY"),
        Strategy("DERISK_SELL_50", "DE_RISKING", derisk=0.5),
        Strategy("DERISK_SELL_100", "DE_RISKING", derisk=1.0),
    ]
    grid = [-0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
    horizons = {k: v / 252 for k, v in ctx.cfg["horizons_trading_days"].items()}
    sm = scenario_matrix(strats, m, grid, horizons, ctx.E, ctx.C)
    ctx.save(sm, "scenario_matrix.csv", index=False)
    ev = event_overlay(strats, m, horizons, ctx.E, ctx.C)
    ctx.save(ev, "event_overlay.csv", index=False)
    # robustness: in how many (x, rates, credit) cells is each strategy within 5% of E of the best
    # downside outcome while keeping upside? -> report rank statistics
    piv = sm[sm["executability"] != "THEORETICAL_ONLY"].copy()
    piv["rank_in_cell"] = piv.groupby(["horizon", "smlt_terminal_return", "rates", "credit"])["pnl_rub"].rank(ascending=False)
    rob = piv.groupby(["horizon", "strategy"]).agg(
        worst_cell_pnl_pct=("pnl_pct_of_smlt_10m", "min"), best_cell_pnl_pct=("pnl_pct_of_smlt_10m", "max"),
        mean_rank=("rank_in_cell", "mean"), max_collateral_shortfall=("comp_collateral_shortfall", "max")).reset_index()
    ctx.save(rob, "scenario_robustness.csv", index=False)


# ============================================================ S7b walk-forward of the optimised overlay
def s7b_overlay_walk_forward(ctx: Ctx) -> None:
    """Weekly min-ES re-optimisation on the trailing 250 daily returns (data < decision day), applied to the
    next week. Sleeves = screened executable/limited representatives. Variants: unrestricted and K <= 2."""
    P, R, V = _panel(ctx)
    uni = pd.read_csv(ctx.out / "instrument_universe.csv")
    S0 = float(P["SMLT"].loc[ctx.cutoff])
    specs = [s for s in _sleeve_specs(ctx, uni, S0) if s["series"] in R and s["cls"] in ("EXECUTABLE", "LIMITED_EXECUTABILITY")]
    cols = ["SMLT"] + [s["series"] for s in specs]
    X = R[cols].dropna()
    per_contract = _costs(ctx).per_contract()
    fut_px = ctx.read("futures_SMLT_FUT.csv", "interim", index_col=0, parse_dates=True)["F_held"]
    mmf = R["LQDT"].fillna(0.0)
    win, step = 250, ctx.cfg["walk_forward"]["rebalance_every_days"]
    start = max(win, int(X.index.searchsorted(pd.Timestamp(ctx.cfg["walk_forward"]["oos_start"]))))
    out = {}
    for label, K in (("OVERLAY_MIN_ES_ALL", None), ("OVERLAY_MIN_ES_K2", 2)):
        h = np.zeros(len(specs))
        rec = []
        for i in range(start, len(X)):
            t = X.index[i]
            r = X.iloc[i]
            pnl = ctx.E * (r["SMLT"] + float(np.dot(h, r[[s["series"] for s in specs]].values)))
            trade_cost = 0.0
            if (i - start) % step == 0:
                hist = X.iloc[i - win:i]                        # strictly before day t
                sl = []
                for s in specs:
                    c1 = per_contract / float(fut_px.get(t, np.nan)) if s["name"] == "SMLT_FUT" else s["cost_6m"] / 6
                    sl.append(Sleeve(s["name"], s["lo"], s["hi"], cost=c1 if np.isfinite(c1) else 0.0,
                                     margin=s["margin"] if not s.get("capital") else 1.0, group=s["group"]))
                prob = Problem(r_s=hist["SMLT"].values, R=hist[[s["series"] for s in specs]].values, sleeves=sl,
                               margin_budget=0.8 * ctx.C / ctx.E, gross_cap=ctx.cfg["optimization"]["max_gross_notional_multiple_of_smlt"])
                sol = min_es(prob) if K is None else cardinality_path(prob, [K])[K]
                new = sol.h if sol is not None else h
                trade_cost = float(np.dot(np.abs(new - h), [x.cost for x in sl])) * ctx.E
                h = new
            carry = ctx.C * float(mmf.get(t, 0.0))
            rec.append({"date": t, "pnl_total": pnl - trade_cost + carry, "pnl_spot": ctx.E * r["SMLT"], "costs": trade_cost,
                        "carry": carry, **{f"h_{s['name']}": v for s, v in zip(specs, h)}})
        out[label] = pd.DataFrame(rec).set_index("date")
    rows = []
    for label, d in out.items():
        unhedged = d["pnl_spot"] + d["carry"]
        hcols = [c for c in d.columns if c.startswith("h_")]
        turnover = float(d[hcols].diff().abs().sum().sum())
        for base_name, base in (("A_per_SMLT_10m", ctx.E), ("B_per_total_capital", ctx.E + ctx.C)):
            m = summary(d["pnl_total"], base_rub=base, bench_pnl_rub=unhedged)
            m.update({"strategy": label, "basis": base_name, "HE_VAR": he_var(d["pnl_total"].values, unhedged.values),
                      "turnover_notional_multiple": turnover, "total_costs_rub": float(d["costs"].sum()),
                      "mean_active_sleeves": float((d[hcols].abs() > 1e-6).sum(axis=1).mean()),
                      **{f"mean_{c}": float(d[c].mean()) for c in hcols}})
            rows.append(m)
        ctx.save(d, f"walkforward_{label}.csv", "interim")
    ctx.save(pd.DataFrame(rows), "walkforward_overlay_metrics.csv", index=False)
