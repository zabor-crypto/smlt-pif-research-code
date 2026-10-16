"""V3 Phase 1: Bitcoin as a candidate diversifier / hedge sleeve (evidence-based; BTC is NOT assumed to be a hedge).

ECONOMIC history : external proxy BTC_PROXY (Binance BTCUSDT, close at 16:00 UTC on each MOEX day; data/crypto.py),
                   cross-checked against Coinbase BTC-USD and the non-synchronous 00:00-UTC close.
EXECUTION history: MOEX instruments only (MOEXBTC index, BTC monthly futures on MOEXBTC, BTCUSDF perpetual, IBIT
                   futures). Only MOEX observations establish executability / liquidity / costs.
Admission rule (pre-declared): BTC enters the main portfolio library ONLY IF
  (a) the TEST-set paired difference ES95(best portfolio with BTC) - ES95(best without BTC), both selected on
      VALIDATION with the same rule, is STABLE_AND_MATERIAL (every admissible model, 20 seeds), AND
  (b) the historical walk-forward (quarterly re-estimated min-ES, data < decision day) with BTC has a lower daily
      ES95 than without BTC with a block-bootstrap 95% CI of the difference that excludes zero.
Otherwise: DIVERSIFIER_ONLY (executable, some diversification, no out-of-sample value) or REJECTED.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import pipeline as pl
from .data import load
from .portfolio import regime_risk as rr
from .portfolio import sleeves as sleeves_mod
from .portfolio import validation as va
from .risk.metrics import es_hist, he_var

BTC_SLEEVE = {"name": "BTC", "series": "BTC_PROXY", "asset": "BTC", "family": "BTC", "lo": -0.3, "hi": 0.3, "rolls_per_year": 12}


# ------------------------------------------------------------------ 1.3 proxy -> executable mapping
def _pair_stats(a: pd.Series, b: pd.Series, name_a: str, name_b: str) -> dict:
    d = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if len(d) < 5:
        return {"proxy": name_a, "target": name_b, "n_overlap": len(d), "status": "INSUFFICIENT_OVERLAP"}
    lag = {}
    for k in (-2, -1, 1, 2):
        x = pd.concat([a, b.shift(-k)], axis=1).dropna()
        lag[f"corr_lag{k:+d}"] = float(x.corr().iloc[0, 1]) if len(x) > 5 else np.nan
    return {"proxy": name_a, "target": name_b, "n_overlap": len(d), "first": str(d.index.min().date()),
            "last": str(d.index.max().date()), "corr": float(d.corr().iloc[0, 1]),
            "tracking_error_ann": float((d["a"] - d["b"]).std() * np.sqrt(252)), "vol_proxy_ann": float(d["a"].std() * np.sqrt(252)),
            "vol_target_ann": float(d["b"].std() * np.sqrt(252)), "mean_diff_daily": float((d["a"] - d["b"]).mean()),
            "beta_target_on_proxy": float(np.cov(d["b"], d["a"])[0, 1] / d["a"].var()), **lag, "status": "OK"}


def proxy_validation(ctx, P, R) -> pd.DataFrame:
    rows = []
    for tgt in ("BTC_PROXY_UTC0", "BTC_COINBASE_UTC0", "MOEXBTC", "BTC_FUT", "BTCUSDF", "IBIT_FUT"):
        if tgt in R:
            rows.append(_pair_stats(R["BTC_PROXY"], R[tgt], "BTC_PROXY(Binance 16:00UTC)", tgt))
    if "BTC_COINBASE_UTC0" in R and "BTC_PROXY_UTC0" in R:
        rows.append(_pair_stats(R["BTC_PROXY_UTC0"], R["BTC_COINBASE_UTC0"], "BINANCE_UTC0", "COINBASE_UTC0"))
    if "MOEXBTC" in R and "BTC_FUT" in R:
        rows.append(_pair_stats(R["MOEXBTC"], R["BTC_FUT"], "MOEXBTC", "BTC_FUT"))
    df = pd.DataFrame(rows)
    # basis of the MOEX futures to the MOEX index (log, annualised by days to expiry)
    fut = ctx.interim / "futures_BTC_FUT.csv"
    if fut.exists() and "MOEXBTC" in P:
        f = pd.read_csv(fut, index_col=0, parse_dates=True)
        lb = np.log(f["F_held"] / P["MOEXBTC"].reindex(f.index)).dropna()
        ann = (lb * 365 / f["days_to_expiry"].clip(lower=1).reindex(lb.index)).dropna()
        df = pd.concat([df, pd.DataFrame([{"proxy": "MOEXBTC", "target": "BTC_FUT basis ln(F/index)", "n_overlap": len(lb),
                                            "mean_diff_daily": float(lb.mean()) if len(lb) else np.nan,
                                            "basis_ann_mean": float(ann.mean()) if len(ann) else np.nan,
                                            "basis_ann_std": float(ann.std()) if len(ann) > 1 else np.nan, "status": "OK" if len(lb) else "NO_DATA"}])],
                       ignore_index=True)
    # alignment sensitivity vs SMLT (synchronised vs +-1 day vs 00:00 UTC)
    for name, s in (("sync_16UTC_lag0", R["BTC_PROXY"]), ("sync_16UTC_btc_lead1", R["BTC_PROXY"].shift(-1)),
                    ("sync_16UTC_btc_lag1", R["BTC_PROXY"].shift(1)), ("utc0_nonsync", R.get("BTC_PROXY_UTC0"))):
        if s is None:
            continue
        d = pd.concat([R["SMLT"], s], axis=1).dropna()
        d = d[d.index >= pd.Timestamp(ctx.cfg["simulation"]["bootstrap_start"])]
        df = pd.concat([df, pd.DataFrame([{"proxy": f"BTC {name}", "target": "SMLT", "n_overlap": len(d),
                                            "corr": float(d.corr().iloc[0, 1]), "status": "ALIGNMENT_SENSITIVITY"}])], ignore_index=True)
    return df


# ------------------------------------------------------------------ 1.2 MOEX contracts
def moex_contracts(ctx, P, V) -> pd.DataFrame:
    from .data.specs import pit_terms
    fc = load.futures_contracts(ctx.root, ctx.cutoff)
    spec = load.futures_specs(ctx.root).set_index("SECID")
    lim = ctx.read("ncc_futures_limits.csv", parse_dates=["tradedate"])
    lim = lim[lim["tradedate"] <= pd.Timestamp(ctx.cutoff)].sort_values("tradedate").groupby("assetcode").tail(1).set_index("assetcode")
    terms = pit_terms(ctx.root, ctx.cutoff).set_index("family")
    t = pd.Timestamp(ctx.cutoff)
    rows = []
    for asset, fam, kind, underlying in (("BTC", "BTC", "monthly futures", "MOEX Bitcoin Index (MOEXBTC)"),
                                         ("BTCUSDF", "BTCUSDF", "perpetual (one-day, auto-prolonged)", "MOEX Bitcoin Index (MOEXBTC)"),
                                         ("IBIT", "IBIT", "quarterly futures", "iShares Bitcoin Trust ETF (IBIT) share price")):
        key = "SECID" if asset == "BTCUSDF" else "ASSETCODE"
        h = fc[fc[key] == asset]
        if h.empty:
            rows.append({"instrument": asset, "status": "DATA_NOT_AVAILABLE"})
            continue
        for secid, g in h.groupby("SECID"):
            g = g.sort_values("TRADEDATE")
            s = spec.loc[secid] if secid in spec.index else pd.Series(dtype=object)
            last = g[g["TRADEDATE"] <= t].tail(1)
            live = bool(len(last) and last["TRADEDATE"].iloc[0] == t)
            tail20 = g[(g["TRADEDATE"] <= t)].tail(20)
            tr = terms.loc[fam] if fam in terms.index else pd.Series(dtype=object)
            mr = float(lim.loc[asset, "mr1"]) if asset in lim.index else np.nan
            notional = float(last["SETTLEPRICE"].iloc[0]) * float(tr.get("point_value_used", np.nan)) if live else np.nan
            rows.append({
                "instrument": asset, "secid": secid, "contract_type": kind, "underlying": underlying,
                "contract_name": s.get("CONTRACTNAME"), "price_unit": s.get("UNIT", "USD"),
                "first_trading_date_spec": s.get("FRSTTRADE"), "first_history_date": str(g["TRADEDATE"].min().date()),
                "last_trading_date": s.get("LSTTRADE"), "trading_on_cutoff": live,
                "settle_on_cutoff": float(last["SETTLEPRICE"].iloc[0]) if live else np.nan,
                "point_value_rub_pit": float(tr.get("point_value_used", np.nan)) if live else np.nan,
                "notional_rub_per_contract_pit": notional,
                "ncc_im_rate_mr1_cutoff": mr, "im_rub_per_contract": notional * mr if live else np.nan,
                "open_interest_cutoff": float(last["OPENPOSITION"].iloc[0]) if live else np.nan,
                "adv20_rub": float(tail20["VALUE"].mean()) if len(tail20) else np.nan,
                "adv20_contracts": float(tail20["VOLUME"].mean()) if len(tail20) else np.nan,
                "n_history_days_to_cutoff": int((g["TRADEDATE"] <= t).sum()),
                "exchange_fee_rate": float(tr.get("exchange_fee_rate", np.nan)),
                "funding_swaprate_mean": float(g["SWAPRATE"].mean()) if asset == "BTCUSDF" and "SWAPRATE" in g else np.nan,
                "rub_pnl_mechanics": "VM in RUB = contracts x dF(USD) x point value; point value = USD/RUB x contract size "
                                     "(moves daily); settlement in RUB, no BTC delivery",
                "bid_ask": "NOT_AVAILABLE_PIT (ISS history has no quotes; post-cutoff board quotes are not used)",
                "weekend_session": "YES (spec WEEKENDSESSION=1); research calendar uses weekday settlements only",
                "status": "OK"})
    rows.append({"instrument": "SPOT_BTC_ON_MOEX", "status": "NOT_VERIFIED",
                 "contract_type": "direct spot bitcoin", "rub_pnl_mechanics": "no MOEX spot BTC instrument found in ISS; not assumed available"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 1.4 relevance tests
def _stats(s: pd.Series, b: pd.Series) -> dict:
    d = pd.concat([s, b], axis=1, keys=["s", "b"]).dropna()
    n = len(d)
    if n < 20:
        return {"n": n}
    q = d["s"].quantile(0.05)
    qb = d["b"].quantile(0.05)
    worst = d["s"] <= q
    co = float(((d["b"] <= qb) & worst).sum() / max(worst.sum(), 1))
    dn = d[d["s"] < 0]
    roll = d["s"].rolling(120).corr(d["b"]).dropna()
    return {"n": n, "corr": float(d.corr().iloc[0, 1]), "downside_corr": float(dn.corr().iloc[0, 1]) if len(dn) > 20 else np.nan,
            "tail_coexceedance_prob": co, "tail_coexceedance_ratio_vs_independence": co / 0.05,
            "mean_btc_ret_on_worst5pct_smlt_days": float(d.loc[worst, "b"].mean()),
            "hstar_smlt_on_btc": float(np.cov(d["s"], d["b"])[0, 1] / d["b"].var()),
            "beta_btc_on_smlt": float(np.cov(d["s"], d["b"])[0, 1] / d["s"].var()),
            "rolling120_corr_mean": float(roll.mean()) if len(roll) else np.nan,
            "rolling120_corr_sd": float(roll.std()) if len(roll) > 1 else np.nan,
            "rolling120_share_positive": float((roll > 0).mean()) if len(roll) else np.nan}


def regime_statistics(ctx, R, reg, ps) -> pd.DataFrame:
    start = pd.Timestamp(ctx.cfg["simulation"]["bootstrap_start"])
    t = pd.Timestamp(ctx.cutoff)
    base = R.loc[(R.index >= start) & (R.index <= t)]
    masks = rr.state_masks(reg, ps)
    sets = {"FULL": base.index, "LAST_250D": base.index[-250:], "LAST_120D": base.index[-120:],
            "SMLT_DOWNSIDE_DAYS": base.index[base["SMLT"] < 0]}
    for k in ("SMLT_UPTREND", "SMLT_RANGE", "SMLT_DOWNTREND", "SMLT_HIGH_VOL_SHOCK", "MARKET_STRESS", "CREDIT_STRESS",
              "RATES_STRESS", "COMBINED_STRESS"):
        m = masks[k].reindex(base.index).fillna(False)
        sets[k] = base.index[m.values]
    rows = []
    for series, label in (("BTC_PROXY", "USD economic proxy (16:00 UTC)"), ("BTC_FUT", "MOEX BTC futures (RUB P&L per RUB notional)")):
        if series not in R:
            continue
        for k, idx in sets.items():
            st = _stats(base.loc[idx, "SMLT"], base.loc[idx, series])
            rows.append({"btc_series": series, "series_label": label, "set": k, **st})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ incremental out-of-sample value
def _wf_btc(ctx, specs, with_btc: bool) -> pd.DataFrame:
    """Quarterly re-estimated min-ES (trailing 250d, demeaned, data < decision day) walk-forward, 2023-01..cutoff."""
    from .phase2 import _env
    from .portfolio import adaptive
    use = [s for s in specs if s.name in (("DIRECT", "MARKET", "BTC") if with_btc else ("DIRECT", "MARKET"))]
    env = _env(ctx, use)
    cols = ["SMLT"] + [s.series for s in use]
    dates = env.R[cols].dropna().index
    dates = dates[(dates >= pd.Timestamp(ctx.cfg["walk_forward"]["oos_start"])) & (dates <= pd.Timestamp(ctx.cutoff))]
    rows = {}
    for q in pd.date_range(dates.min(), dates.max(), freq="QS"):
        d0 = dates[dates >= q]
        if not len(d0):
            continue
        p = env.R.index.get_loc(d0[0])
        rows[d0[0]] = adaptive._lp_target(env, p, 0.40, 0.45, adaptive.CASH_ONLY)
    tg = pd.DataFrame.from_dict(rows, orient="index", columns=[s.name for s in use])
    tg["regime"] = "NORMAL"
    return adaptive.daily_walk(env, tg, dates, 0.0)


def _block_boot_es_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, block: int = 10, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(a)
    nb = int(np.ceil(n / block))
    out = []
    for _ in range(n_boot):
        st = rng.integers(0, n, nb)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        out.append(es_hist(a[idx], 0.95) - es_hist(b[idx], 0.95))
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def s17_btc(ctx) -> None:
    from .v3_stages import _regime_inputs, verdict
    P, R, X, reg, ps, pst = _regime_inputs(ctx)
    V = ctx.read("value_rub.csv", index_col=0, parse_dates=True)
    if "BTC_PROXY" not in R:
        ctx.log["btc"] = "DATA_NOT_AVAILABLE"
        return
    # MOEX BTC futures: RUB P&L per RUB notional already in R["BTC_FUT"] (settle-to-settle within contract)
    ctx.save(proxy_validation(ctx, P, R), "btc_proxy_validation.csv", index=False)
    mc = moex_contracts(ctx, P, V)
    ctx.save(mc, "btc_moex_contracts.csv", index=False)
    rs = regime_statistics(ctx, R, reg, ps)
    ctx.save(rs, "btc_regime_statistics.csv", index=False)
    # ---- executability (MOEX observations only)
    need = abs(BTC_SLEEVE["lo"]) * ctx.E
    fut_live = mc[(mc["instrument"] == "BTC") & (mc["trading_on_cutoff"] == True)]  # noqa: E712
    fut_live = fut_live.sort_values("open_interest_cutoff", ascending=False)
    exe = "REJECTED"
    if len(fut_live):
        f = fut_live.iloc[0]
        liquid = f["adv20_rub"] >= 10 * need and f["open_interest_cutoff"] * f["notional_rub_per_contract_pit"] >= 10 * need
        exe = "EXECUTABLE" if liquid and f["n_history_days_to_cutoff"] >= 60 else "LIMITED_EXECUTABILITY"
    # ---- incremental OOS value (a) TEST paired difference of selected portfolios with vs without BTC
    v3 = ctx.cfg["v3"]
    base_specs = sleeves_mod.build(ctx)
    btc_specs = sleeves_mod.build(ctx, extra=[BTC_SLEEVE])
    cfg_sel = ctx.cfg["smart_portfolio"]["selection"]
    rows, dl = [], []
    for hz_name, hz in (("6m", 126), ("12m", 252)):
        sel = {}
        for tag, specs in (("NO_BTC", base_specs), ("WITH_BTC", btc_specs)):
            fac = va.ScenarioFactory(ctx, pl, specs, extra_cols=("BTC_PROXY",))
            # identical generation rules for both universes (K <= 2 plus the full set; FHS, HS and min-max)
            cands = va.generate(ctx, fac, hz, ctx.C, models=["FHS_FULL", "HS_FULL", "MINMAX"], max_K=2)
            for i, c in enumerate(cands):
                c["cand_id"] = f"{tag}_{hz_name}_{i:04d}"
                c["h"] = {**{s.name: 0.0 for s in btc_specs}, **c["h"]}
            cdf = pd.DataFrame([{"cand_id": c["cand_id"], "family": c["family"],
                                 "K_active": sum(1 for v in c["h"].values() if abs(v) > 1e-4),
                                 **{f"w_{k}": v for k, v in c["h"].items()}} for c in cands])
            ev = va.evaluate(ctx, fac, cands, hz, ctx.C, "VAL", v3["n_val_paths"])
            summ = va.summarise(ev).merge(cdf, on="cand_id")
            for rule in ("static", "robust"):
                r = va.select(summ.drop(columns=["family", "K_active"]), cdf, rule, cfg_sel, "ANY")
                sel[(tag, rule)] = r
                rows.append({"horizon": hz_name, "universe": tag, "rule": rule, **(r.to_dict() if r is not None else {"cand_id": "NONE"})})
        fac = va.ScenarioFactory(ctx, pl, btc_specs, extra_cols=("BTC_PROXY",))
        fin = []
        for (tag, rule), r in sel.items():
            if r is None:
                continue
            h, _ = va.round_contracts(btc_specs, {s.name: float(r.get(f"w_{s.name}", 0.0)) for s in btc_specs}, ctx.E)
            fin.append({"cand_id": f"{rule.upper()}_{tag}", "h": h})
        te = pd.concat([va.evaluate(ctx, fac, fin, hz, ctx.C, "TEST", v3["n_test_paths"], k=k, include_hist=False)
                        for k in range(v3["seeds"]["n_test"])], ignore_index=True)
        for rule in ("STATIC", "ROBUST"):
            for metric in ("ES95", "p_sleeve_breach", "mean_pnl"):
                pm = va.delta_ci(te, f"{rule}_WITH_BTC", f"{rule}_NO_BTC", metric)
                vd = verdict(pm, v3["stability"]["material_rub"] if metric == "ES95" else 0.0, v3["stability"]["min_sign_share"],
                             metric != "mean_pnl")
                for mdl, stt in pm.items():
                    dl.append({"horizon": hz_name, "rule": rule, "metric": metric, "model": mdl, **stt, "verdict_all_models": vd})
    ctx.save(pd.DataFrame(rows), "btc_selected_with_without.csv", index=False)
    dld = pd.DataFrame(dl)
    ctx.save(dld, "btc_incremental_value_ci.csv", index=False)
    # ---- (b) historical walk-forward with vs without BTC
    wf_a, wf_b = _wf_btc(ctx, btc_specs, True), _wf_btc(ctx, btc_specs, False)
    unh = wf_b["pnl_spot"] + wf_b["carry"]
    a, b = wf_a["pnl_total"].values / ctx.E, wf_b["pnl_total"].values / ctx.E
    lo, hi = _block_boot_es_diff(a, b, seed=ctx.cfg["simulation"]["seed"])
    wf = {"wf_ES95_1d_with_btc": es_hist(a, 0.95), "wf_ES95_1d_without_btc": es_hist(b, 0.95), "wf_ES95_diff_ci_lo": lo,
          "wf_ES95_diff_ci_hi": hi, "wf_HE_VAR_with": he_var(wf_a["pnl_total"].values, unh.values),
          "wf_HE_VAR_without": he_var(wf_b["pnl_total"].values, unh.values),
          "wf_mean_btc_weight": float(wf_a["h_BTC"].mean()) if "h_BTC" in wf_a else np.nan,
          "wf_share_days_btc_nonzero": float((wf_a["h_BTC"].abs() > 1e-4).mean()) if "h_BTC" in wf_a else np.nan}
    # ---- verdict
    es_rows = dld[(dld["metric"] == "ES95")]
    test_ok = bool(len(es_rows) and (es_rows["verdict_all_models"] == "STABLE_AND_MATERIAL").any())
    wf_ok = bool(hi < 0)
    full = rs[(rs["btc_series"] == "BTC_PROXY") & (rs["set"] == "FULL")].iloc[0]
    stress = rs[(rs["btc_series"] == "BTC_PROXY") & rs["set"].isin(["MARKET_STRESS", "CREDIT_STRESS", "COMBINED_STRESS", "SMLT_DOWNTREND"])]
    hedge_evidence = bool(full["corr"] > 0.2 or (stress["corr"] > 0.25).any())
    if exe == "REJECTED":
        cls = "REJECTED"
    elif test_ok and wf_ok:
        cls = exe
    else:
        cls = "DIVERSIFIER_ONLY" if exe in ("EXECUTABLE", "LIMITED_EXECUTABILITY") else "REJECTED"
    summary = {"executability_moex": exe, "hedge_evidence": hedge_evidence, "test_incremental_value": test_ok,
               "walkforward_incremental_value": wf_ok, "classification": cls, "admitted_to_main_library": cls in ("EXECUTABLE",
                                                                                                                  "LIMITED_EXECUTABILITY") and test_ok and wf_ok,
               "corr_full": float(full["corr"]), **wf}
    ctx.save(pd.Series(summary).to_frame("value"), "btc_verdict.csv")
    ctx.log["btc"] = {k: (v if isinstance(v, (str, bool)) else float(v)) for k, v in summary.items()}
