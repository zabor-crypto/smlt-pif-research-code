"""V3 research stages (forward scenarios, regime-conditioned risk, BTC extension) on top of the repaired V2 engine.

s16  Phase 0.1/0.2/0.5  independent TRAIN/VALIDATION/TEST selection of static and robust portfolios, integer contracts,
                        multi-seed confidence intervals (model_validation_stability.csv, smart_vs_direct_delta_ci.csv)
Later stages (s17..s23) are defined in this module as well and documented in their docstrings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import pipeline as pl
from .portfolio import sleeves as sleeves_mod
from .portfolio import validation as va
from .strategies.collateral import Collateral

HZ = {"6m": 126, "12m": 252}


def _direct_frac(ctx, B, x: float) -> float:
    """Exact contract notional fraction of E for a share-unit hedge ratio x (round(x * shares) contracts at F0)."""
    return round(x * ctx.E / B["S0"]) * B["F0"] / ctx.E if x else 0.0


def _named_collateral(ctx) -> list[Collateral]:
    return [Collateral(k, v["yield_fraction"], v["haircut"], v["margin_multiplier"]) for k, v in ctx.cfg["collateral"]["named"].items()]


def finalists(ctx, specs, fac, sel: dict, hz_name: str) -> list[dict]:
    """Fixed benchmarks + selected portfolios (continuous and integer-rounded)."""
    names = [s.name for s in specs]
    z = {n: 0.0 for n in names}
    out = [{"cand_id": "CONTROL", "h": dict(z), "kind": "BENCHMARK"}]
    for x in (0.25, 0.5, 0.75, 1.0):
        out.append({"cand_id": f"FUT_{int(x * 100)}", "h": {**z, "DIRECT": -_direct_frac(ctx, fac.B, x)}, "kind": "BENCHMARK"})
    out.append({"cand_id": "DERISK_SELL_50", "h": dict(z), "derisk": 0.5, "kind": "DE_RISKING"})
    for key, r in sel.items():
        if r is None:
            continue
        h = {n: float(r[f"w_{n}"]) for n in names}
        out.append({"cand_id": f"{key}_CONT", "h": h, "kind": "SELECTED_CONTINUOUS", "src": r["cand_id"]})
        hi, _ = va.round_contracts(specs, h, ctx.E)
        out.append({"cand_id": f"{key}_INT", "h": hi, "kind": "SELECTED_INTEGER", "src": r["cand_id"]})
    return out


DELTAS = [("STATIC_SMART_INT", "STATIC_DIRECT_INT"), ("ROBUST_SMART_INT", "ROBUST_DIRECT_INT"),
          ("STATIC_SMART_INT", "FUT_25"), ("STATIC_SMART_INT", "FUT_50"), ("STATIC_SMART_INT", "FUT_75"),
          ("STATIC_SMART_INT", "FUT_100"), ("ROBUST_SMART_INT", "FUT_50"), ("STATIC_SMART_INT", "STATIC_SMART_CONT"),
          ("ROBUST_SMART_INT", "ROBUST_SMART_CONT"), ("STATIC_DIRECT_INT", "FUT_50")]


def verdict(per_model: dict, material: float, min_share: float, better_if_negative: bool = True) -> str:
    """Pre-declared stability rule for a paired difference (a - b) across seeds and admissible models."""
    if not per_model:
        return "NOT_EVALUATED"
    if all(abs(v["mean"]) < 1e-9 and v["sd"] < 1e-9 for v in per_model.values()):
        return "IDENTICAL_PORTFOLIOS"
    sgn = -1 if better_if_negative else 1
    means = [v["mean"] * sgn for v in per_model.values()]
    lo_ok = all((v["ci_hi"] < 0) if better_if_negative else (v["ci_lo"] > 0) for v in per_model.values())
    hi_bad = all((v["ci_lo"] > 0) if better_if_negative else (v["ci_hi"] < 0) for v in per_model.values())
    share = min((v["share_negative"] if better_if_negative else v["share_positive"]) for v in per_model.values())
    if lo_ok and share >= min_share:
        return "STABLE_AND_MATERIAL" if min(means) >= material else "STABLE_BUT_BELOW_MATERIALITY"
    if hi_bad:
        return "STABLY_WORSE" if max(abs(m) for m in means) >= material else "STABLY_WORSE_BELOW_MATERIALITY"
    if any(m > 0 for m in means) and any(m < 0 for m in means):
        return "MODEL_DEPENDENT"
    return "NOT_PROVEN"


def s16_validation(ctx) -> None:
    v3 = ctx.cfg["v3"]
    specs = sleeves_mod.build(ctx)
    C = ctx.C
    cfg_sel = ctx.cfg["smart_portfolio"]["selection"]
    stab = v3["stability"]
    all_c, all_val, sel_rows, int_rows, test_rows, delta_rows = [], [], [], [], [], []
    for hz_name, hz in HZ.items():
        fac = va.ScenarioFactory(ctx, pl, specs)
        ctx.log.setdefault("v3_regime_cond", {})[hz_name] = {"family": fac.regime_family, "n_days": fac.regime_n}
        cands = va.generate(ctx, fac, hz, C)
        for i, c in enumerate(cands):
            c["cand_id"] = f"{c['family']}_{hz_name}_{i:04d}"
        cdf = pd.DataFrame([{"cand_id": c["cand_id"], "family": c["family"], "gen_model": c["gen_model"], "subset": c["subset"],
                             "margin_budget": c["margin_budget"], "upside_floor": c["upside_floor"],
                             "train_obj_es_rub": c["train_obj_es"] * ctx.E if pd.notna(c["train_obj_es"]) else np.nan,
                             "share_hedge_ratio": c.get("share_hedge_ratio", np.nan),
                             "K_active": sum(1 for s in specs if abs(c["h"][s.name]) > 1e-4),
                             **{f"w_{s.name}": c["h"][s.name] for s in specs}} for c in cands])
        cdf["horizon"] = hz_name
        all_c.append(cdf)
        # ---- VALIDATION: every candidate, every admissible model + historical windows
        ev = va.evaluate(ctx, fac, cands, hz, C, "VAL", v3["n_val_paths"])
        all_val.append(ev.assign(horizon=hz_name))
        summ = va.summarise(ev).merge(cdf, on="cand_id")
        ctx.save(summ, f"v3_validation_summary_{hz_name}.csv", "interim", index=False)
        sel = {}
        for rule in ("static", "robust"):
            for scope, tag in (("ANY", "SMART"), ("DIRECT_ONLY", "DIRECT")):
                r = va.select(summ.drop(columns=["family", "K_active"]), cdf, rule, cfg_sel, scope)
                key = f"{rule.upper()}_{tag}"
                sel[key] = r
                sel_rows.append({"horizon": hz_name, "selection": key, "rule": rule, "scope": scope,
                                 **(sel[key].to_dict() if sel[key] is not None else {"cand_id": "NONE"})})
        for key, r in sel.items():
            if r is None:
                continue
            _, rep = va.round_contracts(specs, {s.name: float(r[f"w_{s.name}"]) for s in specs}, ctx.E)
            for x in rep:
                int_rows.append({"horizon": hz_name, "portfolio": key, "src_cand": r["cand_id"], **x})
        # ---- FINAL TEST: untouched paths, n_test independent seeds
        fin = finalists(ctx, specs, fac, {k: sel[k] for k in sel}, hz_name)
        # direct selections are already integer by construction: alias *_DIRECT_INT == *_DIRECT_CONT
        te = []
        for k in range(v3["seeds"]["n_test"]):
            te.append(va.evaluate(ctx, fac, fin, hz, C, "TEST", v3["n_test_paths"], k=k, include_hist=(k == 0)))
        te = pd.concat(te, ignore_index=True).assign(horizon=hz_name)
        # collateral sensitivity on TEST seed 0 (0.4 / Phase 7)
        for coll in _named_collateral(ctx)[1:]:
            te = pd.concat([te, va.evaluate(ctx, fac, fin, hz, C, "TEST", v3["n_test_paths"], k=0, include_hist=False,
                                            coll=coll).assign(horizon=hz_name, collateral=coll.name)], ignore_index=True)
        te["collateral"] = te.get("collateral", pd.Series(index=te.index, dtype=object)).fillna("A_CASH_ONLY_CONSERVATIVE")
        test_rows.append(te)
        base = te[(te["collateral"] == "A_CASH_ONLY_CONSERVATIVE") & (te["model"] != "HIST_WINDOWS")]
        for a, b in DELTAS:
            for metric in ("ES95", "ES99", "p_sleeve_breach", "mean_pnl", "upside_capture"):
                pm = va.delta_ci(base, a, b, metric, stab["ci_level"])
                if not pm:
                    continue
                neg_good = metric in ("ES95", "ES99", "p_sleeve_breach")
                vd = verdict(pm, stab["material_rub"] if metric.startswith("ES") else 0.0, stab["min_sign_share"], neg_good)
                for mdl, st in pm.items():
                    delta_rows.append({"horizon": hz_name, "strategy": a, "comparator": b, "metric": metric, "model": mdl, **st,
                                       "verdict_all_models": vd})
    ctx.save(pd.concat(all_c, ignore_index=True), "v3_candidates.csv", "interim", index=False)
    ctx.save(pd.concat(all_val, ignore_index=True), "v3_validation_metrics.csv", "interim", index=False)
    ctx.save(pd.DataFrame(sel_rows), "v3_selected_portfolios.csv", index=False)
    ctx.save(pd.DataFrame(int_rows), "integer_contract_portfolios.csv", index=False)
    T = pd.concat(test_rows, ignore_index=True)
    ctx.save(T, "v3_test_metrics.csv", "interim", index=False)
    ctx.save(pd.DataFrame(delta_rows), "smart_vs_direct_delta_ci.csv", index=False)
    ctx.save(_stability_table(ctx, T, pd.DataFrame(sel_rows), pd.concat(all_val, ignore_index=True)), "model_validation_stability.csv", index=False)
    ctx.save(collateral_repair_decomposition(ctx), "collateral_repair_decomposition.csv", index=False)


def _stability_table(ctx, T: pd.DataFrame, sel: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    """Per finalist x model: TRAIN objective, VALIDATION value and TEST distribution across seeds."""
    base = T[T["collateral"] == "A_CASH_ONLY_CONSERVATIVE"]
    rows = []
    for (hz, cid, mdl), g in base.groupby(["horizon", "cand_id", "model"]):
        src, s = None, pd.DataFrame()
        tag = cid.rsplit("_", 1)[0] if cid.endswith(("_CONT", "_INT")) else None
        if tag is not None:
            s = sel[(sel["horizon"] == hz) & (sel["selection"] == tag)]
            src = s["cand_id"].iloc[0] if len(s) else None
        vv = val[(val["horizon"] == hz) & (val["cand_id"] == src) & (val["model"] == mdl)] if src else pd.DataFrame()
        row = {"horizon": hz, "strategy": cid, "model": mdl, "n_test_seeds": int(g["seed_k"].nunique()),
               "source_candidate": src or "", "train_lp_objective_ES95_rub": float(s["train_obj_es_rub"].iloc[0]) if len(s) and "train_obj_es_rub" in s else np.nan,
               "val_ES95": float(vv["ES95"].iloc[0]) if len(vv) else np.nan,
               "val_breach": float(vv["p_sleeve_breach"].iloc[0]) if len(vv) else np.nan}
        for m in ("ES95", "ES99", "p_sleeve_breach", "mean_pnl", "upside_capture", "downside_capture", "std_pnl", "p95_max_dd"):
            x = g[m].values
            row.update({f"test_{m}_mean": float(np.mean(x)), f"test_{m}_sd": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
                        f"test_{m}_min": float(np.min(x)), f"test_{m}_max": float(np.max(x))})
        row["val_minus_test_ES95"] = row["val_ES95"] - row["test_ES95_mean"] if np.isfinite(row["val_ES95"]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ================================================================== Phase 2: regime-conditioned risk layer
def _regime_inputs(ctx):
    """Causal macro regimes, SMLT price state, portfolio state and the returns/factor matrix used by Phase 2-6."""
    from .features import factors as fx
    from .portfolio import regime_risk as rr
    from .portfolio import regimes as rgm
    P, R, V = pl._panel(ctx)
    cs = ctx.read("credit_state.csv", index_col=0, parse_dates=True)
    reg = rgm.classify(R, cs, ctx.cfg["smart_portfolio"]["regimes"])
    ps = rr.price_state(R["SMLT"], ctx.cfg["v3"]["price_state"])
    pst = rr.portfolio_state(reg["regime"], ps["price_state"])
    F = fx.factor_frame(R, cs)
    X = R.copy()
    X["SECTOR"], X["CREDIT"] = F["SECTOR"], F["CREDIT"]
    return P, R, X, reg, ps, pst


def s18_regime_risk(ctx) -> None:
    from .portfolio import regime_risk as rr
    v3 = ctx.cfg["v3"]
    P, R, X, reg, ps, pst = _regime_inputs(ctx)
    t = pd.Timestamp(ctx.cutoff)
    start = pd.Timestamp(ctx.cfg["simulation"]["bootstrap_start"])
    cols = [c for c in v3["regime_matrix_columns"] if c in X]
    Xs = X.loc[(X.index >= start) & (X.index <= t), cols].dropna()
    masks = rr.state_masks(reg, ps)
    g_cov = pd.DataFrame(np.cov(Xs.values, rowvar=False, ddof=1), index=cols, columns=cols)
    g_corr = Xs.corr()
    sizes, changes = [], []
    order = ["FULL", "NORMAL", "SMLT_UPTREND", "SMLT_RANGE", "SMLT_DOWNTREND", "SMLT_HIGH_VOL_SHOCK", "MARKET_STRESS",
             "CREDIT_STRESS", "RATES_STRESS", "COMBINED_STRESS"] + rr.PORTFOLIO_STATES
    mats = {}
    for k in order:
        mk = masks[k].reindex(Xs.index).fillna(False)
        res = rr.conditional_matrices(Xs, mk, g_cov, v3["shrink_n0"], ctx.cfg["risk"]["ewma_lambda"])
        mats[k] = res
        sizes.append({"state": k, "n_observations": res["n"], "share_of_sample": res["n"] / len(Xs),
                      "shrink_lambda_to_global": res.get("lambda", 1.0),
                      "estimator_status": "OK" if res["n"] >= 120 else ("SMALL_SAMPLE_SHRUNK" if res["n"] >= 10 else "TOO_FEW")})
        if "corr" not in res:
            continue
        ctx.save(res["corr"], f"regime_correlation_{k}.csv")
        ctx.save(res["corr_shrunk"], f"regime_correlation_shrunk_{k}.csv", "interim")
        for est in ("sample", "oas", "ledoit_wolf", "ewma", "shrunk"):
            if est in res:
                ctx.save(res[est], f"regime_covariance_{k}_{est}.csv", "interim" if k in rr.PORTFOLIO_STATES and est != "shrunk" else "out")
        for a in cols:
            for b in cols:
                if a >= b:
                    continue
                r_k, r_g = float(res["corr"].loc[a, b]), float(g_corr.loc[a, b])
                z, p = rr.fisher_change(r_k, res["n"], r_g, len(Xs))
                d = r_k - r_g
                changes.append({"state": k, "pair": f"{a}~{b}", "involves_smlt": "SMLT" in (a, b), "n": res["n"],
                                "rho_state": r_k, "rho_shrunk": float(res["corr_shrunk"].loc[a, b]), "rho_full": r_g,
                                "delta": d, "fisher_z": z, "p_value": p,
                                "classification": ("MATERIAL_CHANGE" if (abs(d) >= v3["corr_change_material"] and p < 0.05)
                                                   else ("LARGE_BUT_NOT_SIGNIFICANT" if abs(d) >= v3["corr_change_material"] else "STABLE"))})
    ctx.save(pd.DataFrame(sizes), "regime_sample_sizes.csv", index=False)
    ctx.save(pd.DataFrame(changes), "regime_correlation_changes.csv", index=False)
    lab = pd.DataFrame({"macro_regime": reg["regime"], "price_state": ps["price_state"], "portfolio_state": pst}).loc[:t]
    tm = pd.concat([rr.transitions(lab.loc[lab.index >= start, "macro_regime"],
                                   ["NORMAL", "MARKET_STRESS", "CREDIT_STRESS", "RATES_STRESS", "COMBINED_STRESS"]).assign(axis="macro_regime"),
                    rr.transitions(lab.loc[lab.index >= start, "price_state"], rr.PRICE_STATES).assign(axis="smlt_price_state"),
                    rr.transitions(lab.loc[lab.index >= start, "portfolio_state"], rr.PORTFOLIO_STATES).assign(axis="portfolio_state")],
                   ignore_index=True)
    tm["interpretation"] = "DESCRIPTIVE ONLY (empirical daily frequencies 2022-03-28..cutoff); not a forecast"
    ctx.save(tm, "regime_transition_matrix.csv", index=False)
    ctx.save(pd.concat([lab, ps.drop(columns=["price_state"])], axis=1), "regime_states_daily.csv", "interim")
    # price-state threshold sensitivity (frequencies only; thresholds are not tuned)
    sens = []
    for sc in [1.0] + v3["price_state"]["sensitivity_scale"]:
        p2 = rr.price_state(R["SMLT"], v3["price_state"], scale=sc)["price_state"].loc[start:t]
        sens.append({"scale": sc, **{f"freq_{k}": float((p2 == k).mean()) for k in rr.PRICE_STATES},
                     "state_at_cutoff": str(p2.iloc[-1])})
    ctx.save(pd.DataFrame(sens), "price_state_sensitivity.csv", index=False)
    ctx.log["v3_state_at_cutoff"] = {"macro_regime": str(lab["macro_regime"].iloc[-1]), "price_state": str(lab["price_state"].iloc[-1]),
                                     "portfolio_state": str(lab["portfolio_state"].iloc[-1])}
    ctx._regime_mats = mats


# ================================================================== Phase 3: forward scenario path model
def _forward_inputs(ctx):
    from .portfolio import regime_risk as rr
    P, R, X, reg, ps, pst = _regime_inputs(ctx)
    X["MKT"], X["RATES"] = R["IMOEX"], R["RGBI"]
    start, t = pd.Timestamp(ctx.cfg["simulation"]["bootstrap_start"]), pd.Timestamp(ctx.cutoff)
    X = X.loc[(X.index >= start) & (X.index <= t)]
    masks = {k: v.reindex(X.index).fillna(False) for k, v in rr.state_masks(reg, ps).items()}
    return P, R, X, masks


def forward_paths(ctx, cols: list[str], n: int | None = None, H: int = 252, mu_scale: float = 1.0, seed_shift: int = 0) -> dict:
    """Scenario -> {col: (n, H) simple returns} (+ 'SMLT_FUT' structural) using the Phase-3 configuration.
    mu_scale / seed_shift (V4.1 band perturbations) default to the V3/V4 paths exactly."""
    from .scenarios import forward as fw
    from .scenarios.bootstrap import structural_futures
    fcfg = ctx.cfg["v3"]["forward"]
    P, R, X, masks = _forward_inputs(ctx)
    n = n or fcfg["n_paths"]
    base_cols = [c for c in dict.fromkeys(["SMLT"] + [c for c in cols if c != "SMLT_FUT"] + ["MKT", "SECTOR", "RATES", "CREDIT"]) if c in X]
    anc = fw.factor_anchors(X, fcfg["anchors"]["q_low"], fcfg["anchors"]["q_high"])
    lam_n0 = ctx.cfg["v3"]["shrink_n0"]
    sm = np.log1p(X["SMLT"].dropna())
    vol_cut = float(np.sqrt((sm**2).ewm(alpha=1 - ctx.cfg["risk"]["ewma_lambda"], adjust=False).mean().iloc[-1]) * np.sqrt(252))
    vol_med = float((sm.rolling(60).std() * np.sqrt(252)).median())
    out, meta = {}, []
    specs = dict(fcfg["scenarios"])
    for name, sc in specs.items():
        mask = pd.concat([masks[k] for k in sc["regime_set"]], axis=1).any(axis=1)
        betas = fw.regime_betas(X, mask, base_cols, lam_n0)
        sub = sm[mask.reindex(sm.index).fillna(False).values]
        vol_sub = float(sub.std() * np.sqrt(252))
        k = (vol_cut if sc["vol"] == "cutoff_ewma" else vol_med) / vol_sub
        mu = {f: (0.0 if sc["anchor"] == "zero" else mu_scale * float(anc.loc[f, sc["anchor"]])) for f in anc.index}
        meta.append({"scenario": name, "type": "STRUCTURAL", "regime_set": "|".join(sc["regime_set"]), "n_regime_days": int(mask.sum()),
                     "shrink_lambda": betas["SMLT"]["lambda"], "vol_target_ann": vol_cut if sc["vol"] == "cutoff_ewma" else vol_med,
                     "vol_regime_sample_ann": vol_sub, "vol_multiplier": k, "anchor": sc["anchor"],
                     **{f"mu12m_{f}": v for f, v in mu.items()}, **{f"beta_SMLT_{f}": b for f, b in zip(["MKT", "SECTOR", "RATES", "CREDIT"], betas["SMLT"]["beta"])},
                     "smlt_implied_12m_log_drift": float(betas["SMLT"]["beta"] @ np.array([mu[f] for f in ["MKT", "SECTOR", "RATES", "CREDIT"]])),
                     "macro_label_for_policies": sc["macro_label"]})
        specs[name] = {**sc, "mask": mask, "betas": betas, "k": k, "mu": mu}
    seed0 = ctx.cfg["simulation"]["seed"] + fcfg["seed_offset"]
    e_res_all, b_now, e_now = pl._basis_residual(ctx, X.index)
    all_sc = list(specs) + list(fcfg["overlays"])
    for j, name in enumerate(all_sc):
        if name in specs:
            sc, jump = specs[name], None
        else:
            ov = fcfg["overlays"][name]
            sc = specs[ov["base"]]
            jump = {"day": fcfg["event_day"], "series": {"SMLT": float(np.log1p(ov["smlt_jump"])), "CREDIT": float(np.log1p(ov["credit_shock"]))}}
            meta.append({"scenario": name, "type": "EVENT_OVERLAY", "regime_set": "|".join(sc["regime_set"]), "anchor": sc["anchor"],
                         "overlay_base": ov["base"], "event_day": fcfg["event_day"], "smlt_jump": ov["smlt_jump"],
                         "credit_shock": ov["credit_shock"], "macro_label_for_policies": sc["macro_label"]})
        Pth = fw.simulate(X, base_cols, sc["mask"], sc["mu"], sc["k"], sc["betas"], H, n, ctx.cfg["simulation"]["block_length"],
                          seed0 + 101 * j + seed_shift, jump=jump)
        if "SMLT_FUT" in cols:
            e = e_res_all.reindex(Pth["_dates"]).values
            Pth["SMLT_FUT"] = structural_futures(Pth, e, b_now, e_now)
        out[name] = Pth
    return out, pd.DataFrame(meta), anc


def _catalysts(ctx) -> pd.DataFrame:
    """Scheduled catalysts KNOWN at the cutoff (nothing invented)."""
    t = pd.Timestamp(ctx.cutoff)
    rows = []
    b = pd.read_csv(ctx.interim / "smlt_bonds_history.csv", parse_dates=["TRADEDATE"])
    last = b[b["TRADEDATE"] <= t].sort_values("TRADEDATE").groupby("SECID").tail(1)
    last["MATDATE"] = pd.to_datetime(last["MATDATE"], errors="coerce")
    for _, r in last[(last["MATDATE"] > t) & (last["MATDATE"] <= t + pd.Timedelta(days=366))].iterrows():
        rows.append({"date": str(r["MATDATE"].date()), "event": f"Samolet bond maturity {r['SHORTNAME']} ({r['SECID']})",
                     "category": "BOND_MATURITY", "source": "MOEX ISS bonds history (MATDATE, static term)", "status": "CONFIRMED_EXCHANGE_DATA",
                     "last_price_pct_par": r.get("LEGALCLOSEPRICE"), "last_ytm_pct": r.get("YIELDCLOSE")})
    for _, r in last.iterrows():
        od = pd.to_datetime(r.get("OFFERDATE"), errors="coerce")
        if pd.notna(od) and t < od <= t + pd.Timedelta(days=366):
            rows.append({"date": str(od.date()), "event": f"put offer {r['SHORTNAME']}", "category": "BOND_OFFER",
                         "source": "MOEX ISS bonds history (OFFERDATE)", "status": "CONFIRMED_EXCHANGE_DATA"})
    ev = pd.read_csv(ctx.root / "data" / "manual" / "event_chronology.csv")
    for _, r in ev.iterrows():
        if "next meeting" in str(r["event"]):
            d = str(r["event"]).split("next meeting")[-1].strip().strip(";. ")
            rows.append({"date": d, "event": "Bank of Russia key-rate meeting", "category": "CBR_MEETING",
                         "source": f"{r['publisher']} ({r['url']}), published {r['event_date']}",
                         "status": f"MEDIA_REPORTED; primary {r['verification_status']}"})
    rows.append({"date": "", "event": "Samolet IFRS 9M/FY2026 reporting dates", "category": "ISSUER_REPORTING",
                 "source": "not published in any source available to the research environment", "status": "DATA_NOT_AVAILABLE (not invented)"})
    df = pd.DataFrame(rows)
    d = pd.to_datetime(df["date"], errors="coerce")
    df["approx_trading_day"] = [int(np.busday_count(t.date(), x.date())) if pd.notna(x) else np.nan for x in d]
    return df.sort_values("date").reset_index(drop=True)


def s19_forward(ctx) -> None:
    from .scenarios import forward as fw
    fcfg = ctx.cfg["v3"]["forward"]
    S0 = pl._fut_ctx(ctx)[0]
    cat = _catalysts(ctx)
    ctx.save(cat, "forward_catalysts.csv", index=False)
    paths_, meta, anc = forward_paths(ctx, ["SMLT", "SMLT_FUT"])
    nodes = dict(fcfg["nodes"])
    for _, r in cat.dropna(subset=["approx_trading_day"]).iterrows():
        if 0 < r["approx_trading_day"] <= 252:
            nodes[f"{r['date']} {r['category']}"] = int(r["approx_trading_day"])
    q, fp = [], []
    t = pd.Timestamp(ctx.cutoff)
    for name, Pth in paths_.items():
        qt = fw.quantile_table(S0, Pth["SMLT"], nodes)
        qt.insert(0, "scenario", name)
        qt["approx_calendar_date"] = [str((t + pd.offsets.BDay(int(d))).date()) for d in qt["trading_day"]]
        q.append(qt)
        for f in ("MKT", "SECTOR", "RATES", "CREDIT"):
            lvl = np.cumsum(np.log1p(Pth[f]), axis=1)
            for nn, d in fcfg["nodes"].items():
                x = lvl[:, d - 1]
                fp.append({"scenario": name, "factor": f, "node": nn, "trading_day": d,
                           "central_log_path": float(meta.set_index("scenario").get(f"mu12m_{f}", pd.Series()).get(name, np.nan)) * d / 252
                           if name in fcfg["scenarios"] else np.nan,
                           "P05_cum_log": float(np.quantile(x, 0.05)), "P50_cum_log": float(np.quantile(x, 0.5)),
                           "P95_cum_log": float(np.quantile(x, 0.95))})
    ctx.save(pd.concat(q, ignore_index=True), "forward_smlt_path_quantiles.csv", index=False)
    fpd = pd.DataFrame(fp)
    ctx.save(fpd, "forward_scenario_factor_paths.csv", index=False)
    anc = anc.reset_index().rename(columns={"index": "factor"})
    anc["source"] = "MOEX ISS history (IMOEX, PIKK/LSRG/ETLN, RGBI, Samolet bonds) 2022-03-28..cutoff; overlapping 252d log returns"
    ctx.save(anc, "forward_factor_anchors.csv", index=False)
    meta["label"] = "SCENARIO (not a forecast)"
    ctx.save(meta, "scenario_regime_mapping.csv", index=False)
    bands = []
    for k, v in paths_.items():
        lvl = S0 * np.cumprod(1 + v["SMLT"], axis=1)
        qs = np.quantile(lvl, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
        bands.append(pd.DataFrame({"scenario": k, "trading_day": np.arange(1, lvl.shape[1] + 1), "P05": qs[0], "P25": qs[1],
                                   "P50": qs[2], "P75": qs[3], "P95": qs[4]}))
    ctx.save(pd.concat(bands, ignore_index=True), "forward_fan_bands.csv", "interim", index=False)


def collateral_repair_decomposition(ctx) -> pd.DataFrame:
    """Why V2's 'selling does not dominate futures under productive collateral' changed: re-run the regime-C
    comparison with the unified ledger (a) as implemented (a free-liquidity shortfall is financed at the money-market
    rate) and (b) with the V2 treatment (shortfall financed at zero cost). Diagnostic output only."""
    from .scenarios.bootstrap import SleevePos, evaluate_portfolio
    from .strategies import ledger
    from .strategies.futures import contract_position
    c = ctx.cfg["collateral"]["named"]["C_HIGH_COLLATERAL_EFFICIENCY"]
    coll = Collateral("C_HIGH_COLLATERAL_EFFICIENCY", c["yield_fraction"], c["haircut"], c["margin_multiplier"])
    B = pl._bootstrap_inputs(ctx, ["SMLT", "SMLT_FUT"])
    S0, F0, pv = pl._fut_ctx(ctx)
    n_sh = ctx.E / S0
    per = pl._costs(ctx).per_contract() / S0
    dcost = (0.1 + ctx.cfg["costs"]["equity_commission_bps"] * 1e-4 * S0) / S0
    rows = []
    orig = ledger.interest
    try:
        for hz_name, hz in HZ.items():
            P = pl._paths_for(ctx, B, "FHS_FULL", hz, n_paths=10000)
            rolls = int(np.ceil(hz / 252 / 0.25))
            for variant in ("V3_shortfall_financed_at_mmf", "V2_shortfall_free"):
                ledger.interest = orig if variant.startswith("V3") else (
                    lambda f, im, m, cl=coll: np.maximum(f, 0.0) * m + im * cl.earning_per_margin() * m)
                for x in (0.25, 0.5, 0.75, 1.0):
                    N0 = contract_position(x, n_sh, F0, pv)["notional_rub"]
                    f = evaluate_portfolio(P, E=ctx.E, C=ctx.C, sleeves=[SleevePos("SMLT_FUT", -N0, B["mr"], per, rolls)],
                                           mmf_daily=B["daily_y"], collateral=coll)
                    s = evaluate_portfolio(P, E=ctx.E, C=ctx.C, mmf_daily=B["daily_y"], derisk=x, derisk_cost=dcost, collateral=coll)
                    rows.append({"horizon": hz_name, "ledger_variant": variant, "fraction": x,
                                 "ES95_sell_minus_fut": s["ES95"] - f["ES95"], "mean_sell_minus_fut": s["mean_pnl"] - f["mean_pnl"],
                                 "breach_fut": f["p_sleeve_breach"],
                                 "sell_dominates": bool(s["ES95"] <= f["ES95"] and s["mean_pnl"] >= f["mean_pnl"])})
    finally:
        ledger.interest = orig
    return pd.DataFrame(rows)
