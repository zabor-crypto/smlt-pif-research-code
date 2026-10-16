"""Gate R1 outputs: accounting identities, liquidity bridge, margin audit, integer-execution audit and the
re-admission of the frozen V4.1 candidates by the new single authority.

Two demonstrations, neither of which is a portfolio recommendation:
  A. HISTORICAL identity ledgers on exchange data (2023-01-03 .. cutoff): CONTROL (hold), FUT50 (short SMLT futures
     = 50% of the shares, integer, rolled on the contract calendar), SELL50_TO_MMF. Every term as of the date.
  B. RE-ADMISSION of the V4.1 candidate books on the V4.1 independent-seed scenario paths (legacy generator, read
     only), with the new conventions: next-session execution, T+1 SMLT proceeds, LQDT units not margin liquidity,
     5% liquidity minimum, decision-time tail waterfall, whole contracts, margin stress 1.25 / 1.50.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from . import admission as A
from . import engine as E
from . import fx
from . import market as M
from . import orders as O

ROUTES_AUDIT = ["SMLT_FUT", "MIX_FUT", "MXI_FUT", "RGBI_FUT", "BR_FUT", "BTC_FUT", "GLDRUBF", "CNYRUBF"]


def load_cfg(ctx) -> dict:
    return yaml.safe_load((ctx.root / "configs" / "adaptive.yaml").read_text())


# ------------------------------------------------------------------------------------------ A. history
def _smlt_specs(prices: pd.DataFrame, t0: pd.Timestamp, cfg: dict) -> list[M.FundedSpec]:
    return [M.FundedSpec("SMLT", E.FUNDED, float(prices.loc[t0, "SMLT"]), lag=1, lot=1.0, cost_prop=5e-4, cost_unit=0.1,
                         issuer="SMLT", factor="ISSUER_EQUITY"),
            M.FundedSpec("LQDT", E.MMF, float(prices.loc[t0, "LQDT"]), lag=1, lot=1.0)]


def reserve_min(beta: float, vm99: float) -> float:
    """VM-adjusted minimum (fraction of NAV) = beta + VM99; the operating band is [min + h/4 floor, min + h/2 target,
    min + h ceiling] with h = sweep_hysteresis_nav."""
    return beta + vm99


def loss_quantile(r: np.ndarray, level: float) -> float:
    """1-session loss quantile of a unit long position (haircut on sale proceeds between decision and execution)."""
    r = np.asarray(r, float).ravel()
    r = r[np.isfinite(r)]
    return float(max(np.quantile(-r, level), 0.0)) if len(r) else 0.0


def vm99_frac(ret_hist: pd.Series, notional: float, nav: float, level: float) -> float:
    """1-day VM loss quantile of a derivative position (RUB notional, signed) as a fraction of NAV (history <= t)."""
    r = ret_hist.dropna().values
    if not len(r) or notional == 0:
        return 0.0
    loss = -notional * r
    return float(max(np.quantile(loss, level), 0.0) / nav)


def historical_ledgers(ctx, cfg: dict) -> dict:
    prices = pd.read_csv(ctx.proc / "prices.csv", index_col=0, parse_dates=True)
    rets = pd.read_csv(ctx.proc / "returns.csv", index_col=0, parse_dates=True)
    start, end = pd.Timestamp("2023-01-03"), pd.Timestamp(cfg["cutoff"])
    dates = prices.loc[start:end].index
    usd = fx.usd_known(fx.cbr_usd_records(ctx.root), dates)
    fspecs = _smlt_specs(prices, start, cfg)
    fin, din, mkt, terms = M.historical_market(ctx, cfg, fspecs, ["SMLT_FUT"], prices, dates, usd)
    beta = cfg["owner"]["liquidity_min_nav"]
    nav0 = 20_000_000.0
    S0 = float(prices.loc[start, "SMLT"])
    shares = np.floor(10_000_000.0 / S0)
    book0 = E.Book(qf=np.array([shares, 0.0]), qd=np.array([0.0]), cash=nav0 - shares * S0)
    hist_before = rets.loc[:start - pd.Timedelta(days=1), "SMLT_FUT"]
    out = {}
    for name, tgt_shares, tgt_contracts in (("CONTROL_HOLD", shares, 0.0),
                                             ("FUT50_INTEGER", shares, -np.trunc(0.5 * shares)),
                                             ("SELL50_TO_MMF", np.floor(shares / 2), 0.0)):
        F0 = float(mkt.Fn[0, 0, 0]) * float(mkt.pv[0, 0, 0])
        lv = cfg["r1"]["liquidity"]
        res_frac = reserve_min(beta, vm99_frac(hist_before, tgt_contracts * F0, nav0, lv["reserve_var_level"]))
        hc = np.array([loss_quantile(rets.loc[:start - pd.Timedelta(days=1), "SMLT"].values, lv["reserve_var_level"]), 0.0])
        ords, _ = O.build(1, book0.qf.astype(float), book0.qd.astype(float), np.array([tgt_shares, np.nan]),
                          np.array([tgt_contracts]), mkt.P[0, 0, :], np.array([F0]), mkt.mr[0], 1.0, fin,
                          available=float(book0.cash), nav=nav0, target_frac=res_frac + lv["sweep_hysteresis_nav"] / 2,
                          haircut=hc, lag_cycle=int(mkt.lag[1, 0]))
        eng = E.Engine(fin, din, mkt, E.Params(beta=beta, mult=1.0, reserve_frac=res_frac,
                                               hysteresis=lv["sweep_hysteresis_nav"], log=True))
        res = eng.run(book0, orders=ords)
        log = pd.DataFrame(res["log"])
        log.index = dates
        out[name] = {"res": res, "log": log, "reserve_frac": res_frac}
    return {"ledgers": out, "terms": terms, "dates": dates, "usd": usd}


def identity_table(ledgers: dict) -> pd.DataFrame:
    rows = []
    for name, d in ledgers.items():
        lg = d["log"]
        vcols = [c for c in lg if c.startswith("v_")]
        nav_resid = (lg["nav"] - (lg[vcols].sum(axis=1) + lg["cash"] + lg["pending"])).abs()
        pnl_resid = ((lg["nav"] - lg["nav"].iloc[0]) - (lg["price_pnl_cum"] + lg["vm_cum"] + lg["interest_cum"]
                                                        - lg["cost_total_cum"])).abs()
        qcols = [c for c in lg if c.startswith("q_")]
        dcol = [c for c in qcols if c.endswith("_FUT")]
        frac = int(sum((lg[c] - lg[c].round()).abs().gt(1e-9).sum() for c in dcol))
        rows.append({"ledger": name, "days": len(lg), "nav0": lg["nav"].iloc[0], "nav_end": lg["nav"].iloc[-1],
                     "max_abs_nav_identity_residual_rub": float(nav_resid.max()),
                     "max_abs_pnl_identity_residual_rub": float(pnl_resid.max()),
                     "fractional_contract_days": frac, "roll_events": int((lg["roll_cost"] > 0).sum()),
                     "breach_days": int(lg["breach"].sum()), "unresolved_days": int(lg["unresolved"].sum()),
                     "min_available_nav": float((lg["available"] / lg["nav"]).min()),
                     "max_pending_nav": float((lg["pending"].abs() / lg["nav"]).max()),
                     "total_costs_rub": float(lg["cost_total_cum"].iloc[-1]),
                     "vm_cum_rub": float(lg["vm_cum"].iloc[-1]), "price_pnl_cum_rub": float(lg["price_pnl_cum"].iloc[-1]),
                     "identity": "NAV = sum(q*P) + cash + pending; NAV_t - NAV_0 = price P&L + VM + interest - costs",
                     "status": "PASS" if nav_resid.max() < 1e-4 and pnl_resid.max() < 1e-3 and frac == 0 else "FAIL"})
    return pd.DataFrame(rows)


def liquidity_bridge(ledgers: dict) -> pd.DataFrame:
    frames = []
    for name, d in ledgers.items():
        lg = d["log"]
        vq = [c for c in lg if c.startswith("v_") and c != "v_LQDT"]
        f = pd.DataFrame({"ledger": name, "nav": lg["nav"], "immediate_cash": lg["cash"], "mmf_units_value": lg.get("v_LQDT", 0.0),
                          "pending_settlement_net": lg["pending"], "posted_initial_margin": lg["im"],
                          "funded_risky_value": lg[vq].sum(axis=1), "available_immediate": lg["available"],
                          "available_nav": lg["available"] / lg["nav"], "breach": lg["breach"],
                          "reserve_target_nav": d["reserve_frac"]})
        f.index.name = "date"
        frames.append(f.reset_index())
    return pd.concat(frames, ignore_index=True)


def margin_audit(ctx, cfg: dict, dates: pd.DatetimeIndex, usd: pd.Series) -> pd.DataFrame:
    broker = float(ctx.cfg["costs"]["broker_fee_futures_rub_per_contract"])
    rows = []
    for r in ROUTES_AUDIT:
        h = M.deriv_history(ctx, r, dates, usd, broker)
        h = h[h["Fn"].notna()]
        if h.empty:
            rows.append({"route": r, "status": "NO_HISTORY_IN_WINDOW"})
            continue
        ratio_n = h["notional"] / h["notional_cutoff"]
        rows.append({
            "route": r, "first_date": str(h.index.min().date()), "days": len(h),
            "mr1_asof_min": float(h["mr1"].min()), "mr1_asof_max": float(h["mr1"].max()), "mr1_cutoff": float(h["mr1_cutoff"].iloc[0]),
            "share_days_asof_mr1_above_cutoff": float((h["mr1"] > h["mr1_cutoff"] + 1e-12).mean()),
            "notional_asof_min_rub": float(h["notional"].min()), "notional_asof_max_rub": float(h["notional"].max()),
            "notional_cutoff_rub": float(h["notional_cutoff"].iloc[0]),
            "max_ratio_asof_to_cutoff_notional": float(ratio_n.max()),
            "fee_asof_median_rub": float(h["fee"].median()), "fee_cutoff_rub": float(h["fee_cutoff_rub"].iloc[0]),
            "fee_pct_notional_asof_median": float((h["fee"] / h["notional"]).median()),
            "fee_pct_notional_if_cutoff_rub_reused_max": float((h["fee_cutoff_rub"] / h["notional"]).max()),
            "pv_source_counts": "; ".join(f"{k}:{v}" for k, v in h["pv_source"].value_counts().items()),
            "pv_asof_min": float(h["pv"].min()), "pv_asof_max": float(h["pv"].max()), "pv_frozen_cutoff": float(h["pv_frozen_cutoff"].iloc[0]),
            "roll_events": int(h["is_roll"].sum()), "distinct_contracts": int(h["held"].nunique()),
            "mr2_mr3_applied": False, "status": "ASOF_TERMS_USED_IN_REPLAY"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ B. V4.1 re-admission
V41_SERIES = {"SMLT": "SMLT", "BTC_SPOT_EXP": "BTC_SPOT_RUB", "SMLT_FUT": "SMLT_FUT", "MIX": "MIX_FUT", "RGBI": "RGBI_FUT",
              "GOLD": "GLDRUB_TOM", "CNY": "CNYRUB_TOM", "OIL": "BR_FUT", "BTC_FUT": "BTC_PROXY"}


def _v41_paths(ctx, assets):
    from ..pif import final
    sc = ctx.cfg["v41"]["scenario"]
    base = final.path_set(ctx, assets, keep_paths=True)
    alt = final.path_set(ctx, assets, seed_shift=sc["seed_shift"], keep_paths=True)
    return base, alt


def _roll_days(rolls_per_year: int, H: int) -> tuple:
    n = int(np.ceil(rolls_per_year * H / 252 - 1e-9))
    return tuple(int(round(H * (k + 1) / (n + 1))) for k in range(n))     # engine day index (1..H), V4 schedule


def v41_models(ctx, cfg, assets, S, P0_smlt, P0_mmf):
    """Map a V4.1 scenario path set to the engine (V4.1 economics kept: frozen contract terms, spot-series GOLD/CNY)."""
    P = S["P"]
    N, H = P["SMLT"].shape
    fs, ds, Rf, Rd = [], [], [], []
    for a in assets:
        if a.kind == "FUNDED":
            if a.name == "SMLT":
                fs.append(M.FundedSpec("SMLT", E.FUNDED, P0_smlt, lag=1, lot=1.0, cost_prop=5e-4, cost_unit=0.1, issuer="SMLT"))
            else:
                fs.append(M.FundedSpec(a.name, E.FUNDED, 1.0, lag=cfg["r1"]["settlement"]["btc_experimental_lag"], lot=0.0,
                                       cost_prop=a.cost_one_way))
            Rf.append(np.nan_to_num(P[a.series]))
        else:
            ds.append(M.DerivSpec(a.name, F0=a.contract_notional, pv0=1.0, usd_linked=False, mr=a.mr, fee_rate=a.cost_one_way,
                                  fee_fixed_rub=0.0, roll_days=_roll_days(a.rolls_per_year, H) if a.rolls_per_year else ()))
            Rd.append(np.nan_to_num(P[a.series]))
    fs.append(M.FundedSpec("LQDT", E.MMF, P0_mmf, lag=1, lot=1.0))            # forward: T+1 cycle (post-2023)
    Rf.append(np.zeros((N, H)))
    fin, din, mkt = M.forward_market(fs, ds, np.stack(Rf, axis=2), np.stack(Rd, axis=2), None, S["md"])
    return fin, din, mkt


def v41_readmission(ctx, cfg) -> pd.DataFrame:
    from ..pif import final
    E_, C_ = final._EC(ctx)
    NAV0 = E_ + C_
    assets = final.scenario_assets(ctx, E_, C_)
    base, alt = _v41_paths(ctx, assets)
    prices = pd.read_csv(ctx.proc / "prices.csv", index_col=0, parse_dates=True)
    S0 = float(prices["SMLT"].loc[ctx.cutoff])
    mmf0 = float(prices["LQDT"].loc[ctx.cutoff])
    fr = pd.read_csv(ctx.out / "pif_scenario_policy_frontiers.csv")
    cand = fr[fr["row_type"] == "CANDIDATE"]
    names = [a.name for a in assets]
    keep = ["TARGET_CONTINUED_DETERIORATION", "TARGET_STABILISATION_FLAT", "TARGET_RATES_RELIEF_RECOVERY",
            "TARGET_EVENT_DRIVEN_UPSIDE", "TARGET_ADVERSE_CREDIT_EVENT", "STATIC_PIF", "CONTROL_INITIAL_PORTFOLIO",
            "ZERO_SMLT_CASH"]
    state = __import__("json").loads((ctx.interim / "pif_state_at_cutoff.json").read_text())
    cur_sc = state.get("conditional_scenario", "CONTINUED_DETERIORATION")
    rule_main = A.Rule(budget=cfg["r1"]["admission"]["es_budget_nav0"], beta=cfg["owner"]["liquidity_min_nav"],
                       max_breach=cfg["r1"]["admission"]["max_breach_probability"],
                       stress_mults=tuple(cfg["r1"]["admission"]["margin_stress_survival"]),
                       hysteresis=cfg["r1"]["liquidity"]["sweep_hysteresis_nav"])
    shares0 = np.floor(E_ / S0)
    rows = []
    for cname in keep:
        for sc in final.SCENARIOS:
            own = cname.replace("TARGET_", "")
            if cname.startswith("TARGET_") and sc != own:
                continue
            if cname in ("STATIC_PIF", "CONTROL_INITIAL_PORTFOLIO", "ZERO_SMLT_CASH") and sc not in (cur_sc, "ADVERSE_CREDIT_EVENT", "STABILISATION_FLAT"):
                continue
            rr = cand[(cand["candidate"] == cname) & (cand["scenario"] == sc)]
            if rr.empty:
                continue
            x = np.array([float(rr[f"x_{n}"].iloc[0]) for n in names])
            fin, din, mkt = v41_models(ctx, cfg, assets, alt[sc], S0, mmf0)
            gfin, gdin, gmkt = v41_models(ctx, cfg, assets, base[sc], S0, mmf0)
            fnames = [i.name for i in fin]
            dnames = [i.name for i in din]
            wf = {n: x[names.index(n)] for n in fnames if n in names}
            wd = np.array([x[names.index(n)] for n in dnames])
            notional = np.array([float(mkt.Fn[0, 0, j] * mkt.pv[0, 0, j]) for j in range(len(din))])
            Pf0 = mkt.P[0, 0, :]
            lots = np.array([i.lot for i in fin])
            wvec = np.array([wf.get(n, 0.0) for n in fnames])
            qf, qd = E.to_quantities(wvec, wd, NAV0, Pf0, notional, lots)
            # decision-time waterfall ranking and reserve from the GENERATION (base-seed) paths
            G_d = (gmkt.Fn[:, -1, :] - gmkt.Fn[:, 0, :]) * gmkt.pv[:, 0, :]
            G_f = gmkt.P[:, -1, :] - gmkt.P[:, 0, :]
            rel_d = cfg["r1"]["margin"]["stress_multipliers"][0] * notional * gmkt.mr[0] - gmkt.fee[0, 0, :]
            rel_f = Pf0
            wf_rank, rs_rank, _, _ = A.tail_rank(G_d, G_f, qd, qf, rel_d, rel_f, gmkt.fee[0, 0, :], np.array([i.cost_prop for i in fin]))
            rs_rank = tuple(j for j in rs_rank if fin[j].kind != E.MMF)
            dret = (gmkt.Fn[:, 1:, :] / gmkt.Fn[:, :-1, :] - 1) if din else np.zeros((1, 1, 0))
            vm_loss = -(dret * (qd * notional)[None, None, :]).sum(axis=2).ravel()
            lv = cfg["r1"]["liquidity"]
            reserve = reserve_min(rule_main.beta, max(float(np.quantile(vm_loss, lv["reserve_var_level"])), 0.0) / NAV0)
            fret = gmkt.P[:, 1:, :] / gmkt.P[:, :-1, :] - 1
            hc = np.array([loss_quantile(fret[:, :, j], lv["reserve_var_level"]) for j in range(len(fin))])
            jm = fnames.index("LQDT")
            qf[jm] = np.nan                                     # MMF: managed by the engine's cash rule
            book0 = E.Book(qf=np.array([shares0 if n == "SMLT" else 0.0 for n in fnames]), qd=np.zeros(len(din)),
                           cash=NAV0 - shares0 * S0)
            # hard exposure / mandate checks on the EXECUTABLE book (V4 bounds and constraints)
            lo = {a.name: a.lo for a in assets}
            hi = {a.name: a.hi for a in assets}
            wexe = {n: (0.0 if np.isnan(qf[j]) else qf[j]) * Pf0[j] / NAV0 for j, n in enumerate(fnames)}
            wexe.update({n: qd[j] * notional[j] / NAV0 for j, n in enumerate(dnames)})
            funded_risky = sum(v for n, v in wexe.items() if n not in ("LQDT",) and n in fnames)
            long_exp = funded_risky + sum(max(wexe[n], 0) for n in dnames)
            gross_d = sum(abs(wexe[n]) for n in dnames)
            hard = {"asset_bounds": all(lo[n] - 1e-12 <= wexe[n] <= hi[n] + 1e-12 for n in names if n in wexe),
                    "long_exposure<=1": long_exp <= 1.0 + 1e-12, "derivative_gross<=1": gross_d <= 1.0 + 1e-12}
            orders, oinfo = O.build(cfg["owner"]["execution_delay_sessions"], book0.qf, book0.qd, qf, qd, Pf0, notional,
                                    mkt.mr[0], 1.0, fin, available=float(book0.cash), nav=NAV0,
                                    target_frac=reserve + lv["sweep_hysteresis_nav"] / 2, haircut=hc, lag_cycle=1)
            mp = A.ModelPaths(f"V41_{sc}_INDEPENDENT_SEED", fin, din, mkt)
            adm = A.admit([mp], book0, orders, rule_main, reserve, wf_rank, rs_rank, hard_checks=hard)
            legacy = rr.iloc[0]
            for row in adm.rows:
                rows.append({"candidate": cname, "evaluated_scenario": sc, "is_own_scenario": cname.endswith(sc) or not cname.startswith("TARGET_"),
                             **{f"w_exec_{n}": round(v, 6) for n, v in wexe.items()},
                             **{f"contracts_{n}": int(qd[j]) for j, n in enumerate(dnames)},
                             "shares_SMLT": int(qf[fnames.index("SMLT")]), "reserve_min_nav": reserve,
                             "reserve_target_nav": reserve + lv["sweep_hysteresis_nav"] / 2, "two_step_order": oinfo["two_step"],
                             "waterfall_order": ">".join(dnames[j] for j in wf_rank),
                             **row, "legacy_fund_ES95_buffer0": float(legacy["fund_ES95"]),
                             "legacy_fund_breach_buffer0": float(legacy["fund_p_liquidity_breach"]),
                             "hard_checks": "; ".join(f"{k}={v}" for k, v in hard.items()),
                             "ADMISSION": "PASS" if adm.passed else "FAIL", "fail_reasons": " | ".join(adm.reasons),
                             "authority": "adaptive full engine (R1); LP not consulted"})
    return pd.DataFrame(rows)


def integer_audit(adm: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in identity.iterrows():
        rows.append({"object": f"history:{r['ledger']}", "operations": "entry, rolls, restoration, sweeps",
                     "fractional_contract_days": int(r["fractional_contract_days"]), "status": "PASS" if r["fractional_contract_days"] == 0 else "FAIL"})
    g = adm.groupby(["candidate", "evaluated_scenario"])
    for (c, s), d in g:
        rows.append({"object": f"forward:{c}@{s}", "operations": "entry at next session, rolls, forced reductions, restoration",
                     "fractional_contract_days": int(d["frac_contract_path_days"].sum()), "paths_x_models": len(d),
                     "mean_forced_contracts_m1": float(d.loc[d["margin_multiplier"] == 1.0, "mean_forced_contracts"].iloc[0]),
                     "status": "PASS" if d["frac_contract_path_days"].sum() == 0 else "FAIL"})
    return pd.DataFrame(rows)
