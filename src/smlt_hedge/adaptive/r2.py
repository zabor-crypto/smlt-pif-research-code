"""Gate R2 outputs: asset universe with fair-test status, route reconciliation and the matched SMLT route parity.

The parity test runs two EXACT engine ledgers per window on exchange data with the same starting capital:
  A = SMLT shares (n shares) + the rest of the capital in the liquidity sleeve
  B = n SMLT futures contracts (1 contract = 1 share; held contract, rolls, as-of mr1 / fee) + all capital in the
      liquidity sleeve (IM posted from immediate cash)
and decomposes NAV_B - NAV_A with the ledgers' own attribution (identity, zero residual). Route differences are
labelled ROUTE_DIFFERENCE_NOT_ALPHA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import engine as E
from . import market as M
from . import orders as O
from . import routes as RT
from .r1 import loss_quantile, reserve_min, vm99_frac


def parity_engine(ctx, cfg: dict, P: dict, H: int = 126, start: str = "2023-01-03") -> pd.DataFrame:
    prices = pd.read_csv(ctx.proc / "prices.csv", index_col=0, parse_dates=True)
    rets = pd.read_csv(ctx.proc / "returns.csv", index_col=0, parse_dates=True)
    cal = P["cal"]
    idx = cal[cal >= pd.Timestamp(start)]
    windows = [(idx[k], idx[min(k + H, len(idx) - 1)]) for k in range(0, len(idx) - H, H)]
    windows.append((idx[0], idx[-1]))
    lv = cfg["r1"]["liquidity"]
    beta = cfg["owner"]["liquidity_min_nav"]
    rows = []
    for w0, w1 in windows:
        dates = cal[(cal >= w0) & (cal <= w1)]
        fs = [M.FundedSpec("SMLT", E.FUNDED, float(prices.loc[w0, "SMLT"]), lag=1, lot=1.0, cost_prop=5e-4, cost_unit=0.1),
              M.FundedSpec("LQDT", E.MMF, float(prices.loc[w0, "LQDT"]), lag=1, lot=1.0)]
        fin, din, mkt, terms = M.historical_market(ctx, cfg, fs, ["SMLT_FUT"], prices, dates, P["usd"])
        nav0 = 20_000_000.0
        S0 = float(mkt.P[0, 0, 0])
        n = np.floor(10_000_000.0 / S0)
        F0 = float(mkt.Fn[0, 0, 0] * mkt.pv[0, 0, 0])
        hist = rets.loc[:w0 - pd.Timedelta(days=1)]
        out = {}
        for leg, tf, td in (("A_FUNDED", np.array([n, np.nan]), np.array([0.0])),
                            ("B_FUTURES_PLUS_COLLATERAL", np.array([0.0, np.nan]), np.array([n]))):
            res_min = reserve_min(beta, vm99_frac(hist["SMLT_FUT"], td[0] * F0, nav0, lv["reserve_var_level"]))
            hc = np.array([loss_quantile(hist["SMLT"].values, lv["reserve_var_level"]), 0.0])
            ords, _ = O.build(0, np.zeros(2), np.zeros(1), tf, td, mkt.P[0, 0, :], np.array([F0]), mkt.mr[0], 1.0, fin,
                              available=nav0, nav=nav0, target_frac=res_min + lv["sweep_hysteresis_nav"] / 2, haircut=hc,
                              lag_cycle=int(mkt.lag[0, 0]))
            eng = E.Engine(fin, din, mkt, E.Params(beta=beta, reserve_frac=res_min, hysteresis=lv["sweep_hysteresis_nav"]))
            r = eng.run(E.Book(qf=np.zeros(2), qd=np.zeros(1), cash=nav0), orders=ords)
            out[leg] = r
        A, B = out["A_FUNDED"], out["B_FUTURES_PLUS_COLLATERAL"]
        dA = float(A["nav"][0, -1] - nav0)
        dB = float(B["nav"][0, -1] - nav0)
        comp = {"underlying_price_pnl_A": float(A["pnl_funded"][0, 0]), "futures_vm_B": float(B["pnl_deriv"][0, 0]),
                "mmf_carry_A": float(A["pnl_funded"][0, 1]), "mmf_carry_B": float(B["pnl_funded"][0, 1]),
                "costs_A": float(A["costs"][0]), "costs_B": float(B["costs"][0]), "interest_A": float(A["interest"][0]),
                "interest_B": float(B["interest"][0])}
        diff = dB - dA
        basis = comp["futures_vm_B"] - comp["underlying_price_pnl_A"]
        coll = comp["mmf_carry_B"] - comp["mmf_carry_A"]
        full_coll = float(10_000_000.0 * (mkt.P[0, -1, 1] / mkt.P[0, 0, 1] - 1))
        rows.append({"window_start": str(dates[0].date()), "window_end": str(dates[-1].date()), "sessions": len(dates),
                     "full_period": bool(w0 == idx[0] and w1 == idx[-1]), "shares_or_contracts": int(n),
                     "pnl_A_funded": dA, "pnl_B_futures_collateral": dB, "B_minus_A": diff,
                     "component_futures_vs_spot_basis_and_embedded_financing": basis,
                     "component_extra_collateral_yield_net_of_im_and_reserve": coll,
                     "memo_full_collateral_yield_on_10m": full_coll,
                     "memo_im_and_reserve_drag": full_coll - coll,
                     "component_cost_difference": comp["costs_A"] - comp["costs_B"],
                     "component_interest_difference": comp["interest_B"] - comp["interest_A"],
                     "identity_residual": diff - (basis + coll + comp["costs_A"] - comp["costs_B"] + comp["interest_B"] - comp["interest_A"]),
                     "B_minus_A_pct_of_exposure": diff / (n * S0), "rolls_B": int(B["roll_contracts"][0] / max(n, 1)),
                     "breach_A": bool(A["breach"][0]), "breach_B": bool(B["breach"][0]),
                     "unresolved_A": bool(A["unresolved"][0]), "unresolved_B": bool(B["unresolved"][0]),
                     "label": "ROUTE_DIFFERENCE_NOT_ALPHA", **{f"memo_{k}": v for k, v in comp.items()}})
    return pd.DataFrame(rows)


def coverage(R: pd.Series, start: str = "2023-01-03") -> float:
    x = R.loc[start:]
    return float(x.notna().mean()) if len(x) else 0.0


def universe(cfg: dict, P: dict, recon: pd.DataFrame) -> pd.DataFrame:
    """One row per route: economics, mechanics and the pre-declared fair-test / admissibility classification."""
    rr = cfg["r2"]
    cov = {k: coverage(P["R"][k]) for k in P["R"].columns}
    rows = []
    for r in rr["routes"]:
        name = r["route"]
        c = cov.get(name, 0.0)
        mech = r["mechanics"]
        fair_rule = rr["fair_test"]
        full = (c >= fair_rule["min_replay_coverage"]) and all(mech.get(k, False) for k in fair_rule["required_mechanics"])
        status = r.get("status_override") or ("FAIR_TEST" if full and r.get("executed_series", False)
                                               else "PARTIAL_TEST" if c > 0 else "NOT_TESTED")
        adm = r.get("admissibility") or ("ADMISSIBLE" if status in ("FAIR_TEST", "PARTIAL_TEST") else "EVIDENCE_MISSING")
        x = P["R"][name].loc["2023-01-03":] if name in P["R"] else pd.Series(dtype=float)
        mm = P["R"]["MMF"].reindex(x.index)
        exc = (x - (mm if r["kind"] in ("FUNDED", "MMF") else 0.0)).dropna()
        rows.append({"route": name, "exposure": r["exposure"], "kind": r["kind"], "direction": r["direction"],
                     "role": r["role"], "instrument": r["instrument"], "benchmark": r["benchmark"],
                     "availability_from": r["availability_from"], "settlement": r["settlement"],
                     "carry_components": r["carry"], "financing": r["financing"], "margin": r["margin"],
                     "fx": r["fx"], "transaction_cost": r["cost"], "replay_coverage": c,
                     "ann_total_return_2023_cutoff": RT._ann(x) if len(x.dropna()) else np.nan,
                     "ann_excess_vs_mmf_2023_cutoff": float(exc.mean() * 252) if len(exc) else np.nan,
                     "ann_vol_2023_cutoff": float(x.std(ddof=1) * np.sqrt(252)) if len(x.dropna()) > 2 else np.nan,
                     "sample_mean_is_NOT_an_expected_return": True,
                     "evidence_quality": r["evidence"], "FAIR_STATUS": status, "ADMISSIBILITY": adm,
                     "status_reason": r.get("status_reason", ""), "notes": r.get("notes", "")})
    return pd.DataFrame(rows)


def route_rule_table(ctx, cfg: dict, P: dict) -> pd.DataFrame:
    """Re-evaluates the pre-declared selection rules on the archived data and asserts the configured routes match."""
    px = pd.read_csv(ctx.proc / "prices.csv", index_col=0, parse_dates=True)
    val = pd.read_csv(ctx.proc / "value_rub.csv", index_col=0, parse_dates=True)
    start, cutoff = pd.Timestamp("2023-01-03"), pd.Timestamp(cfg["cutoff"])
    rows = []

    def fund_row(exposure, secid, first, v):
        v = v.loc[start:cutoff]
        rows.append({"exposure": exposure, "candidate": secid, "first_trade_in_data": str(first.date()) if first is not None else "",
                     "covers_replay_start": bool(first is not None and first <= start),
                     "zero_value_sessions_in_replay": int((v.fillna(0) <= 0).sum()),
                     "median_daily_value_rub": float(v.median()), "rule": "coverage -> continuity -> max median value"})
    for s in ("SBMX", "TMOS", "EQMX"):
        f = RT.fund_prices(ctx.root, s, cfg["cutoff"])
        fund_row("BROAD_RUSSIAN_EQUITY", s, f.index.min(), f["VALUE"].reindex(P["cal"]))
    f = RT.fund_prices(ctx.root, "SBGB", cfg["cutoff"])
    fund_row("OFZ_DURATION", "SBGB", f.index.min(), f["VALUE"].reindex(P["cal"]))
    for s in ("GOLD", "AKGD", "SBGD", "TGLD"):
        fund_row("GOLD", s, px[s].first_valid_index(), val[s])
    t = pd.DataFrame(rows)
    ok = t[t["covers_replay_start"] & (t["zero_value_sessions_in_replay"] == 0)]
    chosen = ok.sort_values("median_daily_value_rub", ascending=False).groupby("exposure")["candidate"].first()
    for row in rows:
        row["selected"] = bool(chosen.get(row["exposure"]) == row["candidate"])
    want = {"BROAD_RUSSIAN_EQUITY": cfg["r2"]["funds"]["broad_equity"], "OFZ_DURATION": cfg["r2"]["funds"]["ofz"],
            "GOLD": cfg["r2"]["funds"]["gold"]}
    for e, s in want.items():
        assert chosen.get(e) == s, f"configured route {s} for {e} does not follow the declared rule ({chosen.get(e)})"
    # futures hedge contract: notional <= 0.5% NAV0 and median daily value >= 1bn RUB, smallest notional wins
    from . import market as M
    nav0 = 20_000_000.0
    for fam in ("MIX_FUT", "MXI_FUT"):
        h = M.deriv_history(ctx, fam, P["cal"], P["usd"], float(ctx.cfg["costs"]["broker_fee_futures_rub_per_contract"]))
        n = float(h["notional"].dropna().iloc[-1])
        v = val[fam].loc[start:cutoff].median()
        rows.append({"exposure": "BROAD_RUSSIAN_EQUITY_HEDGE", "candidate": fam, "notional_at_cutoff_rub": n,
                     "notional_pct_nav0": n / nav0, "median_daily_value_rub": float(v),
                     "rule": "notional <= 0.5% NAV0 and median value >= 1bn RUB; smallest notional",
                     "selected": bool(n / nav0 <= 0.005 and v >= 1e9)})
    t = pd.DataFrame(rows)
    assert t[(t["exposure"] == "BROAD_RUSSIAN_EQUITY_HEDGE") & t["selected"]]["candidate"].tolist() == [cfg["r2"]["futures"]["equity_hedge"]]
    return t
