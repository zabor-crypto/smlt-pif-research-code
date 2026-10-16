"""V3 Phase 7/8: final decision-date comparison, PURE HEDGE OVERLAY vs ACTIVE EXPOSURE MANAGEMENT.

Every finalist - static or dynamic - runs through the SAME path-dependent engine (portfolio/dynamic.py, unified
ledger, liquidity waterfall) on the SAME untouched TEST paths (20 independent seeds x 4 admissible models) and on the
five forward scenarios. Dynamic policies observe states on each path causally (portfolio/regime_risk.path_labels).
Portfolios are integer MOEX contracts (Phase 0.5); dynamic library states use the integer library of s20.
"""
from __future__ import annotations

import ast

import numpy as np
import pandas as pd

from . import pipeline as pl
from .portfolio import dynamic
from .portfolio import regime_risk as rr
from .portfolio import sleeves as sleeves_mod
from .portfolio import validation as va
from .strategies.collateral import Collateral
from .v3_policy import (STATE_COLS, _eq_cost, active_policy, dyn_sleeves, frozen_thresholds, library_matrix, path_codes,
                        prefix_frame, protective_rank)

HZ = {"6m": 126, "12m": 252}


def _finalists(ctx, specs) -> list[dict]:
    names = [s.name for s in specs]
    z = np.zeros((1, len(names)))
    S0, F0 = pl._fut_ctx(ctx)[:2]
    sel = pd.read_csv(ctx.out / "v3_selected_portfolios.csv")
    lib = library_matrix(ctx, specs)
    smc = ctx.cfg["v3"]["state_machine"]
    sm_sel = pd.read_csv(ctx.out / "state_machine_parameter_selection.csv")["selected"].iloc[-1]
    cf, hd = ast.literal_eval(sm_sel)
    out = [{"branch": "CONTROL", "name": "CONTROL", "pol": lambda hz: dynamic.Policy("c", z, np.array([1.0])), "codes": None}]
    for x in (0.25, 0.5, 0.75, 1.0):
        h = z.copy()
        h[0, names.index("DIRECT")] = -round(x * ctx.E / S0) * F0 / ctx.E
        out.append({"branch": "PURE_HEDGE_OVERLAY", "name": f"FUT_{int(x * 100)}",
                    "pol": (lambda hh: (lambda hz: dynamic.Policy("f", hh, np.array([1.0]))))(h), "codes": None})
    for key, label in (("STATIC_SMART", "STATIC_SMART_INT"), ("ROBUST_SMART", "ROBUST_SMART_INT")):
        def mk(hz, key=key):
            r = sel[(sel["selection"] == key) & (sel["horizon"] == hz)]
            if not len(r) or r["cand_id"].iloc[0] == "NONE":
                return None
            hi, _ = va.round_contracts(specs, {n: float(r[f"w_{n}"].iloc[0]) for n in names}, ctx.E)
            return dynamic.Policy(label, np.array([[hi[n] for n in names]]), np.array([1.0]))
        out.append({"branch": "PURE_HEDGE_OVERLAY", "name": label, "pol": mk, "codes": None})
    out.append({"branch": "PURE_HEDGE_OVERLAY", "name": "REGIME_SWITCHING_SMART", "codes": "portfolio",
                "pol": lambda hz: dynamic.Policy("sm", lib, np.ones(len(rr.PORTFOLIO_STATES)), confirm=cf, hold=hd,
                                                 threshold=smc["threshold"], protective_rank=protective_rank(ctx),
                                                 emergency_shock=smc["emergency_shock"]),
                "meta": f"library of s20, confirm={cf}, hold={hd} (walk-forward selected at the cutoff)"})
    out.append({"branch": "ACTIVE_EXPOSURE_MANAGEMENT", "name": "DERISK_SELL_50", "codes": None,
                "pol": lambda hz: dynamic.Policy("d", z, np.array([0.5]))})
    for v in ctx.cfg["v3"]["active"]["variants"]:
        t = pd.read_csv(ctx.interim / f"active_parameter_selection_{v}.csv")["selected"].iloc[-1]
        acf, ahd = ast.literal_eval(t)
        out.append({"branch": "ACTIVE_EXPOSURE_MANAGEMENT", "name": v, "codes": "active",
                    "pol": (lambda v_, a_, b_: (lambda hz: active_policy(ctx, v_, a_, b_, len(names))))(v, acf, ahd),
                    "meta": f"confirm={acf}, hold={ahd} (walk-forward selected at the cutoff)"})
    return out


def _run_set(ctx, fins, Pth, codes, sl, mmf, eq, coll=None) -> dict:
    res = {}
    N = Pth["SMLT"].shape[0]
    H = Pth["SMLT"].shape[1]
    zeros = np.zeros((N, H + 1), int)
    for f in fins:
        pol = f["pol"](f["_hz"])
        if pol is None:
            continue
        st = zeros if f["codes"] is None else codes[f["codes"]][:, :H + 1]
        kw = {"coll": coll} if coll is not None else {}
        res[f["name"]] = dynamic.run(Pth, st, pol, sl, E=ctx.E, C=ctx.C, mmf_daily=mmf, equity_cost=eq,
                                     shock=codes["shock"][:, :H] if f["codes"] else None, **kw)
    return res


KEYS = ["ES95", "ES99", "std_pnl", "p95_max_dd", "upside_capture", "downside_capture", "mean_pnl", "transaction_costs",
        "turnover", "initial_im", "peak_collateral_p95", "p_sleeve_breach", "mean_switches", "mean_exposure_changes",
        "mean_share_fraction", "p_forced_share_sale", "p_waterfall_cut"]


def s23_final(ctx) -> None:
    v3 = ctx.cfg["v3"]
    specs = sleeves_mod.build(ctx)
    sl = dyn_sleeves(specs)
    fins = _finalists(ctx, specs)
    fz, pref = frozen_thresholds(ctx), prefix_frame(ctx)
    eq = _eq_cost(ctx, pl._fut_ctx(ctx)[0])
    rows = []
    fac = va.ScenarioFactory(ctx, pl, specs, state_cols=STATE_COLS)
    mmf = fac.B["daily_y"]
    for hz_name, hz in HZ.items():
        for f in fins:
            f["_hz"] = hz_name
        for k in range(v3["seeds"]["n_test"]):
            for mdl in v3["admissible_models"]:
                Pth = fac.paths(mdl, hz, v3["n_test_paths"], "TEST", k)
                codes = path_codes(ctx, Pth, fz, pref)
                for name, m in _run_set(ctx, fins, Pth, codes, sl, mmf, eq).items():
                    rows.append({"horizon": hz_name, "set": "TEST", "model": mdl, "seed_k": k, "strategy": name,
                                 **{x: m[x] for x in KEYS}})
        # collateral sensitivity on seed 0, FHS
        Pth = fac.paths("FHS_FULL", hz, v3["n_test_paths"], "TEST", 0)
        codes = path_codes(ctx, Pth, fz, pref)
        for cname, c in ctx.cfg["collateral"]["named"].items():
            coll = Collateral(cname, c["yield_fraction"], c["haircut"], c["margin_multiplier"])
            for name, m in _run_set(ctx, fins, Pth, codes, sl, mmf, eq, coll).items():
                rows.append({"horizon": hz_name, "set": f"COLLATERAL_{cname}", "model": "FHS_FULL", "seed_k": 0, "strategy": name,
                             **{x: m[x] for x in KEYS}})
    # forward scenarios (first 126 / 252 days of each scenario path set)
    from .v3_stages import forward_paths
    cols = ["SMLT"] + [s.series for s in specs if s.series != "SMLT"] + ["SMLT_BOND_IDX"]
    fwd, meta, _ = forward_paths(ctx, cols, n=v3["n_test_paths"])
    for name_sc, Pf in fwd.items():
        Pf = dict(Pf)
        Pf["IMOEX"], Pf["RGBI"] = Pf["MKT"], Pf["RATES"]
        for hz_name, hz in HZ.items():
            for f in fins:
                f["_hz"] = hz_name
            Pc = {c: v[:, :hz] for c, v in Pf.items() if isinstance(v, np.ndarray) and v.ndim == 2}
            codes = path_codes(ctx, Pc, fz, pref)
            for name, m in _run_set(ctx, fins, Pc, codes, sl, mmf, eq).items():
                rows.append({"horizon": hz_name, "set": f"FORWARD_{name_sc}", "model": name_sc, "seed_k": 0, "strategy": name,
                             **{x: m[x] for x in KEYS}})
    D = pd.DataFrame(rows)
    ctx.save(D, "final_v3_all_metrics.csv", "interim", index=False)
    bv = ctx.out / "btc_verdict.csv"
    if bv.exists():
        b = pd.read_csv(bv, index_col=0)["value"]
        btc_note = (f"BTC classification {b['classification']}; admitted_to_main_library={b['admitted_to_main_library']} "
                    f"-> BTC weight 0 in every finalist" if str(b["admitted_to_main_library"]) != "True" else
                    f"BTC admitted ({b['classification']}); see btc_incremental_value_ci.csv")
    else:
        btc_note = "BTC stage not run"
    branch = {f["name"]: f["branch"] for f in fins}
    note = {f["name"]: f.get("meta", "") for f in fins}
    out = []
    test = D[D["set"] == "TEST"]
    for name in branch:
        row = {"branch": branch[name], "strategy": name, "policy_note": note[name]}
        for hz in HZ:
            g = test[(test["strategy"] == name) & (test["horizon"] == hz)]
            if g.empty:
                continue
            fhs = g[g["model"] == "FHS_FULL"]
            for x in KEYS:
                row[f"{x}_{hz}"] = float(fhs[x].mean())
            row[f"ES95_{hz}_seed_sd"] = float(fhs["ES95"].std(ddof=1))
            wm = g.groupby("model")["ES95"].mean()
            row[f"worst_model_{hz}"], row[f"worst_model_ES95_{hz}"] = wm.idxmax(), float(wm.max())
            fw = D[(D["set"].str.startswith("FORWARD_")) & (D["strategy"] == name) & (D["horizon"] == hz)]
            if len(fw):
                w = fw.sort_values("mean_pnl").iloc[0]
                row[f"worst_forward_scenario_{hz}"] = w["model"]
                row[f"worst_forward_scenario_mean_pnl_{hz}"] = float(w["mean_pnl"])
                row[f"worst_forward_scenario_ES95_{hz}"] = float(fw["ES95"].max())
                row[f"worst_forward_scenario_by_ES95_{hz}"] = fw.sort_values("ES95").iloc[-1]["model"]
            for cname in ctx.cfg["collateral"]["named"]:
                c = D[(D["set"] == f"COLLATERAL_{cname}") & (D["strategy"] == name) & (D["horizon"] == hz)]
                if len(c):
                    row[f"breach_{hz}_{cname}"] = float(c["p_sleeve_breach"].iloc[0])
                    row[f"ES95_{hz}_{cname}"] = float(c["ES95"].iloc[0])
        row["btc_contribution"] = btc_note
        out.append(row)
    F = pd.DataFrame(out)
    ctx.save(F, "final_v3_strategy_comparison.csv", index=False)
    # paired differences of the dynamic / smart finalists vs FUT_50 and vs STATIC_SMART_INT (TEST, per model)
    from .v3_stages import verdict
    drows = []
    for a in [n for n in branch if n not in ("CONTROL",)]:
        for b in ("FUT_50", "STATIC_SMART_INT"):
            if a == b:
                continue
            for hz in HZ:
                g = test[test["horizon"] == hz].rename(columns={"strategy": "cand_id"})
                for metric in ("ES95", "p_sleeve_breach", "mean_pnl"):
                    pm = va.delta_ci(g, a, b, metric)
                    if not pm:
                        continue
                    vd = verdict(pm, v3["stability"]["material_rub"] if metric == "ES95" else 0.0, v3["stability"]["min_sign_share"],
                                 metric != "mean_pnl")
                    for mdl, st in pm.items():
                        drows.append({"horizon": hz, "strategy": a, "comparator": b, "metric": metric, "model": mdl, **st,
                                      "verdict_all_models": vd})
    ctx.save(pd.DataFrame(drows), "final_v3_delta_ci.csv", index=False)
