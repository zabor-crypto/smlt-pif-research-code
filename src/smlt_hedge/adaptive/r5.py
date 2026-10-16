"""Gate R5 outputs: policy library across observable states and the stateful decision object at the cutoff.

The library answers "what would the ONE policy do from the actual current book if the confirmed state were X" — it is
the same `policy.decide` call with different state inputs (caps, state views), never a table of mandatory weights.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import engine as E
from . import events as EV
from . import frontier as F
from . import policy as PL
from . import replay as RP
from . import scenarios as SC
from . import state as ST
from . import views as V


def context(ctx, cfg, P, priors) -> dict:
    t = pd.Timestamp(cfg["cutoff"])
    H = cfg["r3"]["frontier"]["horizon_sessions"]
    feat = ST.features(ctx, cfg)
    raw = ST.classify(feat, cfg)
    Ev = EV.register(ctx, cfg)
    severe = pd.Series([EV.state_at(Ev, d)["severe_restriction"] for d in raw.index], index=raw.index)
    conf = PL.confirmed_states(raw, cfg, severe)
    panel = SC.history_panel(P, t, cfg["r3"]["frontier"]["estimation_start"])
    pts, _ = V.registry(ctx, cfg, P, t, H / 252, priors, panel)
    fs = RP.state_forward_table(cfg, P, conf, t, H)
    px = RP.unit_prices(ctx, cfg, P)
    S0 = float(px["SMLT"].loc[t])
    shares = np.floor(cfg["owner"]["smlt_initial_value_rub"] / S0)
    return {"t": t, "H": H, "feat": feat, "raw": raw, "conf": conf, "events": Ev, "panel": panel, "points": pts, "fs": fs,
            "px": px, "shares": shares, "S0": S0}


def initial_book(cfg, fin_names: list, din_names: list, shares: float, capital: float) -> E.Book:
    return E.Book(qf=np.array([shares if n == "SMLT" else 0.0 for n in fin_names]), qd=np.zeros(len(din_names)), cash=capital)


def _decide(ctx, cfg, P, C, states: dict, ev: dict, kappa: float, universe: str, mandate: str, n: int | None, book=None):
    H = C["H"]
    sv = RP.state_views_at(cfg, C["fs"], C["t"], states, H)
    pts = V.add_state_views(C["points"], sv, C["t"], H / 252)
    iv = V.intervals(pts, C["panel"], H / 252).set_index("route")
    routes = list(cfg["r3"]["universes"][universe])
    fin_names = [r for r in routes if r in F.FUNDED] + ["MMF"]
    din_names = [r for r in routes if r not in F.FUNDED]
    capital = float(cfg["owner"]["capital_main_rub"])
    bk = book or initial_book(cfg, fin_names, din_names, C["shares"], capital)
    d = PL.decide(ctx, cfg, P, C["t"], bk, float(np.broadcast_to(bk.cash, (1,))[0]), states, ev, iv, universe, mandate, kappa,
                  cfg["r3"]["frontier"]["budgets"]["primary"], px=C["px"], n_gen=n, n_adm=n)
    return d, sv, iv


_L = {}


def _lib_init(root: str):
    """Worker initialiser (spawn): same inputs as the parent, import guard as in r6."""
    import os
    import sys
    import warnings
    warnings.simplefilter("error", RuntimeWarning)
    from .. import pipeline as pl
    from . import r1, r3
    from . import routes as RT
    ctx = pl.Ctx(root)
    cfg = r1.load_cfg(ctx)
    P = RT.build_panel(ctx, cfg)
    pri = V.priors_pre2022(ctx, cfg)
    bp = r3.priors_btc(ctx, cfg, P, pri)
    if bp:
        pri["BTC_RUB"] = bp
    src = os.path.realpath(os.path.join(root, "src"))
    bad = [m.__name__ for m in list(sys.modules.values()) if getattr(m, "__name__", "").startswith("smlt_hedge")
           and getattr(m, "__file__", None) and not os.path.realpath(m.__file__).startswith(src)]
    if bad:
        raise RuntimeError(f"worker import guard: modules outside {src}: {bad}")
    _L.update({"ctx": ctx, "cfg": cfg, "P": P, "C": context(ctx, cfg, P, pri)})


def _lib_row(task: tuple) -> dict:
    iss, mk, rt, kappa = task
    ctx, cfg, P, C = _L["ctx"], _L["cfg"], _L["P"], _L["C"]
    lib = cfg["r5"]["library"]
    actual = C["conf"].loc[C["t"]].to_dict()
    ev0 = EV.state_at(C["events"], C["t"])
    sev = iss == "SEVERE_VERIFIED_EVENT"
    states = {"ISSUER_CREDIT_RISK": "DISTRESSED" if sev else iss, "MARKET_RISK": mk, "RATES_STATE": rt}
    ev = dict(ev0, severe_restriction=sev)
    d, sv, _ = _decide(ctx, cfg, P, C, states, ev, kappa, "MAIN", "BROAD_PIF_NO_HIDDEN_LEVERAGE", lib["n_paths_per_model"])
    return {"issuer_state": iss, "market_state": mk, "rates_state": rt, "kappa": kappa,
            "is_actual_confirmed_state_at_cutoff": (not sev) and states == actual,
            "hypothetical_state": sev or states != actual,
            "issuer_cap_net_long_nav": d.cap, "state_views_used": "; ".join(f"{k}={v['value']:.4f}" for k, v in sorted(sv.items())),
            "action_from_actual_book": d.action, "current_book_admitted": d.current_admitted,
            "n_candidates": len(d.candidates), "n_admitted": sum(c["passed"] for c in d.candidates),
            **{f"w_tgt_{r}": v for r, v in (d.target_w or {}).items()},
            "released_smlt_capital_goes_to": destination(d),
            "engine_ES95_target_nav0": d.metrics.get("engine_ES95_worst_nav0"),
            "robust_excess_nav0": d.metrics.get("robust_excess_nav0"), "nominal_excess_nav0": d.metrics.get("nominal_excess_nav0"),
            "reasons": " | ".join(d.reasons)}


def library(ctx, cfg, P, priors, C: dict, processes: int = 1) -> pd.DataFrame:
    """Every row is an independent deterministic `decide` call (fixed seeds), so rows are computed in parallel
    processes and assembled in the fixed task order."""
    lib = cfg["r5"]["library"]
    tasks = [(iss, mk, rt, kappa) for iss, mk, rt in itertools.product(lib["issuer_states"], lib["market_states"], lib["rates_states"])
             for kappa in lib["kappas"]]
    if processes <= 1:
        _L.update({"ctx": ctx, "cfg": cfg, "P": P, "C": C})
        return pd.DataFrame([_lib_row(t) for t in tasks])
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(processes, initializer=_lib_init, initargs=(str(ctx.root),)) as pool:
        return pd.DataFrame(pool.map(_lib_row, tasks, chunksize=1))


def destination(d: PL.Decision) -> str:
    if d.target_w is None:
        return "NO_ADMISSIBLE_TARGET"
    released = d.current_w.get("SMLT", 0.0) - d.target_w.get("SMLT", 0.0)
    if released <= 1e-6:
        return "no SMLT capital released"
    other = {r: v - d.current_w.get(r, 0.0) for r, v in d.target_w.items() if r in F.FUNDED and r != "SMLT" and v - d.current_w.get(r, 0.0) > 1e-6}
    mm = released - sum(other.values())
    parts = [f"{r} {v:.3f}" for r, v in sorted(other.items(), key=lambda kv: -kv[1])] + [f"MONEY_MARKET_AND_RESERVE {mm:.3f}"]
    return "; ".join(parts) + " (NAV fractions)"


def smlt_only_scan(su: F.Setup, cfg: dict, budget: float, grid=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)) -> list:
    """Diagnostic only: books holding w NAV of SMLT shares and the money market, evaluated by the full engine from the
    actual book. `risk_pass` ignores the state cap (ES95, breach probability, unresolved deficits); `cap_pass` adds it."""
    out = []
    a = cfg["r1"]["admission"]
    for w in grid:
        x = np.array([w if r == "SMLT" else 0.0 for r in su.routes])
        ex = F.admit_candidate(su, x)
        m = F.summarise(su, ex)
        risk = (m["engine_ES95_worst_nav0"] <= budget and m["p_breach_worst"] <= a["max_breach_probability"]
                and m["p_unresolved_worst"] <= a["max_unresolved_deficit_probability"] and m["p_unresolved_stress_worst"] <= 0.0)
        out.append({"smlt_weight": float(ex["w_exec"].get("SMLT", 0.0)), "engine_ES95_worst_nav0": m["engine_ES95_worst_nav0"],
                    "p_breach_worst": m["p_breach_worst"], "risk_pass": bool(risk),
                    "cap_pass": bool(su.issuer_cap is None or ex["w_exec"].get("SMLT", 0.0) <= su.issuer_cap + 1e-9)})
    return out


def decision_object(ctx, cfg, P, priors, C: dict) -> tuple[dict, pd.DataFrame]:
    """Section X: stateful decision at the cutoff from the ACTUAL book (quantities passed in, not hard-coded weights)."""
    t = C["t"]
    states = C["conf"].loc[t].to_dict()
    raw = C["raw"].loc[t].to_dict()
    ev = EV.state_at(C["events"], t)
    budget = cfg["r3"]["frontier"]["budgets"]["primary"]
    kc = cfg["r3"]["uncertainty_set"]["kappa_canonical"]
    d, sv, iv = _decide(ctx, cfg, P, C, states, ev, kc, "MAIN", "BROAD_PIF_NO_HIDDEN_LEVERAGE", None)
    d0, _, _ = _decide(ctx, cfg, P, C, states, ev, 0.0, "MAIN", "BROAD_PIF_NO_HIDDEN_LEVERAGE", None)
    su = d.su
    adm = [c for c in d.candidates if c["passed"]]
    P0 = {f.name: f.P0 for f in su.fspecs}
    cur_q = {**{n: float(su.book0.qf[j]) for j, n in enumerate(su.fin_names)}, **{n: float(su.book0.qd[j]) for j, n in enumerate(su.din_names)}}
    if d.ex is not None:
        tgt_q = {**{n: (None if np.isnan(d.ex["qf"][j]) else float(d.ex["qf"][j])) for j, n in enumerate(su.fin_names)},
                 **{n: float(d.ex["qd"][j]) for j, n in enumerate(su.din_names)}}
    else:
        tgt_q = dict(cur_q)
    cal = P["cal"]
    nxt = pd.bdate_range(t + pd.Timedelta(days=1), periods=1)[0]
    lo = float(sum(max(d.target_w.get(r, 0), 0) * (iv.loc[F.SERIES.get(r, r), "lo"]) + min(d.target_w.get(r, 0), 0) * (iv.loc[F.SERIES.get(r, r), "hi"])
                   for r in su.routes)) if d.target_w else None
    hi = float(sum(max(d.target_w.get(r, 0), 0) * (iv.loc[F.SERIES.get(r, r), "hi"]) + min(d.target_w.get(r, 0), 0) * (iv.loc[F.SERIES.get(r, r), "lo"])
                   for r in su.routes)) if d.target_w else None
    stress_rows = [r for r in (d.ex["rows"] if d.ex is not None else []) if r["margin_multiplier"] != 1.0]
    scan = smlt_only_scan(su, cfg, budget)
    risk_max = max([r["smlt_weight"] for r in scan if r["risk_pass"]] or [0.0])
    zw = cfg["r3"]["zero_weight"]
    smlt_iv = iv.loc["SMLT"]
    obj = {
        "DATE": str(t.date()), "CURRENT_NAV": su.nav0, "CURRENT_QUANTITIES": cur_q,
        "IMMEDIATE_CASH": float(np.broadcast_to(su.book0.cash, (1,))[0]), "MMF_UNITS": cur_q.get("MMF", 0.0),
        "PENDING_SETTLEMENT": float(sum((su.book0.pending or {}).values())),
        "POSTED_MARGIN": float((np.abs(su.book0.qd) * np.array([su.notional[n] * su.mr[n] for n in su.din_names])).sum()),
        "OBSERVABLE_RISK_STATE": {"raw": raw, "confirmed": states,
                                  "evidence": {k: (None if pd.isna(v) else (float(v) if not isinstance(v, str) else v)) for k, v in C["feat"].loc[t].to_dict().items()}},
        "VERIFIED_EVENT_STATE": ev, "PRIMARY_RISK_BUDGET": {"ES95_6m_max_nav0": budget, "liquidity_min_nav": cfg["owner"]["liquidity_min_nav"]},
        "CURRENT_BOOK_ADMISSION": {"admitted": d.current_admitted, "fail_reasons": d.current_reasons, **d.current_metrics},
        "APPLICABLE_CAPS": {"issuer_net_long_nav": d.cap, "mandate": "BROAD_PIF_NO_HIDDEN_LEVERAGE",
                            "route_bounds": {r: [cfg["r3"]["bounds"][r]["lo"], cfg["r3"]["bounds"][r]["hi"]] for r in su.routes},
                            "note": "caps are ceilings; they never create a purchase"},
        "ADMISSIBLE_FRONTIER_REGION": {"kappa": kc, "n_candidates": len(d.candidates), "n_admitted": len(adm),
                                       "admitted": [{"es_target": c["e"], "robust_excess_nav0": c["robust"], "nominal_excess_nav0": c["nominal"],
                                                     "weights": {r: v for r, v in c["ex"]["w_exec"].items() if abs(v) > 1e-9}} for c in adm]},
        "SELECTED_ACTION": d.action, "TARGET_QUANTITIES": tgt_q, "TARGET_WEIGHTS_NAV": d.target_w,
        "EXECUTION_TIME": f"close of the next executable session after {t.date()} ({nxt.date()}); integer quantities; MMF units by the cash rule after settlement",
        "FULL_ENGINE_ES95": d.metrics.get("engine_ES95_worst_nav0"), "FULL_ENGINE_ES99": d.metrics.get("engine_ES99_worst_nav0"),
        "LIQUIDITY_STRESS": {"p_breach_worst_model": d.metrics.get("p_breach_worst"),
                             "p_unresolved_worst_model": d.metrics.get("p_unresolved_worst"),
                             "p_unresolved_margin_x1.25_x1.50": max([r["p_unresolved"] for r in stress_rows] or [0.0]),
                             "min_available_nav_p05": d.metrics.get("min_available_nav_p05_worst"),
                             "reserve_min_nav": d.metrics.get("reserve_min_nav")},
        "EXPECTED_NET_EXCESS_RETURN_RANGE": {"horizon_sessions": C["H"], "worst_case_kappa1_nav0": d.metrics.get("robust_excess_nav0"),
                                             "centre_nav0": d.metrics.get("nominal_excess_nav0"),
                                             "view_set_position_range_nav0": [lo, hi],
                                             "note": "excess over the money market, per NAV0, net of entry costs and the liquidity reserve drag"},
        "MAIN_REASONS": d.reasons, "STATE_VIEWS_USED": {k: v["source"] for k, v in sv.items()},
        "SENSITIVITY_NOMINAL_KAPPA0": {"action": d0.action, "target_weights": d0.target_w,
                                       "engine_ES95_nav0": d0.metrics.get("engine_ES95_worst_nav0"),
                                       "centre_excess_nav0": d0.metrics.get("nominal_excess_nav0"),
                                       "label": "SENSITIVITY_ONLY (centre of the view set; not the canonical decision)"},
        "WHAT_WOULD_CHANGE_THE_DECISION": {
            "diagnostic_smlt_only_books": scan,
            "largest_smlt_weight_passing_risk_and_liquidity_alone": risk_max,
            "issuer_cap_in_confirmed_state": d.cap,
            "smlt_view_set_6m_excess": {"lo": float(smlt_iv["lo"]), "centre": float(smlt_iv["central"]), "hi": float(smlt_iv["hi"]),
                                        "points_max": float(smlt_iv["points_max"])},
            "conditions": [
                "a substantiated return view with positive excess over the money market for SMLT (or any route) entered in the view "
                "registry: the canonical frontier holds a route only if its worst-case excess net of costs is positive",
                f"a confirmed issuer state other than the present one changes the SMLT ceiling (caps {cfg['r5']['issuer_caps_net_long_nav']}); "
                "a ceiling never creates a purchase",
                "a VERIFIED_PRIMARY severe issuer event sets the ceiling to 0 from the next session",
                "a different primary risk budget (10% / 20% frontiers are reported separately and are never pooled)",
                "evidence that money-market fund units are accepted as same-day margin collateral, or interest on account cash: "
                "reduces the reserve drag, not the direction of the decision",
                f"forced-entry diagnostic size {zw['forced_entry_nav0']:.0%} NAV0 per route: see adaptive_zero_weight_attribution.csv for the "
                "objective loss of holding each route"],
            "note": "diagnostic_smlt_only_books is a risk diagnostic (SMLT + money market, no other route), not a policy candidate"},
        "NEXT_REVIEW": "next weekly decision close; immediately on a verified severe event; model recalibration January / July",
        "AUTHORITY": "full recursive engine (R1); LP generates candidates only",
    }
    rows = []
    for n in list(su.fin_names) + list(su.din_names):
        unit = P0[n] if n in P0 else su.notional[n]
        tq = tgt_q.get(n)
        rows.append({"instrument": n, "kind": "FUNDED" if n in P0 else "DERIVATIVE", "unit_value_rub": unit,
                     "current_quantity": cur_q[n], "current_weight_nav": cur_q[n] * unit / su.nav0,
                     "target_quantity": "by cash rule after settlement" if tq is None else tq,
                     "target_weight_nav": None if tq is None else tq * unit / su.nav0, "action": d.action})
    rows.append({"instrument": "IMMEDIATE_RUB_CASH", "kind": "CASH", "unit_value_rub": 1.0, "current_quantity": obj["IMMEDIATE_CASH"],
                 "current_weight_nav": obj["IMMEDIATE_CASH"] / su.nav0, "target_quantity": "reserve band: floor / target / ceiling above 5% + VM99",
                 "target_weight_nav": None, "action": d.action})
    obj["_decisions"] = (d, d0)
    return obj, pd.DataFrame(rows)


def monte_carlo_error(ctx, cfg, P, C: dict, su: F.Setup, books: dict, offsets: list) -> pd.DataFrame:
    """Section Q: Monte-Carlo error (re-drawing the simulation seeds) reported separately from market-history
    uncertainty (the spread across the three scenario models, which resample ONE history in different ways).
    Seeds say nothing about the market: a small seed spread is not a confidence interval for the fund."""
    rows = []
    for off in offsets:
        sk = F.build(ctx, cfg, P, su.t, "MAIN", su.mandate, su.views, book0=su.book0, seed_offset=int(off), px=C["px"],
                     issuer_cap=su.issuer_cap, available0=su.available0)
        for name, w in books.items():
            x = sk.x0 if w is None else np.array([w.get(r, 0.0) for r in sk.routes])
            ex = F.admit_candidate(sk, x)
            for r in ex["rows"]:
                if r["margin_multiplier"] == 1.0:
                    rows.append({"row": "DRAW", "book": name, "model": r["model"], "seed_offset": int(off),
                                 "ES95_nav0": r["ES95"] / sk.nav0, "p_breach": r["p_breach"]})
    d = pd.DataFrame(rows)
    out = [d]
    g = d.groupby(["book", "model"])["ES95_nav0"].agg(["mean", "std", "min", "max"]).reset_index()
    for _, r in g.iterrows():
        out.append(pd.DataFrame([{"row": "MONTE_CARLO_ERROR_ACROSS_SEEDS", "book": r["book"], "model": r["model"],
                                  "ES95_nav0": r["mean"], "std_across_seeds": r["std"], "min": r["min"], "max": r["max"]}]))
    m = d.groupby(["book", "model"])["ES95_nav0"].mean().reset_index().groupby("book")["ES95_nav0"].agg(["min", "max"]).reset_index()
    for _, r in m.iterrows():
        out.append(pd.DataFrame([{"row": "MODEL_SPREAD_ONE_HISTORY", "book": r["book"], "model": "ALL", "min": r["min"], "max": r["max"],
                                  "ES95_nav0": r["max"], "note": "range of the seed-averaged ES95 across HS_BLOCK / FHS_EWMA / HS_RECENT_1Y"}]))
    return pd.concat(out, ignore_index=True)
