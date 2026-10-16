"""Regime V2 — candidate generation (LP), full-engine admission, frontier refinement and lexicographic selection.

Candidate generator (per NAV0, horizon H, generation paths of every available risk model k):

  maximise   sum_j mu_j x_j - costs(x) - drags(x)            (expected NET excess over the money market)
  s.t.       CVaR_95,k(-pnl(x)) <= e                 for every model k (Rockafellar-Uryasev, paths centred on carry)
             sum over funded non-MMF routes x_j >= floor + operating band          (invested mandate)
             SMLT futures contracts short <= SMLT shares held                       (no naked issuer short; units)
             SMLT shares + SMLT futures <= state ceiling                            (value; ceilings never force a buy)
             short broad-equity / rates futures <= the funded exposure they protect (protective overlays)
             route bounds, long exposure <= 1, derivative gross <= 1, funded + margin + reserve <= 1

The LP never admits. Every candidate is rounded toward zero to shares / units / whole contracts, converted into
settlement-aware next-session orders from the ACTUAL book and evaluated by the R1 engine on independent paths of every
model, at margin x1.0 / x1.25 / x1.50, plus survival on every deterministic stress path. Where two adjacent candidates
bracket PASS / FAIL, the segment between them is bisected in exposure space (re-rounded and re-admitted at every step,
at most 12 steps). Selection among admitted books is lexicographic: expected net excess; within 10 bp, lower engine
ES95; within 5 bp of that, lower expected turnover cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import FunctionType
from weakref import WeakKeyDictionary, ref

import numpy as np
import pandas as pd
import scipy.sparse as sp_
from scipy.optimize import linprog

from ..adaptive import admission as A
from ..adaptive import engine as E
from ..adaptive import frontier as F
from ..adaptive import market as M
from ..adaptive import scenarios as SC
from ..adaptive import stress as SS
from ..adaptive.r1 import loss_quantile
from . import risk as RK
from .cfg import TECH_MANDATE, StopCondition

SERIES = F.SERIES
FUNDED = F.FUNDED

# Provenance only for callbacks created by this native factory, not arbitrary callables.
_NATIVE_STAGE_CHECKS = WeakKeyDictionary()


@dataclass
class Spec:
    """What one frontier / decision is solved under."""
    mandate: dict                      # cfg.mandate(...)
    budget: float
    cap_net: float = 0.50              # ceiling on SMLT shares + SMLT futures (value, NAV)
    cap_gross: float | None = None     # ceiling on SMLT shares (verified severe event: 0)
    kappa: float = 0.0                 # > 0 only for the EXTREME_ROBUST sensitivity (parent view sets)
    delay: int = 1
    fixed: dict = field(default_factory=dict)    # {route: weight} held fixed (hedge-keep comparators)
    floor_suspended: bool = False      # emergency override

    @property
    def floor_target(self) -> float:
        return 0.0 if self.floor_suspended else self.mandate["floor_target"]

    @property
    def floor_hard(self) -> float:
        return 0.0 if self.floor_suspended else self.mandate["floor_hard"]


# ---------------------------------------------------------------------------------------------- setup
def build(ctx, cfg: dict, P: dict, t: pd.Timestamp, universe: str, views: pd.DataFrame, *, book0: E.Book | None = None,
          capital: float | None = None, n_gen: int, n_adm: int, models: tuple | None = None, state_choice: dict | None = None,
          seed_offset: int = 0, panel_end: pd.Timestamp | None = None, px: pd.DataFrame | None = None,
          available0: float | None = None, cost_mult: float = 1.0, generation_only: bool = False, headroom_commitment=None) -> F.Setup:
    """The parent `frontier.build` with the V2 model set and carry-only centring. `views`: returns.views_frame (index =
    scenario series; `central` = expected excess used by the LP, `centre_risk` = mean of the risk paths)."""
    a0_gate = None
    if generation_only or hasattr(ctx, '_headroom_a0'):
        from . import a0_input_authority as A0
    if generation_only or hasattr(ctx, '_headroom_a0'):
        a0_gate, owned = A0.validate_entry(dict(
            ctx=ctx, cfg=cfg, P=P, t=t, universe=universe, views=views,
            book0=book0, capital=capital, n_gen=n_gen, n_adm=n_adm,
            models=models, state_choice=state_choice, seed_offset=seed_offset,
            panel_end=panel_end, px=px, available0=available0,
            cost_mult=cost_mult, generation_only=generation_only,
            headroom_commitment=headroom_commitment))
        (ctx, cfg, P, t, universe, views, book0, capital, n_gen, n_adm,
         models, state_choice, seed_offset, panel_end, px, available0,
         cost_mult, generation_only, headroom_commitment) = (
            owned[name] for name in A0.REQUEST_FIELDS)
        a0_gate.check_request(owned)
    fr = cfg["r3"]["frontier"]
    H = fr["horizon_sessions"]
    L = int(cfg["v2"]["risk"]["block_length"])
    routes = list(cfg["r3"]["universes"][universe])
    panel_now = SC.history_panel(P, t, fr["estimation_start"])
    panel = panel_now if panel_end is None else SC.history_panel(P, panel_end, fr["estimation_start"])
    if a0_gate is not None:
        a0_gate.check_request(owned)
        a0_gate.check_panels(panel_now, panel)
    models = tuple(models) if models is not None else tuple(cfg["v2"]["risk"]["models"])
    if generation_only:
        from . import headroom_a as HA
        HA.check_build_request(cfg, t, n_gen, n_adm, models, seed_offset, headroom_commitment)
        if panel_end is not None or px is not None or cost_mult != 1.:
            raise StopCondition('HEADROOM_UNAUTHORIZED_INPUT_OR_COST_VARIANT')
    tgt = {SERIES.get(r, r): float(views.loc[SERIES.get(r, r), "centre_risk"]) for r in routes}
    usd = P["usd"]

    def _p(name):
        return float(px[name].loc[:t].dropna().iloc[-1]) if px is not None else F._price_at(ctx, P, name, t)
    broker = float(ctx.cfg["costs"]["broker_fee_futures_rub_per_contract"])
    costs = cfg["r3"]["costs"]
    fspecs, dspecs, cost, nroll, mr, notional, unit = [], [], {}, {}, {}, {}, {}
    for r in routes:
        if r in FUNDED:
            if r == "BTC_RUB_EXPERIMENTAL":
                p0 = float(P["B"]["BTC_RUB_LEVEL"].loc[:t].dropna().iloc[-1])
                fs = M.FundedSpec(r, E.FUNDED, p0, lag=cfg["r1"]["settlement"]["btc_experimental_lag"], lot=0.0,
                                  cost_prop=costs["btc_experimental"] * cost_mult)
            elif r == "SMLT":
                fs = M.FundedSpec(r, E.FUNDED, _p(r), lag=1, lot=1.0, cost_prop=costs["smlt"]["bp"] * 1e-4 * cost_mult,
                                  cost_unit=costs["smlt"]["half_tick_rub"] * cost_mult, issuer="SMLT")
            else:
                fs = M.FundedSpec(r, E.FUNDED, _p(r), lag=1, lot=1.0, cost_prop=costs["funds"] * cost_mult)
            if getattr(ctx, "_regime_v2r2_combined", False):
                from .execution_inputs import funded_tape
                key, capacity, _ = funded_tape(ctx, cfg, r, pd.DatetimeIndex([t]))
                fs.execution_key, fs.execution_capacity = key, float(capacity[0])
            fspecs.append(fs)
            cost[r] = fs.cost_prop + fs.cost_unit / fs.P0
            unit[r] = fs.P0
        else:
            h = M.deriv_history(ctx, F.DERIV_FAMILY[r], pd.DatetimeIndex([t]), usd, broker).iloc[-1]
            rd, gap = F._roll_plan(ctx, r, t, H)
            fee_rate = float(h["fee"] / h["notional"]) * cost_mult
            dspecs.append(M.DerivSpec(r, F0=float(h["Fn"]), pv0=float(h["pv"]), usd_linked=(r == "OIL_FUT"),
                                      mr=float(h["mr1"]), fee_rate=fee_rate, fee_fixed_rub=0.0, roll_days=rd, roll_gap=gap))
            if getattr(ctx, "_regime_v2r2_combined", False):
                from .execution_inputs import forecast_info
                dspecs[-1].execution_info = forecast_info(ctx, cfg, F.DERIV_FAMILY[r], t, h, H)
                dspecs[-1].execution_info['cost_multiplier'] = cost_mult
                if r != 'CNY_PERP':
                    dspecs[-1].roll_days = dspecs[-1].execution_info['roll_days']
            cost[r] = fee_rate
            nroll[r] = len(dspecs[-1].roll_days)
            mr[r] = float(h["mr1"])
            notional[r] = float(h["notional"])
            unit[r] = float(h["notional"])
    fspecs.append(M.FundedSpec("MMF", E.MMF, _p("MMF"), lag=1, lot=1.0))
    if getattr(ctx, "_regime_v2r2_combined", False):
        from .execution_inputs import funded_tape
        key, capacity, _ = funded_tape(ctx, cfg, 'MMF', pd.DatetimeIndex([t]))
        fspecs[-1].execution_key, fspecs[-1].execution_capacity = key, float(capacity[0])
        from .session_calendar import forecast
        schedule = forecast(ctx.root, t, H)
        for f in fspecs:
            f.execution_dates = schedule.dates
            f.execution_calendar_evidence = schedule.evidence
    fin_names = [f.name for f in fspecs]
    din_names = [d.name for d in dspecs]

    def to_engine(ps: SC.PathSet):
        idx = {r: k for k, r in enumerate(ps.routes)}
        Rf = np.stack([ps.R[:, :, idx[f.name]] if f.kind != E.MMF else np.zeros(ps.R.shape[:2]) for f in fspecs], axis=2)
        Rd = np.stack([ps.R[:, :, idx[SERIES.get(d.name, d.name)]] for d in dspecs], axis=2) if dspecs else np.zeros((ps.R.shape[0], H, 0))
        return M.forward_market(fspecs, dspecs, Rf, Rd, ps.R[:, :, idx["USDRUB"]], ps.mmf_daily,
                                cfg["r1"]["liquidity"]["immediate_cash_rate"], cny_swap_frac=ps.cny_swap_frac)
    gen, adm, G, used = [], [], {}, []
    seed0 = fr["seed_generation"] + seed_offset
    for mdl in models:
        k = RK.MODELS.index(mdl)                                  # fixed seed slot per model name
        raw = RK.simulate(mdl, panel, n_gen, H, seed0 + 101 * k, L, panel_now, state_choice)
        if raw is None:
            continue
        ps = SC.centre(raw, tgt, dspecs)
        fin, din, mkt = to_engine(ps)
        gen.append((mdl, ps, fin, din, mkt))
        G[mdl] = F.market_horizon_matrix(routes, fin_names, din_names, mkt)
        if not generation_only:
            psa = SC.centre(RK.simulate(mdl, panel, n_adm, H, seed0 + fr["seed_admission_shift"] + 101 * k, L, panel_now, state_choice), tgt, dspecs)
            fa, da, ma = to_engine(psa)
            adm.append(A.ModelPaths(f"{mdl}_ADMISSION", fa, da, ma))
        used.append(mdl)
    mH = float(np.prod(1 + gen[0][1].mmf_daily) - 1)
    lv = cfg["r1"]["liquidity"]["reserve_var_level"]
    q99, hc = {}, []
    for r in routes:
        s = SERIES.get(r, r)
        daily = np.concatenate([g[1].R[:, :, g[1].routes.index(s)].ravel() for g in gen])
        q99[r] = float(max(np.quantile(-daily, lv), np.quantile(daily, lv)))
    for f in fspecs:
        hc.append(0.0 if f.kind == E.MMF else
                  loss_quantile(np.concatenate([g[1].R[:, :, g[1].routes.index(f.name)].ravel() for g in gen]), lv))
    nav_cash = capital if capital is not None else cfg["owner"]["capital_main_rub"]
    if book0 is None:
        shares = np.floor(cfg["owner"]["smlt_initial_value_rub"] / unit["SMLT"])
        book0 = E.Book(qf=np.array([shares if n == "SMLT" else 0.0 for n in fin_names]), qd=np.zeros(len(din_names)), cash=nav_cash)
    nav0 = float((book0.qf * np.array([f.P0 for f in fspecs])).sum() + book0.cash + sum((book0.pending or {}).values()))
    x0 = np.array([(book0.qf[fin_names.index(r)] * unit[r] if r in FUNDED else book0.qd[din_names.index(r)] * unit[r]) / nav0
                   for r in routes])
    su = F.Setup(t, routes, fspecs, dspecs, fin_names, din_names, nav0, book0, x0, views, gen, adm, G, mH, q99, np.array(hc),
                 cost, nroll, mr, notional, unit, cfg, TECH_MANDATE, int(np.ceil(np.sqrt(len(routes)))), None, available0)
    sc = state_choice or {}
    su.v2 = {"models": used, "state_model_available": bool(sc.get("available", False)), "state_model_level": sc.get("level", "NOT_REQUESTED"),
             "state_model_episodes": sc.get("n_episodes", 0), "state_model_block_starts": sc.get("n_eligible_block_starts", 0)}
    if generation_only:
        from . import headroom_a as HA
        HA.check_generation_setup(su, n_gen, n_adm, seed0)

        from ..adaptive import semantics as D26
        pending_identity = D26.identity((panel.to_dict(), panel_now.to_dict(), state_choice, tgt))

        def produce_admission():
            if D26.identity((panel.to_dict(), panel_now.to_dict(), state_choice, tgt)) != pending_identity:
                raise StopCondition("HEADROOM_PENDING_PRODUCER_INPUT_CHANGED")
            output = []
            for mdl in RK.MODELS:
                k = RK.MODELS.index(mdl)
                raw = RK.simulate(mdl, panel, n_adm, H, HA.ADMISSION_SEEDS[k], L, panel_now, state_choice)
                if raw is None:
                    raise StopCondition("HEADROOM_ADMISSION_MODEL_MISSING")
                ps = SC.centre(raw, tgt, dspecs)
                fa, da, ma = to_engine(ps)
                mp = A.ModelPaths(f"{mdl}_ADMISSION", fa, da, ma)
                mp.headroom_producer_identity = dict(model=mdl, seed=HA.ADMISSION_SEEDS[k],
                    role='HEADROOM_A_ADMISSION', runtime=headroom_commitment.read()['runtime'],
                    source_risk_sha256=HA.file_hash(RK.__file__),
                    raw_indices=HA.tensor_identity(raw.indices) if raw.indices is not None else 'NOT_RECORDED_BY_PRODUCER',
                    raw_R=HA.tensor_identity(raw.R), centred_R=HA.tensor_identity(ps.R),
                    centring=D26.canonical(ps.centring), shift=D26.canonical(ps.shift))
                output.append(mp)
            return output

        su._headroom_synthetic = headroom_commitment.read()["mode"] == "SYNTHETIC_CORRECTNESS"
        su._headroom_commitment = headroom_commitment.identity
        su._headroom_admission_producer = produce_admission
        su._headroom_generation_identity = HA.generation_identity(su)
        su._headroom_admission_used = False
    return su


def attach_admission(su: F.Setup, seal, commitment) -> F.Setup:
    """Opt-in PhaseB: verify all identities/list before the first producer call.

    No earlier admission ensemble is reused. Failed/aborted production cannot be
    retried silently because the phase is marked consumed before the first call.
    """
    from . import headroom_a as HA
    HA.verify_phase_b(su, seal, commitment)
    su._headroom_admission_used = True
    adm = su._headroom_admission_producer()
    out = replace(su, adm=adm)
    out.v2 = dict(su.v2)
    out.headroom_context = HA.admission_context(out, seal, commitment)
    return out


# ---------------------------------------------------------------------------------------------- LP
def _vectors(su: F.Setup) -> dict:
    iv = su.views
    s = [SERIES.get(r, r) for r in su.routes]
    return {"c": np.array([float(iv.loc[k, "central"]) for k in s]), "dL": np.array([float(iv.loc[k, "d_long"]) for k in s]),
            "dS": np.array([float(iv.loc[k, "d_short"]) for k in s]), "tc": np.array([su.cost[r] for r in su.routes]),
            "roll": np.array([2 * su.cost[r] * su.nroll.get(r, 0) for r in su.routes]),
            "drag": np.array([0.0 if r in FUNDED else (su.mr[r] + su.q99[r]) * su.mH for r in su.routes])}


def reserve_drag(su: F.Setup) -> float:
    return (su.cfg["owner"]["liquidity_min_nav"] + su.cfg["r1"]["liquidity"]["sweep_hysteresis_nav"] / 2) * su.mH


def expected_net(su: F.Setup, x: np.ndarray, kappa: float = 0.0, traded: bool = True) -> float:
    """Expected net excess over the money market of a FIXED weight vector, per NAV0: expected excess minus roll costs,
    margin and reserve drag and (if `traded`) the cost of moving there from the actual book."""
    v = _vectors(su)
    x = np.asarray(x, float)
    out = float(v["c"] @ x - (v["roll"] + v["drag"]) @ np.abs(x) - reserve_drag(su))
    if traded:
        out -= float(v["tc"] @ np.abs(x - su.x0))
    if kappa > 0:
        dev = np.where(x > 0, v["dL"] * x, -v["dS"] * x)
        out -= kappa * float(np.sort(dev)[::-1][:su.gamma].sum())
    return out


def turnover_cost(su: F.Setup, x: np.ndarray) -> float:
    return float(_vectors(su)["tc"] @ np.abs(np.asarray(x, float) - su.x0))


def solve(su: F.Setup, spec: Spec, e: float | None, objective: str = "max_return", force: dict | None = None, *,
          ofz_upper_bound: float | None = None) -> dict:
    """`objective`: 'max_return' (CVaR of every model <= e) or 'min_risk' (minimise the worst model's CVaR)."""
    cfg = su.cfg
    J = len(su.routes)
    R = su.routes
    beta = cfg["owner"]["liquidity_min_nav"]
    h = cfg["r1"]["liquidity"]["sweep_hysteresis_nav"]
    lo = np.array([cfg["r3"]["bounds"][r]["lo"] for r in R])
    hi = np.array([cfg["r3"]["bounds"][r]["hi"] for r in R])
    if not spec.mandate["overlay_rule"]:
        for r, b in cfg["v2"].get("bounds_net_short_sensitivity", {}).items():
            if r in R:
                lo[R.index(r)], hi[R.index(r)] = float(b["lo"]), float(b["hi"])
    if spec.cap_gross is not None and "SMLT" in R:
        hi[R.index("SMLT")] = min(hi[R.index("SMLT")], spec.cap_gross)
    for r, w in spec.fixed.items():
        lo[R.index(r)] = hi[R.index(r)] = float(w)
    if ofz_upper_bound is not None:
        if ("OFZ_FUNDED" not in R or not np.isfinite(ofz_upper_bound)
                or not 0 <= ofz_upper_bound <= min(.40, cfg["r3"]["bounds"]["OFZ_FUNDED"]["hi"])
                or ("OFZ_FUNDED" in spec.fixed and spec.fixed["OFZ_FUNDED"] > ofz_upper_bound)):
            raise StopCondition("HEADROOM_OFZ_CONSTRUCTION_BOUND_INVALID")
        hi[R.index("OFZ_FUNDED")] = min(hi[R.index("OFZ_FUNDED")], ofz_upper_bound)
    v = _vectors(su)
    c, dL, dS, tc, roll, drag = v["c"], v["dL"], v["dS"], v["tc"], v["roll"], v["drag"]
    funded = np.array([r in FUNDED for r in R])
    models = list(su.G)
    S = {m: su.G[m].shape[0] for m in models}
    nx = 4 * J + 1 + J
    off, n = {}, nx
    for m in models:
        off[m] = n
        n += 1 + S[m]
    ZR = n                                         # worst-model CVaR (min_risk objective)
    n += 1
    XP, XM, TP, TM, Z, PP = 0, J, 2 * J, 3 * J, 4 * J, 4 * J + 1
    alpha = cfg["owner"]["es_alpha"]
    ret = np.zeros(n)
    ret[XP:XP + J] = -(c - roll - drag)
    ret[XM:XM + J] = c + roll + drag
    ret[TP:TP + J] = tc
    ret[TM:TM + J] = tc
    ret[Z] = spec.kappa * su.gamma
    ret[PP:PP + J] = spec.kappa
    if objective == "min_risk":
        obj = 1e-3 * ret
        obj[ZR] = 1.0
    else:
        obj = ret
    rows, cols, vals, b_ub = [], [], [], []
    nr = 0

    def add(cidx, cval, rhs):
        nonlocal nr
        rows.extend([nr] * len(cidx)); cols.extend(cidx); vals.extend(cval); b_ub.append(rhs); nr += 1
    eq = sp_.lil_matrix((J, n))
    b_eq = []
    for j in range(J):
        eq[j, XP + j] = 1; eq[j, XM + j] = -1; eq[j, TP + j] = -1; eq[j, TM + j] = 1
        b_eq.append(su.x0[j])
        add([XP + j, XM + j, PP + j, Z], [dL[j], dS[j], -1.0, -1.0], 0.0)
    const = su.mH - (beta + h / 2) * su.mH
    adj = np.where(funded, su.mH, 0.0)
    for m in models:
        Gm = su.G[m] - adj[None, :]
        k = off[m]
        Sm = S[m]
        blk = np.hstack([-Gm + roll + drag, Gm + roll + drag, np.tile(tc, (Sm, 1)), np.tile(tc, (Sm, 1))])
        xi = np.arange(Sm)
        r0 = nr
        rows.extend(np.repeat(xi + r0, 4 * J).tolist()); cols.extend(np.tile(np.arange(4 * J), Sm).tolist()); vals.extend(blk.ravel().tolist())
        rows.extend((xi + r0).tolist()); cols.extend([k] * Sm); vals.extend([-1.0] * Sm)
        rows.extend((xi + r0).tolist()); cols.extend((k + 1 + xi).tolist()); vals.extend([-1.0] * Sm)
        b_ub.extend([const] * Sm)
        nr += Sm
        ci = [k] + list(range(k + 1, k + 1 + Sm))
        cv = [1.0] + [1.0 / ((1 - alpha) * Sm)] * Sm
        if objective == "min_risk":
            add(ci + [ZR], cv + [-1.0], 0.0)
        else:
            add(ci, cv, float(e))
    pc = cfg["r3"]["portfolio_constraints"]
    ci, cv = [], []
    for j, rt in enumerate(R):
        if rt in FUNDED:
            ci += [XP + j]; cv += [1.0]
        else:
            ci += [XP + j, XM + j]; cv += [su.mr[rt] + su.q99[rt]] * 2
    add(ci, cv, pc["funded_plus_margin_plus_reserve_max_nav0"] - (beta + h / 2))
    add(list(range(XP, XP + J)), [1.0] * J, pc["long_exposure_max_nav0"])
    ci = [c_ for j, rt in enumerate(R) if rt not in FUNDED for c_ in (XP + j, XM + j)]
    add(ci, [1.0] * len(ci), pc["derivative_gross_max_nav0"])

    def net(names, scale=None):
        ci, cv = [], []
        for rt in names:
            if rt in R:
                j = R.index(rt)
                s_ = 1.0 if scale is None else scale[rt]
                ci += [XP + j, XM + j]; cv += [s_, -s_]
        return ci, cv
    issuer_scale = {"SMLT":1.0,"SMLT_FUT":su.unit_price["SMLT"] * next((d.pv0 for d in su.dspecs if d.name == "SMLT_FUT"),1.0) / su.unit_price.get("SMLT_FUT",su.unit_price["SMLT"])}
    ci, cv = net(("SMLT", "SMLT_FUT"), issuer_scale)
    if ci:
        add(ci, cv, spec.cap_net)                                                         # issuer ceiling (value)
        if spec.mandate["overlay_rule"]:
            if "SMLT_FUT" in R:                                                           # units: short contracts <= shares
                ci2, cv2 = net(("SMLT", "SMLT_FUT"), {"SMLT": 1.0 / su.unit_price["SMLT"], "SMLT_FUT": next(d.pv0 for d in su.dspecs if d.name == "SMLT_FUT") / su.unit_price["SMLT_FUT"]})
                sc_ = su.unit_price["SMLT"]
                add(ci2, [-v_ * sc_ for v_ in cv2], 0.0)
        else:
            add(ci, [-v_ for v_ in cv], -spec.mandate["issuer_net_min"])
    if spec.mandate["overlay_rule"]:
        for hedge, prot in (("EQ_HEDGE_FUT", ("EQ_FUNDED", "SMLT", "SMLT_FUT", "EQ_HEDGE_FUT")),
                            ("RATES_HEDGE_FUT", ("OFZ_FUNDED", "SMLT", "SMLT_FUT", "RATES_HEDGE_FUT"))):
            if hedge in R:
                ci, cv = net(prot, {rt:issuer_scale.get(rt,1.0) for rt in prot})
                add(ci, [-v_ for v_ in cv], 0.0)
    if spec.floor_target > 0:
        ci = [XP + j for j, rt in enumerate(R) if rt in spec.mandate["floor_set"]]
        add(ci, [-1.0] * len(ci), -spec.floor_target)
    A_ub = sp_.csr_matrix((vals, (rows, cols)), shape=(nr, n))
    bounds = [(max(lo[j], 0.0), max(hi[j], 0.0)) for j in range(J)] + [(max(-hi[j], 0.0), max(-lo[j], 0.0)) for j in range(J)]
    for rt, mn in (force or {}).items():
        j = R.index(rt)
        if hi[j] > 0:
            bounds[j] = (min(max(mn, bounds[j][0]), hi[j]), bounds[j][1])
        else:
            bounds[J + j] = (min(max(mn, bounds[J + j][0]), -lo[j]), bounds[J + j][1])
    bounds += [(0, None)] * (2 * J) + [(0, None)] + [(0, None)] * J
    for m in models:
        bounds += [(None, None)] + [(0, None)] * S[m]
    bounds += [(None, None)]
    res = linprog(obj, A_ub=A_ub, b_ub=np.array(b_ub), A_eq=eq.tocsr(), b_eq=np.array(b_eq), bounds=bounds, method="highs")
    if res.status != 0:
        return {"status": res.message, "x": None}
    z = res.x
    x = z[XP:XP + J] - z[XM:XM + J]
    x = np.where(np.abs(x) < 1e-9, 0.0, x)
    from .lk import candidate_guard, CandidateRejected, serialise
    try:
        lk_reports = candidate_guard(su, x)
    except CandidateRejected as exc:
        return {"status": str(exc), "x": x, "lk_rejected": True, "lk_reports": serialise(exc.reports)}
    return {"lk_reports": serialise(lk_reports), "status": "optimal", "x": x, "expected_net": expected_net(su, x, spec.kappa),
            "cvar": {m: float(z[off[m]] + z[off[m] + 1:off[m] + 1 + S[m]].sum() / ((1 - alpha) * S[m])) for m in models}}


# ---------------------------------------------------------------------------------------------- admission
def _rule(cfg: dict, budget: float, spec: Spec | None = None) -> A.Rule:
    return A.Rule(budget=budget, beta=cfg["owner"]["liquidity_min_nav"], max_breach=cfg["r1"]["admission"]["max_breach_probability"],
                  stress_mults=tuple(cfg["r1"]["admission"]["margin_stress_survival"]),
                  hysteresis=cfg["r1"]["liquidity"]["sweep_hysteresis_nav"],
                  issuer_floor=bool(spec and spec.mandate["overlay_rule"]),
                  funded_floor=spec.mandate["floor_hard"] if spec else 0.0,
                  emergency_floor_override=bool(spec and spec.floor_suspended), restoration_telemetry=True, recovery_clock=True,
                  floor_names=tuple(spec.mandate["floor_set"]) if spec else None)


def snap(su: F.Setup, x: np.ndarray) -> np.ndarray:
    """A weight equal to the current holding must round to the current quantity (never trade one unit by float error)."""
    x = np.asarray(x, float).copy()
    same = np.abs(x - su.x0) < 1e-9
    x[same] = su.x0[same] * (1 + 1e-10)
    return x


def book_key(ex: dict) -> tuple:
    return tuple(np.round(np.nan_to_num(ex["qf"], nan=-1.0), 6)) + tuple(np.round(ex["qd"], 6))


def executable(su: F.Setup, x: np.ndarray, delay: int, spec: Spec) -> dict:
    """Quantity conversion -> settlement/cash scaling -> executable quantities at reference prices."""
    from .lk import candidate_guard
    lk_reports = candidate_guard(su, x)
    ex = F.executable(su, x, delay)
    from .lk import serialise
    ex["lk_reports"] = serialise(lk_reports)
    ex["nominal_qf"], ex["nominal_qd"] = ex["qf"].copy(), ex["qd"].copy()
    from ..adaptive import semantics as D26
    from . import reference as REF
    H = int(su.cfg['r3']['frontier']['horizon_sessions'])
    if not hasattr(su,'d26_content_identity'):
        su.d26_content_identity = D26.content_identity()
    endpoint_f = np.asarray(su.book0.qf,float).copy()
    endpoint_d = np.asarray(su.book0.qd,float).copy()
    for o in ex['orders']:
        if o.qf is not None:
            endpoint_f = np.where(np.isfinite(o.qf),o.qf,endpoint_f)
        if o.qd is not None:
            endpoint_d = np.where(np.isfinite(o.qd),o.qd,endpoint_d)
    plan = D26.ExecutionPlan.create(origin=dict(date=su.t,book=su.book0),nav=su.nav0,
        endpoint=dict(qf=endpoint_f,qd=endpoint_d),orders=ex['orders'],spec=spec,
        terms=dict(funded=su.fspecs,derivatives=su.dspecs),
        mechanics=dict(reserve=ex['reserve_min'],waterfall=ex['wf_rank'],restoration=ex['restore_rank'],
                       beta=su.cfg['owner']['liquidity_min_nav'],
                       hysteresis=su.cfg['r1']['liquidity']['sweep_hysteresis_nav']),
        sources=su.d26_content_identity,horizon=H)
    ex['orders'] = [replace(o,plan=plan,stage_sequence=i,stage_check=_execution_check(su,replace(spec,mandate=dict(spec.mandate),fixed=dict(spec.fixed)))) for i,o in enumerate(ex['orders'])]
    ex['execution_plan'],ex['execution_plan_id'],ex['spec'] = plan,plan.execution_plan_id,spec
    ex['authorized_qf'],ex['authorized_qd'] = endpoint_f,endpoint_d
    if hasattr(su, '_headroom_commitment'):
        from . import headroom_a as HA
        ex = HA.certify_reference(su, ex)
    else:
        ex.update(REF.evaluate(su,ex,spec))
    ex['w_exec'] = {r:float(ex['qf'][su.fin_names.index(r)]*su.unit_price[r]/su.nav0)
                    if r in FUNDED else float(ex['reference_notional'][su.din_names.index(r)]/su.nav0)
                    for r in su.routes}
    ex['net_issuer_notional_nav0'] = ex['w_exec'].get('SMLT',0.)+ex['w_exec'].get('SMLT_FUT',0.)
    ex['directional_short'] = not spec.mandate['overlay_rule'] and ex['net_issuer'] < 0
    ex['score_plan_id'] = ex['turnover_plan_id'] = ex['reference_plan_id'] = ex['admission_plan_id'] = plan.execution_plan_id
    ex['score_basis'] = 'UNCHANGED_FIXED_VECTOR_PRIOR_CARRY_STATE_AT_COMPLETED_REFERENCE' if ex['reference_eligible'] else 'INELIGIBLE_REFERENCE_NO_ORDINARY_SCORE'

    return ex


def hard_checks(su: F.Setup, w: dict, spec: Spec, qf=None, qd=None, current: bool = False, funded_fraction: float | None = None,
                net_issuer=None, issuer_units=None, derivative_gross=None) -> dict:
    """Mandate and exposure rules on the executable (integer) book. For the CURRENT book the protective-overlay
    proportion rule is not tested (it constrains new targets; price drift of a held hedge is not a violation)."""
    b = su.cfg["r3"]["bounds"]
    tol = spec.mandate["tol"]
    issuer_scale = su.unit_price["SMLT"] * next((d.pv0 for d in su.dspecs if d.name == "SMLT_FUT"),1.0) / su.unit_price.get("SMLT_FUT",su.unit_price["SMLT"])
    net = w.get("SMLT", 0.0) + issuer_scale * w.get("SMLT_FUT", 0.0) if net_issuer is None else float(net_issuer)
    if net_issuer is None and qf is not None and hasattr(su,'nav0'):
        shares=float(qf[su.fin_names.index('SMLT')]) if 'SMLT' in su.fin_names else 0.
        hedge=float(qd[su.din_names.index('SMLT_FUT')])*next(d.pv0 for d in su.dspecs if d.name=='SMLT_FUT') if 'SMLT_FUT' in su.din_names else 0.
        units=shares+hedge if issuer_units is None else float(issuer_units)
        net=units*su.unit_price.get('SMLT',0.)/su.nav0
    funded = sum(v for r, v in w.items() if r in spec.mandate["floor_set"])
    long_exp = sum(v for r, v in w.items() if r in FUNDED) + sum(max(v, 0) for r, v in w.items() if r not in FUNDED)
    gross = sum(abs(v) for r, v in w.items() if r not in FUNDED) if derivative_gross is None else float(derivative_gross)
    out = {"issuer_net_ceiling": net <= spec.cap_net + tol, "long_exposure<=1": long_exp <= 1 + 1e-9,
           "derivative_gross<=1": gross <= 1 + 1e-9}
    if spec.floor_hard > 0:
        out["funded_investment_floor"] = (funded if funded_fraction is None else funded_fraction) >= spec.floor_hard - 1e-12
    if spec.cap_gross is not None:
        out["issuer_gross_ceiling"] = w.get("SMLT", 0.0) <= spec.cap_gross + tol
    if spec.mandate["overlay_rule"]:
        if qf is not None and "SMLT_FUT" in su.din_names:
            units = float(qf[su.fin_names.index("SMLT")] + qd[su.din_names.index("SMLT_FUT")] * next(d.pv0 for d in su.dspecs if d.name == "SMLT_FUT")) if issuer_units is None else float(issuer_units)
            out["no_naked_issuer_short_units"] = bool(units >= -1e-9)
    else:
        out["issuer_net_floor"] = net >= spec.mandate["issuer_net_min"] - tol
    if current:
        # M01/M02/M03: passive magnitudes are diagnostic; sign/unit safety stays live.
        out = {k:v for k,v in out.items() if k in ('no_naked_issuer_short_units','issuer_net_floor')}
        out['route_sign_safety'] = all((v>=-1e-9 if b[r]['lo']>=0 else v<=1e-9 if b[r]['hi']<=0 else True) for r,v in w.items())
        return out
    out["route_bounds"] = all(b[r]["lo"] - 1e-9 <= v <= b[r]["hi"] + 1e-9 for r, v in w.items() if r not in spec.fixed)
    if spec.mandate["overlay_rule"]:
        if "EQ_HEDGE_FUT" in w:
            out["equity_overlay_protective"] = w.get("EQ_FUNDED", 0.0) + net + w["EQ_HEDGE_FUT"] >= -tol
        if "RATES_HEDGE_FUT" in w:
            out["rates_overlay_protective"] = w.get("OFZ_FUNDED", 0.0) + net + w["RATES_HEDGE_FUT"] >= -tol
    return out


def _continuous_check(su,spec):
    def check(qf,qd,prices,eng,nav):
        passed = np.isfinite(qf).all(axis=1) & np.isfinite(qd).all(axis=1)
        passed &= (qf >= -1e-9).all(axis=1) & (np.abs(qd-np.round(qd)) <= 1e-9).all(axis=1)
        for r in su.routes:
            if r in su.din_names:
                quantity = qd[:,su.din_names.index(r)]
                bound=su.cfg['r3']['bounds'][r]
                if bound['lo']>=0: passed &= quantity>=-1e-9
                if bound['hi']<=0: passed &= quantity<=1e-9
        if spec.mandate['overlay_rule']:
            passed &= eng.issuer_units(qf,qd,eng._current_day)>=-1e-9
        return passed
    return check


def _allocation_drift(su,spec):
    """Diagnostics only; never enters a target applicability mask or verdict."""
    def check(qf,qd,prices,eng,nav,origin_nav):
        den=np.maximum(origin_nav,1e-12)
        notional=eng.execution.signed_notional() if eng.execution else qd*eng._Fn(eng._current_day)*eng._pv(eng._current_day)
        weights={r:(qf[:,su.fin_names.index(r)]*prices[:,su.fin_names.index(r)] if r in su.fin_names else
                   notional[:,su.din_names.index(r)])/den for r in su.routes}
        bad=np.zeros(len(nav),bool)
        for r,w in weights.items():
            b=su.cfg['r3']['bounds'][r]
            if r not in spec.fixed:bad |= (w<b['lo']-1e-9)|(w>b['hi']+1e-9)
        long=sum((w if r in FUNDED else np.maximum(w,0.) for r,w in weights.items()),np.zeros(len(nav)))
        gross=eng.execution.gross() if eng.execution else np.abs(notional).sum(axis=1)
        bad |= (long>1+1e-9)|(gross/den>1+1e-9)
        units=eng.issuer_units(qf,qd,eng._current_day)
        direct=units*prices[:,su.fin_names.index('SMLT')]/den if 'SMLT' in su.fin_names else np.zeros(len(nav))
        bad |= direct>spec.cap_net+spec.mandate['tol']
        for hedge,protected in [('EQ_HEDGE_FUT','EQ_FUNDED'),('RATES_HEDGE_FUT','OFZ_FUNDED')]:
            if spec.mandate['overlay_rule'] and hedge in weights:
                bad |= weights.get(protected,0.)+direct+weights[hedge] < -spec.mandate['tol']
        return bad
    return check


def _execution_check(su, spec):
    """M01--M05: actual stage marks, captured origin NAV, identical floor record."""
    from ..adaptive import semantics as D26
    def check(stage):
        N=len(stage.nav)
        rules={}
        details={}
        safety_names={'no_naked_issuer_short_units','issuer_net_floor'}
        for n in range(N):
            den=float(stage.decision_origin_nav[n])
            valid=np.isfinite(den) and den>0 and np.isfinite(stage.nav[n]) and stage.nav[n]>0
            den=max(den,1e-12)
            w={r:float(stage.qf[n,su.fin_names.index(r)]*stage.prices[n,su.fin_names.index(r)]/den)
                 if r in su.fin_names else float(stage.notional[n,su.din_names.index(r)]/den) for r in su.routes}
            spot=stage.prices[n,su.fin_names.index('SMLT')] if 'SMLT' in su.fin_names else 0.
            h=hard_checks(su,w,spec,stage.qf[n],stage.qd[n],
                funded_fraction=float(stage.floor_numerator[n]/max(stage.nav[n],1e-12)),
                net_issuer=stage.issuer_units[n]*spot/den,issuer_units=stage.issuer_units[n],
                derivative_gross=stage.derivative_gross[n]/den)
            h['funded_investment_floor']=bool(stage.floor_predicate[n])
            h['stage_NAV_evaluable']=valid
            def detail(name,numerator,denominator,threshold,epsilon,economic=0.):
                row=details.setdefault(name,[np.zeros(N),np.ones(N),threshold,epsilon,economic])
                row[0][n]=numerator;row[1][n]=denominator
            detail('funded_investment_floor',stage.floor_numerator[n],stage.nav[n],spec.floor_hard,1e-12)
            detail('issuer_net_ceiling',stage.issuer_units[n]*spot,stage.decision_origin_nav[n],spec.cap_net,0.,spec.mandate['tol'])
            detail('no_naked_issuer_short_units',stage.issuer_units[n],1.,0.,1e-9)
            detail('long_exposure<=1',sum((v if r in FUNDED else max(v,0.) for r,v in w.items()))*den,den,1.,1e-9)
            detail('derivative_gross<=1',stage.derivative_gross[n],den,1.,1e-9)
            for hedge,protected,name in [('EQ_HEDGE_FUT','EQ_FUNDED','equity_overlay_protective'),('RATES_HEDGE_FUT','OFZ_FUNDED','rates_overlay_protective')]:
                if name in h:detail(name,(w.get(protected,0.)+stage.issuer_units[n]*spot/den+w.get(hedge,0.))*den,den,0.,0.,spec.mandate['tol'])
            if 'issuer_gross_ceiling' in h:detail('issuer_gross_ceiling',w.get('SMLT',0.)*den,den,spec.cap_gross,0.,spec.mandate['tol'])
            for r,value in w.items():
                if r in spec.fixed:continue
                bound=su.cfg['r3']['bounds'][r]
                for side,ok in [('lower',value>=bound['lo']-1e-9),('upper',value<=bound['hi']+1e-9)]:
                    name='route_'+side+':'+r;h[name]=bool(ok)
                    detail(name,value*den,den,bound['lo' if side=='lower' else 'hi'],1e-9)
            for name,ok in h.items():
                mask=(bool(stage.floor_applicable[n]) if name=='funded_investment_floor'
                      else True if name in safety_names or name=='stage_NAV_evaluable' else bool(stage.ordinary[n]))
                results,app=rules.setdefault(name,(np.full(N,'N/A',dtype='<U16'),np.zeros(N,bool)))
                app[n]=mask
                results[n]='PASS' if ok else 'FAIL' if valid else 'NOT_EVALUABLE'
                if not mask:results[n]='N/A'
        if stage.intent=='SEVERE_ISSUER_EXIT':
            zero=stage.issuer_units==0
            if 'SMLT' in su.fin_names:zero &= stage.qf[:,su.fin_names.index('SMLT')]==0
            if 'SMLT_FUT' in su.din_names:zero &= stage.qd[:,su.din_names.index('SMLT_FUT')]==0
            rules['severe_zero_quantity']=(np.where(zero,'PASS','FAIL'),stage.local_attempt)
        records=tuple((name,D26.frozen(result),D26.frozen(mask)) for name,(result,mask) in sorted(rules.items()))
        passed=np.ones(N,bool)
        for _,result,mask in records:passed &= ~mask | (result=='PASS')
        numeric=tuple((name,D26.frozen(v[0]),D26.frozen(v[1]),*v[2:]) for name,v in sorted(details.items()))
        return passed,records,numeric
    # This is the sole registration site: the returned native function itself,
    # with the actual factory closure referents and canonical module environment.
    def origin_ref(value):
        try:
            return ref(value)
        except TypeError:
            # Some small legacy synthetic setups are SimpleNamespace objects.
            return lambda: value
    _NATIVE_STAGE_CHECKS[check] = (check.__code__, check.__globals__,
        tuple(origin_ref(c.cell_contents) for c in check.__closure__))
    return check


def _native_stage_callback_origin(fn):
    """Read-only factory provenance check; no callable registration API."""
    if not isinstance(fn, FunctionType):
        raise StopCondition('HEADROOM_NON_NATIVE_STAGE_CALLBACK_MISSING')
    codes = [c for c in _execution_check.__code__.co_consts
             if hasattr(c, 'co_code') and c.co_name == 'check']
    if (not codes or fn.__code__ is not codes[0]
            or fn.__globals__ is not _execution_check.__globals__):
        raise StopCondition('HEADROOM_NON_NATIVE_STAGE_CALLBACK_ENVIRONMENT')
    record = _NATIVE_STAGE_CHECKS.get(fn)
    if (record is None or record[0] is not fn.__code__ or record[1] is not fn.__globals__
            or fn.__closure__ is None or len(fn.__closure__) != len(record[2])
            or any(expected() is None or expected() is not cell.cell_contents
                   for expected, cell in zip(record[2], fn.__closure__))):
        raise StopCondition('HEADROOM_NON_NATIVE_STAGE_CALLBACK_FACTORY_OR_REFERENT')
    return dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__)))


def _headroom_order_supplier(su, ex, orders, spec):
    """Private per-invocation directives, from existing native authority only."""
    if not (hasattr(su, '_headroom_commitment') or '_headroom_native_reference' in ex):
        return None
    from . import headroom_a as HA

    def validate_source():
        HA.require_native_reference(ex)
        HA.require_native_reference(dict(ex, orders=orders, spec=spec))

    def detached(items):
        def quantity(value):
            if value is None:
                return None
            out = np.array(value, copy=True)
            out.setflags(write=False)
            return out
        # All scalar mechanics and the original frozen plan/function referents
        # remain exact. Neither the list, Orders nor quantity storage is shared.
        return [replace(o, qf=quantity(o.qf), qd=quantity(o.qd)) for o in items]

    validate_source()
    baseline = tuple(detached(orders))  # private; never exposed to Engine/observer

    def snapshot():
        # An observer's alias mutation cannot silently become new authority.
        validate_source()
        actual = detached(baseline)  # fresh even if a previous consumer retained aliases
        HA.require_native_reference(dict(ex, orders=actual, spec=spec))
        return actual
    return snapshot


def _engine_rows(su: F.Setup, ex: dict, orders: list, has_deriv: bool, spec: Spec, *, observer=None) -> list:
    order_supplier = _headroom_order_supplier(su, ex, orders, spec)
    rule = _rule(su.cfg, 1.0, spec)
    rule.execution_check = _execution_check(su,spec)
    rule.continuous_check = _continuous_check(su,spec)
    rule.drift_check = _allocation_drift(su,spec)
    if has_deriv:
        if observer is None and order_supplier is None:
            return A.evaluate_all(su.adm, su.book0, orders, rule, ex["reserve_min"], ex["wf_rank"], ex["restore_rank"])
        rows = []
        for mp in su.adm:
            for mult in (1.0, *rule.stress_mults):
                actual_orders = order_supplier() if order_supplier is not None else orders
                m = A.evaluate(mp, su.book0, actual_orders, rule, mult, ex["reserve_min"], ex["wf_rank"], ex["restore_rank"])
                res = m.pop("_res")
                if observer is not None:
                    observer.observe(res, mp.name, mult)
                rows.append({"model": mp.name, "margin_multiplier": mult, **m,
                             "ES95_nav0": m["ES95"] / m["NAV0"],
                             "frac_contract_path_days": int(res["frac_contract_days"].sum())})
        return rows
    rows = []                       # no derivative is ever held: posted margin is zero, the multipliers cannot matter
    for mp in su.adm:
        actual_orders = order_supplier() if order_supplier is not None else orders
        m = A.evaluate(mp, su.book0, actual_orders, rule, 1.0, ex["reserve_min"], ex["wf_rank"], ex["restore_rank"])
        res = m.pop("_res")
        if observer is not None:
            observer.observe(res, mp.name, 1.0)
        row = {"model": mp.name, "margin_multiplier": 1.0, **m, "ES95_nav0": m["ES95"] / m["NAV0"],
               "frac_contract_path_days": int(res["frac_contract_days"].sum())}
        rows.append(row)
        rows.extend({**row, "margin_multiplier": mult} for mult in rule.stress_mults)
    return rows


def det_survival(su: F.Setup, x: np.ndarray, spec: Spec, ex: dict, no_trade: bool = False, *,
                 execution_case_provider=None, observer=None) -> list:
    """One engine run per deterministic path and margin multiplier (no probabilities): does the book survive the path?"""
    cfg = su.cfg
    st = cfg["stress"]
    exs = {spec.delay: ex}
    rows = []
    for name, sc in st["scenarios"].items():
        d = max(spec.delay, int(sc.get("order_delay_sessions", spec.delay)))
        if execution_case_provider is None:
            if d not in exs:
                exs[d] = executable(su, x, d, spec)
            e_ = exs[d]
        else:
            e_ = execution_case_provider(name, sc, ex, spec)
            from . import headroom_a as HA
            HA.require_case_reference(e_)
            if e_["spec"] != spec:
                raise StopCondition("HEADROOM_CASE_SPEC_MISMATCH")
        if execution_case_provider is not None and not e_['reference_eligible']:
            # Valid native negative S09 reference: preserve FAIL; do not recertify or fabricate a score.
            for mult in st['margin_multipliers']:
                rows.append(dict(path=name, margin_multiplier=mult,
                    FULL_DETERMINISTIC_ADMISSION_PASS='FAIL',
                    REFERENCE_CHILD_ELIGIBILITY='FAIL', reference_plan_state=e_['reference_plan_state'],
                    reference_plan_id=e_['execution_plan_id'], reference_snapshot_id=e_['reference_state_snapshot_id'],
                    reason='NATIVE_S09_REFERENCE_INELIGIBLE', native_reference_engine_rerun=False))
            continue
        order_supplier = _headroom_order_supplier(su, e_, e_["orders"], spec)
        ps = SS.path(su, sc, st["route_groups"])
        mmH = float(np.prod(1 + ps.mmf_daily))
        for mult in st["margin_multipliers"]:
            fin, din, mkt = SS.to_market(su, ps, sc, mult)
            prm = E.Params(beta=cfg["owner"]["liquidity_min_nav"], mult=mult, reserve_frac=e_["reserve_min"],
                           hysteresis=cfg["r1"]["liquidity"]["sweep_hysteresis_nav"], wf_rank=e_["wf_rank"], restore_rank=e_["restore_rank"], issuer_floor=spec.mandate["overlay_rule"],
                           funded_floor=spec.mandate["floor_hard"], emergency_floor_override=spec.floor_suspended, restoration_telemetry=True,
                           execution_check=_execution_check(su,spec),continuous_check=_continuous_check(su,spec),drift_check=_allocation_drift(su,spec),floor_names=tuple(spec.mandate["floor_set"]))
            actual_orders = order_supplier() if order_supplier is not None else e_["orders"]
            res = E.Engine(fin, din, mkt, prm).run(su.book0, orders=[] if no_trade else actual_orders)
            if observer is not None:
                observer.observe(res, name, mult)
            nav = res["nav"][0]
            peak = np.maximum.accumulate(nav)
            stages=res['execution_records']
            count=sum(int(s.ordinary.sum()) for s in stages)
            target='N/A' if not count else 'PASS' if all(not ids for s in stages for ids in s.failure_ids) else 'FAIL'
            cash_ok=not bool(res['unresolved'].any())
            safety_ok=bool(res['continuous_safety_pass'].all())
            event_ok=not any('SEVERE_EXIT_DUE_BLOCKED' in r['status'] for r in res['severe_exit_records'])
            full=cash_ok and safety_ok and target in ('PASS','N/A') and event_ok
            rows.append({'DETERMINISTIC_CASH_MARGIN_SURVIVAL':'PASS' if cash_ok else 'FAIL',
                         'DETERMINISTIC_CONTINUOUS_SAFETY_PASS':'PASS' if safety_ok else 'FAIL',
                         'DETERMINISTIC_TARGET_STAGE_COMPLIANCE':target,'target_stage_applicable_count':count,
                         'ALLOCATION_DRIFT_DIAGNOSTIC':bool(res['allocation_drift_diagnostic'].any()),
                         'FULL_DETERMINISTIC_ADMISSION_PASS':'PASS' if full else 'FAIL',
                         "path": name, "margin_multiplier": mult, "unresolved_deficit": bool(res["unresolved"][0]),
                         "actual_constraints_pass":bool(res["actual_constraints_pass"].all()),
                         "pnl_nav0": float(nav[-1] / nav[0] - 1), "excess_over_money_market_nav0": float(nav[-1] / nav[0] - mmH),
                         "min_nav_nav0": float(nav.min() / nav[0]), "max_drawdown": float(((peak - nav) / peak).max()),
                         "min_available_nav": float((res["available"][0] / nav).min()), "breach_days": int(res["breach_days"][0]),
                         "forced_contracts": float(res["forced_contracts"][0])})
    return rows


def admit(su: F.Setup, x: np.ndarray, spec: Spec, cache: dict | None = None) -> dict:
    """Executable integer book -> full engine on every model and multiplier -> deterministic paths -> hard checks."""
    xs = snap(su, x)
    from .lk import candidate_guard
    candidate_guard(su, xs)
    from ..adaptive import semantics as D26
    ex = executable(su,xs,spec.delay,spec)
    key = (ex["execution_plan_id"],D26.identity(spec),D26.VERSION,ex["reference_state_snapshot_id"])
    from . import reference as REF
    expected_plan_id, expected_snapshot_id = ex['execution_plan_id'], ex['reference_state_snapshot_id']
    if cache is not None and key in cache:
        ex = dict(cache[key])
        certification = REF.certify_cached(ex)
        if not certification and (ex.get('execution_plan_id') != expected_plan_id
                                  or ex.get('reference_state_snapshot_id') != expected_snapshot_id):
            certification = REF._invalid('NOT_EVALUABLE_PLAN_ID_MISMATCH')
        ex.update(certification)
    else:
        has_deriv = bool(np.any(ex["qd"] != 0) or np.any(np.asarray(su.book0.qd) != 0))
        ex["rows"] = _engine_rows(su, ex, ex["orders"], has_deriv, spec)
        ex["det"] = det_survival(su, xs, spec, ex)
        ex["x"] = xs
        if cache is not None:
            cache[key] = dict(ex)
    ex.update(REF.certify_cached(ex))
    completed = REF._completion(ex, ex.get('execution_plan_id'))
    if ex['reference_plan_state'] == 'COMPLETED' and completed is not None:
        stage = completed['stage']
        origin_nav = ex['execution_plan'].decision_origin_nav
        ex["hard"] = hard_checks(su, ex["w_exec"], spec, ex["qf"], ex["qd"],
            funded_fraction=ex["funded_fraction_at_execution"],
            net_issuer=ex['net_issuer'], issuer_units=ex['net_issuer_units'],
            derivative_gross=float(stage.derivative_gross[0] / origin_nav))
    else:
        # Missing or invalid completion cannot supply canonical final predicates.
        ex["hard"] = {}
    if not ex['reference_eligible']:
        ex['score_basis'] = 'INELIGIBLE_REFERENCE_NO_ORDINARY_SCORE'
        ex['target_published'] = False
    ex["hard"]["deterministic_stress_survival"] = all(r["FULL_DETERMINISTIC_ADMISSION_PASS"] == "PASS" for r in ex["det"])
    ex["hard"]["reference_plan_eligibility"] = ex["reference_eligible"]
    return ex


def current_eval(su: F.Setup, spec: Spec) -> dict:
    """Engine evaluation of the book AS IT IS (no order)."""
    from .lk import actual_guard, serialise
    actual_reports = actual_guard(su)
    xs = snap(su, su.x0)
    from ..adaptive import semantics as D26
    ex = F.executable(su,xs,spec.delay)
    ex['spec']=spec
    ex['execution_plan_id']=D26.identity(dict(semantics=D26.VERSION,kind='HOLD',origin=su.book0,
                                             origin_nav=su.nav0,date=su.t,spec=spec))
    ex['reference_plan_state']='HOLD_NO_TARGET'
    ex['REFERENCE_TARGET_MANDATE_PASS']='N/A'
    ex['reference_eligible']=False
    ex["lk_reports"] = serialise(actual_reports)
    ex["orders"] = []
    ex["rows"] = _engine_rows(su, ex, [], bool(np.any(np.asarray(su.book0.qd) != 0)), spec)
    ex["det"] = det_survival(su, xs, spec, ex, no_trade=True)
    ex["w_exec"] = {r: float(su.x0[j]) for j, r in enumerate(su.routes)}
    ex["x"] = su.x0.copy()
    ex["qf"], ex["qd"] = np.asarray(su.book0.qf,float).copy(), np.asarray(su.book0.qd,float).copy()
    ex["execution_nav"] = su.nav0
    ex["execution_cash"] = float(su.book0.cash)
    ex["execution_pending"] = float(sum(su.book0.pending.values()))
    ex["execution_margin"] = float(sum(abs(ex["qd"][j])*d.F0*d.pv0*d.mr for j,d in enumerate(su.dspecs)))
    ex["funded_fraction_at_execution"] = float(sum(ex["qf"][su.fin_names.index(r)]*su.unit_price[r] for r in spec.mandate["floor_set"] if r in su.fin_names)/su.nav0)
    shares = ex["qf"][su.fin_names.index("SMLT")] if "SMLT" in su.fin_names else 0.0
    hedge = origin_issuer_units(su)-shares
    ex["net_issuer_units"] = float(shares+hedge)
    ex["net_issuer"] = float((shares+hedge)*su.unit_price.get("SMLT",0.0)/su.nav0)
    ex["hard"] = hard_checks(su, ex["w_exec"], spec, np.asarray(su.book0.qf, float), np.asarray(su.book0.qd, float), current=True,
                            net_issuer=ex['net_issuer'],issuer_units=ex['net_issuer_units'])
    ex['issuer_cap_status']='ISSUER_CAP_REDUCTION_DUE' if ex['net_issuer']>spec.cap_net+spec.mandate['tol'] else 'PASS'
    ex['hard']['issuer_event_reduction_due']=ex['issuer_cap_status']=='PASS'
    ex['hard']['current_regime_eligibility']=bool(spec.floor_suspended or np.all(np.asarray(su.book0.regime)=='ORDINARY'))
    if spec.cap_gross==0:
        ex['hard']['severe_issuer_exit_due']=ex['net_issuer_units']==0
    ex["hard"]["deterministic_stress_survival"] = all(r['FULL_DETERMINISTIC_ADMISSION_PASS']=='PASS' for r in ex['det'])
    return ex


def origin_issuer_units(su):
    """Actual origin shares and certified per-series contracts; no price ratio."""
    from . import reference
    fin,din,m=reference.market(su.fspecs,su.dspecs,1)
    eng=E.Engine(fin,din,m,E.Params(beta=su.cfg['owner']['liquidity_min_nav']))
    qf=np.asarray(su.book0.qf,float).reshape(1,-1)
    qd=np.asarray(su.book0.qd,float).reshape(1,-1)
    if m.execution is not None:
        from .execution import Execution
        eng.execution=Execution(eng,qd,su.book0.series,su.book0.capacity)
    return float(eng.issuer_units(qf,qd,0)[0])


def verdict(su: F.Setup, ex: dict, budget: float) -> A.Admission:
    return A.judge(ex["rows"], _rule(su.cfg, budget, ex.get("spec")), ex["hard"])


def es_of(su: F.Setup, ex: dict, key: str = "ES95") -> float:
    return max(r[key] for r in ex["rows"] if r["margin_multiplier"] == 1.0) / su.nav0


def candidate(su: F.Setup, spec: Spec, x: np.ndarray, source: str, e: float | None, cache: dict | None, cid: str) -> dict:
    from .lk import actual_guard, CandidateRejected, TIER, ACTUAL_AMBIGUOUS, NOT_CERTIFIED, serialise
    actual_guard(su)  # Outside the catch: an unsafe held book always stops.
    try:
        ex = admit(su, x, spec, cache)
    except StopCondition as exc:
        if not isinstance(exc, CandidateRejected) and not str(exc).startswith((TIER, ACTUAL_AMBIGUOUS)):
            raise
        reason = str(exc).replace(ACTUAL_AMBIGUOUS, NOT_CERTIFIED)
        return {"id": cid, "source": source, "es_target": e, "x": np.asarray(x,float),
                "passed": False, "reasons": [reason], "admission_status": reason.split(':')[0],
                "lk_rejected": True, "lk_reports": serialise(exc.reports),
                "expected_net": None, "es95": None, "es99": None, "turnover_cost": None,
                "key": ('LK_REJECTED', tuple(np.asarray(x,float)))}
    v = verdict(su, ex, spec.budget)
    xe = np.array([ex["w_exec"][r] for r in su.routes])
    return {"id": cid, "source": source, "es_target": e, "x": np.asarray(ex["x"], float), "x_exec": xe, "ex": ex, "passed": v.passed and ex['reference_eligible'],
            "reasons": v.reasons, "admission_status": v.status, "expected_net": expected_net(su, xe, spec.kappa) if ex["reference_eligible"] else None, "es95": es_of(su, ex), "es99": es_of(su, ex, "ES99"),
            "turnover_cost": turnover_cost(su, xe) if ex["reference_eligible"] else None, "key": book_key(ex),
            "ordinary_score_eligible":ex["reference_eligible"],
            "execution_plan_id":ex.get('execution_plan_id'),
            "scored_reference_state_snapshot_id":ex.get('reference_state_snapshot_id'),
            "target_published":False}


# ---------------------------------------------------------------------------------------------- frontier
def refine(su: F.Setup, spec: Spec, a: dict, b: dict, cache: dict | None, log: list, tag: str) -> list:
    """Deterministic one-dimensional bisection in exposure space between a PASS (`a`) and a FAIL (`b`) candidate.
    Stops when the best passing book is within 0.25% NAV of the ES budget, when the integer book no longer changes, or
    after 12 steps. Every step is re-rounded and re-admitted by the full engine."""
    rc = su.cfg["v2"]["frontier"]["refinement"]
    gap = float(rc["stop_es_gap_nav"])
    lo, hi = 0.0, 1.0
    seen = {a["key"], b["key"]}
    out, best = [], a
    stop = "ES_WITHIN_GAP_OF_BUDGET" if spec.budget - a["es95"] < gap else None
    it = 0
    while stop is None:
        if it >= int(rc["max_iterations_per_bracket"]):
            stop = "MAX_ITERATIONS"
            break
        it += 1
        mid = 0.5 * (lo + hi)
        c = candidate(su, spec, a["x"] + mid * (b["x"] - a["x"]), "REFINEMENT", None, cache, f"{tag}#r{it}")
        if c.get("lk_rejected"):
            hi = mid
            log.append({"bracket": tag, "iteration": it, "lambda": mid, "verdict": c["admission_status"],
                        "fail_reasons": " | ".join(c["reasons"]), "lk_reports": c["lk_reports"]})
            out.append(c)
            continue
        if c["key"] in seen and rc["stop_if_integer_book_unchanged"]:
            log.append({"bracket": tag, "iteration": it, "lambda": mid, "verdict": "SAME_INTEGER_BOOK", "es95_nav0": c["es95"],
                        "expected_net_nav0": c["expected_net"]})
            stop = "INTEGER_BOOK_UNCHANGED"
            break
        seen.add(c["key"])
        log.append({"bracket": tag, "iteration": it, "lambda": mid, "verdict": "PASS" if c["passed"] else "FAIL", "es95_nav0": c["es95"],
                    "expected_net_nav0": c["expected_net"], "fail_reasons": " | ".join(c["reasons"])})
        if c["passed"]:
            lo, best = mid, c
            out.append(c)
            if spec.budget - c["es95"] < gap:
                stop = "ES_WITHIN_GAP_OF_BUDGET"
        else:
            hi = mid
    log.append({"bracket": tag, "iteration": "STOP", "lambda": lo, "verdict": stop, "es95_nav0": best["es95"],
                "expected_net_nav0": best["expected_net"], "gap_to_budget_nav0": spec.budget - best["es95"], "steps": it})
    return out


def failure_classes(reasons: list) -> set:
    """Classify full-admission blockers without confusing floor, ES and liquidity."""
    out = set()
    for reason in reasons:
        if "executable funded investment floor failed" in reason or reason == "hard constraint failed: funded_investment_floor":
            out.add("EXECUTABLE_FUNDED_FLOOR_FAIL")
        elif "ES95" in reason and "> budget" in reason:
            out.add("ES_BUDGET_FAIL")
        elif "liquidity breach probability" in reason:
            out.add("LIQUIDITY_BREACH_FAIL")
        elif "unresolved deficit" in reason or "unserviceable obligation" in reason:
            out.add("UNRESOLVED_DEFICIT_FAIL")
        elif reason == "hard constraint failed: deterministic_stress_survival":
            out.add("DETERMINISTIC_STRESS_FAIL")
        elif "hard constraint failed" in reason or "MANDATE_FAIL" in reason:
            out.add("MANDATE_FAIL")
        else:
            out.add("OTHER")
    return out


def opposing_constraint_bridge(su: F.Setup, spec: Spec, a: dict, b: dict, cache: dict | None, log: list, tag: str) -> list:
    """Bounded rescue only on an existing floor-only FAIL -> ES-only FAIL segment.

    Each trial uses normal integer execution and full admission. No rescue is
    attempted for a third constraint. A first admitted book is retained; the
    usual lexicographic selector decides whether to use it.
    """
    if (a.get("lk_rejected") or b.get("lk_rejected") or a["passed"] or b["passed"] or
        failure_classes(a["reasons"]) != {"EXECUTABLE_FUNDED_FLOOR_FAIL"} or
        failure_classes(b["reasons"]) != {"ES_BUDGET_FAIL"} or
        b["expected_net"] <= a["expected_net"]):
        return []
    lo, hi, seen, out = 0., 1., {a["key"], b["key"]}, []
    stop, evaluations = "NO_RESCUE_MAX_ITERATIONS", 0
    limit = min(12, int(su.cfg["v2"]["frontier"]["refinement"]["max_iterations_per_bracket"]))
    for iteration in range(1, limit + 1):
        mid = (lo + hi) / 2
        c = candidate(su, spec, a["x"] + mid * (b["x"] - a["x"]),
                      "OPPOSING_CONSTRAINT_BRIDGE_REFINEMENT", None, cache, f"{tag}#b{iteration}")
        evaluations += 1
        if c.get("lk_rejected"):
            hi = mid
            out.append(c)
            log.append({"bracket": tag, "method": "OPPOSING_CONSTRAINT_BRIDGE_REFINEMENT",
                        "iteration": iteration, "lambda": mid, "verdict": c["admission_status"],
                        "fail_reasons": " | ".join(c["reasons"]), "lk_reports": c["lk_reports"]})
            continue
        classes = failure_classes(c["reasons"])
        repeated = c["key"] in seen
        log.append({"bracket": tag, "method": "OPPOSING_CONSTRAINT_BRIDGE_REFINEMENT", "iteration": iteration,
                    "lambda": mid, "verdict": "PASS" if c["passed"] else "FAIL", "es95_nav0": c["es95"],
                    "expected_net_nav0": c["expected_net"], "fail_reasons": " | ".join(c["reasons"]),
                    "repeated_integer_book": repeated})
        if repeated:
            stop = "NO_RESCUE_INTEGER_BOOK_UNCHANGED"
            break
        if c["passed"]:
            out.append(c)
            stop = "ADMITTED_INTERIOR_BOOK"
            break
        seen.add(c["key"])
        if classes == {"EXECUTABLE_FUNDED_FLOOR_FAIL"}:
            lo = mid
        elif classes == {"ES_BUDGET_FAIL"}:
            hi = mid
        else:
            stop = "NO_RESCUE_THIRD_CONSTRAINT"
            break
    log.append({"bracket": tag, "method": "OPPOSING_CONSTRAINT_BRIDGE_REFINEMENT", "iteration": "STOP",
                "verdict": stop, "steps": evaluations})
    return out


def frontier(su: F.Setup, spec: Spec, cache: dict | None = None) -> dict:
    """Candidates: min-risk anchor (the funded-floor anchor of the invested mandate), LP grid, refinement points, and the
    operational safe anchor when the mandate allows it (floor 0)."""
    fc = su.cfg["v2"]["frontier"]
    cands, log, infeasible, keys = [], [], [], set()

    def push(c):
        if c["key"] in keys:
            return None
        keys.add(c["key"])
        cands.append(c)
        return c
    chain = []
    a = solve(su, spec, None, "min_risk")
    if a["x"] is not None:
        c = push(candidate(su, spec, a["x"], "FLOOR_ANCHOR_MIN_RISK" if spec.floor_target > 0 else "MIN_RISK_ANCHOR", None, cache, "anchor"))
        if c is not None:
            chain.append(c)
    else:
        infeasible.append({"es_target": None, "objective": "min_risk", "status": a["status"]})
    for f_ in fc["es_grid_fraction_of_budget"]:
        e = float(f_) * spec.budget
        s = solve(su, spec, e)
        if s["x"] is None:
            infeasible.append({"es_target": e, "objective": "max_return", "status": s["status"]})
            continue
        c = push(candidate(su, spec, s["x"], "LP", e, cache, f"lp@{e:.4f}"))
        if c is not None:
            chain.append(c)
    for c1, c2 in zip(chain[:-1], chain[1:]):
        if c1["passed"] != c2["passed"]:
            p_, f_ = (c1, c2) if c1["passed"] else (c2, c1)
            for c in refine(su, spec, p_, f_, cache, log, f"{p_['id']}->{f_['id']}"):
                push(c)
        elif not c1["passed"] and not c2["passed"]:
            for c in opposing_constraint_bridge(su, spec, c1, c2, cache, log, f"{c1['id']}->{c2['id']}"):
                push(c)
    if spec.floor_target <= 0 and not spec.fixed:
        sa = safe_anchor(su, spec, cache)
        if sa["key"] in keys:                        # same integer book as another candidate: keep both labels
            sa["id"] = "safe_anchor(=" + next(c["id"] for c in cands if c["key"] == sa["key"]) + ")"
        cands.append(sa)
        keys.add(sa["key"])
    relevant = [c for c in cands if c["source"] != "OPERATIONAL_SAFE_ANCHOR"]
    lk_status = ("ALL_ECONOMICALLY_RELEVANT_CANDIDATES_LK_UNCERTIFIED_OR_BREACHING"
                 if relevant and all(c.get("lk_rejected") for c in relevant) else "LK_CANDIDATES_CERTIFIED_OR_OTHER_ADMISSION_FAILURES")
    return {"candidates": cands, "refinement_log": log, "lp_infeasible": infeasible, "lk_status": lk_status}


def safe_anchor(su: F.Setup, spec: Spec, cache: dict | None = None) -> dict:
    """OPERATIONAL_SAFE_ANCHOR: every risky holding sold / closed at the next session; money-market units by the cash rule
    after settlement; the same 5% immediate-liquidity rule, zero-yield immediate cash, costs and execution delay."""
    sp = replace(spec, floor_suspended=True, fixed={})
    c = candidate(su, sp, np.zeros(len(su.routes)), "OPERATIONAL_SAFE_ANCHOR", None, cache, "safe_anchor")
    return c


def select(cands: list, cfg: dict) -> dict | None:
    """Lexicographic: max expected net excess; within 10 bp lower ES95; within 5 bp of that lower turnover cost."""
    adm = [c for c in cands if c["passed"]]
    if not adm:
        return None
    lx = cfg["v2"]["frontier"]["lexicographic"]
    top = max(c["expected_net"] for c in adm)
    a1 = [c for c in adm if c["expected_net"] >= top - float(lx["step2_within_nav0"])]
    es_min = min(c["es95"] for c in a1)
    a2 = [c for c in a1 if c["es95"] <= es_min + float(lx["step3_within_nav0"])]
    return min(a2, key=lambda c: (c["turnover_cost"], -c["expected_net"], c["es95"], c["id"]))


def min_es_fallback(cands: list) -> dict | None:
    """Hedge-keep comparators only: the lowest-ES book with no unresolved deficit when no book is admitted."""
    ok = [c for c in cands if not c.get("lk_rejected") and all(r["p_unresolved"] == 0 for r in c["ex"]["rows"]) and c["ex"]["hard"].get("deterministic_stress_survival", True)]
    return min(ok, key=lambda c: (c["es95"], c["id"])) if ok else None


def summarise(su: F.Setup, ex: dict) -> dict:
    rows = [r for r in ex["rows"] if r["margin_multiplier"] == 1.0]
    inherited = F.summarise(su, ex)
    inherited["risk_path_mean_excess"] = inherited.pop("engine_mean_excess_worst_nav0")
    inherited["risk_path_mean_pnl_min_nav0"] = inherited.pop("engine_mean_pnl_min_nav0")
    deterministic=ex.get('det',[])
    def det_status(key):
        values=[r.get(key,'NOT_EVALUABLE') for r in deterministic]
        return 'PASS' if values and all(v=='PASS' for v in values) else 'N/A' if values and all(v=='N/A' for v in values) else 'FAIL'
    return {**inherited,
            'execution_plan_id':ex.get('execution_plan_id'),
            'REFERENCE_TARGET_MANDATE_PASS':ex.get('REFERENCE_TARGET_MANDATE_PASS','NOT_EVALUABLE'),
            'active_stage_applicable_count':sum(r.get('target_stage_applicable_count',0) for r in ex['rows']),
            'ACTIVE_EXECUTION_STAGE_MANDATE_PASS':('N/A' if not sum(r.get('target_stage_applicable_count',0) for r in ex['rows']) else 'PASS' if all(r.get('ACTIVE_EXECUTION_STAGE_MANDATE_PASS') in ('PASS','N/A') for r in ex['rows']) else 'FAIL'),
            'PATHWISE_CONTINUOUS_SAFETY_PASS':('PASS' if all(r.get('PATHWISE_CONTINUOUS_SAFETY_PASS')=='PASS' for r in ex['rows']) else 'FAIL'),
            'DETERMINISTIC_CASH_MARGIN_SURVIVAL':det_status('DETERMINISTIC_CASH_MARGIN_SURVIVAL'),
            'DETERMINISTIC_TARGET_STAGE_COMPLIANCE':det_status('DETERMINISTIC_TARGET_STAGE_COMPLIANCE'),
            'ALLOCATION_DRIFT_DIAGNOSTIC':any(r.get('ALLOCATION_DRIFT_DIAGNOSTIC',False) for r in deterministic+ex['rows']),
            'FULL_ENGINE_ADMISSION_PASS':'PASS' if verdict(su,ex,ex.get('spec',Spec({},1)).budget).passed else 'FAIL',
            'target_published':False,
            "funded_fraction_at_execution": ex["funded_fraction_at_execution"],
            "net_smlt_issuer_units": ex["net_issuer_units"],
            "p_breach_raw_worst": max(r.get("p_breach_raw", r["p_breach"]) for r in rows),
            "p_inherited_pre_action_breach_worst": max(r.get("p_inherited_pre_action_breach", 0) for r in rows),
            "p_transition_unavoidable_breach_worst": max(r.get("p_transition_unavoidable_breach", 0) for r in rows),
            "p_pre_action_unserviceable_worst": max(r.get("p_pre_action_unserviceable", 0) for r in ex["rows"])}


def risk_contribution(su: F.Setup, x: np.ndarray, alpha: float) -> pd.DataFrame:
    """Euler contributions of each sleeve to ES95 of the static horizon P&L, on the worst model's tail paths."""
    adj = np.array([su.mH if r in FUNDED else 0.0 for r in su.routes])
    best = None
    for m, G in su.G.items():
        contrib = (G - adj[None, :]) * np.asarray(x, float)[None, :]
        pnl = su.mH + contrib.sum(axis=1)
        k = max(1, int(np.ceil((1 - alpha) * len(pnl))))
        tail = np.argsort(pnl)[:k]
        es = float(-pnl[tail].mean())
        if best is None or es > best[0]:
            best = (es, m, -contrib[tail].mean(axis=0))
    es, m, cvec = best
    rows = [{"worst_model": m, "sleeve": r, "weight_nav0": float(x[j]), "es_contribution_nav0": float(cvec[j]), "book_ES95_static_nav0": es}
            for j, r in enumerate(su.routes)]
    rows.append({"worst_model": m, "sleeve": "MONEY_MARKET_AND_RESERVE", "weight_nav0": float(1 - sum(x[j] for j, r in enumerate(su.routes) if r in FUNDED)),
                 "es_contribution_nav0": float(-su.mH), "book_ES95_static_nav0": es})
    return pd.DataFrame(rows)


def stop_if_no_safe_anchor(c: dict) -> None:
    if not c["passed"]:
        raise StopCondition("the operational safe anchor is not admitted by the full engine: " + "; ".join(c["reasons"]))
