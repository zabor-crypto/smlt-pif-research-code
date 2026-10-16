"""Gate R3 — robust frontier: LP candidate generator + full-engine admission (the single authority).

LP (per NAV0, horizon H, generation paths of every model k):
  pnl_s(x) = m_H + sum_f x_f (G_fs - m_H) + sum_d x_d G_ds - costs(x) - drags(x)
    G_fs  funded horizon return per RUB invested; G_ds cumulative VM per RUB of initial notional (as the engine books it)
    costs = sum_j c_j |x_j - x0_j| + sum_d 2 c_d n_roll_d |x_d|          (entry + rolls; mark-to-market, no exit charge)
    drags = [ (beta + h/2) + sum_d (mr_d + q99_d) |x_d| ] * m_H            (immediate reserve and posted IM earn 0)
  maximise   sum_j c_j^view x_j - kappa * (Gamma z + sum_j p_j) - costs - drags
  s.t.       p_j + z >= dL_j x_j^+ + dS_j x_j^-                             (Bertsimas-Sim budgeted worst case)
             CVaR_95,k(-pnl) <= e   for every model k                         (Rockafellar-Uryasev)
             bounds, issuer net limits (mandate), long exposure <= 1, derivative gross <= 1,
             funded + IM + reserve <= 1
The LP never admits. Every candidate is rounded toward zero to executable quantities, converted to settlement-aware
orders (next-session execution) and evaluated by the R1 engine on INDEPENDENT admission paths.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from . import derivative_units as U

import scipy.sparse as sp
from scipy.optimize import linprog

from . import admission as A
from . import engine as E
from . import market as M
from . import orders as O
from . import scenarios as SC
from .r1 import loss_quantile, reserve_min

SERIES = {"OIL_FUT": "OIL_FUT_USD"}          # LP route -> scenario series
FUNDED = {"SMLT", "EQ_FUNDED", "OFZ_FUNDED", "GOLD_FUNDED", "BTC_RUB_EXPERIMENTAL"}
DERIV_FAMILY = {"SMLT_FUT": "SMLT_FUT", "EQ_HEDGE_FUT": "MXI_FUT", "RATES_HEDGE_FUT": "RGBI_FUT", "CNY_PERP": "CNYRUBF",
                "OIL_FUT": "BR_FUT"}
PRICE_SERIES = {"SMLT": ("proc", "SMLT"), "EQ_FUNDED": ("fund", "TMOS"), "OFZ_FUNDED": ("fund", "SBGB"),
                "GOLD_FUNDED": ("proc", "GOLD"), "MMF": ("proc", "LQDT")}


@dataclass
class Setup:
    t: pd.Timestamp
    routes: list            # LP routes (universe order)
    fspecs: list
    dspecs: list
    fin_names: list
    din_names: list
    nav0: float
    book0: E.Book
    x0: np.ndarray          # current weights per LP route
    views: pd.DataFrame     # interval table (route index = series names)
    gen: list               # [(model, PathSet, fin, din, mkt)]
    adm: list               # [ModelPaths]
    G: dict                 # model -> (S, J) horizon matrix (LP routes order)
    mH: float
    q99: dict
    haircut: np.ndarray
    cost: dict
    nroll: dict
    mr: dict
    notional: dict
    unit_price: dict
    cfg: dict
    mandate: str
    gamma: int
    issuer_cap: float | None = None      # state cap on SMLT net long exposure (ceiling; R5)
    available0: float | None = None      # immediately available liquidity of the actual book (engine state)


def market_horizon_matrix(routes: list, funded_names: list, derivative_names: list, market) -> np.ndarray:
    """Candidate G: funded price horizon and shared gross fixed-count derivative kernel. No optimisation."""
    dg = U.horizon(market) if derivative_names else np.empty((len(market.P), 0))
    return np.stack([market.P[:, -1, funded_names.index(r)] / market.P[:, 0, funded_names.index(r)] - 1
                     if r in FUNDED else dg[:, derivative_names.index(r)] for r in routes], axis=1)


def _price_at(ctx, P, name: str, t) -> float:
    kind, s = PRICE_SERIES[name]
    if kind == "proc":
        px = pd.read_csv(ctx.proc / "prices.csv", index_col=0, parse_dates=True)[s]
        return float(px.loc[:t].dropna().iloc[-1])
    from .routes import fund_prices
    return float(fund_prices(ctx.root, s, str(t.date()))["PX"].loc[:t].iloc[-1])


def _roll_plan(ctx, route: str, t: pd.Timestamp, H: int, cal_days: int = 5) -> tuple:
    """Listed-expiry roll days within the horizon (engine day index) and the log gap next/front at t."""
    if route == "CNY_PERP":
        return (), 0.0
    from ..data import load
    if not hasattr(ctx, "_adaptive_fut"):
        ctx._adaptive_fut = load.futures_contracts(ctx.root, ctx.cutoff)
    fc = ctx._adaptive_fut
    code = {"SMLT_FUT": "SMLT", "EQ_HEDGE_FUT": "MXI", "RATES_HEDGE_FUT": "RGBI", "OIL_FUT": "BR"}[route]
    d = fc[(fc["ASSETCODE"] == code) & (fc["TRADEDATE"] == t) & fc["SETTLEPRICE"].notna() & (fc["LSTTRADE"] > t)]
    d = d.sort_values("LSTTRADE")
    days, bd = [], pd.bdate_range(t, periods=H + 1)
    # Resolve the true expiry insertion position before applying the roll offset.
    # A horizon-limited searchsorted saturates at H+1 and manufactures H+1-cal_days.
    if not d.empty:
        bd = pd.bdate_range(t, max(bd[-1], d["LSTTRADE"].max() + pd.offsets.BDay(1)))
    for lt in d["LSTTRADE"]:
        k = int(bd.searchsorted(lt) - cal_days)
        if not 1 <= k <= H:             # OUT_OF_HORIZON (or already rolled); never clip
            continue
        days.append(k)                  # IN_HORIZON; duplicate protection is separate below
    gap = float(np.log(d["SETTLEPRICE"].iloc[1] / d["SETTLEPRICE"].iloc[0])) if len(d) >= 2 else 0.0
    return tuple(sorted(set(days))), gap


def build(ctx, cfg: dict, P: dict, t: pd.Timestamp, universe: str, mandate: str, views_iv: pd.DataFrame,
          capital: float | None = None, book0: E.Book | None = None, n_gen: int | None = None,
          n_adm: int | None = None, models: tuple | None = None, seed_offset: int = 0,
          panel_end: pd.Timestamp | None = None, px: pd.DataFrame | None = None, issuer_cap: float | None = None,
          available0: float | None = None, cost_mult: float = 1.0) -> Setup:
    """`panel_end`: last recalibration date (bootstrap sample frozen there; FHS volatility and the money-market rate
    are taken at t). `px`: cached unit prices of the funded routes. `book0`: the ACTUAL current book (quantities, cash,
    pending settlements). `cost_mult`: cost sensitivity."""
    fr = cfg["r3"]["frontier"]
    H = fr["horizon_sessions"]
    routes = list(cfg["r3"]["universes"][universe])
    panel_now = SC.history_panel(P, t, fr["estimation_start"])
    panel = panel_now if panel_end is None else SC.history_panel(P, panel_end, fr["estimation_start"])
    models = models or tuple(fr["models"])
    n_gen, n_adm = n_gen or fr["n_generation_per_model"], n_adm or fr["n_admission_per_model"]
    central = {r: float(views_iv.loc[SERIES.get(r, r), "central"]) for r in routes}
    tgt = {SERIES.get(r, r): central[r] for r in routes}
    usd = P["usd"]

    def _p(name):
        return float(px[name].loc[:t].dropna().iloc[-1]) if px is not None else _price_at(ctx, P, name, t)
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
                p0 = _p(r)
                fs = M.FundedSpec(r, E.FUNDED, p0, lag=1, lot=1.0, cost_prop=costs["smlt"]["bp"] * 1e-4 * cost_mult,
                                  cost_unit=costs["smlt"]["half_tick_rub"] * cost_mult, issuer="SMLT")
            else:
                p0 = _p(r)
                fs = M.FundedSpec(r, E.FUNDED, p0, lag=1, lot=1.0, cost_prop=costs["funds"] * cost_mult)
            fspecs.append(fs)
            cost[r] = fs.cost_prop + fs.cost_unit / fs.P0
            unit[r] = fs.P0
        else:
            h = M.deriv_history(ctx, DERIV_FAMILY[r], pd.DatetimeIndex([t]), usd, broker).iloc[-1]
            rd, gap = _roll_plan(ctx, r, t, H)
            fee_rate = float(h["fee"] / h["notional"]) * cost_mult
            dspecs.append(M.DerivSpec(r, F0=float(h["Fn"]), pv0=float(h["pv"]), usd_linked=(r == "OIL_FUT"),
                                      mr=float(h["mr1"]), fee_rate=fee_rate, fee_fixed_rub=0.0, roll_days=rd, roll_gap=gap))
            cost[r] = fee_rate
            nroll[r] = len(rd)
            mr[r] = float(h["mr1"])
            notional[r] = float(h["notional"])
            unit[r] = float(h["notional"])
    mm0 = _p("MMF")
    fspecs.append(M.FundedSpec("MMF", E.MMF, mm0, lag=1, lot=1.0))
    fin_names = [f.name for f in fspecs]
    din_names = [d.name for d in dspecs]

    def to_engine(ps: SC.PathSet):
        idx = {r: k for k, r in enumerate(ps.routes)}
        Rf = np.stack([ps.R[:, :, idx[f.name]] if f.kind != E.MMF else np.zeros(ps.R.shape[:2]) for f in fspecs], axis=2)
        Rd = np.stack([ps.R[:, :, idx[SERIES.get(d.name, d.name)]] for d in dspecs], axis=2) if dspecs else np.zeros((ps.R.shape[0], H, 0))
        Rfx = ps.R[:, :, idx["USDRUB"]]
        return M.forward_market(fspecs, dspecs, Rf, Rd, Rfx, ps.mmf_daily, cfg["r1"]["liquidity"]["immediate_cash_rate"], cny_swap_frac=ps.cny_swap_frac)
    gen, adm, G = [], [], {}
    seed0 = fr["seed_generation"] + seed_offset
    for k, mdl in enumerate(models):
        ps = SC.centre(SC.simulate(panel, mdl, n_gen, H, seed0 + 101 * k, fr["block_length"], panel_now), tgt, dspecs)
        fin, din, mkt = to_engine(ps)
        gen.append((mdl, ps, fin, din, mkt))
        G[mdl] = market_horizon_matrix(routes, fin_names, din_names, mkt)
        psa = SC.centre(SC.simulate(panel, mdl, n_adm, H, seed0 + fr["seed_admission_shift"] + 101 * k, fr["block_length"],
                                    panel_now), tgt, dspecs)
        fa, da, ma = to_engine(psa)
        adm.append(A.ModelPaths(f"{mdl}_ADMISSION", fa, da, ma))
    mH = float(np.prod(1 + gen[0][1].mmf_daily) - 1)
    lv = cfg["r1"]["liquidity"]["reserve_var_level"]
    q99, hc = {}, []
    for r in routes:
        s = SERIES.get(r, r)
        daily = np.concatenate([g[1].R[:, :, g[1].routes.index(s)].ravel() for g in gen])
        q99[r] = float(max(np.quantile(-daily, lv), np.quantile(daily, lv)))
    for f in fspecs:
        if f.kind == E.MMF:
            hc.append(0.0)
        else:
            hc.append(loss_quantile(np.concatenate([g[1].R[:, :, g[1].routes.index(f.name)].ravel() for g in gen]), lv))
    nav_cash = capital if capital is not None else cfg["owner"]["capital_main_rub"]
    if book0 is None:
        shares = np.floor(cfg["owner"]["smlt_initial_value_rub"] / unit["SMLT"])
        book0 = E.Book(qf=np.array([shares if n == "SMLT" else 0.0 for n in fin_names]), qd=np.zeros(len(din_names)),
                       cash=nav_cash)
    nav0 = float((book0.qf * np.array([f.P0 for f in fspecs])).sum() + book0.cash + sum((book0.pending or {}).values()))
    x0 = np.array([(book0.qf[fin_names.index(r)] * unit[r] if r in FUNDED else book0.qd[din_names.index(r)] * unit[r]) / nav0
                   for r in routes])
    return Setup(t, routes, fspecs, dspecs, fin_names, din_names, nav0, book0, x0, views_iv, gen, adm, G, mH, q99,
                 np.array(hc), cost, nroll, mr, notional, unit, cfg, mandate, int(np.ceil(np.sqrt(len(routes)))),
                 issuer_cap, available0)


# ---------------------------------------------------------------------------------------------- LP
def solve(su: Setup, e: float, kappa: float, caps: dict | None = None, force: dict | None = None,
          extra_views: pd.DataFrame | None = None) -> dict:
    """caps: {route: (lo, hi)} tighter state caps; force: {route: minimum |x| in its allowed direction}."""
    cfg = su.cfg["r3"]
    iv = extra_views if extra_views is not None else su.views
    J = len(su.routes)
    beta = su.cfg["owner"]["liquidity_min_nav"]
    h = su.cfg["r1"]["liquidity"]["sweep_hysteresis_nav"]
    lo = np.array([cfg["bounds"][r]["lo"] for r in su.routes])
    hi = np.array([cfg["bounds"][r]["hi"] for r in su.routes])
    for r, (a, b) in (caps or {}).items():
        j = su.routes.index(r)
        lo[j], hi[j] = max(lo[j], a), min(hi[j], b)
        if hi[j] < lo[j]:
            hi[j] = lo[j]
    funded = np.array([r in FUNDED for r in su.routes])
    c = np.array([float(iv.loc[SERIES.get(r, r), "central"]) for r in su.routes])
    dL = np.array([float(iv.loc[SERIES.get(r, r), "d_long"]) for r in su.routes])
    dS = np.array([float(iv.loc[SERIES.get(r, r), "d_short"]) for r in su.routes])
    tc = np.array([su.cost[r] for r in su.routes])
    roll = np.array([2 * su.cost[r] * su.nroll.get(r, 0) for r in su.routes])
    drag = np.array([0.0 if r in FUNDED else (su.mr[r] + su.q99[r]) * su.mH for r in su.routes])
    models = list(su.G)
    S = {m: su.G[m].shape[0] for m in models}
    # variable layout: xp(J) xm(J) tp(J) tm(J) z p(J) [eta_k, xi_k(S_k)] per model
    nx = 4 * J + 1 + J
    off, n = {}, nx
    for m in models:
        off[m] = n
        n += 1 + S[m]
    XP, XM, TP, TM, Z, PP = 0, J, 2 * J, 3 * J, 4 * J, 4 * J + 1
    alpha = su.cfg["owner"]["es_alpha"]
    obj = np.zeros(n)
    obj[XP:XP + J] = -(c - roll - drag)
    obj[XM:XM + J] = c + roll + drag
    obj[TP:TP + J] = tc
    obj[TM:TM + J] = tc
    obj[Z] = kappa * su.gamma
    obj[PP:PP + J] = kappa
    rows, cols, vals, b_ub = [], [], [], []
    nr = 0

    def add(cidx, cval, rhs):
        nonlocal nr
        rows.extend([nr] * len(cidx)); cols.extend(cidx); vals.extend(cval); b_ub.append(rhs); nr += 1
    eq = sp.lil_matrix((J, n))
    b_eq = []
    for j in range(J):                                          # x - x0 = tp - tm
        eq[j, XP + j] = 1; eq[j, XM + j] = -1; eq[j, TP + j] = -1; eq[j, TM + j] = 1
        b_eq.append(su.x0[j])
        add([XP + j, XM + j, PP + j, Z], [dL[j], dS[j], -1.0, -1.0], 0.0)      # Bertsimas-Sim
    const = su.mH - (beta + h / 2) * su.mH
    adj = np.where(funded, su.mH, 0.0)
    for m in models:                                            # CVaR per model (Rockafellar-Uryasev)
        Gm = su.G[m] - adj[None, :]
        k = off[m]
        Sm = S[m]
        blk_x = np.hstack([-Gm + roll + drag, Gm + roll + drag, np.tile(tc, (Sm, 1)), np.tile(tc, (Sm, 1))])
        xi = np.arange(Sm)
        r0 = nr
        rr = np.repeat(np.arange(Sm) + r0, 4 * J)
        cc = np.tile(np.arange(4 * J), Sm)
        rows.extend(rr.tolist()); cols.extend(cc.tolist()); vals.extend(blk_x.ravel().tolist())
        rows.extend((xi + r0).tolist()); cols.extend([k] * Sm); vals.extend([-1.0] * Sm)
        rows.extend((xi + r0).tolist()); cols.extend((k + 1 + xi).tolist()); vals.extend([-1.0] * Sm)
        b_ub.extend([const] * Sm)
        nr += Sm
        add([k] + list(range(k + 1, k + 1 + Sm)), [1.0] + [1.0 / ((1 - alpha) * Sm)] * Sm, e)
    pc = cfg["portfolio_constraints"]
    ci, cv = [], []
    for j, rt in enumerate(su.routes):                          # funded + IM + reserve <= 1
        if rt in FUNDED:
            ci += [XP + j]; cv += [1.0]
        else:
            ci += [XP + j, XM + j]; cv += [su.mr[rt] + su.q99[rt]] * 2
    add(ci, cv, pc["funded_plus_margin_plus_reserve_max_nav0"] - (beta + h / 2))
    add(list(range(XP, XP + J)), [1.0] * J, pc["long_exposure_max_nav0"])      # long exposure
    ci = [c_ for j, rt in enumerate(su.routes) if rt not in FUNDED for c_ in (XP + j, XM + j)]
    add(ci, [1.0] * len(ci), pc["derivative_gross_max_nav0"])                # derivative gross
    md = cfg["mandates"][su.mandate]
    ci, cv = [], []
    for rt in ("SMLT", "SMLT_FUT"):
        if rt in su.routes:
            j = su.routes.index(rt); ci += [XP + j, XM + j]; cv += [1.0, -1.0]
    add(ci, cv, md["issuer_net_max"] if su.issuer_cap is None else min(md["issuer_net_max"], su.issuer_cap))
    add(ci, [-v_ for v_ in cv], -md["issuer_net_min"])
    A_ub = sp.csr_matrix((vals, (rows, cols)), shape=(nr, n))
    bounds = []
    for j in range(J):
        bounds.append((0, max(hi[j], 0.0)))
    for j in range(J):
        bounds.append((0, max(-lo[j], 0.0)))
    for (rt, mn) in (force or {}).items():
        j = su.routes.index(rt)
        if hi[j] > 0:
            bounds[j] = (min(mn, hi[j]), bounds[j][1])
        else:
            bounds[J + j] = (min(mn, -lo[j]), bounds[J + j][1])
    bounds += [(0, None)] * (2 * J) + [(0, None)] + [(0, None)] * J
    for m in models:
        bounds += [(None, None)] + [(0, None)] * S[m]
    res = linprog(obj, A_ub=A_ub, b_ub=np.array(b_ub), A_eq=eq.tocsr(), b_eq=np.array(b_eq), bounds=bounds,
                  method="highs")
    if res.status != 0:
        return {"status": res.message, "x": None}
    v = res.x
    x = v[XP:XP + J] - v[XM:XM + J]
    x = np.where(np.abs(x) < 1e-9, 0.0, x)
    turn = v[TP:TP + J] + v[TM:TM + J]
    nominal = float(c @ x - roll @ np.abs(x) - drag @ np.abs(x) - tc @ turn - (beta + h / 2) * su.mH)
    robust = float(-res.fun - (beta + h / 2) * su.mH)
    return {"status": "optimal", "x": x, "robust_excess": robust, "nominal_excess": nominal,
            "cvar": {m: float(v[off[m]] + v[off[m] + 1:off[m] + 1 + S[m]].sum() / ((1 - alpha) * S[m])) for m in models}}


# ---------------------------------------------------------------------------------------------- admission of a candidate
def executable(su: Setup, x: np.ndarray, delay: int) -> dict:
    """LP weights -> executable integer book, orders, reserve, waterfall; hard checks on the executable book."""
    cfg = su.cfg
    wf = np.zeros(len(su.fin_names))
    wd = np.zeros(len(su.din_names))
    for j, r in enumerate(su.routes):
        if r in FUNDED:
            wf[su.fin_names.index(r)] = x[j]
        else:
            wd[su.din_names.index(r)] = x[j]
    P0 = np.array([f.P0 for f in su.fspecs])
    notional = np.array([su.notional[d] for d in su.din_names])
    lots = np.array([f.lot for f in su.fspecs])
    qf, qd = E.to_quantities(wf, wd, su.nav0, P0, notional, lots)
    jm = su.fin_names.index("MMF")
    qf[jm] = np.nan
    mkt0 = su.gen[0][4]
    # decision-time VM99 of the derivative book and waterfall ranking (generation paths of every model)
    vm, Gd, Gf = [], [], []
    for (_, _, fin, din, mkt) in su.gen:
        if din:
            dret = mkt.Fp[:, 1:, :] * mkt.pv[:, 1:, :] / (mkt.Fn[:, :-1, :] * mkt.pv[:, :-1, :]) - 1
            vm.append(-(dret * (qd * notional)[None, None, :]).sum(axis=2).ravel())
            Gd.append(((mkt.Fp[:, 1:, :] - mkt.Fn[:, :-1, :]) * mkt.pv[:, 1:, :]).sum(axis=1))
        Gf.append(mkt.P[:, -1, :] - mkt.P[:, 0, :])
    lv = cfg["r1"]["liquidity"]["reserve_var_level"]
    vm99 = max(float(np.quantile(np.concatenate(vm), lv)), 0.0) / su.nav0 if vm else 0.0
    res_min = reserve_min(cfg["owner"]["liquidity_min_nav"], vm99)
    qf_rank = np.nan_to_num(qf)
    rel_d = notional * mkt0.mr[0] - mkt0.fee[0, 0, :] if su.din_names else np.zeros(0)
    wf_rank, rs_rank, relief_d, relief_f = A.tail_rank(np.concatenate(Gd) if Gd else np.zeros((1, 0)),
                                                       np.concatenate(Gf), qd, qf_rank, rel_d, P0,
                                                       mkt0.fee[0, 0, :] if su.din_names else np.zeros(0),
                                                       np.array([f.cost_prop for f in su.fspecs]))
    rs_rank = tuple(j for j in rs_rank if su.fspecs[j].kind != E.MMF)
    avail0 = float(np.broadcast_to(su.book0.cash, (1,))[0]) if su.available0 is None else float(su.available0)
    orders, oinfo = O.build(delay, su.book0.qf.astype(float), su.book0.qd.astype(float), qf, qd, P0, notional,
                            mkt0.mr[0], 1.0, [E.Instrument(f.name, f.kind, f.lot, f.cost_prop, f.cost_unit) for f in su.fspecs],
                            available=avail0, nav=su.nav0,
                            target_frac=res_min + cfg["r1"]["liquidity"]["sweep_hysteresis_nav"] / 2,
                            haircut=su.haircut, lag_cycle=1)
    wexe = {}
    for j, r in enumerate(su.routes):
        if r in FUNDED:
            wexe[r] = float(qf[su.fin_names.index(r)] * P0[su.fin_names.index(r)] / su.nav0)
        else:
            wexe[r] = float(qd[su.din_names.index(r)] * notional[su.din_names.index(r)] / su.nav0)
    b = cfg["r3"]["bounds"]
    md = cfg["r3"]["mandates"][su.mandate]
    net_iss = wexe.get("SMLT", 0.0) + wexe.get("SMLT_FUT", 0.0)
    long_exp = sum(v for r, v in wexe.items() if r in FUNDED) + sum(max(v, 0) for r, v in wexe.items() if r not in FUNDED)
    gross = sum(abs(v) for r, v in wexe.items() if r not in FUNDED)
    hard = {"route_bounds": all(b[r]["lo"] - 1e-9 <= v <= b[r]["hi"] + 1e-9 for r, v in wexe.items()),
            "issuer_net_limits": md["issuer_net_min"] - 1e-9 <= net_iss <= (md["issuer_net_max"] if su.issuer_cap is None
                                                                             else min(md["issuer_net_max"], su.issuer_cap)) + 1e-9,
            "long_exposure<=1": long_exp <= 1 + 1e-9, "derivative_gross<=1": gross <= 1 + 1e-9}
    return {"qf": qf, "qd": qd, "orders": orders, "order_info": oinfo, "reserve_min": res_min, "vm99": vm99,
            "wf_rank": wf_rank, "restore_rank": rs_rank, "w_exec": wexe, "hard": hard, "net_issuer": net_iss,
            "directional_short": net_iss < md.get("label_directional_short_if_net_below", 0.0) - 1e-9}


def admit_candidate(su: Setup, x: np.ndarray, delay: int = 1) -> dict:
    ex = executable(su, x, delay)
    cfg = su.cfg
    rule = A.Rule(budget=1.0, beta=cfg["owner"]["liquidity_min_nav"],
                  max_breach=cfg["r1"]["admission"]["max_breach_probability"],
                  stress_mults=tuple(cfg["r1"]["admission"]["margin_stress_survival"]),
                  hysteresis=cfg["r1"]["liquidity"]["sweep_hysteresis_nav"])
    rows = A.evaluate_all(su.adm, su.book0, ex["orders"], rule, ex["reserve_min"], ex["wf_rank"], ex["restore_rank"])
    ex["rows"] = rows
    return ex


def verdict(ex: dict, budget: float, cfg: dict) -> A.Admission:
    rule = A.Rule(budget=budget, beta=cfg["owner"]["liquidity_min_nav"],
                  max_breach=cfg["r1"]["admission"]["max_breach_probability"],
                  stress_mults=tuple(cfg["r1"]["admission"]["margin_stress_survival"]),
                  hysteresis=cfg["r1"]["liquidity"]["sweep_hysteresis_nav"])
    return A.judge(ex["rows"], rule, ex["hard"])


def summarise(su: Setup, ex: dict) -> dict:
    m1 = [r for r in ex["rows"] if r["margin_multiplier"] == 1.0]
    st = [r for r in ex["rows"] if r["margin_multiplier"] != 1.0]
    nav0 = su.nav0
    return {"engine_ES95_worst_nav0": max(r["ES95"] for r in m1) / nav0, "engine_ES99_worst_nav0": max(r["ES99"] for r in m1) / nav0,
            "engine_mean_excess_worst_nav0": min(r["mean_excess_pnl"] for r in m1) / nav0,
            "engine_mean_pnl_min_nav0": min(r["mean_pnl"] for r in m1) / nav0,
            "p_breach_worst": max(r["p_breach"] for r in m1), "p_unresolved_worst": max(r["p_unresolved"] for r in m1),
            "p_unresolved_stress_worst": max([r["p_unresolved"] for r in st] or [0.0]),
            "p_forced_worst": max(r["p_forced"] for r in m1), "peak_im_nav_p95_worst": max(r["peak_im_nav_p95"] for r in m1),
            "min_available_nav_p05_worst": min(r["min_available_nav_p05"] for r in m1),
            "mdd_p95_worst": max(r["mdd_p95"] for r in m1), "ES95_liquidation_worst_nav0": max(r["ES95_liquidation_nav"] for r in m1) / nav0,
            "vm99_nav": ex["vm99"], "reserve_min_nav": ex["reserve_min"], "two_step_order": ex["order_info"]["two_step"],
            "net_issuer_exec": ex["net_issuer"], "directional_short": ex["directional_short"]}
