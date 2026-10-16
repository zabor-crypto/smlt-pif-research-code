"""Frozen-source producers for the V2R2 shared execution ledger.

No capture time is a publication clock. Intraday tariff W/R qualifiers are
retained; exchange daily observed conversion is the accepted projection.
Missing capacity is zero, never an unlimited assumed tape.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import clocks
from .execution import component
from .lk import AGGREGATION, ALIASES, FORECAST_ASSUMPTION, forecast_terms, terms as lk_terms
from .cfg import StopCondition

FEE_ROUTE = {'SMLT_FUT':'SMLT_FUT','MXI_FUT':'EQ_HEDGE_FUT','RGBI_FUT':'RATES_HEDGE_FUT','BR_FUT':'OIL_FUT','CNYRUBF':'CNY_PERP'}
FAMILY = {'SMLT_FUT': 'SMLT', 'MXI_FUT': 'MXI', 'RGBI_FUT': 'RGBI', 'BR_FUT': 'BR', 'CNYRUBF': 'CNYRUBTOM/CNYRUBF'}


@lru_cache(maxsize=8)
def tariffs(root):
    return pd.read_csv(Path(root)/'data/regime_v2r2_combined/fee_authority.csv')


def fee_terms(ctx, route, history, date, pv, pit, tick, broker):
    table = tariffs(str(ctx.root))
    rows = table[(table.route == FEE_ROUTE[route]) & (pd.to_datetime(table.effective_from) <= date)
                 & (pd.to_datetime(table.effective_to) >= date)]
    if rows.empty:
        raise StopCondition(f'R03_COMPONENT_AUTHORITY_MISSING:{route}:{date.date()}')
    fields = rows.set_index('field').value_or_rule.to_dict()
    prior = history.loc[history.index < date, 'SETTLEPRICE'].dropna()
    # First-day initial settlement is not inferred from the day's final price.
    if prior.empty:
        return np.nan, np.nan, np.nan, fields
    def price_for(authority):
        if not authority.startswith(('PRIOR_TRADING_DAY_EVENING_SETTLEMENT', 'PRIOR_TRADING_DAY_MAIN_SESSION_SETTLEMENT')):
            raise StopCondition('R03_UNSUPPORTED_COMPONENT_FEE_PRICE_AUTHORITY:' + authority)
        return float(prior.iloc[-1]) # separately selected daily-mark projection for each component
    exchange = component(price_for(fields['exchange_fee_price']), pv, float(fields['exchange_rate_percent']))
    clearing = component(price_for(fields['clearing_fee_price']), pv, float(fields['clearing_rate_percent']))
    current = float(history.loc[date, 'SETTLEPRICE']) if date in history.index else np.nan
    academic_broker = broker if route == 'SMLT_FUT' else float(pit['exchange_fee_rate'])*current*pv
    slippage = .5*tick*pv
    return exchange+clearing+academic_broker+slippage, exchange, clearing, fields


def repair_terms(ctx, route, out, pit, broker):
    if route not in FAMILY:
        raise StopCondition('V2R2_UNSUPPORTED_DERIVATIVE_ROUTE:' + route)
    out = out.copy()
    mask = out.index >= pd.Timestamp('2023-01-03')
    if mask.any():
        out.loc[mask, ['mr1','mr1_policy_label','mr1_clearing','mr1_clearing_label']] = clocks.apply(ctx.root, FAMILY[route], out.loc[mask])[['mr1','mr1_policy_label','mr1_clearing','mr1_clearing_label']]
    for date, row in out.loc[mask].iterrows():
        key = row.held
        if pd.isna(key):
            continue
        hist = ctx._adaptive_fut[(ctx._adaptive_fut.SECID == key) & (~ctx._adaptive_fut.weekend)].set_index('TRADEDATE').sort_index()
        pv = float(row.pv)
        if route == 'BR_FUT' and row.pv_source != 'EXCHANGE_OI_IMPLIED':
            raise StopCondition(f'R03_BR_OBSERVED_CONVERSION_MISSING:{key}:{date.date()}')
        fee, ex, cl, fields = fee_terms(ctx, route, hist, date, pv, pit, float(row.get('tick', pit['min_step'])), broker)
        out.loc[date, 'fee'] = fee
        out.loc[date, 'exchange_component'] = ex
        out.loc[date, 'clearing_component'] = cl
        out.loc[date, 'fee_rate_status'] = 'HISTORICAL_COMPONENTS_PLUS_CUTOFF_ACADEMIC_BROKER'
        out.loc[date, 'fee_evidence'] = 'HISTORICAL_SEPARATE_COMPONENTS_PLUS_FROZEN_ACADEMIC_COSTS'
        out.loc[date, 'component_price_authority'] = fields['exchange_fee_price'] + ' / ' + fields['clearing_fee_price']
        out.loc[date, 'component_conversion_clock'] = fields['tick_value_input_clock']
        observation = hist.reindex([date]).iloc[0]
        out.loc[date, 'execution_capacity'] = observation.VOLUME
        out.loc[date, 'executable'] = bool(observation.NUMTRADES > 0 and observation.VOLUME > 0 and observation.CLOSE > 0 and np.isfinite(observation.CLOSE))
    return out


def funded_tape(ctx, cfg, name, dates):
    from ..data import load
    from ..adaptive import routes
    mapping = {'EQ_FUNDED': cfg['r2']['funds']['broad_equity'], 'OFZ_FUNDED': cfg['r2']['funds']['ofz'],
               'GOLD_FUNDED': cfg['r2']['funds']['gold'], 'MMF': 'LQDT', 'SMLT': 'SMLT'}
    key = mapping.get(name, name)
    if name == 'BTC_RUB_EXPERIMENTAL':
        from ..data import crypto
        if not hasattr(ctx,'_v2r2_btc_tape'):
            ctx._v2r2_btc_tape = crypto.binance_hourly(ctx.root,ctx.cutoff)
        tape = ctx._v2r2_btc_tape
        closes = (tape.index+pd.Timedelta(hours=1)).tz_convert(None)
        marks = pd.DatetimeIndex(dates)+pd.Timedelta(hours=cfg['r2']['btc']['bar_close_hour_utc'])
        indices = closes.searchsorted(marks,side='right')-1
        capacity, eligible = np.zeros(len(dates)), np.zeros(len(dates),bool)
        for k,index in enumerate(indices):
            if index >= 0 and marks[k]-closes[index] <= pd.Timedelta(hours=6):
                row = tape.iloc[index]
                eligible[k] = bool(row.volume > 0 and row.close > 0 and np.isfinite(row.close))
                capacity[k] = float(row.volume) if eligible[k] else 0.
        return 'BINANCE_BTCUSDT_FROZEN_ACADEMIC_PROXY',capacity,eligible
    f = load._frames(ctx.root, 'shares_history', key)
    a = routes.adaptive_frames(ctx.root, 'shares_history', key)
    if not a.empty:
        f = pd.concat([f,a], ignore_index=True)
    if f.empty:
        return key, np.zeros(len(dates)), np.zeros(len(dates), bool)
    f['TRADEDATE'] = pd.to_datetime(f.TRADEDATE)
    f = f.sort_values(['TRADEDATE','VALUE']).drop_duplicates('TRADEDATE', keep='last').set_index('TRADEDATE').reindex(dates)
    cap = f.VOLUME.fillna(0).to_numpy(float)
    ok = ((f.NUMTRADES > 0) & (f.VOLUME > 0) & (f.CLOSE > 0) & np.isfinite(f.CLOSE)).to_numpy()
    return key, np.where(ok, cap, 0.), ok


def historical(ctx, cfg, funded, routes, dates, mkt, terms):
    keys, cap_f = [], []
    for j, spec in enumerate(funded):
        key, cap, ok = funded_tape(ctx, cfg, spec.name, dates)
        keys.append(key); cap_f.append(cap)
        mkt.tradable_f[:, j] &= ok
    series = []
    policy, clearing, lk1 = [], [], []
    vm_lag = np.zeros((len(dates),len(routes)), int)
    daily, _ = clocks.authority(str(ctx.root))
    for j, route in enumerate(routes):
        h = terms[route]
        policy.append(h.mr1.to_numpy(float)); clearing.append(h.mr1_clearing.to_numpy(float))
        d = daily[daily.family == FAMILY[route]].drop_duplicates('requested_date', keep='last').set_index('requested_date')
        lk1.append(d.lk1.reindex(d.index.union(dates)).ffill().reindex(dates).to_numpy(float))
        pit = ctx._adaptive_pit.loc['CNYRUBF' if route == 'CNYRUBF' else FAMILY[route]]
        family = {}
        for key in h.held.dropna().unique():
            raw = ctx._adaptive_fut[(ctx._adaptive_fut.SECID == key) & (~ctx._adaptive_fut.weekend)].set_index('TRADEDATE').sort_index()
            rows = raw.reindex(dates)
            mark = raw.SETTLEPRICE.reindex(raw.index.union(dates)).ffill().reindex(dates)
            pv = float(pit.point_value_static) * np.ones(len(dates))
            if route == 'BR_FUT':
                observed = raw.OPENPOSITIONVALUE/(raw.OPENPOSITION*raw.SETTLEPRICE)
                observed = observed.where((raw.OPENPOSITION > 0) & np.isfinite(observed))
                # Conversion belongs to the actual date/series; no CBR fallback.
                pv = observed.reindex(dates).to_numpy(float)
            fees = np.full(len(dates), np.nan)
            for k, date in enumerate(dates):
                if date in raw.index and np.isfinite(pv[k]):
                    tick = float(h.tick.iloc[k])
                    fees[k] = fee_terms(ctx, route, raw, date, pv[k], pit, tick,
                                       float(ctx.cfg['costs']['broker_fee_futures_rub_per_contract']))[0]
            swap = rows.SWAPRATE.fillna(0) if route == 'CNYRUBF' else 0.
            ok = ((rows.NUMTRADES > 0)&(rows.VOLUME > 0)&(rows.CLOSE > 0)&np.isfinite(rows.CLOSE)).to_numpy()
            expiry = raw.LSTTRADE.dropna().iloc[-1] if not raw.LSTTRADE.dropna().empty else None
            family[key] = {'mark':mark.to_numpy(float)[None,:], 'vm_mark':(mark-swap).to_numpy(float)[None,:],
                           'pv':pv[None,:], 'fee':fees[None,:], 'capacity':rows.VOLUME.fillna(0).to_numpy(float),
                           'executable':ok, 'expiry':None if route == 'CNYRUBF' else expiry,
                           'rounded_vm':route in ('SMLT_FUT','BR_FUT'),
                           **lk_terms(ctx.root, route, key, dates[0]),
                           'lk_classification_by_session':[lk_terms(ctx.root, route, key, date)['lk_classification'] for date in dates]}
        series.append(family)
        mkt.tradable_d[:,j] = h.executable.fillna(False).to_numpy(bool)
        if route in ('SMLT_FUT','BR_FUT'):
            vm_lag[:,j] = (dates >= pd.Timestamp('2026-03-23')).astype(int)
    cal = pd.read_csv(Path(ctx.root)/'data/regime_v2r2_combined/cny_qconv_calendar.csv')
    return {'dates':dates, 'active':mkt.contract, 'series':series, 'funded_keys':keys,
            'capacity_f':np.stack(cap_f,axis=1) if funded else np.zeros((len(dates),0)),
            'mr_policy':np.stack(policy,axis=1) if routes else np.zeros((len(dates),0)),
            'mr_clearing':np.stack(clearing,axis=1) if routes else np.zeros((len(dates),0)),
            'vm_lag':vm_lag, 'lk1':np.stack(lk1,axis=1) if routes else np.zeros((len(dates),0)),
            'lk_aggregation':AGGREGATION,
            'lk_bases':[FAMILY[r].split('/')[0] for r in routes],
            'lk_units':[next(iter(f.values()))['lk_units'] if f else None for f in series], 'qconv':cal.to_dict('records'),
            'cny_indices':[j for j,r in enumerate(routes) if r == 'CNYRUBF']}


def forecast_info(ctx, cfg, route, date, terms, horizon):
    """Conditional admission inputs known at the decision cut; no future tape."""
    from .session_calendar import forecast
    date = pd.Timestamp(date)
    listed = ctx._adaptive_fut[(ctx._adaptive_fut.ASSETCODE == FAMILY[route]) &
                              (ctx._adaptive_fut.TRADEDATE == date)].sort_values('LSTTRADE') if route != 'CNYRUBF' else pd.DataFrame()
    cal = pd.read_csv(Path(ctx.root)/'data/regime_v2r2_combined/cny_qconv_calendar.csv')
    plans = cal[pd.to_datetime(cal.contract_listed_from) <= date].to_dict('records')
    # Already-listed quarterly contracts beyond the historical event register
    # supply expiry terms at the origin, never future marks/tape or notices.
    quarters = ctx._adaptive_fut[(ctx._adaptive_fut.ASSETCODE == 'CNY') &
                                 (ctx._adaptive_fut.TRADEDATE == date)]
    registered_expiries = set(pd.to_datetime(cal.expiration_date))
    for row in quarters.itertuples():
        if row.FRSTTRADE > date or row.LSTTRADE <= date or row.LSTTRADE in registered_expiries:
            continue
        plans.append({'quarter_contract':row.SECID, 'expiration_date':str(row.LSTTRADE.date()),
                      'contract_listed_from':str(row.FRSTTRADE.date()),
                      'expiration_source':f'FROZEN_ORIGIN_LISTED_TERMS:{row.SECID}:{date.date()}'})
    bounds = [pd.Timestamp(r['expiration_date']) for r in plans]
    if not listed.empty:
        bounds.extend(listed.LSTTRADE.tolist())
    schedule = forecast(ctx.root, date, horizon, through=max(bounds) if bounds else None)
    calendar, dates = schedule.sessions, schedule.dates
    table = tariffs(str(ctx.root))
    fields = table[(table.route == FEE_ROUTE[route])&(pd.to_datetime(table.effective_from) <= date)&(pd.to_datetime(table.effective_to) >= date)].set_index('field').value_or_rule
    # A conditional forecast uses the rule known at its origin. It does not
    # backcast April's ten-session notice into an earlier admission.
    events = []
    for row in plans:
        expiry = pd.Timestamp(row['expiration_date'])
        lead = 10 if date >= pd.Timestamp('2025-04-15') and expiry >= pd.Timestamp('2025-04-21') else 3
        before = calendar[calendar < expiry]
        if len(before) < lead+2:
            continue
        event = before[-lead]
        exit_day = calendar[calendar < event][-1]
        decision = calendar[calendar < exit_day][-1]
        events.append({**row, 'lead_sessions':lead,
                       'rule_known_at':'2025-04-15T10:03:00+03:00' if lead == 10 else '2022-10-20T17:40:00+03:00',
                       'rule_authority':'MOEX_N89495_PRIMARY' if lead == 10 else 'MOEX_N52418_PRIMARY',
                       'rule_effective_from':'2025-04-21' if lead == 10 else '2022-12-05',
                       'calendar_source':'B01_CAUSAL_FORECAST_REGULAR_SESSION_CALENDAR',
                       'validation_notice':'',
                       'status':'CONDITIONAL_PLAN_USING_ORIGIN_KNOWN_RULE', 'event_session':str(event.date()), 'exit_execution_session':str(exit_day.date()),
                       'maintenance_decision_session':str(decision.date())})
    expiries, capacities = [], []
    if not listed.empty:
        current = listed[listed.SECID == terms.held]
        if current.empty:
            raise StopCondition('R03_FORECAST_HELD_EXPIRY_UNKNOWN')
        listed = listed[listed.LSTTRADE >= current.LSTTRADE.iloc[0]]
        expiries = [str(x.date()) for x in listed.LSTTRADE]
        capacities = [float(r.VOLUME) if r.NUMTRADES > 0 and r.CLOSE > 0 and np.isfinite(r.CLOSE) else 0.
                      for r in listed.itertuples()]
    residual_series = []
    if route != 'CNYRUBF':
        older = ctx._adaptive_fut[(ctx._adaptive_fut.ASSETCODE == FAMILY[route]) &
                                 (ctx._adaptive_fut.TRADEDATE == date) &
                                 (ctx._adaptive_fut.LSTTRADE > date) &
                                 (ctx._adaptive_fut.LSTTRADE < pd.Timestamp(expiries[0]))]
        pit = ctx._adaptive_pit.loc[FAMILY[route]]
        for row in older.itertuples():
            point = float(pit.point_value_static)
            if route == 'BR_FUT':
                point = row.OPENPOSITIONVALUE/(row.OPENPOSITION*row.SETTLEPRICE) if row.OPENPOSITION > 0 and row.SETTLEPRICE > 0 else np.nan
            raw = ctx._adaptive_fut[(ctx._adaptive_fut.SECID == row.SECID) & (~ctx._adaptive_fut.weekend)].set_index('TRADEDATE').sort_index()
            fee = fee_terms(ctx,route,raw,date,point,pit,float(terms.tick),
                            float(ctx.cfg['costs']['broker_fee_futures_rub_per_contract']))[0] if np.isfinite(point) else np.nan
            residual_series.append({'key':row.SECID,'price':float(row.SETTLEPRICE),'pv':point,'fee':fee,
                'capacity':float(row.VOLUME) if row.NUMTRADES > 0 and row.CLOSE > 0 and np.isfinite(row.CLOSE) else 0.,
                'expiry':str(row.LSTTRADE.date()), **lk_terms(ctx.root, route, row.SECID, date)})
    offset = int(calendar.get_loc(date))
    roll_days = tuple(sorted({int(calendar.searchsorted(pd.Timestamp(x))-5-offset) for x in expiries
                             if 0 < calendar.searchsorted(pd.Timestamp(x))-5-offset <= horizon}))
    series_keys = listed.SECID.tolist() if not listed.empty else [str(terms.held)]
    series_lk = [lk_terms(ctx.root, route, key, date) for key in series_keys]
    return {'held_key':str(terms.held),'residual_series':residual_series,
            'series_keys':series_keys, 'series_lk':series_lk,
            'forecast_lk':forecast_terms(ctx.root, route, terms.held, date),
            **lk_terms(ctx.root, route, terms.held, date),
            'expiries':expiries, 'series_capacities':capacities, 'roll_days':roll_days,
            'dates':dates, 'calendar_evidence':schedule.evidence,
            'capacity':float(terms.execution_capacity) if bool(terms.executable) else 0.,
            'rates':(float(fields.exchange_rate_percent),float(fields.clearing_rate_percent)),
            'broker':float(ctx.cfg['costs']['broker_fee_futures_rub_per_contract']) if route == 'SMLT_FUT' else None,
            'broker_coefficient':float(ctx._adaptive_pit.loc['CNYRUBF' if route == 'CNYRUBF' else FAMILY[route],'exchange_fee_rate']), 'tick':float(terms.tick),
            'origin_fee':float(terms.fee), 'vm_lag':int(route in ('SMLT_FUT','BR_FUT') and date >= pd.Timestamp('2026-03-23')),
            'qconv':events, 'lk1':float(clocks.authority(str(ctx.root))[0].loc[lambda d: (d.family == FAMILY[route]) & (d.requested_date <= date), 'lk1'].iloc[-1])}


def forward(funded, derivatives, mkt):
    """Translate the same authority to scenario paths without simulated volume."""
    info = next((d.execution_info for d in derivatives if d.execution_info is not None), None)
    if info is None:
        dates = funded[0].execution_dates
    else:
        dates = info['dates']
    N,T,_ = mkt.P.shape
    if len(dates) != T or any(d.execution_info is not None and not dates.equals(d.execution_info['dates']) for d in derivatives) or any(
            f.execution_dates is not None and not dates.equals(f.execution_dates) for f in funded):
        raise StopCondition('B01_FORWARD_CALENDAR_CONSUMER_MISMATCH')
    active = [[None]*len(derivatives) for _ in range(T)]
    series = []
    aliases = []
    vm_lag = np.zeros((T,len(derivatives)),int)
    cny = []
    events = info.get('qconv',[]) if info else []
    for j,d in enumerate(derivatives):
        inf = d.execution_info
        # Model-generated roll segments are not the origin-listed SECID list.
        # Validate the explicit producer contract before assigning its frozen
        # origin-unit convention to every internal synthetic continuation.
        forecast_lk = inf.get('forecast_lk', {})
        expected = {'lk_route':ALIASES.get(d.name, d.name),
                    'lk_base':inf['lk_base'], 'lk_units':inf['lk_units'],
                    'lk_origin_series':inf['held_key'],
                    'lk_origin_classification':inf['lk_classification'],
                    'lk_source_sha256':inf['lk_source_sha256'],
                    'lk_classification':FORECAST_ASSUMPTION}
        if any(forecast_lk.get(k) != v for k,v in expected.items()):
            raise StopCondition('LK_UNIT_MAPPING_UNKNOWN:' + d.name + ':FORECAST_IDENTITY')
        family = {}
        segment = np.cumsum(mkt.roll[:,j]).astype(int)
        # Each forecast series has the frozen gap; residual earlier contracts
        # keep their own mark rather than inheriting a completed roll.
        for k in range(int(segment[-1])+1):
            key = f'{d.name}:FORECAST_SERIES:{k}'
            mark = mkt.Fn[:,:,j]*np.exp((k-segment)[None,:]*d.roll_gap)
            vm_mark = mark.copy()
            if d.name == 'CNY_PERP':
                vm_mark = mkt.Fp[:,:,j].copy()
                if j not in cny:
                    cny.append(j)
            fees = np.empty((N,T))
            ex_rate, cl_rate = inf['rates']
            for n in range(N):
                for t in range(T):
                    # Forecast component prices use the previous same-series
                    # mark. Day zero is the observed decision-origin fee.
                    price = mark[n,max(0,t-1)]
                    pv = mkt.pv[n,t,j]
                    br = inf['broker'] if inf['broker'] is not None else inf['broker_coefficient']*mark[n,t]*pv
                    fees[n,t] = inf['origin_fee'] if t == 0 else component(price,pv,ex_rate)+component(price,pv,cl_rate)+br+.5*inf['tick']*pv
            capacity = inf['series_capacities'][k] if k < len(inf['series_capacities']) else (inf['capacity'] if d.name == 'CNY_PERP' else 0.)
            expiry = inf['expiries'][k] if k < len(inf['expiries']) else None
            fees *= inf.get('cost_multiplier',1.)
            family[key] = {'mark':mark,'vm_mark':vm_mark,'pv':mkt.pv[:,:,j], 'fee':fees,
                           'executable':np.full(T,capacity > 0), 'capacity':np.full(T,capacity),
                           'expiry':expiry, 'rounded_vm':d.name in ('SMLT_FUT','OIL_FUT'),
                           **forecast_lk}
            for t in np.flatnonzero(segment == k):
                active[t][j] = key
        alias = {inf['held_key']:active[0][j]}
        for old in inf.get('residual_series',[]):
            # Preserve the actual origin quote/point value. Future movement is
            # the existing shared route factor with the frozen roll gaps
            # removed, not a new spread or expected-return model.
            factor = mkt.Fn[:,:,j]/d.F0*np.exp(-segment[None,:]*d.roll_gap)
            mark = old['price']*factor
            point = mkt.pv[:,:,j]*(old['pv']/d.pv0)
            fees = np.empty((N,T))
            for n in range(N):
                for t in range(T):
                    price = mark[n,max(0,t-1)]
                    pv = point[n,t]
                    br = inf['broker'] if inf['broker'] is not None else inf['broker_coefficient']*mark[n,t]*pv
                    fees[n,t] = old['fee'] if t == 0 else component(price,pv,ex_rate)+component(price,pv,cl_rate)+br+.5*inf['tick']*pv
            family[old['key']] = {'mark':mark,'vm_mark':mark,'pv':point,
                'fee':fees*inf.get('cost_multiplier',1.), 'executable':np.full(T,old['capacity'] > 0),
                'capacity':np.full(T,old['capacity']), 'expiry':old['expiry'],
                'rounded_vm':d.name in ('SMLT_FUT','OIL_FUT'),
                **{k:old[k] for k in ('lk_base','lk_units','lk_classification','lk_source_sha256')}}
            alias[old['key']] = old['key']
        aliases.append(alias)
        series.append(family)
        vm_lag[:,j] = inf['vm_lag']
    cap_f = np.tile([f.execution_capacity for f in funded],(T,1))
    mkt.tradable_f &= cap_f > 0
    for j in range(len(derivatives)):
        mkt.tradable_d[:,j] &= np.array([series[j][active[t][j]]['executable'][t] for t in range(T)])
    return {'forecast':True,'series_alias':aliases, 'lk_aggregation':AGGREGATION,
            'lk_bases':[d.execution_info['lk_base'] for d in derivatives],
            'calendar_evidence':info.get('calendar_evidence') if info else getattr(funded[0], 'execution_calendar_evidence', None),
            'dates':dates, 'active':active, 'series':series, 'funded_keys':[f.execution_key for f in funded],
            'capacity_f':cap_f, 'mr_policy':mkt.mr, 'mr_clearing':mkt.mr, 'vm_lag':vm_lag,
            'lk1':np.tile([d.execution_info['lk1'] for d in derivatives],(T,1)),
            'lk_units':[d.execution_info['lk_units'] for d in derivatives], 'qconv':events,'cny_indices':cny}
