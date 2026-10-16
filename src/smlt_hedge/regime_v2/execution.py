"""Bounded V2R2 execution authority used by the shared quantity Engine.

Valuation is independent of executable availability. Budgets are per real
instrument/session and per simulated account, at the maximum observed tape
bound. Per-series holdings survive partial rolls; no delivery is fabricated.
"""
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd

from .cfg import StopCondition


def rounded(x, places):
    return float(Decimal(str(float(x))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def component(price, point_value, percent):
    return max(.01, rounded(rounded(abs(price) * rounded(point_value, 5), 2) * percent / 100, 2))


class Execution:
    def __init__(self, engine, qd, series=None, capacity=None):
        self.e, self.a = engine, engine.m.execution
        self.N = engine.N
        self.positions = [{} for _ in range(engine.Ad)]
        self.records = []
        self.lk_records = []
        self.initial_capacity = capacity or {}
        self.t = 0
        self.margin_clock = 'policy'
        aliases = self.a.get('series_alias',[{} for _ in range(engine.Ad)])
        for j in range(engine.Ad):
            source = series[j] if series is not None else {self.active(0,j):qd[:,j]}
            for rawkey, quantity in source.items():
                q = np.broadcast_to(np.asarray(quantity,float),(self.N,)).copy()
                if not np.any(q):
                    continue
                key = aliases[j].get(rawkey,rawkey)
                if key not in self.a['series'][j]:
                    raise StopCondition('FA_B_ACTUAL_HELD_SERIES_INPUT_MISSING:' + str(rawkey))
                self.positions[j][key] = self.positions[j].get(key,np.zeros(self.N))+q
        if not np.array_equal(self.quantity(),qd):
            raise StopCondition('FA_B_ACTUAL_SERIES_QUANTITY_MISMATCH')
        self.begin(0)

    def active(self, t, j):
        return self.a['active'][t][j]

    def begin(self, t):
        self.t = t
        for j, instrument in enumerate(self.e.din):
            if instrument.name == 'SMLT_FUT':
                keys = {self.active(t,j), *self.positions[j]}
                keys.update(key for key,data in self.a['series'][j].items() if data['executable'][t])
                for key in keys:
                    self.share_multiplier(j,key)
        self.e._fill_scope=None
        self.margin_clock = 'policy'
        self.budget = {}
        # Shared identifiers make repeated routes/actions spend the same budget.
        for j, series in enumerate(self.a['series']):
            for key, data in series.items():
                cap = float(data['capacity'][t]) if data['executable'][t] else 0.
                if not np.isfinite(cap) or cap < 0:
                    cap = 0.
                self.budget[key] = np.full(self.N, cap)
        for j, key in enumerate(self.a['funded_keys']):
            cap = float(self.a['capacity_f'][t, j]) if self.e.m.tradable_f[t, j] else 0.
            if not np.isfinite(cap) or cap < 0:
                cap = 0.
            self.budget[key] = np.full(self.N, cap)
        if t == 0 and self.initial_capacity:
            aliases = {raw:key for group in self.a.get('series_alias',[]) for raw,key in group.items()}
            for raw,remaining in self.initial_capacity.items():
                key = aliases.get(raw,raw)
                if key in self.budget:
                    self.budget[key] = np.minimum(self.budget[key],np.broadcast_to(np.asarray(remaining,float),(self.N,)))
        date = pd.Timestamp(self.a['dates'][t])
        for j, positions in enumerate(self.positions):
            for key, q in positions.items():
                expiry = self.a['series'][j][key].get('expiry')
                if expiry is not None and date >= pd.Timestamp(expiry) and np.any(q != 0):
                    raise StopCondition(f'UNMODELED_EXCEPTIONAL_SETTLEMENT:{key}:expiry:{date.date()}')
        for row in self.a.get('qconv', []):
            if date == pd.Timestamp(row['event_session']):
                for j in self.a.get('cny_indices', []):
                    if np.any(self.quantity()[:, j] != 0):
                        raise StopCondition('UNMODELED_EXCEPTIONAL_SETTLEMENT:CNY_QCONV:' + str(date.date()))
        self.lk_guard()

    def quantity(self):
        return np.stack([sum(p.values(), np.zeros(self.N)) for p in self.positions], axis=1) if self.positions else np.zeros((self.N, 0))

    def blocked(self, j):
        date = pd.Timestamp(self.a['dates'][self.t])
        return j in self.a.get('cny_indices', []) and any(
            pd.Timestamp(r['maintenance_decision_session']) <= date <= pd.Timestamp(r['event_session'])
            for r in self.a.get('qconv', []))

    def maintenance(self):
        date = pd.Timestamp(self.a['dates'][self.t])
        return [j for j in self.a.get('cny_indices', []) if any(
            pd.Timestamp(r['exit_execution_session']) <= date < pd.Timestamp(r['event_session'])
            for r in self.a.get('qconv', []))]

    def data(self, j, key, field):
        x = np.asarray(self.a['series'][j][key][field])
        return np.broadcast_to(x[:, self.t] if x.ndim == 2 else x[self.t], (self.N,))

    def lk_guard(self):
        from .lk import guard
        reports = guard(self.a, self.positions, self.t, self.N)
        self.lk_records.append({'day': self.t, 'underlyings': reports})
        return reports

    def margin(self):
        self.lk_guard()
        rates = self.a['mr_policy'] if self.margin_clock == 'policy' else self.a['mr_clearing']
        return self.e.p.mult * sum((sum((np.abs(q) * self.data(j, key, 'mark') * self.data(j, key, 'pv')
                  for key, q in p.items() if np.any(q)), np.zeros(self.N)) * rates[self.t, j]
                  for j, p in enumerate(self.positions)), np.zeros(self.N))

    def gross(self):
        return sum((sum((np.abs(q) * self.data(j, key, 'mark') * self.data(j, key, 'pv')
                         for key, q in p.items() if np.any(q)), np.zeros(self.N)) for j, p in enumerate(self.positions)), np.zeros(self.N))

    def signed_notional(self):
        return np.stack([sum((q*self.data(j,key,'mark')*self.data(j,key,'pv')
                            for key,q in p.items() if np.any(q)),np.zeros(self.N))
                         for j,p in enumerate(self.positions)],axis=1) if self.positions else np.zeros((self.N,0))

    def share_multiplier(self, j, key):
        """Frozen certified SMLT LK unit map is contractual shares, not RUB pv.

        REGIME_V2_METHOD_FREEZE section 1 and the per-series Master unit map
        certify this identity. Explicit missing series authority never falls back.
        """
        data = self.a['series'][j][key]
        value = data.get('lk_units')
        try:
            valid = bool(value is not None and np.isfinite(value) and value > 0)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise StopCondition('LK_UNIT_MAPPING_UNKNOWN:SMLT_SHARE_EQUIVALENT_NOT_EVALUABLE:'+str(key))
        return float(value)

    def share_equivalents(self, j):
        self.share_multiplier(j, self.active(self.t,j))
        return sum((q*self.share_multiplier(j,key) for key,q in self.positions[j].items()),np.zeros(self.N))

    def vm(self):
        total = np.zeros((self.N, self.e.Ad))
        if self.t == 0:
            return total
        for j, p in enumerate(self.positions):
            for key, q in p.items():
                if not np.any(q):
                    continue
                data = self.a['series'][j][key]
                current = self.data(j, key, 'vm_mark')
                previous = np.broadcast_to(np.asarray(data['mark'])[:, self.t-1], (self.N,))
                pv = self.data(j, key, 'pv')
                if not np.isfinite(current[q != 0]).all() or not np.isfinite(previous[q != 0]).all() or not np.isfinite(pv[q != 0]).all():
                    raise StopCondition('R03_HELD_SERIES_MARK_MISSING:' + key)
                if data.get('rounded_vm', False):
                    legs = np.array([rounded(c*rounded(v,5), 2)-rounded(o*rounded(v,5), 2) for c, o, v in zip(current, previous, pv)])
                    total[:, j] += q * legs
                else:
                    total[:, j] += q * (current - previous) * pv
        return total

    def consume(self, key, wanted, lot=1.):
        remaining = self.budget[key]
        amount = np.minimum(np.abs(wanted), remaining)
        if lot > 0:
            amount = np.floor(amount / lot + 1e-12) * lot
        fill = np.sign(wanted) * amount
        self.budget[key] -= np.abs(fill)
        return fill

    def funded(self, j, wanted):
        return self.consume(self.a['funded_keys'][j], wanted, self.e.lot[j])

    def fill(self, j, key, wanted, tag):
        filled = self.consume(key, wanted)
        p = self.positions[j]
        p[key] = p.get(key, np.zeros(self.N)) + filled
        fee_input = self.data(j, key, 'fee')
        if not np.isfinite(fee_input[filled != 0]).all():
            raise StopCondition('R03_COMPONENT_PRICE_OR_CONVERSION_MISSING:' + key)
        fee = np.abs(filled) * np.where(filled != 0, fee_input, 0.)
        turnover = np.abs(filled) * np.where(filled != 0, self.data(j, key, 'mark') * self.data(j, key, 'pv'), 0.)
        self.records.append({'day': self.t, 'instrument': key, 'tag': tag, 'requested': wanted.copy(),
                             'filled': filled.copy(), 'fee': fee.copy(), 'capacity_left': self.budget[key].copy(),
                             'local_intent':np.asarray(getattr(self.e,'_fill_scope',None),bool).copy()
                                if getattr(self.e,'_fill_scope',None) is not None else (wanted!=0).copy()})
        return fee, turnover

    def target(self, j, target, tag):
        current = self.quantity()[:, j]
        target = np.asarray(target, float).copy()
        if self.blocked(j):
            # Maintenance has priority over alpha, while all closes still need tape.
            target = np.where((np.sign(target) != np.sign(current)) | (np.abs(target) > np.abs(current)), 0., target)
        cost, turnover = np.zeros(self.N), np.zeros(self.N)
        remaining = target - current
        active = self.active(self.t, j)
        # Exit residual old contracts first; all orders share the same capacity.
        for key in sorted(self.positions[j], key=lambda k: (k == active, k)):
            q = self.positions[j][key]
            close = np.where(np.sign(remaining) == -np.sign(q), np.sign(remaining)*np.minimum(np.abs(remaining), np.abs(q)), 0.)
            before = q.copy()
            c, tr = self.fill(j, key, close, tag)
            remaining -= self.positions[j][key] - before
            cost += c; turnover += tr
        # Cannot open the opposite sign while a close is still unfilled.
        current = self.quantity()[:, j]
        add = np.where((current == 0) | (np.sign(remaining) == np.sign(current)), remaining, 0.)
        if self.blocked(j):
            add[:] = 0.
        if active in self.a['series'][j]:
            c, tr = self.fill(j, active, add, tag)
            cost += c; turnover += tr
        self.lk_guard()
        return cost, turnover

    def rolls(self):
        cost, count = np.zeros(self.N), np.zeros(self.N)
        if self.t == 0 and self.a.get('forecast',False):
            return cost,count
        for j, p in enumerate(self.positions):
            active = self.active(self.t, j)
            for old in list(p):
                if old == active or not np.any(p[old]):
                    continue
                # Both real legs must fit their separate tape budgets. Preserve
                # all unmatched old quantities and mark them on their own series.
                if active not in self.budget:
                    continue
                qty = np.minimum(np.abs(p[old]), np.minimum(self.budget[old], self.budget[active]))
                qty = np.floor(qty)
                signed = np.sign(p[old]) * qty
                c1, _ = self.fill(j, old, -signed, 'ROLL_CLOSE')
                c2, _ = self.fill(j, active, signed, 'ROLL_OPEN')
                cost += c1+c2; count += qty
        return cost, count
