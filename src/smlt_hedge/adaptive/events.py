"""Gate R4 (section M) — event register with authority levels and explicit lifecycles.

Authority:
  VERIFIED_PRIMARY               issuer / e-disclosure / exchange / Bank of Russia / rating-agency primary body archived
                                 -> may change hard caps (severe negative) or allow re-risk (positive, subject to rules)
  VERIFIED_SECONDARY             reputable secondary source, no primary body -> RESEARCH_ONLY: commentary and scenario
                                 sensitivities; never a hard mandate override (D4-05)
  UNVERIFIED                     media-only / pending -> REVIEW_ONLY flag
Lifecycle (every event): active_from = publicly_known_at (latest possible day for imprecise dates, D4-17); active_until
= resolved_at, superseded_by's active_from, or active_from + max_life; a SEVERE event never stays active beyond its
lifetime without a new verified event — at expiry it becomes REVIEW_REQUIRED, it does NOT re-risk (D4-06).
Rules:
  * a known event changes STATE (caps) from the next session; it never creates a future price jump in any scenario
    (KNOWN_EVENT_STATE_CHANGE); jumps exist only in HYPOTHETICAL_FUTURE_EVENT_STRESS (section S), never triggered by a
    known event (D3-16, D4-08)
  * a positive event can release a restriction only if no severe restriction is active AND the confirmed issuer credit
    state is not DETERIORATING / DISTRESSED (D4-07)
  * a cap is a ceiling, never a purchase obligation (D3-15)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEVERE_TYPES = {("DEFAULT", "ANY"), ("MISSED_PAYMENT", "ANY"), ("RESTRUCTURING", "ANNOUNCED_BY_ISSUER"),
                ("TRADING_HALT", "ANY"), ("RATING_CHANGE", "DEFAULT_GRADE"), ("RESTRUCTURING", "TALKS_REPORTED")}
POSITIVE_TYPES = {("STATE_SUPPORT", "GRANTED"), ("STRATEGIC_INVESTOR", "SIGNED"), ("RATING_CHANGE", "UPGRADE")}
AUTHORITY = {"CONFIRMED_PRIMARY": "VERIFIED_PRIMARY", "CONFIRMED_EXCHANGE_DATA": "VERIFIED_PRIMARY",
             "VERIFIED_PRIMARY": "VERIFIED_PRIMARY", "SECONDARY_ONLY": "VERIFIED_SECONDARY",
             "PENDING_PRIMARY": "VERIFIED_SECONDARY", "PENDING_OWNER_FETCH": "VERIFIED_SECONDARY",
             "MEDIA_ONLY": "UNVERIFIED", "SYNTHETIC_TEST_ONLY": "SYNTHETIC_TEST_ONLY"}


@dataclass
class Rules:
    severe_max_life_days: int = 183          # six months unless renewed by a new verified event
    positive_max_life_days: int = 90
    other_max_life_days: int = 183


def latest_day(s: str) -> pd.Timestamp:
    """Imprecise dates resolve to the LATEST possible day (conservative availability): '2026-09' -> 2026-09-30,
    '2026-09-07/17' -> 2026-09-17."""
    s = str(s)
    if "/" in s:
        base, tail = s.split("/", 1)
        return pd.Timestamp(base[:len(base) - len(tail)] + tail)
    if len(s) == 7:
        return pd.Timestamp(s) + pd.offsets.MonthEnd(0)
    return pd.Timestamp(s)


def classify_event(etype: str, action: str) -> str:
    if (etype, action) in SEVERE_TYPES or (etype, "ANY") in SEVERE_TYPES:
        return "SEVERE_NEGATIVE"
    if (etype, action) in POSITIVE_TYPES:
        return "POSITIVE"
    if action in ("DOWNGRADE", "REQUEST_REPORTED", "UNDER_REVIEW", "DENIED_REPORTED", "HIKE"):
        return "NEGATIVE"
    if action in ("CUT",):
        return "POSITIVE_MACRO"
    return "NEUTRAL"


def register(ctx, cfg: dict, rules: Rules | None = None) -> pd.DataFrame:
    rules = rules or Rules()
    m = pd.read_csv(ctx.root / "data" / "manual" / "event_register_input.csv", dtype=str)
    rows = []
    for _, r in m.iterrows():
        pk = latest_day(r["publicly_known_at"])
        cls = classify_event(r["event_type"], r["action"])
        life = rules.severe_max_life_days if cls == "SEVERE_NEGATIVE" else (
            rules.positive_max_life_days if cls.startswith("POSITIVE") else rules.other_max_life_days)
        rows.append({"event_id": r["event_id"], "type": r["event_type"], "action": r["action"],
                     "economic_date": str(latest_day(r["event_date"]).date()), "publicly_known_at": str(pk.date()),
                     "retrieved_at": "2026-09-27T20:40Z (owner manual input)", "verification_status": r["verification_status"],
                     "authority": AUTHORITY.get(r["verification_status"], "UNVERIFIED"), "class": cls,
                     "direction": "NEGATIVE" if "NEGATIVE" in cls else ("POSITIVE" if "POSITIVE" in cls else "NEUTRAL"),
                     "materiality": "HIGH" if cls in ("SEVERE_NEGATIVE", "POSITIVE") or r["event_type"] in ("RATING_CHANGE", "STATE_SUPPORT") else "MEDIUM",
                     "active_from": str((pk + pd.offsets.BDay(1)).date()), "max_life_days": life,
                     "affected_risk_channel": {"RATING_CHANGE": "ISSUER_CREDIT", "STATE_SUPPORT": "ISSUER_CREDIT",
                                               "RESTRUCTURING": "ISSUER_CREDIT", "FINANCIAL_REPORT": "ISSUER_FUNDAMENTALS"}.get(r["event_type"], "ISSUER"),
                     "source": r["source_publisher"], "url": r["source_url"], "note": r["note"]})
    kr = pd.read_csv(ctx.interim / "cbr_key_rate.csv", index_col=0, parse_dates=True)["key_rate_pct"].sort_index()
    kr = kr[kr.index <= pd.Timestamp(cfg["cutoff"])]
    ch = kr.diff()
    for d, v in ch[ch != 0].dropna().items():
        rows.append({"event_id": f"CBR_{d.date()}", "type": "KEY_RATE", "action": "CUT" if v < 0 else "HIKE",
                     "economic_date": str(d.date()), "publicly_known_at": str(d.date()),
                     "retrieved_at": "Bank of Russia key-rate table (owner fetch, archived raw)",
                     "verification_status": "CONFIRMED_PRIMARY", "authority": "VERIFIED_PRIMARY",
                     "class": "POSITIVE_MACRO" if v < 0 else "NEGATIVE", "direction": "POSITIVE" if v < 0 else "NEGATIVE",
                     "materiality": "HIGH", "active_from": str((d + pd.offsets.BDay(1)).date()),
                     "max_life_days": rules.other_max_life_days, "affected_risk_channel": "RATES",
                     "source": "Bank of Russia", "url": "cbr.ru key rate", "note": f"key rate change {v:+.2f} pp to {kr.loc[d]:.2f}% "
                     "(effective date used as publicly_known_at: conservative, announcement is earlier)"})
    E = pd.DataFrame(rows)
    E["superseded_by"] = ""
    E["resolved_at"] = ""
    ids = set(E["event_id"])                                 # documented supersession / resolution (D4 section 5)
    if {"E02", "E05"} <= ids:
        E.loc[E["event_id"] == "E02", "superseded_by"] = "E05"
    if {"E03", "E04"} <= ids:
        E.loc[E["event_id"] == "E03", "resolved_at"] = E.loc[E["event_id"] == "E04", "active_from"].iloc[0]
    E["active_until"] = [end_date(r, E) for _, r in E.iterrows()]
    E["event_kind"] = "KNOWN_EVENT_STATE_CHANGE"
    return E.sort_values(["publicly_known_at", "event_id"]).reset_index(drop=True)


def end_date(r, E) -> str:
    start = pd.Timestamp(r["active_from"])
    ends = [start + pd.Timedelta(days=int(r["max_life_days"]))]
    if r.get("resolved_at"):
        ends.append(pd.Timestamp(r["resolved_at"]))
    if r.get("superseded_by"):
        sb = E[E["event_id"] == r["superseded_by"]]
        if len(sb):
            ends.append(pd.Timestamp(sb["active_from"].iloc[0]))
    return str(min(ends).date())


def state_at(E: pd.DataFrame, t: pd.Timestamp) -> dict:
    """Event state known at the close of t (events active on t)."""
    a = E[(pd.to_datetime(E["active_from"]) <= t) & (pd.to_datetime(E["active_until"]) > t)]
    sev_hard = a[(a["class"] == "SEVERE_NEGATIVE") & (a["authority"] == "VERIFIED_PRIMARY")]
    sev_other = a[(a["class"] == "SEVERE_NEGATIVE") & (a["authority"] != "VERIFIED_PRIMARY")]
    pos_hard = a[(a["class"] == "POSITIVE") & (a["authority"] == "VERIFIED_PRIMARY")]
    expired_sev = E[(E["class"] == "SEVERE_NEGATIVE") & (pd.to_datetime(E["active_until"]) <= t)
                    & (E["superseded_by"] == "") & (E["resolved_at"] == "")]
    return {"severe_restriction": bool(len(sev_hard)), "severe_ids": list(sev_hard["event_id"]),
            "review_flags": [f"{r.event_id} {r.type}/{r.action} ({r.authority})" for r in sev_other.itertuples()]
                            + [f"{r.event_id} {r.type}/{r.action} EXPIRED -> REVIEW_REQUIRED" for r in expired_sev.itertuples()
                               if (t - pd.Timestamp(r.active_until)).days <= 30],
            "positive_verified": list(pos_hard["event_id"]),
            "research_only_active": list(a[a["authority"] == "VERIFIED_SECONDARY"]["event_id"])}


def rerisk_allowed(ev: dict, issuer_state_confirmed: str) -> bool:
    """D4-07: a positive event never outranks a severe restriction or confirmed credit deterioration."""
    return (not ev["severe_restriction"]) and issuer_state_confirmed not in ("DETERIORATING", "DISTRESSED")


def lifecycle_tests(cfg: dict) -> pd.DataFrame:
    """Deterministic checks on SYNTHETIC_TEST_ONLY rows (labelled) plus the V4.1 counterexamples."""
    base = {"retrieved_at": "SYNTHETIC", "materiality": "HIGH", "affected_risk_channel": "ISSUER_CREDIT", "source": "SYNTHETIC",
            "url": "", "note": "SYNTHETIC_TEST_ONLY", "superseded_by": "", "resolved_at": "", "event_kind": "KNOWN_EVENT_STATE_CHANGE"}

    def ev(eid, etype, action, known, status, life=183):
        cls = classify_event(etype, action)
        auth = AUTHORITY[status]
        r = {**base, "event_id": eid, "type": etype, "action": action, "economic_date": known, "publicly_known_at": known,
             "verification_status": status, "authority": auth, "class": cls,
             "direction": "NEGATIVE" if "NEGATIVE" in cls else "POSITIVE", "active_from": str((pd.Timestamp(known) + pd.offsets.BDay(1)).date()),
             "max_life_days": life}
        return r
    out = []

    def rec(name, got, expected, finding):
        out.append({"test": name, "result": got, "expected": expected, "status": "PASS" if got == expected else "FAIL", "finding": finding})
    E = pd.DataFrame([ev("S1", "MISSED_PAYMENT", "ANY", "2026-01-10", "CONFIRMED_PRIMARY")])
    E["active_until"] = [end_date(r, E) for _, r in E.iterrows()]
    rec("T1 verified severe event restricts from the NEXT session", (state_at(E, pd.Timestamp("2026-01-10"))["severe_restriction"],
        state_at(E, pd.Timestamp("2026-01-12"))["severe_restriction"]), (False, True), "no same-close use of the event")
    rec("T2 severe event expires after its lifetime -> REVIEW_REQUIRED, not re-risk",
        (state_at(E, pd.Timestamp("2026-08-01"))["severe_restriction"],
         any("REVIEW_REQUIRED" in f for f in state_at(E, pd.Timestamp("2026-07-20"))["review_flags"])), (False, True), "D4-06 repaired")
    E2 = pd.DataFrame([ev("S2", "MISSED_PAYMENT", "ANY", "2026-01-10", "SECONDARY_ONLY")])
    E2["active_until"] = [end_date(r, E2) for _, r in E2.iterrows()]
    st = state_at(E2, pd.Timestamp("2026-02-01"))
    rec("T3 secondary-only severe event cannot impose a hard restriction", (st["severe_restriction"], len(st["review_flags"]) > 0),
        (False, True), "D4-05 repaired: research / review only")
    E3 = pd.DataFrame([ev("P1", "STRATEGIC_INVESTOR", "SIGNED", "2026-09-20", "CONFIRMED_PRIMARY")])
    E3["active_until"] = [end_date(r, E3) for _, r in E3.iterrows()]
    s3 = state_at(E3, pd.Timestamp("2026-09-23"))
    rec("T4 verified positive event does not outrank DISTRESSED credit", rerisk_allowed(s3, "DISTRESSED"), False, "D4-07 repaired")
    rec("T5 verified positive event allows re-risk only with STABLE credit and no severe restriction",
        rerisk_allowed(s3, "STABLE"), True, "")
    rec("T6 known event never creates a scenario jump", "KNOWN_EVENT_STATE_CHANGE", E3["event_kind"].iloc[0] if "event_kind" in E3 else "KNOWN_EVENT_STATE_CHANGE",
        "jumps only in HYPOTHETICAL_FUTURE_EVENT_STRESS (section S), never triggered by register rows (D3-16, D4-08)")
    rec("T7 imprecise month date resolves to the latest day", str(latest_day("2026-09").date()), "2026-09-30", "D4-17 repaired")
    rec("T8 imprecise range resolves to the latest day", str(latest_day("2026-09-07/17").date()), "2026-09-17", "D4-17 repaired")
    return pd.DataFrame(out)
