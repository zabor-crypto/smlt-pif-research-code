"""Causal non-numeric context. No axis, return coefficient, score or issuer cap."""
import json
from pathlib import Path

import pandas as pd

REPLAY_QUALIFICATION = ("Replay is conditional on captured primary issuer/rating chronology and bounded "
                        "official index coverage. Uncaptured primary severe events may alter issuer caps.")
ISSUER_CAP_SCOPE = "DIRECT_SMLT_ONLY"


def issuer_disclosure(direct_shares, direct_futures, indirect=None):
    return {"issuer_cap_scope": ISSUER_CAP_SCOPE,
            "direct_smlt_exposure": direct_shares + direct_futures,
            "indirect_smlt_exposure": "UNKNOWN" if indirect is None else indirect,
            "wording": "Direct SMLT exposure; diversified-product indirect exposure is separately disclosed."}


def housing(events, at, rates_state):
    cut = pd.Timestamp(at)
    known = [e for e in events if e.get("authority") == "VERIFIED_PRIMARY"
             and e.get("published_at") and e.get("effective_at")
             and pd.Timestamp(e["published_at"]) <= cut
             and pd.Timestamp(e["effective_at"]) <= cut
             and pd.Timestamp(e.get("active_from", e["published_at"])) <= cut
             and (not e.get("resolved_at") or cut < pd.Timestamp(e["resolved_at"]))]
    conflict = rates_state == "EASING" and any(e["action"] == "TIGHTENING" for e in known)
    return {"HOUSING_POLICY_CONFLICT": conflict,
            "HOUSING_CONTEXT_STATUS": "VERIFIED_PRIMARY_CONTEXT" if known else "UNKNOWN",
            "RATE_INTERPRETATION": "HOUSING_RELIEF_NOT_CONFIRMED" if conflict else ("REVIEW_CONTEXT_REQUIRED" if known else "NO_HOUSING_AUTHORITY"),
            "positive_re_risk_confirmation": False if conflict else None,
            "event_ids": [e["event_id"] for e in known]}


def housing_at(root, at, rates_state):
    p = Path(root) / "data/regime_v2r2_combined/housing_events.json"
    return housing(json.loads(p.read_text()) if p.exists() else [], at, rates_state)


def maturity_review(calendar, at, sessions):
    """Only public causal rows; issue face value never becomes outstanding debt."""
    cut = pd.Timestamp(at)
    known = calendar[pd.to_datetime(calendar.KNOWN_AT_DATE, errors="coerce") <= cut].copy()
    dates = pd.to_datetime(known.scheduled_date, errors="coerce")
    future = pd.DatetimeIndex(sessions)[pd.DatetimeIndex(sessions) > cut]
    horizons = {}
    for n in (21, 63, 126):
        end = future[n - 1] if len(future) >= n else None
        selected = known[(dates > cut) & (dates <= end)] if end is not None else known.iloc[:0]
        horizons[n] = {"end_session": str(end.date()) if end is not None else "UNKNOWN",
                       "events": selected.to_dict("records"), "outstanding_amount_RUB": "UNKNOWN"}
    return {"status": "REVIEW_ONLY", "mandate_acknowledgement": "Original debt/maturity-wall channel retained",
            "coverage": "PARTIAL_PIT_COVERAGE", "horizons": horizons}


def corrected_register(ctx, cfg):
    from ..adaptive import events as EV
    result = EV.register(ctx, cfg)
    corrections = pd.read_csv(Path(ctx.root) / "data/regime_v2r2_combined/rating_chronology.csv").set_index("event_id")
    for eid in ("E01", "E03", "E04", "E07"):
        r = corrections.loc[eid]
        mask = result.event_id == eid
        result.loc[mask, "publicly_known_at"] = r.publicly_known_at
        result.loc[mask, "economic_date"] = r.publicly_known_at
        result.loc[mask, "active_from"] = r.proposed_active_from
        result.loc[mask, "verification_status"] = r.verification_status
        result.loc[mask, "authority"] = EV.AUTHORITY[r.verification_status]
        if eid != "E07":
            result.loc[mask, "url"] = r.source_url
            result.loc[mask, "source"] = "ACRA"
        result.loc[mask, "note"] = r.action + "; " + r.qualification
    if (result.event_id == "E03").any():
        result.loc[result.event_id == "E03", "resolved_at"] = "2026-03-03"
    result["active_until"] = [EV.end_date(r, result) for _, r in result.iterrows()]
    result["replay_qualification"] = REPLAY_QUALIFICATION
    return result


def event_context(register, at):
    cut = pd.Timestamp(at)
    active = register[(pd.to_datetime(register.active_from) <= cut) & (pd.to_datetime(register.active_until) > cut)]
    return {"active_ids": active.event_id.tolist(),
            "primary_active_ids": active.loc[active.authority == "VERIFIED_PRIMARY", "event_id"].tolist(),
            "secondary_only_ids": active.loc[active.authority == "VERIFIED_SECONDARY", "event_id"].tolist(),
            "qualification": REPLAY_QUALIFICATION}
