"""Bounded D11 corrections and explicit historical execution-evidence qualifications.

The official MOEX announcement proves the CNY tick change; its extracted tool response
is archived. Publisher HTML could not be retrieved, so that raw-body evidence is missing.
Historical tariff RATE remains the predeclared cutoff assumption, not an as-of claim.
"""
from pathlib import Path
import numpy as np
import pandas as pd


def repair(ctx, route, out, pit, broker_fee_smlt):
    out = out.copy()
    out["tick"] = float(pit["min_step"])
    out["tick_status"] = "CUTOFF_REUSED"
    out["fee_rate"] = float(pit["exchange_fee_rate"])
    out["fee_rate_status"] = "CUTOFF_REUSED"
    out["fee_evidence"] = "HISTORICAL_COST_ASSUMPTION_EVIDENCE_DOWNGRADED"
    if route == "CNYRUBF":
        src = pd.read_csv(Path(ctx.root)/"data/regime_v2r1_derivative_term_overrides.csv")
        for row in src.itertuples():
            if row.route != route: continue
            mask = (out.index >= pd.Timestamp(row.valid_from)) & (out.index <= pd.Timestamp(row.valid_to))
            out.loc[mask,"tick"] = row.tick
            out.loc[mask,"tick_status"] = "AS_OF_VERIFIED"
        out["tick_raw_publisher_body_status"] = "EVIDENCE_MISSING"
    # Exchange OI/notional provides a contemporaneous cross-check of RUB point value.
    if route != "BR_FUT":
        implied = out["oi_value"] / (out["oi"] * out["Fn"])
        valid = np.isfinite(implied) & (out["oi"] > 0)
        same = np.isclose(implied, float(pit["point_value_static"]), rtol=1e-7, atol=1e-9)
        out.loc[valid & ~same,"pv"] = implied[valid & ~same]
        out.loc[valid,"pv_source"] = np.where(same[valid],"STATIC_BUT_EVIDENCED_INVARIANT","EXCHANGE_OI_IMPLIED")
    notional = out["Fn"] * out["pv"]
    broker = broker_fee_smlt if route == "SMLT_FUT" else out["fee_rate"] * notional
    out["fee"] = out["fee_rate"] * notional + broker + .5 * out["tick"] * out["pv"]
    out["notional"] = notional
    return out
