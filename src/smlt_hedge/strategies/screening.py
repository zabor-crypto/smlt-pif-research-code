"""Hard instrument gates G1-G5 and executability classification.

Classification rules (applied in order):
  THEORETICAL_ONLY  : OTC route with no public, point-in-time evidence of quote/counterparty for the notional.
  REJECTED          : fails G1 (no instrument), G3 (data), G4 (liquidity) or, for a HEDGE role, G5 (relevance);
                      also exchange items with no tradable wrapper (index-only).
  LIMITED_EXECUTABILITY : passes hard gates but with a binding practical constraint
                      (broker short-borrow not verifiable; only short-dated options; dividend-unadjusted data).
  EXECUTABLE        : passes G1-G5 with no binding caveat.
Only EXECUTABLE and LIMITED_EXECUTABILITY instruments may enter the optimiser; LIMITED ones only in
sensitivity runs unless explicitly allowed. THEORETICAL_ONLY never enters the main optimiser.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HEDGE_ROLES = {"DIRECT_HEDGE", "INDEX_CROSS_HEDGE", "SECTOR_CROSS_HEDGE", "RATE_FACTOR_HEDGE", "CREDIT_HEDGE"}


def screen(universe: list[dict], gates: dict, stats: pd.DataFrame, P: pd.DataFrame, R: pd.DataFrame,
           V: pd.DataFrame, fut_oi: dict, contract_notional: dict, cutoff: str, exposure_rub: float,
           special: dict | None = None, hedge_capital_rub: float | None = None) -> pd.DataFrame:
    special = special or {}
    st = stats.set_index("series") if "series" in stats else stats
    t = pd.Timestamp(cutoff)
    two_y = R.index >= t - pd.Timedelta(days=730)
    out = []
    for c in universe:
        s = c.get("series")
        row = {k: c.get(k) for k in ("id", "series", "role", "sleeve", "access", "hedge_direction", "asset", "notes")}
        reasons = []
        # ---------------- G1 existence
        if c["access"] == "OTC":
            g1 = "UNVERIFIED"
            reasons.append("OTC: legal possibility only; no public PIT quote/counterparty evidence")
        elif c["id"] in special and "G1" in special[c["id"]]:
            g1 = special[c["id"]]["G1"]
        elif c["id"] in special:
            g1 = special[c["id"]].get("G1", "PASS")
        elif s is None or s not in P or pd.isna(P[s].get(t)):
            g1 = "FAIL"
            reasons.append("no price on decision date")
        else:
            g1 = "PASS"
        # ---------------- G2 accessibility
        if c["access"] == "EXCHANGE":
            g2 = "PASS"
        elif c["access"] == "EXCHANGE_SHORT_SPOT":
            g2 = "CONDITIONAL"
            reasons.append("requires broker securities lending; borrow availability/fee not verifiable point-in-time")
        elif c["access"] == "NONE":
            g2 = "FAIL"
            reasons.append("index only: no tradable futures/ETF wrapper")
        elif c["access"] == "INSTRUCTOR_AUTHORIZED":
            g2 = "ASSUMED (instructor-authorised)"
        elif c["access"] == "BORROW_UNEVIDENCED":
            g2 = "FAIL_UNEVIDENCED"
            reasons.append("short requires borrowing the specific issue; no evidenced lending/repo route for the investor")
        else:
            g2 = "FAIL_UNEVIDENCED"
        # ---------------- G3 data quality
        g3, g4 = "N/A", "N/A"
        if s is not None and s in R:
            r2 = R.loc[two_y, s]
            first = P[s].first_valid_index()
            live = r2[r2.index >= first] if first is not None else r2
            n = int(live.notna().sum())
            miss = float(live.isna().mean()) if len(live) else 1.0
            row.update({"obs_last_2y": n, "missing_share_last_2y": miss})
            g3 = "PASS" if (n >= gates["G3_min_obs_last_2y"] and miss <= gates["G3_max_missing_share"]) else "FAIL"
            if g3 == "FAIL":
                reasons.append(f"data: {n} obs in 2y, missing {miss:.1%}")
                if special.get(c["id"], {}).get("proxy_history"):
                    g3 = "LIMITED"          # short EXECUTION history; economic history carried by a validated proxy
                    reasons.append(special[c["id"]]["proxy_history"])
            if c["role"] == "DIVERSIFIER" and not s.endswith(("_FUT", "_TOM", "RUBF")) and s not in ("IMOEXF",):
                g3 = "LIMITED"
                reasons.append("single-stock price return: dividend register unavailable from ISS -> ex-div gaps unadjusted")
            # ---------------- G4 liquidity
            adv = float(V[s].loc[V.index <= t].tail(60).mean()) if s in V else np.nan
            row["adv60_rub"] = adv
            need = exposure_rub * abs(st.loc[s, "hstar_250d"]) if (s in st.index and "hstar_250d" in st and
                                                                   pd.notna(st.loc[s, "hstar_250d"])) else exposure_rub
            need = max(need, 0.25 * exposure_rub)
            if c["role"] == "LIQUIDITY_SLEEVE":
                need = hedge_capital_rub if hedge_capital_rub else exposure_rub
            row["planned_notional_rub"] = need
            ok_adv = pd.notna(adv) and adv >= gates["G4_min_adv_multiple_of_hedge_notional"] * need
            if c["asset"] in fut_oi and c["asset"] in contract_notional and contract_notional[c["asset"]] > 0:
                n_contracts = need / contract_notional[c["asset"]]
                oi = fut_oi[c["asset"]]
                row.update({"contracts_needed": n_contracts, "open_interest_front": oi})
                ok_oi = oi >= gates["G4_min_oi_multiple_of_contracts"] * n_contracts
            else:
                ok_oi = True
            g4 = "PASS" if (ok_adv and ok_oi) else "FAIL"
            if pd.isna(adv) and ok_oi:
                g4 = "NO_VOLUME_DATA"
                reasons.append("liquidity: traded value not available from ISS for this market -> not verified")
            if g4 == "FAIL":
                reasons.append(f"liquidity: ADV60 {adv/1e6 if pd.notna(adv) else float('nan'):.1f}m RUB vs need {need/1e6:.1f}m"
                               + ("" if ok_oi else "; open interest too small"))
        elif c["id"] in special:
            g3 = special[c["id"]].get("G3", "N/A")
            g4 = special[c["id"]].get("G4", "N/A")
            reasons += special[c["id"]].get("reasons", [])
        # ---------------- G5 economic relevance
        g5 = "N/A"
        if s is not None and s in st.index and c["role"] != "LIQUIDITY_SLEEVE":
            rho = st.loc[s, "corr_250d"]
            rmin = st.loc[s, "corr120_min_2y"]
            row.update({"corr_250d": rho, "corr120_min_2y": rmin, "hstar_250d": st.loc[s, "hstar_250d"],
                        "downside_corr": st.loc[s, "downside_corr"], "corr_market_stress": st.loc[s, "corr_market_stress"],
                        "mean_ret_on_smlt_worst5pct": st.loc[s, "mean_ret_on_smlt_worst5pct"]})
            if c["role"] in HEDGE_ROLES:
                ok = pd.notna(rho) and rho >= gates["G5_min_abs_corr_hedge"] and pd.notna(rmin) and rmin > 0
                g5 = "PASS" if ok else "FAIL"
                if not ok:
                    reasons.append(f"relevance: corr250={rho:.2f}, min rolling-120d corr (2y)={rmin:.2f}")
            elif c["role"] == "DIVERSIFIER":
                dc, sc = st.loc[s, "downside_corr"], st.loc[s, "corr_market_stress"]
                evid = (pd.notna(rho) and rho < 0) or (pd.notna(st.loc[s, "mean_ret_on_smlt_worst5pct"])
                                                       and st.loc[s, "mean_ret_on_smlt_worst5pct"] > 0)
                g5 = "DIVERSIFIER_EVIDENCE" if evid else "NO_HEDGE_EVIDENCE"
                reasons.append(f"diversifier: corr250={rho:.2f}, stress corr={sc:.2f}, downside corr={dc:.2f}")
        elif c["role"] == "LIQUIDITY_SLEEVE" and s in R:
            vol = float(R[s].loc[two_y].std() * np.sqrt(252))
            row["ann_vol_2y"] = vol
            g5 = "PASS" if vol <= gates["liquidity_sleeve_max_ann_vol"] else "FAIL"
            if g5 == "FAIL":
                reasons.append(f"sleeve vol {vol:.2%} too high")
        elif c["id"] in special:
            g5 = special[c["id"]].get("G5", "N/A")
        if c["id"] in special and s is not None:
            reasons += [r for r in special[c["id"]].get("reasons", []) if r not in reasons]
        # ---------------- classification
        if c["role"] in ("NOT_A_HEDGE",):
            cls = "REJECTED"
            reasons.append("classification: same-issuer exposure, not a hedge or diversifier")
        elif c["access"] in ("OTC", "BORROW_UNEVIDENCED"):
            cls = "THEORETICAL_ONLY"
        elif g1 == "FAIL" or g2 == "FAIL" or g3 == "FAIL" or g4 == "FAIL" or (c["role"] in HEDGE_ROLES and g5 == "FAIL"):
            cls = "REJECTED"
        elif c["id"] == "SMLT_SPOT_SHORT":
            cls = "REJECTED"
            reasons.append("shorting the held stock nets against the position = selling -> DE_RISKING branch")
        elif g2 == "CONDITIONAL" or g3 == "LIMITED" or g4 == "NO_VOLUME_DATA" or special.get(c["id"], {}).get("limited"):
            cls = "LIMITED_EXECUTABILITY"
        else:
            cls = "EXECUTABLE"
        if c["role"] == "EXPERIMENTAL":
            cls = "APPENDIX_ONLY"
        elif c["role"] == "ECONOMIC_PROXY":
            cls = "ECONOMIC_PROXY_ONLY"
            reasons.append("economic history only; execution (if any) through the MOEX instrument")
        elif c["role"] == "EXPERIMENTAL_FUNDED":
            cls = "INSTRUCTOR_AUTHORIZED_EXPERIMENTAL"
            reasons.append("academic assumption authorised by the instructor; tradability for a real PIF not verified")
        row.update({"G1_existence": g1, "G2_access": g2, "G3_data": g3, "G4_liquidity": g4, "G5_relevance": g5,
                    "classification": cls, "reasons": " | ".join(reasons)})
        out.append(row)
    return pd.DataFrame(out)
