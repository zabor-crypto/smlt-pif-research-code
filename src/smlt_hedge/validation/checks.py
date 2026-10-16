"""Deterministic data-validation checks. Output: one row per series with pass/flag columns."""
from __future__ import annotations

import numpy as np
import pandas as pd


def validate_panel(P: pd.DataFrame, R: pd.DataFrame, cutoff: str, jump: float = 0.25) -> pd.DataFrame:
    rows = []
    for c in R.columns:
        r = R[c]
        p = P[c] if c in P else pd.Series(dtype=float)
        first = p.first_valid_index()
        last = p.last_valid_index()
        live = r.loc[first:] if first is not None else r
        big = live[live.abs() > jump]
        rows.append({
            "series": c,
            "first_date": first,
            "last_date": last,
            "n_price_obs": int(p.notna().sum()),
            "n_return_obs": int(r.notna().sum()),
            "missing_share_since_first": float(live.isna().mean()) if len(live) else np.nan,
            "nonpositive_prices": int((p <= 0).sum()),
            "n_abs_return_gt_jump": int(len(big)),
            "max_abs_return": float(live.abs().max()) if live.notna().any() else np.nan,
            "jump_dates": ";".join(d.strftime("%Y-%m-%d") for d in big.index[:10]),
            "after_cutoff_rows": int((r.index > pd.Timestamp(cutoff)).sum()),
            "has_cutoff_price": bool(pd.Timestamp(cutoff) in p.index and pd.notna(p.get(pd.Timestamp(cutoff)))),
            "duplicate_dates": int(r.index.duplicated().sum()),
        })
    out = pd.DataFrame(rows)
    out["PASS_no_lookahead"] = out["after_cutoff_rows"] == 0
    out["PASS_no_duplicates"] = out["duplicate_dates"] == 0
    out["PASS_positive_prices"] = out["nonpositive_prices"] == 0
    return out


def validate_manifest(man: pd.DataFrame, root) -> pd.DataFrame:
    """Re-hash every raw file and compare with the manifest checksum."""
    import hashlib
    from pathlib import Path
    res = []
    for _, r in man.iterrows():
        p = Path(root) / r["raw_path"]
        ok = p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == r["checksum"]
        res.append({"source_id": r["source_id"], "exists": p.exists(), "checksum_ok": ok})
    return pd.DataFrame(res)
