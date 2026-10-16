"""D23 weekly consumer only. The general Friday helper and margin feed are unchanged."""
from pathlib import Path
import pandas as pd

HOLIDAY_FIXES = {"2024-02-23": "2024-02-22", "2024-03-08": "2024-03-07",
                 "2025-05-09": "2025-05-08", "2026-01-02": "2025-12-30",
                 "2026-05-01": "2026-04-30", "2026-06-12": "2026-06-11"}


def weekly_sessions(start, end, sessions):
    """Same first-Friday convention plus cutoff, mapped only backwards."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    cal = pd.DatetimeIndex(sessions).sort_values().unique()
    anchors = list(pd.date_range(start, end, freq="W-FRI")) + [end]
    out = []
    for anchor in anchors:
        prior = cal[cal <= anchor]
        if len(prior) and prior[-1] not in out:
            out.append(prior[-1])
    return pd.DatetimeIndex(out)


def read(ctx):
    original = pd.read_csv(ctx.proc / "gcurve_yearyields.csv", index_col=0, parse_dates=True)
    if not getattr(ctx, "_regime_v2r2_combined", False):
        return original
    repaired = pd.read_csv(Path(ctx.root) / "data/regime_v2r2_combined/gcurve_weekly.csv", index_col=0, parse_dates=True)
    # The authority exports 1.0 while the parent names this tenor 1; match tenor
    # values without creating extra columns or rewriting the parent's names.
    names = {float(c): c for c in original.columns}
    repaired = repaired.rename(columns={c: names[float(c)] for c in repaired.columns})[original.columns]
    # Only these six omissions; preserve every original fixing and warm-up value.
    for friday, prior in HOLIDAY_FIXES.items():
        if weekly_sessions(friday,friday,repaired.index).tolist() != [pd.Timestamp(prior)]:
            raise ValueError("D23_FROZEN_WEEKLY_EXPORT_MISMATCH:" + friday)
    additions = repaired.loc[pd.to_datetime(list(HOLIDAY_FIXES.values()))]
    return pd.concat([original, additions[~additions.index.isin(original.index)]]).sort_index()
