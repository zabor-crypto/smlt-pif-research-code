"""Regime V2 configuration: the accepted parent configuration (configs/adaptive.yaml, read unchanged) plus the V2 block.

`load` returns ONE dict. The parent keys stay where the reused R1 / R2 helpers expect them; the V2 rules live under
`cfg["v2"]`. Three parent structures that the reused helpers read are filled from the V2 block (universes, route bounds,
stress paths), and one technical mandate is added so that the parent's executable-book check does not apply its own
issuer rule: V2 applies its own hard checks (`frontier.hard_checks`).
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

TECH_MANDATE = "V2_HARD_CHECKS_IN_REGIME_V2"


class StopCondition(RuntimeError):
    """A stop condition of the brief (section 38): the run must stop and report, never improvise."""


def load(root) -> dict:
    root = Path(root)
    base = yaml.safe_load((root / "configs" / "adaptive.yaml").read_text())
    v2 = yaml.safe_load((root / "configs" / "regime_v2.yaml").read_text())
    if base["cutoff"] != v2["cutoff"]:
        raise StopCondition("cutoff of regime_v2.yaml differs from adaptive.yaml")
    cfg = copy.deepcopy(base)
    cfg["v2"] = v2
    # Names change; every numeric research parameter is inherited unchanged.
    def repaired_name(name):
        return name.replace("ADAPTIVE_REGIME_PIF_V2", "ADAPTIVE_REGIME_PIF_V2R1").replace("STATIC_INVESTED_PIF_V2", "STATIC_INVESTED_PIF_V2R1")
    cfg["v2"]["policy"]["name"] = repaired_name(v2["policy"]["name"])
    cfg["v2"]["replay"]["ledgers"] = [{**s, "name": repaired_name(s["name"])} for s in v2["replay"]["ledgers"]]
    cfg["v2"]["validation"]["incremental_value"]["pair"] = [repaired_name(n) for n in v2["validation"]["incremental_value"]["pair"]]
    cfg["v2r1"] = {"output_namespace": "regime_v2r1_", "derivative_terms_repair": True,
                    "historical_cost_evidence": "CUTOFF_RATE_ASSUMPTION_DOWNGRADED",
                    "checkpoint_resume": "RESTART_FROM_START"}
    cfg["r3"]["universes"] = {k: list(v) for k, v in v2["universe"].items()}
    cfg["r3"]["bounds"] = {r: {"lo": float(b["lo"]), "hi": float(b["hi"]), "why": b.get("why", "")} for r, b in v2["bounds"].items()}
    cfg["r3"]["mandates"][TECH_MANDATE] = {"issuer_net_min": -10.0, "issuer_net_max": 10.0}
    cfg["stress"]["scenarios"] = {**cfg["stress"]["scenarios"], **v2["scenarios"]["extra_paths"]}
    return cfg


def mandate(cfg: dict, name: str, floor: float | None = None) -> dict:
    """Resolved mandate parameters: hard floor, target floor (hard + operating band), issuer floor, overlay rule."""
    m = cfg["v2"]["mandates"]
    d = m[name]
    hard = float(d["funded_floor_nav"] if floor is None else floor)
    band = float(m["HEDGED_INVESTED_PIF"]["funded_floor_operating_band_nav"]) if hard > 0 else 0.0
    return {"name": name, "floor_hard": hard, "floor_target": hard + band, "issuer_net_min": float(d["issuer_net_min"]),
            "overlay_rule": bool(d["protective_overlays_cannot_exceed_protected_funded_exposure"]),
            "floor_set": list(m["funded_assets_counting_toward_floor"]), "tol": float(m["integer_rounding_tolerance_nav"])}
