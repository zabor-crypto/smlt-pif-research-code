"""Generated documentation: SOURCE_REGISTER.md and DATA_DICTIONARY.md (built from files on disk)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

FILE_DESC = {
    "data/source_manifest.csv": "One row per immutable raw snapshot (provenance, SHA-256, PIT status).",
    "data/manual/issuer_facts.csv": "Issuer financial/debt facts with source rank, retrieval method and verification status.",
    "data/manual/event_chronology.csv": "Dated issuer/market events used for context and event-scenario anchoring.",
    "data/processed/prices.csv": "Daily prices on the master calendar (official close for shares; settlement for futures; index close).",
    "data/processed/returns.csv": "Daily simple returns (NaN on gaps, closures and excluded corporate-action days).",
    "data/processed/value_rub.csv": "Daily traded value in RUB (futures: volume x settle of the held contract).",
    "data/processed/credit_state.csv": "Daily Samolet bond credit-stress state (median Z-spread in pp, YTM in %, clean price in % of par).",
    "data/processed/gcurve_yearyields.csv": "MOEX G-curve zero-coupon yields (% p.a.) by tenor (years), weekly samples.",
    "data/processed/ncc_futures_limits.csv": "NCC market-risk rates (mr1..mr3) and concentration limits by asset and date.",
    "data/processed/im_rate_mr1.csv": "Daily PIT IM rate mr1 per asset (last published value <= date).",
    "data/processed/series_meta.csv": "Source/status of every panel series and the list of explicit exclusions.",
    "outputs/instrument_universe.csv": "G1-G5 gate results and executability classification per candidate.",
    "outputs/candidate_stats.csv": "Empirical candidate statistics vs SMLT (vol, rolling corr/h*, stability, stress, tails).",
    "outputs/rolling_hedge_ratios.csv": "Trailing 60/120/250d h* and correlation time series.",
    "outputs/smlt_market_dossier.csv": "SMLT market-risk facts at the decision date.",
    "outputs/smlt_options_20260925.csv": "Listed SMLT options on 2026-09-25 with parity-implied forward, IV and Greeks.",
    "outputs/covariance_oos.csv": "Walk-forward covariance estimator comparison (QLIKE, bias ratio).",
    "outputs/walkforward_daily_metrics.csv": "Daily constant-exposure walk-forward metrics per policy/threshold (bases A and B).",
    "outputs/threshold_selection_walkforward.csv": "Threshold chosen at each quarter from prior OOS data only.",
    "outputs/episodes.csv": "Per-episode (6m/12m, monthly starts) results with full futures mechanics.",
    "outputs/episodes_summary.csv": "Episode distribution summary incl. liquidity-at-risk.",
    "outputs/bootstrap_strategy_metrics.csv": "Decision-date bootstrap metrics per strategy/capital/horizon/variant (RUB).",
    "outputs/risk_metrics.csv": "bootstrap metrics normalised by basis A (RUB 10m SMLT) and basis B (total capital).",
    "outputs/strategy_comparison.csv": "Main-capital strategy comparison: risk, cost, upside, liquidity + walk-forward columns.",
    "outputs/capital_budget_sensitivity.csv": "Capital-efficiency frontier: best feasible hedge per capital level.",
    "outputs/optimizer_solutions.csv": "Min-ES / min-variance / max-Sharpe (diagnostic) overlay solutions.",
    "outputs/complexity_frontier.csv": "Best min-ES overlay for each cardinality K, re-evaluated on paths.",
    "outputs/pareto_es_upside.csv": "Min-ES subject to a minimum upside-capture constraint (Pareto frontier).",
    "outputs/market_inputs_at_cutoff.csv": "Market inputs to the deterministic scenario engine (all PIT-derived).",
    "outputs/scenario_matrix.csv": "Deterministic payoff surface: SMLT terminal return x rates x credit x strategy.",
    "outputs/event_overlay.csv": "Event scenarios: day-1 VM shock, liquidity breach, terminal P&L.",
    "outputs/scenario_robustness.csv": "Worst/best cell and mean rank per strategy across the deterministic grid.",
    "outputs/data_validation.csv": "Panel validation (coverage, gaps, jumps, look-ahead).",
    "outputs/raw_checksum_report.csv": "Re-hash of every raw snapshot vs manifest checksum.",
    "outputs/factor_decomposition.csv": "A1/B3: Shapley variance shares, betas and Newey-West t-stats of the 4-factor SMLT model by window.",
    "outputs/factor_loadings.csv": "Factor betas / t-stats per window.",
    "outputs/rolling_factor_loadings.csv": "Trailing 60/120/250d factor betas and R2 (not shifted).",
    "outputs/factor_risk_contributions.csv": "Euler variance contributions of each factor + residual.",
    "outputs/futures_contract_spec_audit.csv": "A4: point value, lot, notional formula and exchange-value verification per futures family.",
    "outputs/collateral_sensitivity.csv": "A2: FUT_x / DERISK_x metrics under every collateral regime, capital, model and horizon.",
    "outputs/collateral_conclusions.csv": "A2: per regime - does selling dominate futures, best futures ratio under breach limits.",
    "outputs/fhs_basis_audit.csv": "A7: historical vs simulated basis statistics and dislocation stress.",
    "outputs/correlation_full.csv": "B2: correlation matrix, full sample (also _250d, _120d, _current_regime_60d, _downside, _market_stress, _credit_stress).",
    "outputs/correlation_stability.csv": "B2: SMLT pair correlations across sets and persist/break/reverse classification.",
    "outputs/phase_b_sleeves.csv": "B1: Phase-B sleeve specifications (margin rate, cost, contract notional).",
    "outputs/smart_candidates_summary.csv": "B5: every candidate overlay (MULTI / DIRECT_ONLY) with FHS and worst-model metrics.",
    "outputs/static_smart_constrained_solutions.csv": "B5: best MULTI vs DIRECT_ONLY solution under each constraint profile.",
    "outputs/smart_selected_portfolios.csv": "B5/B6: pre-declared static and robust selections by horizon and capital.",
    "outputs/robust_optimizer_solutions.csv": "B6: min worst-model-ES selections.",
    "outputs/complexity_multidim.csv": "B11: best portfolio per K under breach/upside constraints and Pareto test vs K=1.",
    "outputs/adaptive_walkforward_metrics.csv": "B7/B12: daily walk-forward metrics of adaptive and benchmark policies.",
    "outputs/adaptive_episodes.csv": "B12: 6m/12m historical episodes with margin, breach and liquidity actions.",
    "outputs/liquidity_waterfall_results.csv": "B10: waterfall vs immediate-cut liquidity policy by strategy and horizon.",
    "outputs/regime_sensitivity.csv": "B8: regime frequencies and adaptive results under threshold scale 0.75 / 1.0 / 1.25.",
    "outputs/smart_portfolio_comparison.csv": "B15: decision table (all required rows and columns).",
    "outputs/risk_contributions.csv": "B9: per-sleeve variance/ES/mean/up-down contributions, margin and costs.",
    "outputs/key_findings.json": "Every number quoted in RESEARCH_REPORT.md (computed by reporting/findings.py).",
}

COL_DESC = {
    "ES95": "Expected Shortfall 95% of horizon P&L (RUB, loss positive; Rockafellar-Uryasev definition)",
    "VaR95": "Value-at-Risk 95% of horizon P&L (RUB, loss positive)",
    "ES99": "Expected Shortfall 99% (RUB, loss positive)", "VaR99": "VaR 99% (RUB, loss positive)",
    "mean_pnl": "mean horizon P&L (RUB) incl. costs, carry and sleeve interest",
    "upside_capture": "mean P&L in paths/days where SMLT rose / mean CONTROL gain in the same states",
    "downside_capture": "mean P&L where SMLT fell / mean CONTROL loss in the same states",
    "p_sleeve_breach": "share of paths where free liquidity sleeve < 0 at any day (hedge would be cut)",
    "max_cum_vm_drain_p99": "99th percentile of max peak-to-trough cumulative variation-margin outflow (RUB)",
    "initial_im": "initial margin at t0 (RUB)", "peak_collateral_p95": "95th pct of max(IM + VM drain) over the path (RUB)",
    "HE_VAR": "1 - Var(hedged P&L)/Var(unhedged P&L)", "HE_PNL": "-sum(dV_hedge)/sum(dV_spot)",
    "hstar_250d": "trailing 250d minimum-variance hedge ratio Cov(S,H)/Var(H)",
    "classification": "EXECUTABLE | LIMITED_EXECUTABILITY | THEORETICAL_ONLY | REJECTED | APPENDIX_ONLY",
    "zspread_median_pp": "median Z-spread over MOEX G-curve of traded Samolet bonds (percentage points, ISS ZSPREAD)",
    "mr1": "NCC base market-risk rate (IM per unit notional)", "iv_from_settle": "Black-76 IV backed out of the exchange settlement price",
    "F_implied_parity": "forward implied by put-call parity at the most-traded strike",
}


def source_register(root: Path) -> str:
    root = Path(root)
    man = pd.read_csv(root / "data" / "source_manifest.csv", dtype=str).drop_duplicates("source_id")
    g = man.groupby(["publisher", "dataset", "point_in_time_valid"]).agg(
        snapshots=("source_id", "size"), instruments=("instrument", "nunique"),
        first_retrieved=("retrieval_timestamp", "min"), last_retrieved=("retrieval_timestamp", "max")).reset_index()
    facts = pd.read_csv(root / "data" / "manual" / "issuer_facts.csv")
    ev = pd.read_csv(root / "data" / "manual" / "event_chronology.csv")
    own = yaml.safe_load((root / "configs" / "owner_fetch.yaml").read_text())
    out = ["# Source register", "",
           "Generated by `smlt_hedge.reporting.docs` from `data/source_manifest.csv` and `data/manual/`. "
           "Source hierarchy: (1) issuer / e-disclosure, (2) MOEX / ISS / NCC, (3) Bank of Russia, (4) rating agencies, "
           "(5) MinFin / state, (6) reputable financial databases, (7) media (chronology only).", "",
           f"## 1. Raw machine snapshots ({len(man)} files, SHA-256 in manifest; re-verified by pipeline stage S1)", "",
           g.to_markdown(index=False), "",
           "## 2. Manually recorded facts (secondary sources; NOT raw snapshots)", "",
           "WEB_SUMMARY = page retrieved through a web-fetch tool that returns processed text, so no byte-level checksum "
           "exists; each fact is therefore marked PENDING_PRIMARY / PENDING_OWNER_FETCH until the owner-run fetchers "
           "store the primary document.", "",
           facts[["fact_id", "variable", "value", "unit", "as_of", "publication_date", "source_rank", "publisher",
                  "verification_status"]].to_markdown(index=False), "",
           "## 3. Event chronology sources", "", ev.to_markdown(index=False), "",
           "## 4. Owner-run fetches pending (blocked from the research environment)", ""]
    rows = [{"section": k, "id": i["id"], "url": i["url"], "blocks": ", ".join(i.get("blocks", []))}
            for k, v in own.items() for i in v]
    out += [pd.DataFrame(rows).to_markdown(index=False), "",
            "Commands (run from the project root without VPN):", "", "```bash",
            ".venv/bin/python scripts/fetch_cbr.py", ".venv/bin/python scripts/fetch_issuer.py",
            ".venv/bin/python scripts/fetch_rating_data.py", "```", ""]
    return "\n".join(out)


def data_dictionary(root: Path) -> str:
    root = Path(root)
    out = ["# Data dictionary", "", "Generated from the files on disk. Units: RUB unless stated; returns are simple "
           "daily returns; 'loss positive' sign convention for VaR/ES.", ""]
    for rel, desc in FILE_DESC.items():
        p = root / rel
        if not p.exists():
            out += [f"## `{rel}`", "", f"{desc}", "", "_not present in this run_", ""]
            continue
        try:
            df = pd.read_csv(p, nrows=5)
        except Exception:  # noqa: BLE001
            continue
        out += [f"## `{rel}`", "", desc, "", "| column | description |", "|---|---|"]
        for c in df.columns:
            d = COL_DESC.get(c, "")
            if not d and c.startswith("w_"):
                d = "optimal signed notional of sleeve as fraction of RUB 10m (short < 0)"
            if not d and c.startswith("path_"):
                d = "metric of this solution re-evaluated on full bootstrap paths (see bootstrap columns)"
            if not d and c.startswith("comp_"):
                d = "P&L component (RUB)"
            out.append(f"| `{c}` | {d} |")
        out.append("")
    return "\n".join(out)


def write_docs(ctx) -> None:
    (ctx.root / "SOURCE_REGISTER.md").write_text(source_register(ctx.root))
    (ctx.root / "DATA_DICTIONARY.md").write_text(data_dictionary(ctx.root))
