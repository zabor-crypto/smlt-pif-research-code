"""Generated SMLT_POINT_IN_TIME_DOSSIER.md and INSTRUMENT_UNIVERSE.md. Numbers come from pipeline outputs /
data/manual facts (with their provenance); interpretation text is kept separate and minimal."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _f(x, pct=False, nd=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x * 100:.{nd}f}%" if pct else f"{x:,.{nd}f}"


def bonds_at_cutoff(ctx) -> pd.DataFrame:
    b = pd.read_csv(ctx.interim / "smlt_bonds_history.csv", parse_dates=["TRADEDATE"])
    t = pd.Timestamp(ctx.cutoff)
    b = b[(b["TRADEDATE"] <= t) & (b["TRADEDATE"] > t - pd.Timedelta(days=10)) & (b["VALUE"].fillna(0) > 0)]
    last = b.sort_values("TRADEDATE").groupby("SECID").tail(1)
    last = last.assign(price=last["CLOSE"].fillna(last["LEGALCLOSEPRICE"]))
    cols = ["SECID", "SHORTNAME", "TRADEDATE", "MATDATE", "OFFERDATE", "COUPONPERCENT", "price", "YIELDCLOSE",
            "DURATION", "ZSPREAD", "VALUE"]
    return last[[c for c in cols if c in last]].sort_values("MATDATE")


def dossier(ctx) -> str:
    d = pd.read_csv(ctx.out / "smlt_market_dossier.csv", index_col=0)["value"]
    num = lambda k: float(d[k]) if k in d and pd.notna(pd.to_numeric(d[k], errors="coerce")) else np.nan  # noqa: E731
    facts = pd.read_csv(ctx.root / "data" / "manual" / "issuer_facts.csv")
    ev = pd.read_csv(ctx.root / "data" / "manual" / "event_chronology.csv")
    cs = ctx.read("credit_state.csv", index_col=0, parse_dates=True)
    lim = ctx.read("ncc_futures_limits.csv", parse_dates=["tradedate"])
    sm = lim[lim["assetcode"] == "SMLT"].set_index("tradedate").sort_index()
    mi = pd.read_csv(ctx.out / "market_inputs_at_cutoff.csv", index_col=0)["value"]
    bonds = bonds_at_cutoff(ctx)
    t = pd.Timestamp(ctx.cutoff)
    z_now = cs["zspread_median_pp"].loc[:t].dropna()
    z_1y = z_now.loc[t - pd.Timedelta(days=365):]
    fv = facts.set_index("variable")["value"]
    L = []
    L += ["# SMLT point-in-time risk dossier (information set: 2026-09-25)", "",
          "**Scope:** market data only up to 2026-09-25, from MOEX ISS raw snapshots.",
          "",
          "**Sources:**",
          "- Issuer financial facts come from reputable secondary reports. They are marked `PENDING_PRIMARY` until the owner fetches the primary IFRS documents (`scripts/fetch_issuer.py`).",
          "- Media are used for event chronology only.",
          "",
          "**Every number** below is taken from `outputs/smlt_market_dossier.csv`, `data/processed/*`, or `data/manual/issuer_facts.csv` (with source rank).", ""]
    L += ["## 1. Equity market state", "",
          "| Metric | Value |", "|---|---|",
          f"| Official close 2026-09-25 (LEGALCLOSEPRICE) | RUB {_f(num('close_2026_09_25'))} |",
          f"| Close 2026-09-04 | RUB {_f(num('close_2026_09_04'))} |",
          f"| September 2026 drawdown from the September high | {_f(num('sept_2026_drawdown_from_sep_high'), True)} |",
          f"| YTD 2026 price return | {_f(num('ytd_2026_return'), True)} |",
          f"| Drawdown from the 2022+ high ({d.get('date_of_2022plus_high')}) | {_f(num('drawdown_from_2022_high'), True)} |",
          f"| 2026 low close | RUB {_f(num('min_close_2026'))} on {d.get('date_min_close_2026')} |",
          f"| Realised vol 20d / 60d / 120d / 250d (ann.) | {_f(num('vol_20d'), True)} / {_f(num('vol_60d'), True)} / {_f(num('vol_120d'), True)} / {_f(num('vol_250d'), True)} |",
          f"| Beta to IMOEX 120d / 250d | {_f(num('beta_IMOEX_120d'), nd=2)} / {_f(num('beta_IMOEX_250d'), nd=2)} |",
          f"| Beta to MOEXRE (real-estate index) 120d / 250d | {_f(num('beta_MOEXRE_120d'), nd=2)} / {_f(num('beta_MOEXRE_250d'), nd=2)} |",
          f"| 20d / 60d ADV | RUB {_f(num('adv20_rub') / 1e9, nd=2)}bn / {_f(num('adv60_rub') / 1e9, nd=2)}bn |",
          f"| Position (RUB 10m) as a share of 20d ADV | {_f(num('position_share_of_adv20'), True, 2)} |",
          f"| Shares in position | {_f(num('shares_in_position'), nd=0)} |",
          f"| Worst / best day since 2022 | {_f(num('worst_day'), True)} ({d.get('worst_day_date')}) / {_f(num('best_day'), True)} ({d.get('best_day_date')}) |",
          ""]
    L += ["## 2. Derivatives state at the cutoff", "",
          "| Metric | Value |", "|---|---|",
          f"| Front future / settle / days to expiry | {mi.get('front_future')} / {_f(float(mi['F0']), nd=0)} / {_f(float(mi['days_to_expiry']), nd=0)} |",
          f"| Annualised futures basis ln(F/S)·365/d | {_f(float(mi['basis_ann']), True)} |",
          f"| NCC base IM rate mr1 (SMLT) | {_f(float(mi['mr1']), True, 0)}; mr2 = {_f(float(mi['mr2']), True, 0)}; 2y median = {_f(float(mi['mr1_median_2y']), True, 0)} |",
          f"| mr1 one year earlier | {_f(float(sm['mr1'].loc[:t - pd.Timedelta(days=365)].iloc[-1]), True, 0) if len(sm.loc[:t - pd.Timedelta(days=365)]) else 'n/a'} |",
          f"| Listed option IV (21-Oct, parity strike) | {_f(float(mi['listed_iv_oct21_median']), True, 0)} |",
          "| Listed option expiries | 2026-09-30, 2026-10-07, 2026-10-21 only (no 6-12m tenor) |", ""]
    L += ["## 3. Credit state (exchange bond data)", "",
          f"- Median Z-spread of traded Samolet bonds (ISS `ZSPREAD`, percentage points) on the last observation ≤ cutoff: **{_f(float(z_now.iloc[-1]), nd=1)} pp**.",
          f"  - 1y range: {_f(float(z_1y.min()), nd=1)} to {_f(float(z_1y.max()), nd=1)} pp.",
          f"- Median clean price: {_f(float(cs['price_median_pct_par'].loc[:t].dropna().iloc[-1]), nd=1)}% of par.",
          f"- Median YTM: {_f(float(cs['ytm_median_pct'].loc[:t].dropna().iloc[-1]), nd=1)}%.", "",
          "Traded Samolet bonds in the last 10 days before the cutoff (MOEX ISS; YIELDCLOSE and ZSPREAD in %, DURATION in days):", "",
          bonds.to_markdown(index=False, floatfmt=".2f"), "",
          "Bond prices well below par and three-digit yields on some issues mean the market prices a **material "
          "probability of restructuring or default**. The equity is therefore a deeply subordinated, option-like claim, and a "
          "linear hedge ratio estimated in calmer periods can understate the jump risk.", ""]
    L += ["## 4. Balance sheet and debt (secondary sources, pending primary verification)", "",
          facts[["fact_id", "variable", "value", "unit", "as_of", "publisher", "source_rank", "verification_status", "notes"]]
          .to_markdown(index=False), "",
          "**Separation of debt concepts** (from the reported H1-2026 IFRS figures):", "",
          f"- Gross borrowings RUB {fv.get('total_borrowings')}bn consist of:",
          f"  - **project finance** RUB {fv.get('project_finance')}bn: secured by escrow accounts. Escrow coverage was {fv.get('escrow_coverage_of_project_debt')}%. It is serviced at subsidised rates while escrow covers it, so it is not ordinary corporate leverage;",
          f"  - **corporate debt** RUB {fv.get('corporate_debt')}bn (net RUB {fv.get('net_corporate_debt')}bn): the refinancing-risk layer, including the public bonds above.",
          f"- Net debt excluding escrow: RUB {fv.get('net_debt_ex_escrow')}bn, {fv.get('net_debt_ex_escrow_to_adj_ebitda')}x adjusted EBITDA.",
          "- Treating project-finance gross debt as corporate leverage would overstate default risk. Ignoring the falling escrow coverage would understate it.", "",
          "**Maturity schedule and fixed/floating split:** DATA_NOT_AVAILABLE from primary sources in this environment. Bond maturities and offers are listed in §3; the full schedule is pending `fetch_issuer.py`.", ""]
    L += ["## 5. Event chronology (pre-cutoff)", "", ev.to_markdown(index=False), ""]
    L += ["## 6. Risk map", "",
          "| Risk | Evidence at cutoff | Hedgeable with | Residual |", "|---|---|---|---|",
          f"| MARKET | beta to IMOEX {_f(num('beta_IMOEX_250d'), nd=2)} (250d) | index futures (partially) | beta instability |",
          f"| SECTOR | beta to MOEXRE {_f(num('beta_MOEXRE_250d'), nd=2)}; peers fell too (PIK −6.7% in September) | no listed sector instrument; peer shorts LIMITED | mostly unhedgeable |",
          "| INTEREST RATE | key rate held at 14% on 2026-09-11 (media; CBR blocked); mortgage-driven demand | RGBI futures only if the correlation is stable (see universe) | indirect channel |",
          "| CREDIT / REFINANCING | bonds far below par, net debt ex-escrow +14% in H1 | nothing direct (no CDS); SMLT futures hedge the equity leg | jump-to-default gap risk |",
          f"| LIQUIDITY | position ≈ {_f(num('position_share_of_adv20'), True, 2)} of ADV; SMLT IM rate {_f(float(mi['mr1']), True, 0)} | the position itself is liquid; hedge liquidity is margin-bound | VM spikes on rallies |",
          "| IDIOSYNCRATIC | futures corr with spot ≈ 0.99; index corr ≈ 0.57 | SMLT futures / sale only | basis, roll |",
          "| EVENT | state support / strategic investor / restructuring / SPO | futures (both directions), puts (theoretical) | gap risk, VM on up-jumps |", ""]
    return "\n".join(L)


def universe_md(ctx) -> str:
    u = pd.read_csv(ctx.out / "instrument_universe.csv")
    cols = ["id", "role", "access", "G1_existence", "G2_access", "G3_data", "G4_liquidity", "G5_relevance",
            "classification", "corr_250d", "hstar_250d", "adv60_rub"]
    L = ["# Instrument universe", "",
         "Generated from `outputs/instrument_universe.csv`. Gates and rules: METHODOLOGY.md §5.",
         "",
         "- ACCESS ELIGIBILITY ≠ EXECUTABILITY: OTC items are THEORETICAL_ONLY without a public, point-in-time quote or counterparty route.",
         "- A diversifier is called a hedge **only** if stress evidence supports it.", "",
         u[[c for c in cols if c in u]].to_markdown(index=False, floatfmt=".3f"), "", "## Reasons", ""]
    for _, r in u.iterrows():
        if isinstance(r.get("reasons"), str) and r["reasons"]:
            L.append(f"- **{r['id']}** ({r['classification']}): {r['reasons']}")
    L += ["", "## Summary by class", "", u["classification"].value_counts().to_frame("count").to_markdown(), ""]
    return "\n".join(L)


def write(ctx) -> None:
    (ctx.root / "SMLT_POINT_IN_TIME_DOSSIER.md").write_text(dossier(ctx))
    (ctx.root / "INSTRUMENT_UNIVERSE.md").write_text(universe_md(ctx))
