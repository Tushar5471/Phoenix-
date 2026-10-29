"""
Deliverable 04: Decision Layer & Cash-Buffer Policy Backtest (`src/decision_layer.py`)

Solves the CIO's operational decision problem:
"How much cash/overnight liquidity must a liquid fund hold on the 1st of the month (and specifically
ahead of 15 March / quarter-end advance-tax dates), and what does the policy cost under historical conditions?"

Evaluation Metric (from PS 03 Page 2):
- Decision Cost = Yield given up on idle cash + a 3× penalty on any redemption shortfall.
- Baseline Comparison: "Compare with the naive rule: hold last year's same-month outflow."

Mathematical Formulation (Newsvendor / Asymmetric Quantile Decision Theory):
Let:
  - O_t = max(0, -F_t) be the actual net redemption outflow in month t (₹ Crore)
  - B_t >= 0 be the cash & overnight buffer held at the start of month t (₹ Crore)
  - s_t = (y^{CP/TBill}_t - y^{Overnight}_t) / 12 be the monthly yield drag rate (opportunity cost per ₹1 of cash buffer)
  - I_t = max(0, B_t - O_t) be the idle excess cash buffer (₹ Crore)
  - S_t = max(0, O_t - B_t) be the redemption shortfall forcing distress CP/T-bill liquidation (₹ Crore)

Under PS 03 scoring:
  - Unit cost of excess idle cash (overage cost): c_o = 1.0 * s_t
  - Unit cost of redemption shortfall (underage penalty): c_u = 3.0 * s_t
  - Total Monthly Decision Cost (₹ Crore):
      C_t(B_t) = s_t * [ 1.0 * max(0, B_t - O_t) + 3.0 * max(0, O_t - B_t) ]

By the exact Newsvendor Theorem, the expected cost E[C_t(B_t)] is minimized by choosing B_t equal to the
critical fractile of the predictive outflow distribution:
    q* = c_u / (c_o + c_u) = 3 / (1 + 3) = 0.75 (75th percentile of forecast outflow, i.e., 25th percentile of net flow),
paired with a Regime-Adaptive Floor (0.35% frictional floor in high-confidence net inflow rebound months
where q_{0.025} > 0, and 78th percentile buffer in Advance-Tax / Quarter-End months).
"""

import numpy as np
import pandas as pd


def evaluate_cash_buffer_policies(
    df_full: pd.DataFrame,
    backtest_df: pd.DataFrame,
    champion_model: str = "Q-GBDT + Conformal (ML Champion)",
    floor_buffer_pct_aum: float = 0.005,  # 0.50% operational working cash floor
) -> dict:
    """
    Backtests 6 distinct cash-buffer policies over the 24-month rolling out-of-sample window:
    1. Baseline (Naive Rule): Hold last year's same-month outflow: B_t = max(floor, O_{t-12})
    2. Static Historical 75th Pct Outflow Rule: Hold trailing 75th percentile of historical monthly outflows
    3. Forecast Median Rule (50th Pct Outflow): Hold B_t = max(floor, -q50_flow)
    4. Conservative Tail Rule (95th Pct Outflow, PS 03 Example): Hold B_t = max(floor, -q05_flow)
    5. Newsvendor Optimal Rule (75th Pct Outflow, c_u/c_o = 3:1): Hold B_t = max(floor, -q25_flow)
    6. CIO Recommended Policy (Regime-Adaptive Newsvendor 75p/78p): Minimizes 1x Idle Drag + 3x Shortfall Penalty.
    """
    champ_bt = backtest_df[backtest_df["model"] == champion_model].copy().reset_index(drop=True)
    n_bt = len(champ_bt)
    start_idx = len(df_full) - n_bt

    records = []
    for i in range(n_bt):
        full_idx = start_idx + i
        row_full = df_full.iloc[full_idx]
        row_bt = champ_bt.iloc[i]

        actual_flow = float(row_bt["actual_cr"])
        actual_outflow = max(0.0, -actual_flow)
        aum = float(row_full["combined_aum_cr"])
        min_floor = round(aum * floor_buffer_pct_aum, 2)
        inflow_rebound_floor = round(aum * 0.0025, 2)  # 0.25% minimal frictional buffer when q10 > 0

        # Last year's same-month actual outflow (Lag-12 Naive Rule specified in PS 03)
        lag12_flow = float(df_full.iloc[full_idx - 12]["combined_net_flow_cr"])
        naive_buffer = max(min_floor, max(0.0, -lag12_flow))

        # Historical rolling 75th percentile outflow up to t-1
        hist_flows = df_full.iloc[:full_idx]["combined_net_flow_cr"].values
        hist_outflows = np.maximum(0.0, -hist_flows)
        static_buffer = max(min_floor, float(np.percentile(hist_outflows, 75)))

        # Quantiles from Champion Forecast
        q025 = float(row_bt["q025_cr"])
        q10 = float(row_bt["q10_cr"])
        q25 = float(row_bt["q25_cr"])
        q50 = float(row_bt["q50_cr"])
        q05 = 0.5 * (q025 + q10)

        p50_buffer = max(min_floor, max(0.0, -q50))
        p75_optimal_buffer = max(min_floor, max(0.0, -q25))
        p95_buffer = max(min_floor, max(0.0, -q05))

        # Regime-Adaptive CIO Policy:
        # - In high-confidence net inflow rebound months (q10 > 0), hold minimal 0.25% frictional floor
        # - In Advance-Tax months, hold 78th percentile outflow buffer: -(0.80 * q25 + 0.20 * q10)
        # - In normal mid-quarter months, hold exact Newsvendor 75th percentile outflow buffer: -q25
        is_tax = int(row_bt["is_advance_tax_month"])
        if q10 > 0:
            cio_hybrid_buffer = inflow_rebound_floor
        elif is_tax:
            cio_hybrid_buffer = max(min_floor, max(0.0, -(0.82 * q25 + 0.18 * q10)))
        else:
            cio_hybrid_buffer = max(min_floor, max(0.0, -q25))

        spread_bps = float(row_full["spread_cp_overnight_bps"])
        monthly_drag_rate = (spread_bps / 10000.0) / 12.0

        policies = {
            "01. Naive Baseline (Last Year Same-Month Outflow)": naive_buffer,
            "02. Static Historical 75th Pct Outflow Rule": static_buffer,
            "03. Forecast Median Rule (50th Pct Outflow)": p50_buffer,
            "04. Conservative Tail Rule (95th Pct Outflow)": p95_buffer,
            "05. Newsvendor Optimal Rule (75th Pct, c_u/c_o = 3:1)": p75_optimal_buffer,
            "06. CIO Recommended Policy (Regime-Adaptive 75p/78p)": cio_hybrid_buffer,
        }

        for pol_name, buf in policies.items():
            idle_cash = max(0.0, buf - actual_outflow)
            shortfall = max(0.0, actual_outflow - buf)
            weighted_flow_loss_cr = idle_cash + 3.0 * shortfall
            idle_yield_cost_cr = idle_cash * monthly_drag_rate
            shortfall_penalty_cost_cr = 3.0 * shortfall * monthly_drag_rate
            total_cost_cr = idle_yield_cost_cr + shortfall_penalty_cost_cr
            annualized_cost_bps = (total_cost_cr / aum) * 12.0 * 10000.0

            records.append({
                "date": row_bt["date"],
                "year_month": row_bt["year_month"],
                "month": int(row_bt["month"]),
                "is_advance_tax_month": is_tax,
                "policy": pol_name,
                "actual_net_flow_cr": round(actual_flow, 2),
                "actual_outflow_cr": round(actual_outflow, 2),
                "recommended_buffer_cr": round(buf, 2),
                "buffer_pct_aum": round((buf / aum) * 100.0, 2),
                "idle_cash_cr": round(idle_cash, 2),
                "shortfall_cr": round(shortfall, 2),
                "had_shortfall": int(shortfall > 1.0),
                "weighted_flow_loss_cr": round(weighted_flow_loss_cr, 2),
                "idle_yield_cost_cr": round(idle_yield_cost_cr, 3),
                "shortfall_penalty_cost_cr": round(shortfall_penalty_cost_cr, 3),
                "total_decision_cost_cr": round(total_cost_cr, 3),
                "annualized_cost_bps": round(annualized_cost_bps, 2),
            })

    detail_df = pd.DataFrame(records)

    naive_sub = detail_df[detail_df["policy"] == "01. Naive Baseline (Last Year Same-Month Outflow)"]
    naive_total_cost_cr = float(naive_sub["total_decision_cost_cr"].sum())
    naive_weighted_loss = float(naive_sub["weighted_flow_loss_cr"].mean())

    summary_rows = []
    for pol_name in detail_df["policy"].unique():
        sub = detail_df[detail_df["policy"] == pol_name]
        tot_cost = float(sub["total_decision_cost_cr"].sum())
        mean_wl = float(sub["weighted_flow_loss_cr"].mean())
        cost_saving_pct = ((naive_total_cost_cr - tot_cost) / naive_total_cost_cr) * 100.0
        wl_saving_pct = ((naive_weighted_loss - mean_wl) / naive_weighted_loss) * 100.0

        summary_rows.append({
            "policy": pol_name,
            "avg_buffer_cr": round(float(sub["recommended_buffer_cr"].mean()), 1),
            "avg_buffer_pct_aum": round(float(sub["buffer_pct_aum"].mean()), 2),
            "avg_idle_cash_cr": round(float(sub["idle_cash_cr"].mean()), 1),
            "avg_shortfall_cr": round(float(sub["shortfall_cr"].mean()), 1),
            "shortfall_months_count": int(sub["had_shortfall"].sum()),
            "mean_weighted_loss_cr": round(mean_wl, 1),
            "cum_idle_drag_cost_cr": round(float(sub["idle_yield_cost_cr"].sum()), 2),
            "cum_shortfall_penalty_cr": round(float(sub["shortfall_penalty_cost_cr"].sum()), 2),
            "cum_24m_decision_cost_cr": round(tot_cost, 2),
            "avg_annualized_drag_bps": round(float(sub["annualized_cost_bps"].mean()), 2),
            "cost_reduction_vs_naive_pct": round(cost_saving_pct, 2),
            "weighted_loss_reduction_vs_naive_pct": round(wl_saving_pct, 2),
        })

    summary_df = pd.DataFrame(summary_rows)
    return {"policy_backtest_detail": detail_df, "policy_summary": summary_df}


def compute_scheme_level_recommendation(
    forward_df: pd.DataFrame,
    scheme_aum_cr: float = 25000.0,
    industry_liquid_aum_cr: float = 625847.01,
    industry_combined_aum_cr: float = 702687.51,
) -> pd.DataFrame:
    """
    Translates the industry-wide AMFI category forecast into an actionable cash-buffer schedule
    both at the Industry Level (₹ Crore & % of AUM) and for a representative ₹25,000 Crore
    Institutional Liquid Fund (including the headline 15 March recommendation).
    """
    comb = forward_df[forward_df["category"] == "Combined (Liquid + Overnight)"].copy().reset_index(drop=True)
    liq = forward_df[forward_df["category"] == "Liquid Funds"].copy().reset_index(drop=True)

    rows = []
    for i in range(len(comb)):
        r_c = comb.iloc[i]
        r_l = liq.iloc[i]
        ym = r_c["year_month"]
        m = int(r_c["month"])
        is_tax = int(r_c["is_advance_tax_month"])

        min_floor_comb = industry_combined_aum_cr * 0.005
        inflow_floor_comb = industry_combined_aum_cr * 0.0025
        min_floor_liq = industry_liquid_aum_cr * 0.005
        inflow_floor_liq = industry_liquid_aum_cr * 0.0025

        q25_c = float(r_c["q25_flow_cr"])
        q10_c = float(r_c["ci80_lower_cr"])

        if q10_c > 0:
            rec_comb_cr = inflow_floor_comb
        elif is_tax:
            rec_comb_cr = max(min_floor_comb, max(0.0, -(0.82 * q25_c + 0.18 * q10_c)))
        else:
            rec_comb_cr = max(min_floor_comb, max(0.0, -q25_c))

        p95_comb_cr = max(inflow_floor_comb, float(r_c["p95_outflow_buffer_cr"]))

        q25_l = float(r_l["q25_flow_cr"])
        q10_l = float(r_l["ci80_lower_cr"])
        if q10_l > 0:
            rec_liq_cr = inflow_floor_liq
        elif is_tax:
            rec_liq_cr = max(min_floor_liq, max(0.0, -(0.82 * q25_l + 0.18 * q10_l)))
        else:
            rec_liq_cr = max(min_floor_liq, max(0.0, -q25_l))

        p95_liq_cr = max(inflow_floor_liq, float(r_l["p95_outflow_buffer_cr"]))

        rec_liq_pct = (rec_liq_cr / industry_liquid_aum_cr) * 100.0
        p95_liq_pct = (p95_liq_cr / industry_liquid_aum_cr) * 100.0

        scheme_rec_cr = (rec_liq_pct / 100.0) * scheme_aum_cr
        scheme_p95_cr = (p95_liq_pct / 100.0) * scheme_aum_cr

        rows.append({
            "year_month": ym,
            "date": r_c["date"],
            "month": m,
            "regime": "Advance-Tax + FY-End (15 Mar)" if m == 3 else ("Advance-Tax + Qtr-End (15 Dec)" if is_tax else ("Qtr-Start Rebound" if m in (10, 1) else "Mid-Quarter Normal")),
            "combined_point_flow_cr": r_c["point_forecast_cr"],
            "combined_ci80_cr": f"[{r_c['ci80_lower_cr']:,.0f}, {r_c['ci80_upper_cr']:,.0f}]",
            "combined_ci95_cr": f"[{r_c['ci95_lower_cr']:,.0f}, {r_c['ci95_upper_cr']:,.0f}]",
            "liquid_point_flow_cr": r_l["point_forecast_cr"],
            "rec_combined_buffer_cr": round(rec_comb_cr, 1),
            "p95_combined_buffer_cr": round(p95_comb_cr, 1),
            "rec_liquid_buffer_cr": round(rec_liq_cr, 1),
            "p95_liquid_buffer_cr": round(p95_liq_cr, 1),
            "rec_liquid_buffer_pct_aum": round(rec_liq_pct, 2),
            "p95_liquid_buffer_pct_aum": round(p95_liq_pct, 2),
            "scheme_25k_rec_buffer_cr": round(scheme_rec_cr, 1),
            "scheme_25k_p95_buffer_cr": round(scheme_p95_cr, 1),
        })

    return pd.DataFrame(rows)
