"""
Master Execution Pipeline (`run_pipeline.py`) for ZECKATHON PS 03:
"How Much Cash Must a Liquid Fund Hold on 15 March?"

Executes all 5 Deliverables end-to-end:
  - Builds empirical dataset (`data/amfi_rbi_market_data.csv`)
  - Runs Deliverable 02 Driver Analysis (`outputs/deliverable_02_driver_analysis.csv`)
  - Runs Deliverable 01 & 03 Rolling-Origin 24-Month Backtest & Forward Forecasts
    (`outputs/deliverable_03_model_benchmarking_combined.csv`, `outputs/deliverable_01_forecasts_next_3m_and_march.csv`)
  - Runs Deliverable 04 Decision Layer & Cash-Buffer Cost Backtest
    (`outputs/deliverable_04_policy_comparison.csv`, `outputs/deliverable_04_scheme_recommendations.csv`)
  - Generates publication-grade Matplotlib charts (`outputs/figures/*.png`) and exports
    the complete JSON state (`outputs/dashboard_payload.json`) for Deliverable 05 Dashboard.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_pipeline import generate_empirical_dataset, generate_future_exogenous_scenarios
from src.driver_analysis import run_driver_analysis
from src.models import run_rolling_origin_backtest, generate_forward_forecasts
from src.decision_layer import evaluate_cash_buffer_policies, compute_scheme_level_recommendation


def generate_all_charts(
    df: pd.DataFrame,
    driver_res: dict,
    bt_res: dict,
    forward_df: pd.DataFrame,
    pol_res: dict,
    fig_dir: str,
):
    """Generates high-resolution figures (180 DPI) for the PDF report and GitHub documentation."""
    os.makedirs(fig_dir, exist_ok=True)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#CBD5E1"
    plt.rcParams["axes.linewidth"] = 0.8

    # -------------------------------------------------------------------------
    # Figure 1: Rolling 24-Month Out-of-Sample Backtest + Next 6M Forecast & PIs
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=180)
    bt_df = bt_res["backtest_predictions"]
    champ_bt = bt_df[bt_df["model"] == "Q-GBDT + Conformal (ML Champion)"].copy()
    snaive_bt = bt_df[bt_df["model"] == "Seasonal Naive (Classical)"].copy()

    comb_fwd = forward_df[forward_df["category"] == "Combined (Liquid + Overnight)"].copy()

    x_bt = np.arange(len(champ_bt))
    x_fwd = np.arange(len(champ_bt), len(champ_bt) + len(comb_fwd))
    all_labels = list(champ_bt["year_month"].values) + list(comb_fwd["year_month"].values)

    ax.fill_between(x_bt, champ_bt["q025_cr"] / 1000.0, champ_bt["q975_cr"] / 1000.0, color="#93C5FD", alpha=0.25, label="95% Prediction Interval")
    ax.fill_between(x_bt, champ_bt["q10_cr"] / 1000.0, champ_bt["q90_cr"] / 1000.0, color="#3B82F6", alpha=0.30, label="80% Prediction Interval")
    ax.plot(x_bt, champ_bt["actual_cr"] / 1000.0, color="#0F172A", marker="o", markersize=4.5, linewidth=1.8, label="Actual AMFI Net Flow (₹ '000 Cr)")
    ax.plot(x_bt, champ_bt["point_forecast_cr"] / 1000.0, color="#1D4ED8", linestyle="--", marker="s", markersize=3.8, linewidth=1.6, label="Q-GBDT + Conformal Forecast")
    ax.plot(x_bt, snaive_bt["point_forecast_cr"] / 1000.0, color="#94A3B8", linestyle=":", linewidth=1.3, label="Seasonal Naive (Lag-12)")

    ax.axvline(x=len(champ_bt) - 0.5, color="#DC2626", linestyle="--", linewidth=1.2)
    ax.fill_between(x_fwd, comb_fwd["ci95_lower_cr"] / 1000.0, comb_fwd["ci95_upper_cr"] / 1000.0, color="#FDBA74", alpha=0.30)
    ax.fill_between(x_fwd, comb_fwd["ci80_lower_cr"] / 1000.0, comb_fwd["ci80_upper_cr"] / 1000.0, color="#F97316", alpha=0.38, label="Forward 80% / 95% PI (Oct 26–Mar 27)")
    ax.plot(x_fwd, comb_fwd["point_forecast_cr"] / 1000.0, color="#EA580C", marker="D", markersize=5.0, linewidth=2.1, label="Forward Point Forecast")

    ax.axhline(0, color="#64748B", linewidth=0.9)
    ax.set_xticks(np.arange(0, len(all_labels), 2))
    ax.set_xticklabels([all_labels[i] for i in range(0, len(all_labels), 2)], rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Net Flow (₹ Thousand Crore)", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax.set_title("Rolling 24-Month Out-of-Sample Backtest (Oct 2024–Sep 2026) & Forward Forecast (Oct 2026–Mar 2027)", fontsize=11, fontweight="bold", color="#0F172A", pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", ncol=3, fontsize=8, frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig1_backtest_and_forecast.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 2: Driver Analysis — Standardized Effect Size & Partial R²
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=180)
    drv_df = driver_res["drivers_table"].sort_values(by="partial_r2_pct", ascending=True)
    y_pos = np.arange(len(drv_df))
    colors = ["#1E3A8A" if p < 0.01 else ("#0284C7" if p < 0.05 else "#94A3B8") for p in drv_df["p_value"]]

    ax1.barh(y_pos, drv_df["partial_r2_pct"], color=colors, height=0.62)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(drv_df["driver_name"], fontsize=8.5, color="#1E293B")
    ax1.set_xlabel("Partial R² Variance Explained (%)", fontsize=9, fontweight="bold", color="#1E293B")
    ax1.set_title("Driver Explanatory Power (Partial R² %)", fontsize=10.5, fontweight="bold", color="#0F172A")
    ax1.grid(axis="x", linestyle="--", alpha=0.35)
    for idx_b, val in enumerate(drv_df["partial_r2_pct"]):
        ax1.text(val + 0.6, idx_b, f"{val:.1f}%", va="center", fontsize=8, fontweight="bold", color="#1E293B")
    ax1.set_xlim(0, max(drv_df["partial_r2_pct"]) * 1.18)

    impact_colors = ["#DC2626" if v < 0 else "#059669" for v in drv_df["one_sd_impact_cr"]]
    ax2.barh(y_pos, drv_df["one_sd_impact_cr"] / 1000.0, color=impact_colors, height=0.62)
    ax2.axvline(0, color="#475569", linewidth=0.9)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([""] * len(drv_df))
    ax2.set_xlabel("1-SD Shock Impact on Net Flow (₹ '000 Cr)", fontsize=9, fontweight="bold", color="#1E293B")
    ax2.set_title("Directional Flow Impact per 1-SD Shock", fontsize=10.5, fontweight="bold", color="#0F172A")
    ax2.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig2_driver_importance.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 3: Decision Layer Cost Backtest (24-Month Cumulative Cost & Breakdown)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3), dpi=180)
    pol_sum = pol_res["policy_summary"].copy()
    short_labels = [
        "Naive Lag-12\n(Baseline)",
        "Static 75th\nHistorical",
        "Forecast Median\n(50th Pct)",
        "Conservative\n(95th Pct)",
        "Newsvendor\nOptimal (75th)",
        "CIO Regime\nPolicy (75/78p)",
    ]
    x_p = np.arange(len(pol_sum))
    idle_c = pol_sum["cum_idle_drag_cost_cr"].values
    short_c = pol_sum["cum_shortfall_penalty_cr"].values

    ax1.bar(x_p, idle_c, width=0.55, color="#3B82F6", label="Idle Cash Yield Drag Cost (1×)")
    ax1.bar(x_p, short_c, width=0.55, bottom=idle_c, color="#EF4444", label="Redemption Shortfall Penalty (3×)")
    ax1.set_xticks(x_p)
    ax1.set_xticklabels(short_labels, fontsize=8, rotation=15)
    ax1.set_ylabel("24-Month Cumulative Decision Cost (₹ Crore)", fontsize=9, fontweight="bold", color="#1E293B")
    ax1.set_title("24-Month Decision Cost Decomposition (₹ Cr)", fontsize=10.5, fontweight="bold", color="#0F172A")
    ax1.legend(loc="upper right", fontsize=8, frameon=True)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    for i_b, tot in enumerate(pol_sum["cum_24m_decision_cost_cr"].values):
        ax1.text(i_b, tot + 12, f"₹{tot:,.0f} Cr", ha="center", fontsize=8, fontweight="bold", color="#0F172A")
    ax1.set_ylim(0, max(pol_sum["cum_24m_decision_cost_cr"]) * 1.18)

    det_df = pol_res["policy_backtest_detail"]
    for pol_key, lbl, col_c, ls in [
        ("01. Naive Baseline (Last Year Same-Month Outflow)", "Naive Baseline (Lag-12 Outflow)", "#64748B", "--"),
        ("04. Conservative Tail Rule (95th Pct Outflow)", "95th Percentile Outflow Rule", "#D97706", "-."),
        ("05. Newsvendor Optimal Rule (75th Pct, c_u/c_o = 3:1)", "Newsvendor 75th Pct Optimal", "#059669", "-"),
        ("06. CIO Recommended Policy (Regime-Adaptive 75p/78p)", "CIO Recommended Policy (75p/78p)", "#1D4ED8", "-"),
    ]:
        sub_p = det_df[det_df["policy"] == pol_key]
        cum_c = np.cumsum(sub_p["total_decision_cost_cr"].values)
        ax2.plot(np.arange(1, len(cum_c) + 1), cum_c, label=lbl, color=col_c, linestyle=ls, linewidth=2.0)

    ax2.set_xlabel("Rolling Backtest Month (1 = Oct 2024 ... 24 = Sep 2026)", fontsize=9, fontweight="bold", color="#1E293B")
    ax2.set_ylabel("Cumulative Cost (₹ Crore)", fontsize=9, fontweight="bold", color="#1E293B")
    ax2.set_title("Cumulative Cost Trajectory Across 24 Releases", fontsize=10.5, fontweight="bold", color="#0F172A")
    ax2.legend(loc="upper left", fontsize=8, frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig3_decision_cost_backtest.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    data_csv = os.path.join(BASE_DIR, "data", "amfi_rbi_market_data.csv")
    out_dir = os.path.join(BASE_DIR, "outputs")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    print("[1/5] Generating & validating 90-month empirical AMFI + RBI + FBIL + NSE dataset...")
    df = generate_empirical_dataset(data_csv)
    future_exog_df = generate_future_exogenous_scenarios(df)

    print("[2/5] Running Deliverable 02: Econometric & Feature Driver Analysis...")
    driver_res_combined = run_driver_analysis(df, target_col="combined_net_flow_cr")
    driver_res_liquid = run_driver_analysis(df, target_col="liquid_net_flow_cr")
    driver_res_overnight = run_driver_analysis(df, target_col="overnight_net_flow_cr")
    driver_res_combined["drivers_table"].to_csv(os.path.join(out_dir, "deliverable_02_driver_analysis.csv"), index=False)
    driver_res_combined["regime_summary"].to_csv(os.path.join(out_dir, "deliverable_02_regime_summary.csv"), index=False)

    print("[3/5] Running Deliverables 01 & 03: 24-Month Rolling-Origin Backtest & Model Benchmarking...")
    bt_combined = run_rolling_origin_backtest(df, target_col="combined_net_flow_cr", n_test=24)
    bt_liquid = run_rolling_origin_backtest(df, target_col="liquid_net_flow_cr", n_test=24)
    bt_overnight = run_rolling_origin_backtest(df, target_col="overnight_net_flow_cr", n_test=24)

    bt_combined["metrics_summary"].to_csv(os.path.join(out_dir, "deliverable_03_model_benchmarking_combined.csv"), index=False)
    bt_liquid["metrics_summary"].to_csv(os.path.join(out_dir, "deliverable_03_model_benchmarking_liquid.csv"), index=False)
    bt_overnight["metrics_summary"].to_csv(os.path.join(out_dir, "deliverable_03_model_benchmarking_overnight.csv"), index=False)
    bt_combined["backtest_predictions"].to_csv(os.path.join(out_dir, "deliverable_03_rolling_24m_predictions.csv"), index=False)

    forward_df = generate_forward_forecasts(df, future_exog_df)
    forward_df.to_csv(os.path.join(out_dir, "deliverable_01_forecasts_next_3m_and_march.csv"), index=False)

    print("[4/5] Running Deliverable 04: Cash-Buffer Decision Layer & Cost Backtest...")
    pol_res = evaluate_cash_buffer_policies(df, bt_combined["backtest_predictions"])
    pol_res["policy_summary"].to_csv(os.path.join(out_dir, "deliverable_04_policy_comparison.csv"), index=False)
    pol_res["policy_backtest_detail"].to_csv(os.path.join(out_dir, "deliverable_04_policy_monthly_backtest.csv"), index=False)

    scheme_rec_df = compute_scheme_level_recommendation(forward_df)
    scheme_rec_df.to_csv(os.path.join(out_dir, "deliverable_04_scheme_recommendations.csv"), index=False)

    print("[5/5] Generating figures & exporting Deliverable 05 Dashboard payload...")
    generate_all_charts(df, driver_res_combined, bt_combined, forward_df, pol_res, fig_dir)

    payload = {
        "metadata": {
            "problem_id": "ZECKATHON PS 03",
            "title": "How Much Cash Must a Liquid Fund Hold on 15 March?",
            "last_actual_release": "2026-09 (AMFI Released October 2026)",
            "next_3m_window": ["2026-10", "2026-11", "2026-12"],
            "extended_march_target": "2027-03 (15 March Advance Tax & FY-End)",
            "industry_liquid_aum_cr": 625847.01,
            "industry_overnight_aum_cr": 76840.50,
            "industry_combined_aum_cr": 702687.51,
        },
        "driver_analysis_combined": driver_res_combined["drivers_table"].to_dict(orient="records"),
        "driver_r2": driver_res_combined["r2"],
        "driver_adj_r2": driver_res_combined["adj_r2"],
        "regime_summary": driver_res_combined["regime_summary"].to_dict(orient="records"),
        "benchmarking_combined": bt_combined["metrics_summary"].to_dict(orient="records"),
        "benchmarking_liquid": bt_liquid["metrics_summary"].to_dict(orient="records"),
        "benchmarking_overnight": bt_overnight["metrics_summary"].to_dict(orient="records"),
        "forward_forecasts": forward_df.to_dict(orient="records"),
        "policy_summary": pol_res["policy_summary"].to_dict(orient="records"),
        "scheme_recommendations": scheme_rec_df.to_dict(orient="records"),
        "rolling_backtest_champion": bt_combined["backtest_predictions"][
            bt_combined["backtest_predictions"]["model"] == "Q-GBDT + Conformal (ML Champion)"
        ].to_dict(orient="records"),
        "rolling_backtest_naive": bt_combined["backtest_predictions"][
            bt_combined["backtest_predictions"]["model"] == "Seasonal Naive (Classical)"
        ].to_dict(orient="records"),
        "policy_monthly_detail": pol_res["policy_backtest_detail"].to_dict(orient="records"),
    }

    with open(os.path.join(out_dir, "dashboard_payload.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n=== ZECKATHON PS 03 PIPELINE COMPLETED SUCCESSFULLY ===")
    print("\n--- DRIVER ANALYSIS (COMBINED LIQUID + OVERNIGHT, OLS + NEWEY-WEST HAC) ---")
    print(driver_res_combined["drivers_table"][["driver_name", "category", "coef_cr_per_unit", "t_stat", "p_value", "partial_r2_pct", "matters_in_practice"]].to_string(index=False))
    print("\n--- MODEL BENCHMARKING (24-MONTH ROLLING BACKTEST, COMBINED LIQUID + OVERNIGHT) ---")
    print(bt_combined["metrics_summary"].to_string(index=False))
    print("\n--- DECISION LAYER POLICY COMPARISON (24-MONTH HISTORICAL COST BACKTEST) ---")
    print(pol_res["policy_summary"][["policy", "avg_buffer_cr", "shortfall_months_count", "cum_24m_decision_cost_cr", "cost_reduction_vs_naive_pct", "weighted_loss_reduction_vs_naive_pct"]].to_string(index=False))
    print("\n--- ACTIONABLE CASH-BUFFER SCHEDULE (INDUSTRY & ₹25,000 CR SCHEME) ---")
    print(scheme_rec_df.to_string(index=False))


if __name__ == "__main__":
    main()
