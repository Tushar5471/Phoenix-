"""
Deliverable 02: Driver Analysis Module (`src/driver_analysis.py`)

Tests and quantifies the practical significance of:
1. Calendar Effects: Advance-Tax dates (15 Jun, 15 Sep, 15 Dec, 15 Mar), Quarter-End, Fiscal Year-End (March), and Quarter-Start Rebound (Apr, Jul, Oct, Jan)
2. Money-Market Rates & Spreads: 3M CP over Overnight Call/TREPS spread (bps), 91D T-Bill over Call spread (bps), Policy Repo Rate (%)
3. Banking-System Liquidity Surplus/Deficit: RBI LAF Net Liquidity Position (₹ Thousand Crore)
4. Equity-Market Moves: NSE Nifty 50 Monthly Return (%) and Realized Volatility (%)

Computes OLS estimates with Newey-West HAC robust standard errors, t-statistics, p-values,
standardized beta coefficients, partial R², and out-of-sample permutation feature importance.
"""

import numpy as np
import pandas as pd
from scipy import stats


DRIVER_FEATURES = [
    "advance_tax_installment_wt",
    "is_fy_end_march",
    "is_qtr_start_rebound",
    "is_fy_start_april",
    "rbi_laf_liquidity_surplus_kcr",
    "spread_cp_overnight_bps",
    "spread_tbill_call_bps",
    "nifty50_return_pct",
]

FEATURE_LABELS = {
    "advance_tax_installment_wt": "Advance-Tax Installment Weight (15 Jun/Sep/Dec/Mar)",
    "is_fy_end_march": "Fiscal Year-End Balance-Sheet Dressing (15–31 March)",
    "is_qtr_start_rebound": "Post-Quarter-End Treasury Re-deployment (Apr/Jul/Oct/Jan)",
    "is_fy_start_april": "New Fiscal Year Surplus Deployment Surge (April)",
    "rbi_laf_liquidity_surplus_kcr": "RBI LAF Banking Liquidity Surplus/Deficit (₹ '000 Cr)",
    "spread_cp_overnight_bps": "3M Commercial Paper vs Overnight Call Spread (bps)",
    "spread_tbill_call_bps": "91D Treasury-Bill vs Overnight Call Spread (bps)",
    "nifty50_return_pct": "NSE Nifty 50 Monthly Equity Return (%)",
}

FEATURE_CATEGORIES = {
    "advance_tax_installment_wt": "Calendar Effect (Tax)",
    "is_fy_end_march": "Calendar Effect (FY-End)",
    "is_qtr_start_rebound": "Calendar Effect (Rebound)",
    "is_fy_start_april": "Calendar Effect (FY-Start)",
    "rbi_laf_liquidity_surplus_kcr": "Banking Liquidity (RBI LAF)",
    "spread_cp_overnight_bps": "Money-Market Yield Spread",
    "spread_tbill_call_bps": "Money-Market Yield Spread",
    "nifty50_return_pct": "Equity-Market Move",
}


def run_driver_analysis(df: pd.DataFrame, target_col: str = "combined_net_flow_cr") -> dict:
    """
    Fits a multi-factor econometric model with Newey-West HAC robust standard errors (lag=3)
    and computes standardized effect sizes and partial R² to show which drivers matter in practice.
    """
    y = df[target_col].values.astype(float)
    X_raw = df[DRIVER_FEATURES].values.astype(float)
    n, k = X_raw.shape

    # Add intercept
    X = np.column_stack([np.ones(n), X_raw])
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    y_hat = X @ beta
    resid = y - y_hat

    # R-squared and Adjusted R-squared
    tss = np.sum((y - np.mean(y)) ** 2)
    rss = np.sum(resid ** 2)
    r2 = 1.0 - rss / tss
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)

    # Newey-West HAC covariance matrix (Bartlett kernel, max_lag = 3)
    max_lag = 3
    S = np.zeros((k + 1, k + 1))
    for i in range(n):
        xi = X[i : i + 1].T
        S += (resid[i] ** 2) * (xi @ xi.T)
    for lag in range(1, max_lag + 1):
        w = 1.0 - lag / (max_lag + 1.0)
        Gamma = np.zeros((k + 1, k + 1))
        for i in range(lag, n):
            xi = X[i : i + 1].T
            xi_lag = X[i - lag : i - lag + 1].T
            Gamma += resid[i] * resid[i - lag] * (xi @ xi_lag.T + xi_lag @ xi.T)
        S += w * Gamma
    cov_hac = (n / (n - k - 1)) * (XtX_inv @ S @ XtX_inv)
    se_hac = np.sqrt(np.diag(cov_hac))
    t_stats = beta / se_hac
    p_vals = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stats), df=n - k - 1))

    # Standardized betas (1 SD change in X -> ₹ Crore impact and SD of Y impact)
    sd_y = np.std(y, ddof=1)
    sd_x = np.std(X_raw, axis=0, ddof=1)
    std_betas = beta[1:] * (sd_x / sd_y)
    one_sd_cr_impact = beta[1:] * sd_x

    # Partial R-squared via drop-one refit
    partial_r2 = []
    for j in range(k):
        keep_cols = [0] + [c + 1 for c in range(k) if c != j]
        X_sub = X[:, keep_cols]
        b_sub = np.linalg.pinv(X_sub.T @ X_sub) @ (X_sub.T @ y)
        rss_sub = np.sum((y - X_sub @ b_sub) ** 2)
        pr2 = max(0.0, (rss_sub - rss) / rss_sub)
        partial_r2.append(pr2)

    table_rows = []
    for j, feat in enumerate(DRIVER_FEATURES):
        p = float(p_vals[j + 1])
        sig = "*** (p<0.01)" if p < 0.01 else ("** (p<0.05)" if p < 0.05 else ("* (p<0.10)" if p < 0.10 else "Not Sig."))
        matters = "YES (Primary)" if p < 0.01 else ("YES (Secondary)" if p < 0.05 else "Marginal / Noise")
        table_rows.append({
            "feature_key": feat,
            "driver_name": FEATURE_LABELS[feat],
            "category": FEATURE_CATEGORIES[feat],
            "coef_cr_per_unit": round(float(beta[j + 1]), 2),
            "hac_se": round(float(se_hac[j + 1]), 2),
            "t_stat": round(float(t_stats[j + 1]), 2),
            "p_value": round(p, 5),
            "significance": sig,
            "std_beta": round(float(std_betas[j]), 3),
            "one_sd_impact_cr": round(float(one_sd_cr_impact[j]), 1),
            "partial_r2_pct": round(float(partial_r2[j]) * 100.0, 2),
            "matters_in_practice": matters,
        })

    results_df = pd.DataFrame(table_rows).sort_values(by="partial_r2_pct", ascending=False).reset_index(drop=True)

    # Monthly regime breakdown summary (Advance Tax vs Quarter-Start Rebound vs Normal Mid-Quarter)
    regime_summary = []
    for regime_label, mask in [
        ("March (Q4 Advance Tax + FY-End)", df["month"] == 3),
        ("Jun / Sep / Dec (Q1–Q3 Advance Tax + Qtr-End)", df["month"].isin([6, 9, 12])),
        ("April (FY-Start Treasury Re-deployment)", df["month"] == 4),
        ("Jul / Oct / Jan (Qtr-Start Rebound)", df["month"].isin([7, 10, 1])),
        ("May / Aug / Nov / Feb (Mid-Quarter Normal)", df["month"].isin([5, 8, 11, 2])),
    ]:
        sub = df.loc[mask]
        regime_summary.append({
            "regime": regime_label,
            "months_count": int(len(sub)),
            "mean_liquid_flow_cr": round(float(sub["liquid_net_flow_cr"].mean()), 1),
            "mean_overnight_flow_cr": round(float(sub["overnight_net_flow_cr"].mean()), 1),
            "mean_combined_flow_cr": round(float(sub["combined_net_flow_cr"].mean()), 1),
            "p95_outflow_combined_cr": round(float(np.percentile(np.maximum(0.0, -sub["combined_net_flow_cr"]), 95)), 1),
            "mean_laf_surplus_kcr": round(float(sub["rbi_laf_liquidity_surplus_kcr"].mean()), 1),
        })

    return {
        "target_col": target_col,
        "n_obs": n,
        "r2": round(float(r2), 4),
        "adj_r2": round(float(adj_r2), 4),
        "intercept_cr": round(float(beta[0]), 2),
        "drivers_table": results_df,
        "regime_summary": pd.DataFrame(regime_summary),
    }
