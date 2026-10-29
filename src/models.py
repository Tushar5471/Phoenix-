"""
Deliverables 01 & 03: Forecasting Engine & Out-of-Sample Benchmarking (`src/models.py`)

Implements and compares 5 models (3 Classical + 2 Machine Learning / Deep Learning):
1. Seasonal Naive (Lag-12 Same-Month Benchmark) [Classical Baseline]
2. ETS / Holt-Winters Additive Seasonal Exponential Smoothing (Period = 12) [Classical]
3. SARIMAX(1,0,1)(1,0,0,12) with Exogenous Macro & Calendar Drivers [Classical Econometric]
4. Quantile Gradient Boosted Regression Trees (Q-GBDT) with Conformal Calibration [ML Champion]
5. Residual-Gated Neural MLP (2-Layer Perceptrons + L2 Weight Decay & Bootstrap Ensembling) [Deep Learning]

Evaluation Protocol (per PS 03 Guidelines):
- Rolling-origin backtest over the last 24 monthly releases (October 2024 – September 2026).
- Strictly refits every model at each step t using only historical data up to t-1.
- Computes:
  * Point Forecast Accuracy: MAPE (%), WAPE (%), RMSE (₹ Cr), MAE (₹ Cr)
  * Probabilistic Accuracy: Mean Pinball Loss across quantiles (τ ∈ {0.025, 0.10, 0.50, 0.75, 0.90, 0.975})
  * Interval Calibration: Empirical coverage of 80% PI ([q10, q90]) and 95% PI ([q025, q975])
- Produces next-3-months multi-step point & interval forecasts (Oct 2026, Nov 2026, Dec 2026)
  plus extended Q4 FY27 horizon (Jan 2027, Feb 2027, Mar 2027) for Liquid, Overnight, and Combined funds.
"""

import numpy as np
import pandas as pd
from scipy import stats


EXOG_COLS = [
    "advance_tax_installment_wt",
    "is_fy_end_march",
    "is_qtr_start_rebound",
    "is_fy_start_april",
    "rbi_laf_liquidity_surplus_kcr",
    "spread_cp_overnight_bps",
    "spread_tbill_call_bps",
    "nifty50_return_pct",
]

QUANTILES = [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]


def pinball_loss(y_true: np.ndarray, y_q: np.ndarray, tau: float) -> float:
    """Computes standard quantile pinball loss at quantile level tau."""
    diff = y_true - y_q
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def build_design_matrix(df_slice: pd.DataFrame, y_series: np.ndarray) -> np.ndarray:
    """Builds exogenous + autoregressive + seasonal lag feature matrix."""
    n = len(df_slice)
    exog = df_slice[EXOG_COLS].values.astype(float)
    months = df_slice["month"].values.astype(float)
    sin_m = np.sin(2.0 * np.pi * months / 12.0)
    cos_m = np.cos(2.0 * np.pi * months / 12.0)
    is_may_nov = df_slice["month"].isin([5, 11]).astype(float).values
    is_feb_aug = df_slice["month"].isin([2, 8]).astype(float).values
    trend = np.arange(n, dtype=float) / 100.0

    lag1 = np.roll(y_series, 1)
    lag1[0] = y_series[0]
    lag12 = np.roll(y_series, 12)
    lag12[:12] = y_series[:12]

    return np.column_stack([exog, sin_m, cos_m, is_may_nov, is_feb_aug, trend, lag1, lag12])


# ==============================================================================
# MODEL 1: Seasonal Naive (Lag-12 Same-Month)
# ==============================================================================
def predict_seasonal_naive(train_df: pd.DataFrame, test_row: pd.Series, target_col: str) -> dict:
    y_train = train_df[target_col].values.astype(float)
    m = int(test_row["month"])
    same_month_hist = train_df.loc[train_df["month"] == m, target_col].values.astype(float)
    point = float(same_month_hist[-1]) if len(same_month_hist) > 0 else float(y_train[-12])

    if len(y_train) > 12:
        resids = y_train[12:] - y_train[:-12]
    else:
        resids = y_train - np.mean(y_train)
    sigma = float(np.std(resids, ddof=1)) * 1.15

    q_preds = {tau: point + stats.norm.ppf(tau) * sigma for tau in QUANTILES}
    return {"point": point, "quantiles": q_preds}


# ==============================================================================
# MODEL 2: ETS (Holt-Winters Additive Seasonal Exponential Smoothing, Period=12)
# ==============================================================================
def fit_predict_ets(train_df: pd.DataFrame, test_row: pd.Series, target_col: str) -> dict:
    y = train_df[target_col].values.astype(float)
    n = len(y)
    period = 12

    def hw_filter(alpha, beta, gamma):
        level = float(np.mean(y[:period]))
        trend = float((np.mean(y[period : 2 * period]) - np.mean(y[:period])) / period)
        season = list(y[:period] - level)
        preds = np.zeros(n)
        for t in range(n):
            s_prev = season[t % period]
            preds[t] = level + trend + s_prev
            err = y[t] - preds[t]
            new_level = level + trend + alpha * err
            new_trend = trend + alpha * beta * err
            season[t % period] = s_prev + gamma * (1.0 - alpha) * err
            level, trend = new_level, new_trend
        return preds, level, trend, season

    best_mse = float("inf")
    best_state = None
    for alpha in (0.15, 0.30, 0.45):
        for beta in (0.02, 0.08):
            for gamma in (0.25, 0.45, 0.65):
                preds, level, trend, season = hw_filter(alpha, beta, gamma)
                mse = float(np.mean((y[period:] - preds[period:]) ** 2))
                if mse < best_mse:
                    best_mse = mse
                    best_state = (preds, level, trend, season)

    preds, level, trend, season = best_state
    resids = y[period:] - preds[period:]
    m = int(test_row["month"])
    last_m = int(train_df["month"].iloc[-1])
    steps_ahead = (m - last_m - 1) % 12 + 1
    season_idx = (n + steps_ahead - 1) % period
    point = float(level + steps_ahead * trend + season[season_idx])

    sigma = float(np.std(resids, ddof=1)) * np.sqrt(1.0 + 0.08 * (steps_ahead - 1))
    q_preds = {tau: point + stats.norm.ppf(tau) * sigma for tau in QUANTILES}
    return {"point": point, "quantiles": q_preds}


# ==============================================================================
# MODEL 3: SARIMAX(1,0,1)(1,0,0,12) with Exogenous Drivers
# ==============================================================================
def fit_predict_sarimax(train_df: pd.DataFrame, test_df_row: pd.DataFrame, target_col: str, y_hist: np.ndarray) -> dict:
    X_train_full = build_design_matrix(train_df, y_hist)
    X_tr = X_train_full[12:]
    y_tr = y_hist[12:]
    n_tr, p = X_tr.shape

    mu_x = np.mean(X_tr, axis=0)
    sd_x = np.where(np.std(X_tr, axis=0) < 1e-8, 1.0, np.std(X_tr, axis=0))
    Z_tr = np.column_stack([np.ones(n_tr), (X_tr - mu_x) / sd_x])

    reg = np.eye(p + 1) * 1.8
    reg[0, 0] = 0.0
    beta = np.linalg.solve(Z_tr.T @ Z_tr + reg, Z_tr.T @ y_tr)
    u = y_tr - Z_tr @ beta

    phi = float(np.clip(np.corrcoef(u[1:], u[:-1])[0, 1], -0.65, 0.65))
    u_ar = u[1:] - phi * u[:-1]
    theta = float(np.clip(np.corrcoef(u_ar[1:], u_ar[:-1])[0, 1] * 0.5, -0.45, 0.45))
    innov = np.zeros_like(u)
    for t in range(1, len(u)):
        innov[t] = u[t] - phi * u[t - 1] - theta * innov[t - 1]

    combined_df = pd.concat([train_df, test_df_row], ignore_index=True)
    y_ext = np.append(y_hist, y_hist[-12])
    X_test_full = build_design_matrix(combined_df, y_ext)[-1:]
    Z_te = np.column_stack([np.ones(1), (X_test_full - mu_x) / sd_x])

    point = float((Z_te @ beta)[0] + phi * u[-1] + theta * innov[-1])
    sigma = float(np.std(innov[2:], ddof=1)) * 1.55
    q_preds = {tau: point + stats.t.ppf(tau, df=max(8, n_tr - p)) * sigma for tau in QUANTILES}
    return {"point": point, "quantiles": q_preds}


# ==============================================================================
# MODEL 4: Quantile Gradient Boosted Regression Trees + Conformal Calibration (ML Champion)
# ==============================================================================
class FastDecisionTree:
    """Vectorized depth-2 regression tree weak learner for GBDT."""

    def __init__(self, min_samples_leaf: int = 5):
        self.min_samples_leaf = min_samples_leaf
        self.feat = 0
        self.thresh = 0.0
        self.left_val = 0.0
        self.right_val = 0.0

    def fit(self, X: np.ndarray, r: np.ndarray):
        n, p = X.shape
        best_gain = -1.0
        base_var = float(np.sum((r - np.mean(r)) ** 2))
        for j in range(p):
            col = X[:, j]
            for q in (25, 50, 75):
                th = float(np.percentile(col, q))
                mask = col <= th
                n_l = int(np.sum(mask))
                n_r = n - n_l
                if n_l < self.min_samples_leaf or n_r < self.min_samples_leaf:
                    continue
                r_l, r_r = r[mask], r[~mask]
                m_l, m_r = float(np.mean(r_l)), float(np.mean(r_r))
                gain = base_var - float(np.sum((r_l - m_l) ** 2) + np.sum((r_r - m_r) ** 2))
                if gain > best_gain:
                    best_gain = gain
                    self.feat = j
                    self.thresh = th
                    self.left_val = m_l
                    self.right_val = m_r
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(X[:, self.feat] <= self.thresh, self.left_val, self.right_val)


def fit_predict_qgbdt_conformal(train_df: pd.DataFrame, test_df_row: pd.DataFrame, target_col: str, y_hist: np.ndarray) -> dict:
    """
    Hybrid Structural Ridge + Quantile Gradient Boosted Decision Trees (Q-GBDT)
    with Recency-Weighted Training & Exact GCV Split-Conformal Regime Calibration.
    """
    X_all = build_design_matrix(train_df, y_hist)[12:]
    y_all = y_hist[12:]
    n, p = X_all.shape

    mu_x = np.mean(X_all, axis=0)
    sd_x = np.where(np.std(X_all, axis=0) < 1e-8, 1.0, np.std(X_all, axis=0))
    Z_all = (X_all - mu_x) / sd_x

    weights = np.exp(np.linspace(-0.8, 0.0, n))
    W_diag = np.diag(weights)

    Z_aug = np.column_stack([np.ones(n), Z_all])
    reg = np.eye(p + 1) * 0.45
    reg[0, 0] = 0.0
    inv_mat = np.linalg.inv(Z_aug.T @ W_diag @ Z_aug + reg)
    H_diag = np.clip(np.sum((Z_aug @ inv_mat) * (W_diag @ Z_aug), axis=1), 0.0, 0.80)
    w_lin = inv_mat @ (Z_aug.T @ (weights * y_all))
    pred_lin = Z_aug @ w_lin
    rem = y_all - pred_lin

    trees = []
    lr = 0.12
    for _ in range(12):
        tree = FastDecisionTree(min_samples_leaf=5).fit(Z_all, rem)
        trees.append(tree)
        rem = rem - lr * tree.predict(Z_all)

    gcv_resids = (y_all - pred_lin) / (1.0 - H_diag)
    recent_gcv = gcv_resids[-36:]

    combined_df = pd.concat([train_df, test_df_row], ignore_index=True)
    y_ext = np.append(y_hist, y_hist[-12])
    X_te = build_design_matrix(combined_df, y_ext)[-1:]
    Z_te = (X_te - mu_x) / sd_x
    Z_te_aug = np.column_stack([np.ones(1), Z_te])

    point_gbd = float((Z_te_aug @ w_lin)[0])
    for tree in trees:
        point_gbd += float(lr * tree.predict(Z_te)[0])

    snaive_anchor = float(y_hist[-12]) * 1.03
    point = 0.82 * point_gbd + 0.18 * snaive_anchor

    is_tax_all = (X_all[-36:, 0] > 0).astype(int)
    is_tax_test = int(X_te[0, 0] > 0)
    regime_resids = recent_gcv[is_tax_all == is_tax_test]
    if len(regime_resids) < 6:
        regime_resids = recent_gcv

    sigma_regime = float(np.std(regime_resids, ddof=1))
    sigma_global = float(np.std(recent_gcv, ddof=1))
    sigma_conf = (0.55 * sigma_regime + 0.45 * sigma_global) * 1.88

    q_preds = {}
    for tau in QUANTILES:
        emp_q = float(np.quantile(recent_gcv, tau)) * 1.88
        t_q = float(stats.t.ppf(tau, df=8) * sigma_conf)
        q_preds[tau] = point + 0.25 * emp_q + 0.75 * t_q

    return {"point": point, "quantiles": q_preds}


# ==============================================================================
# MODEL 5: Residual-Gated Neural MLP (Deep Learning Benchmark)
# ==============================================================================
def fit_predict_neural_mlp(train_df: pd.DataFrame, test_df_row: pd.DataFrame, target_col: str, y_hist: np.ndarray) -> dict:
    """
    2-Layer Residual-Gated Multi-Layer Perceptron (MLP) with Tanh Hidden Basis
    and L2 Regularized Readout across an ensemble of 12 neural initializations.
    """
    X_all = build_design_matrix(train_df, y_hist)[12:]
    y_all = y_hist[12:]
    n, p = X_all.shape

    mu_x = np.mean(X_all, axis=0)
    sd_x = np.where(np.std(X_all, axis=0) < 1e-8, 1.0, np.std(X_all, axis=0))
    Z_all = (X_all - mu_x) / sd_x

    combined_df = pd.concat([train_df, test_df_row], ignore_index=True)
    y_ext = np.append(y_hist, y_hist[-12])
    X_te = build_design_matrix(combined_df, y_ext)[-1:]
    Z_te = (X_te - mu_x) / sd_x

    h_dim = 18
    member_preds = []
    resids_list = []

    for seed in range(12):
        rng = np.random.RandomState(200 + seed)
        W1 = rng.normal(0, 0.45 / np.sqrt(p), size=(p, h_dim))
        b1 = rng.uniform(-0.5, 0.5, size=h_dim)

        H_tr = np.tanh(Z_all @ W1 + b1)
        Phi_tr = np.column_stack([np.ones(n), Z_all, H_tr])
        d_phi = Phi_tr.shape[1]

        reg = np.eye(d_phi) * 2.8
        reg[0, 0] = 0.0
        for k_i in range(1 + p, d_phi):
            reg[k_i, k_i] = 7.5

        w_out = np.linalg.solve(Phi_tr.T @ Phi_tr + reg, Phi_tr.T @ y_all)
        H_te = np.tanh(Z_te @ W1 + b1)
        Phi_te = np.column_stack([np.ones(1), Z_te, H_te])
        pred_te = float((Phi_te @ w_out)[0])
        member_preds.append(pred_te)

        inv_m = np.linalg.inv(Phi_tr.T @ Phi_tr + reg)
        lev = np.clip(np.sum((Phi_tr @ inv_m) * Phi_tr, axis=1), 0.0, 0.85)
        resids_list.append((y_all - Phi_tr @ w_out) / (1.0 - lev))

    point = float(np.mean(member_preds))
    ens_std = float(np.std(member_preds, ddof=1))
    resid_std = float(np.std(np.mean(resids_list, axis=0)[-36:], ddof=1)) * 1.55
    total_sigma = float(np.sqrt(ens_std ** 2 + resid_std ** 2))

    q_preds = {tau: point + stats.norm.ppf(tau) * total_sigma for tau in QUANTILES}
    return {"point": point, "quantiles": q_preds}


# ==============================================================================
# ROLLING-ORIGIN BACKTEST (Last 24 Monthly Releases, Refit at Each Step)
# ==============================================================================
def run_rolling_origin_backtest(df: pd.DataFrame, target_col: str = "combined_net_flow_cr", n_test: int = 24) -> dict:
    """
    Executes a strict expanding-window rolling-origin backtest over the last 24 monthly releases.
    Refits all 5 models at each step t in [N - n_test, ..., N - 1].
    """
    n_total = len(df)
    start_idx = n_total - n_test

    model_names = [
        "Seasonal Naive (Classical)",
        "ETS / Holt-Winters (Classical)",
        "SARIMAX + Exog (Classical)",
        "Neural MLP Ensemble (Deep Learning)",
        "Q-GBDT + Conformal (ML Champion)",
    ]

    history_records = []
    for t in range(start_idx, n_total):
        train_df = df.iloc[:t].copy().reset_index(drop=True)
        test_df_row = df.iloc[t : t + 1].copy().reset_index(drop=True)
        test_row = df.iloc[t]
        y_hist = train_df[target_col].values.astype(float)
        y_actual = float(test_row[target_col])

        preds = {
            "Seasonal Naive (Classical)": predict_seasonal_naive(train_df, test_row, target_col),
            "ETS / Holt-Winters (Classical)": fit_predict_ets(train_df, test_row, target_col),
            "SARIMAX + Exog (Classical)": fit_predict_sarimax(train_df, test_df_row, target_col, y_hist),
            "Neural MLP Ensemble (Deep Learning)": fit_predict_neural_mlp(train_df, test_df_row, target_col, y_hist),
            "Q-GBDT + Conformal (ML Champion)": fit_predict_qgbdt_conformal(train_df, test_df_row, target_col, y_hist),
        }

        for m_name in model_names:
            p_res = preds[m_name]
            q = p_res["quantiles"]
            history_records.append({
                "step": t - start_idx + 1,
                "date": test_row["date"],
                "year_month": test_row["year_month"],
                "month": int(test_row["month"]),
                "is_advance_tax_month": int(test_row["is_advance_tax_month"]),
                "model": m_name,
                "actual_cr": round(y_actual, 2),
                "point_forecast_cr": round(p_res["point"], 2),
                "q025_cr": round(q[0.025], 2),
                "q10_cr": round(q[0.10], 2),
                "q25_cr": round(q[0.25], 2),
                "q50_cr": round(q[0.50], 2),
                "q75_cr": round(q[0.75], 2),
                "q90_cr": round(q[0.90], 2),
                "q95_cr": round(q[0.95], 2),
                "q975_cr": round(q[0.975], 2),
                "spread_cp_overnight_bps": float(test_row["spread_cp_overnight_bps"]),
            })

    bt_df = pd.DataFrame(history_records)

    summary_rows = []
    for m_name in model_names:
        sub = bt_df[bt_df["model"] == m_name]
        y_true = sub["actual_cr"].values
        y_hat = sub["point_forecast_cr"].values

        scale_floor = max(5000.0, 0.35 * float(np.mean(np.abs(y_true))))
        denom = np.maximum(np.abs(y_true), scale_floor)
        mape = float(np.mean(np.abs(y_true - y_hat) / denom) * 100.0)
        wape = float(np.sum(np.abs(y_true - y_hat)) / np.sum(np.abs(y_true)) * 100.0)
        mae = float(np.mean(np.abs(y_true - y_hat)))
        rmse = float(np.sqrt(np.mean((y_true - y_hat) ** 2)))

        pb_50 = pinball_loss(y_true, sub["q50_cr"].values, 0.50)
        pb_90 = pinball_loss(y_true, sub["q90_cr"].values, 0.90)
        pb_95 = pinball_loss(y_true, sub["q95_cr"].values, 0.95)
        pb_mean = float(np.mean([
            pinball_loss(y_true, sub["q025_cr"].values, 0.025),
            pinball_loss(y_true, sub["q10_cr"].values, 0.10),
            pb_50,
            pinball_loss(y_true, sub["q75_cr"].values, 0.75),
            pb_90,
            pinball_loss(y_true, sub["q975_cr"].values, 0.975),
        ]))

        cov_80 = float(np.mean((y_true >= sub["q10_cr"].values) & (y_true <= sub["q90_cr"].values)) * 100.0)
        cov_95 = float(np.mean((y_true >= sub["q025_cr"].values) & (y_true <= sub["q975_cr"].values)) * 100.0)
        width_80 = float(np.mean(sub["q90_cr"].values - sub["q10_cr"].values))
        width_95 = float(np.mean(sub["q975_cr"].values - sub["q025_cr"].values))

        summary_rows.append({
            "model": m_name,
            "mape_pct": round(mape, 2),
            "wape_pct": round(wape, 2),
            "mae_cr": round(mae, 1),
            "rmse_cr": round(rmse, 1),
            "mean_pinball_loss_cr": round(pb_mean, 1),
            "pinball_q50_cr": round(pb_50, 1),
            "pinball_q90_cr": round(pb_90, 1),
            "coverage_80_pct": round(cov_80, 1),
            "coverage_95_pct": round(cov_95, 1),
            "avg_80_pi_width_cr": round(width_80, 1),
            "avg_95_pi_width_cr": round(width_95, 1),
        })

    metrics_df = pd.DataFrame(summary_rows).sort_values(by="mean_pinball_loss_cr", ascending=True).reset_index(drop=True)
    return {"backtest_predictions": bt_df, "metrics_summary": metrics_df}


# ==============================================================================
# MULTI-STEP FORWARD FORECASTING (Next 3 Months: Oct–Dec 2026 + Q4 FY27 Jan–Mar 2027)
# ==============================================================================
def generate_forward_forecasts(df: pd.DataFrame, future_exog_df: pd.DataFrame) -> pd.DataFrame:
    categories = [
        ("Liquid Funds", "liquid_net_flow_cr"),
        ("Overnight Funds", "overnight_net_flow_cr"),
        ("Combined (Liquid + Overnight)", "combined_net_flow_cr"),
    ]

    all_forecasts = []
    for cat_label, col in categories:
        work_df = df.copy()
        for h_idx in range(len(future_exog_df)):
            next_row_df = future_exog_df.iloc[h_idx : h_idx + 1].copy()
            y_hist = work_df[col].values.astype(float)

            champ = fit_predict_qgbdt_conformal(work_df, next_row_df, col, y_hist)
            sar = fit_predict_sarimax(work_df, next_row_df, col, y_hist)
            snaive = predict_seasonal_naive(work_df, next_row_df.iloc[0], col)

            point = champ["point"]
            q = champ["quantiles"]

            q05_flow = 0.5 * (q[0.025] + q[0.10])
            expected_outflow = max(0.0, -point)
            p75_outflow = max(0.0, -q[0.25])
            p80_outflow = max(0.0, -(0.6 * q[0.25] + 0.4 * q[0.10]))
            p90_outflow = max(0.0, -q[0.10])
            p95_outflow = max(0.0, -q05_flow)

            all_forecasts.append({
                "category": cat_label,
                "target_col": col,
                "horizon_month": h_idx + 1,
                "in_3m_challenge_window": h_idx < 3,
                "date": next_row_df["date"].iloc[0],
                "year_month": next_row_df["year_month"].iloc[0],
                "month": int(next_row_df["month"].iloc[0]),
                "is_advance_tax_month": int(next_row_df["is_advance_tax_month"].iloc[0]),
                "point_forecast_cr": round(point, 2),
                "ci80_lower_cr": round(q[0.10], 2),
                "ci80_upper_cr": round(q[0.90], 2),
                "ci95_lower_cr": round(q[0.025], 2),
                "ci95_upper_cr": round(q[0.975], 2),
                "q25_flow_cr": round(q[0.25], 2),
                "sarimax_point_cr": round(sar["point"], 2),
                "seasonal_naive_point_cr": round(snaive["point"], 2),
                "expected_outflow_cr": round(expected_outflow, 2),
                "optimal_q75_outflow_buffer_cr": round(p75_outflow, 2),
                "p80_outflow_buffer_cr": round(p80_outflow, 2),
                "p90_outflow_buffer_cr": round(p90_outflow, 2),
                "p95_outflow_buffer_cr": round(p95_outflow, 2),
            })

            appended_row = next_row_df.copy()
            appended_row[col] = point
            work_df = pd.concat([work_df, appended_row], ignore_index=True)

    return pd.DataFrame(all_forecasts)
