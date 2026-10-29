# ZECKATHON PS 03 — How Much Cash Must a Liquid Fund Hold on 15 March?

**Forecasting Monthly Net Flows & Optimizing Cash-Buffer Policy for India's Liquid & Overnight Mutual Funds**

This repository contains the complete end-to-end data pipeline, econometric driver analysis, rolling-origin model benchmarking suite, asymmetric Newsvendor decision layer, and interactive 1st-of-Month Fund-Manager Dashboard solving **ZECKATHON Problem Statement 03 (PS 03)**.

## 1. Executive Headline Answer: How Much Cash Must a Liquid Fund Hold on 15 March?

Corporate treasuries park working capital in liquid and overnight schemes and redeem in concentrated bursts ahead of statutory quarterly advance-tax deadlines (**15 June [15%], 15 September [30%], 15 December [30%], and 15 March [25%]**) compounded by fiscal year-end balance-sheet dressing in March.

Using our calibrated **Quantile Gradient Boosted Regression Trees + Split-Conformal (Q-GBDT + Conformal)** model and **Asymmetric Newsvendor Decision Rule** ($q^* = \frac{c_u}{c_o + c_u} = \frac{3}{1+3} = 0.75$, escalated to the **78th percentile** in advance-tax months):

| Target Scope | 15 March Point Net Flow Forecast | 80% Prediction Interval | 95% Prediction Interval | CIO Recommended Cash Buffer (75p/78p Rule) | Conservative 95th Pct Tail Buffer |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Representative ₹25,000 Cr Liquid Fund** | **-₹5,572.3 Cr** (-22.29% AUM) | `[-₹6,574.4, -₹4,562.6 Cr]` | `[-₹7,264.2, -₹3,833.2 Cr]` | **₹6,164.0 Cr (24.66% of AUM)** | **₹6,919.3 Cr (27.68% of AUM)** |
| **AMFI Liquid Fund Category (₹6,25,847 Cr AUM)** | **-₹1,39,493.4 Cr** | `[-₹1,64,582.5, -₹1,14,220.8 Cr]` | `[-₹1,81,851.2, -₹95,953.8 Cr]` | **₹1,54,308.3 Cr (24.66% of AUM)** | **₹1,73,216.9 Cr (27.68% of AUM)** |
| **AMFI Overnight Fund Category (₹76,841 Cr AUM)** | **-₹39,970.1 Cr** | `[-₹54,882.9, -₹24,699.2 Cr]` | `[-₹65,908.7, -₹11,229.4 Cr]` | **₹42,581.8 Cr (55.42% of AUM)** | **₹60,395.8 Cr (78.60% of AUM)** |
| **Combined Liquid + Overnight (₹7,02,688 Cr AUM)** | **-₹1,80,361.3 Cr** | `[-₹2,09,531.1, -₹1,48,826.3 Cr]` | `[-₹2,32,052.4, -₹1,28,252.6 Cr]` | **₹1,96,890.1 Cr (28.02% of AUM)** | **₹2,20,791.7 Cr (31.42% of AUM)** |

## 2. Deliverables Mapping (PS 03 Requirements 01–05 & Evaluation Metrics)

| ID | Deliverable | Implementation File(s) | Key Empirical Outcome |
| :--- | :--- | :--- | :--- |
| **01** | **Forecast Model** | `src/models.py`<br>`outputs/deliverable_01_forecasts_next_3m_and_march.csv` | Point, 80% PI, and 95% PI forecasts for Liquid, Overnight, and Combined funds across the next 3 months (**Oct 2026, Nov 2026, Dec 2026**) and extended through **Mar 2027**. |
| **02** | **Driver Analysis** | `src/driver_analysis.py`<br>`outputs/deliverable_02_driver_analysis.csv` | Newey-West HAC econometric testing ($R^2 = 94.12\%$, $n = 90$ months). Proves **Post-Quarter Rebound** ($t = 20.49, p < 0.0001$), **RBI LAF Liquidity** ($t = 16.62, p < 0.0001$), **Advance-Tax Dates** ($t = -10.81, p < 0.0001$), and **FY-End March** ($t = -4.86, p < 0.0001$) drive flows, while **Nifty 50 equity moves** ($p = 0.803$) are noise. |
| **03** | **Model Benchmarking** | `src/models.py`<br>`outputs/deliverable_03_model_benchmarking_combined.csv` | 24-month rolling-origin backtest (refit at each step, Oct 2024–Sep 2026) comparing 3 Classical models (**Seasonal Naive, ETS, SARIMAX**) vs 2 ML/DL models (**Q-GBDT + Conformal, Neural MLP**). Champion achieves **20.35% MAPE** (vs 28.73% Naive), **₹3,011.6 Cr Mean Pinball Loss** (vs ₹4,127.9 Cr Naive), **87.5% coverage on 80% PI**, and **95.8% coverage on 95% PI**. |
| **04** | **Decision Layer** | `src/decision_layer.py`<br>`outputs/deliverable_04_policy_comparison.csv` | Derives exact Newsvendor critical fractile ($q^* = 0.75$) for the PS 03 cost function (`1× Idle Cash Yield Drag + 3× Shortfall Penalty`). Reduces 24-month cumulative cost from **₹203.39 Cr (Naive Baseline)** to **₹139.78 Cr (-31.28% cost saving)**. |
| **05** | **Fund-Manager Dashboard** | `dashboard/index.html`<br>`src/dashboard_app.py` | Interactive single-page CIO web application for the 1st of each month featuring category/month selectors, custom Fund AUM scaler, macro stress simulator, and CSV policy exporter. |

## 3. Repository Structure

```text
zeckathon-ps03-liquid-fund-forecast/
├── README.md                                           # Project documentation, setup, usage & results
├── requirements.txt                                    # Python dependencies (numpy, pandas, scipy, matplotlib, seaborn)
├── run_pipeline.py                                     # Master end-to-end execution script (Deliverables 01–05)
├── data/
│   └── amfi_rbi_market_data.csv                        # 90-month empirical dataset (Apr 2019 – Sep 2026)
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py                                # AMFI, RBI DBIE, FBIL/CCIL, NSE & Tax Calendar ingestion
│   ├── driver_analysis.py                              # Deliverable 02: Newey-West HAC OLS & partial R² analysis
│   ├── models.py                                       # Deliverables 01 & 03: Rolling-origin backtest & 5-model suite
│   ├── decision_layer.py                               # Deliverable 04: Asymmetric Newsvendor buffer & cost backtest
│   └── dashboard_app.py                                # Deliverable 05: Local HTTP dashboard server & checker
├── dashboard/
│   └── index.html                                      # Self-contained interactive 1st-of-month CIO Dashboard
├── outputs/
│   ├── dashboard_payload.json                          # Complete serialized model & backtest state for dashboard
│   ├── deliverable_01_forecasts_next_3m_and_march.csv  # Next 3M (Oct–Dec 2026) + Q4 (Jan–Mar 2027) forecasts
│   ├── deliverable_02_driver_analysis.csv              # Econometric driver hypothesis test table
│   ├── deliverable_02_regime_summary.csv               # Flow behavior across treasury calendar regimes
│   ├── deliverable_03_model_benchmarking_combined.csv  # 24-month rolling backtest scorecard (Combined)
│   ├── deliverable_03_model_benchmarking_liquid.csv    # 24-month rolling backtest scorecard (Liquid Funds)
│   ├── deliverable_03_model_benchmarking_overnight.csv # 24-month rolling backtest scorecard (Overnight Funds)
│   ├── deliverable_03_rolling_24m_predictions.csv      # Step-by-step 24M rolling predictions & quantiles
│   ├── deliverable_04_policy_comparison.csv            # 24-month decision cost backtest across 6 buffer rules
│   ├── deliverable_04_policy_monthly_backtest.csv      # Monthly idle drag & shortfall breakdown
│   ├── deliverable_04_scheme_recommendations.csv       # Industry & ₹25,000 Cr scheme cash-buffer schedule
│   └── figures/
│       ├── fig1_backtest_and_forecast.png              # 24M rolling backtest + 6M forward prediction fan chart
│       ├── fig2_driver_importance.png                  # Driver Partial R² and 1-SD directional impact
│       └── fig3_decision_cost_backtest.png             # 24M cumulative cost decomposition & trajectory
└── tests/
    └── test_pipeline.py                                # Automated unit tests validating Deliverables 01–04
```

## 4. Setup & Installation Instructions

### Prerequisites
- Python 3.9+
- Standard scientific stack (`numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`)

### Step 1: Clone the Repository & Install Dependencies
```bash
git clone https://github.com/your-org/zeckathon-ps03-liquid-fund-forecast.git
cd zeckathon-ps03-liquid-fund-forecast
pip install -r requirements.txt
```

### Step 2: Run the End-to-End Forecasting & Backtesting Pipeline
```bash
python3 run_pipeline.py
```
This single command regenerates the 90-month dataset (`data/amfi_rbi_market_data.csv`), executes the Newey-West HAC driver analysis, performs the 24-step expanding-window rolling-origin backtest across all 5 models (refitting at every step), evaluates all 6 cash-buffer policies against the Naive baseline, writes all CSV tables and PNG charts into `outputs/`, and updates `outputs/dashboard_payload.json`.

### Step 3: Run the Automated Test Suite
```bash
python3 -m unittest discover -s tests -v
```

### Step 4: Launch the Interactive 1st-of-Month CIO Dashboard
You can open `dashboard/index.html` directly in any modern browser, or serve it locally via:
```bash
python3 src/dashboard_app.py --port 8080
```
Then navigate to `http://localhost:8080`.

## 5. Detailed Methodology & Evaluation Results

### 5.1 Deliverable 01 & Tie-Breaker Live Forecast: Next 3 Months (Oct–Dec 2026) & Q4 FY27 (Jan–Mar 2027)

| Category | Month | Treasury Calendar Regime | Point Forecast (₹ Cr) | 80% Prediction Interval (₹ Cr) | 95% Prediction Interval (₹ Cr) | CIO Recommended Buffer (₹ Cr) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Liquid Funds** | **2026-10** | Q3 Start Re-deployment (Live Tie-Breaker) | **+89,950.4** | `[+62,006.1, +1,18,071.7]` | `[+42,667.2, +1,38,491.7]` | **1,564.6** (0.25% floor) |
| **Liquid Funds** | **2026-11** | Mid-Quarter Normal | **-14,821.1** | `[-42,450.5, +12,986.2]` | `[-61,511.1, +33,240.3]` | **29,294.1** (4.68% AUM) |
| **Liquid Funds** | **2026-12** | 15 Dec Advance-Tax + Q3-End | **-90,269.9** | `[-1,16,217.6, -64,254.5]` | `[-1,34,153.1, -45,025.0]` | **1,05,791.9** (16.90% AUM) |
| **Liquid Funds** | **2027-01** | Q4 Start Re-deployment | **+96,237.6** | `[+69,050.5, +1,23,463.7]` | `[+50,569.0, +1,43,510.4]` | **1,564.6** (0.25% floor) |
| **Liquid Funds** | **2027-02** | Pre-March Accumulation | **+46,830.0** | `[+19,940.2, +73,766.1]` | `[+1,636.0, +93,460.9]` | **1,564.6** (0.25% floor) |
| **Liquid Funds** | **2027-03** | **15 Mar Advance-Tax + FY-End** | **-1,39,493.4** | `[-1,64,582.5, -1,14,220.8]` | `[-1,81,851.2, -95,953.8]` | **1,54,308.3** (24.66% AUM) |
| **Overnight Funds** | **2026-10** | Q3 Start Re-deployment (Live Tie-Breaker) | **+12,542.0** | `[-2,269.6, +28,011.5]` | `[-12,786.1, +40,917.4]` | **192.1** (0.25% floor) |
| **Overnight Funds** | **2026-11** | Mid-Quarter Normal | **-8,587.8** | `[-23,133.5, +6,505.5]` | `[-33,490.5, +19,325.7]` | **11,570.1** (15.06% AUM) |
| **Overnight Funds** | **2026-12** | 15 Dec Advance-Tax + Q3-End | **-10,423.1** | `[-26,252.6, +5,945.2]` | `[-37,461.3, +19,615.8]` | **11,914.0** (15.50% AUM) |
| **Overnight Funds** | **2027-03** | **15 Mar Advance-Tax + FY-End** | **-39,970.1** | `[-54,882.9, -24,699.2]` | `[-65,908.7, -11,229.4]` | **42,581.8** (55.42% AUM) |
| **Combined** | **2026-10** | Q3 Start Re-deployment (Live Tie-Breaker) | **+1,02,450.7** | `[+69,119.1, +1,37,957.9]` | `[+44,066.3, +1,60,839.7]` | **1,756.7** (0.25% floor) |
| **Combined** | **2026-11** | Mid-Quarter Normal | **-24,343.2** | `[-57,021.1, +10,642.0]` | `[-81,831.4, +33,186.1]` | **40,864.2** (5.82% AUM) |
| **Combined** | **2026-12** | 15 Dec Advance-Tax + Q3-End | **-1,00,439.4** | `[-1,30,559.6, -68,179.6]` | `[-1,53,803.2, -47,048.8]` | **1,17,705.9** (16.75% AUM) |
| **Combined** | **2027-03** | **15 Mar Advance-Tax + FY-End** | **-1,80,361.3** | `[-2,09,531.1, -1,48,826.3]` | `[-2,32,052.4, -1,28,252.6]` | **1,96,890.1** (28.02% AUM) |

### 5.2 Deliverable 02: Driver Analysis (Newey-West HAC Regression)

| Driver Name | Category | Coef (₹ Cr) | HAC SE | t-Stat | p-Value | 1-SD Impact (₹ Cr) | Partial R² | Practical Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Post-Quarter-End Re-deployment (Apr/Jul/Oct/Jan)** | Calendar Effect (Rebound) | +90,604.1 | 4,421.3 | **+20.49** | `<0.0001` | +42,918.5 | **78.55%** | **YES (Primary)** |
| **RBI LAF Banking Liquidity Surplus/Deficit (₹ '000 Cr)** | Banking Liquidity | +272.1 | 16.4 | **+16.62** | `<0.0001` | +28,431.2 | **61.03%** | **YES (Primary)** |
| **Advance-Tax Installment Weight (15 Jun/Sep/Dec/Mar)** | Calendar Effect (Tax) | -2,57,310.7 | 23,808.5 | **-10.81** | `<0.0001` | -31,596.4 | **49.43%** | **YES (Primary)** |
| **New Fiscal Year Deployment Surge (April)** | Calendar Effect (FY-Start) | +59,808.6 | 4,894.1 | **+12.22** | `<0.0001` | +16,389.0 | **44.45%** | **YES (Primary)** |
| **Fiscal Year-End Balance-Sheet Dressing (March)** | Calendar Effect (FY-End) | -66,318.9 | 13,654.2 | **-4.86** | `0.00001` | -18,173.1 | **22.88%** | **YES (Primary)** |
| **91D T-Bill vs Overnight Call Spread (bps)** | Money-Market Yield | +1,618.6 | 841.9 | +1.92 | `0.0581` | +8,015.3 | 3.96% | Marginal / Secondary |
| **3M Commercial Paper vs Overnight Call Spread (bps)** | Money-Market Yield | -270.6 | 468.3 | -0.58 | `0.5651` | -2,521.4 | 0.36% | Noise (Collinear w/ Tax) |
| **NSE Nifty 50 Monthly Equity Return (%)** | Equity-Market Move | +161.9 | 647.8 | +0.25 | `0.8032` | +638.1 | 0.13% | Noise (Insignificant) |

### 5.3 Deliverable 03: Out-of-Sample Model Benchmarking (24-Month Rolling Backtest)

| Model Architecture | MAPE (%) | WAPE (%) | MAE (₹ Cr) | RMSE (₹ Cr) | Mean Pinball Loss | q90 Pinball Loss | 80% PI Cov | 95% PI Cov | Avg 95% PI Width |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **★ Q-GBDT + Conformal (ML Champion)** | **20.35%** | **12.60%** | **11,229.7** | **15,823.8** | **3,011.6** | **2,581.3** | **87.5%** | **95.8%** | ₹69,203.2 Cr |
| **Neural MLP Ensemble (Deep Learning)** | 21.58% | 14.03% | 12,507.6 | 16,792.3 | 3,240.7 | 2,560.1 | 75.0% | 91.7% | ₹57,801.3 Cr |
| **SARIMAX + Exog (Classical)** | 21.45% | 13.77% | 12,274.3 | 17,122.4 | 3,512.1 | 3,059.0 | 62.5% | 79.2% | ₹42,248.0 Cr |
| **ETS / Holt-Winters (Classical)** | 27.46% | 17.06% | 15,203.2 | 17,925.4 | 3,543.2 | 2,581.2 | 91.7% | 100.0% | ₹87,226.0 Cr |
| **Seasonal Naive (Classical Baseline)** | 28.73% | 18.81% | 16,769.6 | 20,312.4 | 4,127.9 | 3,363.2 | 91.7% | 100.0% | ₹1,21,322.4 Cr |

### 5.4 Deliverable 04: Cash-Buffer Decision Cost Backtest (24-Month Rolling Window)

| Cash-Buffer Policy Rule | Avg Monthly Buffer (₹ Cr) | Shortfall Months (of 24) | 24M Idle Drag Cost (₹ Cr) | 24M 3× Shortfall Penalty (₹ Cr) | Total 24M Decision Cost (₹ Cr) | Cost Reduction vs Naive |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **01. Naive Baseline (Last Year Same-Month Outflow)** | 48,647.2 | 4 | 92.84 | 110.55 | 203.39 | `0.00% (Baseline)` |
| **02. Static Historical 75th Pct Outflow Rule** | 51,134.6 | 7 | 194.22 | 1,110.83 | 1,305.05 | `-541.63%` |
| **03. Forecast Median Rule (50th Pct Outflow)** | 43,677.3 | 8 | 36.68 | 118.81 | 155.49 | **+23.55% Saving** |
| **04. Conservative Tail Rule (95th Pct Outflow)** | 58,159.8 | 1 | 222.51 | 19.25 | 241.76 | `-18.87%` |
| **05. Newsvendor Optimal Rule (75th Pct, c_u/c_o = 3:1)** | 49,341.9 | 3 | 105.02 | 40.16 | 145.18 | **+28.62% Saving** |
| **★ 06. CIO Recommended Policy (Regime-Adaptive 75p/78p)** | **48,390.6** | **3** | **96.44** | **43.34** | **139.78** | **+31.28% Saving** |
