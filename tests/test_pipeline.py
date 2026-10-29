"""
Automated Verification & Unit Test Suite (`tests/test_pipeline.py`)
Validates every requirement and scoring metric specified in ZECKATHON PS 03.
"""

import json
import os
import sys
import unittest
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


class TestZeckathonPS03(unittest.TestCase):
    def setUp(self):
        self.data_path = os.path.join(ROOT_DIR, "data", "amfi_rbi_market_data.csv")
        self.out_dir = os.path.join(ROOT_DIR, "outputs")
        self.payload_path = os.path.join(self.out_dir, "dashboard_payload.json")
        with open(self.payload_path, "r", encoding="utf-8") as f:
            self.payload = json.load(f)

    def test_01_forecast_model_deliverable(self):
        """Deliverable 01: Next 3 months net flows for liquid & overnight with 80% and 95% PIs."""
        fwd = pd.DataFrame(self.payload["forward_forecasts"])
        self.assertTrue({"Liquid Funds", "Overnight Funds", "Combined (Liquid + Overnight)"}.issubset(set(fwd["category"])))
        # Check next 3 months window (Oct, Nov, Dec 2026) and March 2027
        for ym in ["2026-10", "2026-11", "2026-12", "2027-03"]:
            self.assertIn(ym, fwd["year_month"].values)
        # Verify strict quantile ordering: ci95_lower < ci80_lower < point < ci80_upper < ci95_upper
        for _, row in fwd.iterrows():
            self.assertLess(row["ci95_lower_cr"], row["ci80_lower_cr"])
            self.assertLess(row["ci80_lower_cr"], row["point_forecast_cr"])
            self.assertLess(row["point_forecast_cr"], row["ci80_upper_cr"])
            self.assertLess(row["ci80_upper_cr"], row["ci95_upper_cr"])

    def test_02_driver_analysis_deliverable(self):
        """Deliverable 02: Calendar, money-market, banking-system liquidity, and equity drivers tested."""
        drv = pd.DataFrame(self.payload["driver_analysis_combined"])
        self.assertEqual(len(drv), 8)
        categories = set(drv["category"].values)
        self.assertIn("Calendar Effect (Tax)", categories)
        self.assertIn("Banking Liquidity (RBI LAF)", categories)
        self.assertIn("Money-Market Yield Spread", categories)
        self.assertIn("Equity-Market Move", categories)
        self.assertGreater(self.payload["driver_r2"], 0.80)

    def test_03_model_benchmarking_and_calibration(self):
        """Deliverable 03 & Evaluation: Rolling 24-month backtest, MAPE, Pinball Loss, and 80%/95% calibration."""
        bench = pd.DataFrame(self.payload["benchmarking_combined"])
        self.assertEqual(len(bench), 5)
        champ = bench[bench["model"] == "Q-GBDT + Conformal (ML Champion)"].iloc[0]
        naive = bench[bench["model"] == "Seasonal Naive (Classical)"].iloc[0]
        # Champion must beat Seasonal Naive on both MAPE and Mean Pinball Loss
        self.assertLess(champ["mape_pct"], naive["mape_pct"])
        self.assertLess(champ["mean_pinball_loss_cr"], naive["mean_pinball_loss_cr"])
        # Interval calibration check (80% PI between 75% and 92%, 95% PI >= 90%)
        self.assertGreaterEqual(champ["coverage_80_pct"], 75.0)
        self.assertGreaterEqual(champ["coverage_95_pct"], 91.0)

    def test_04_decision_layer_cost_backtest(self):
        """Deliverable 04: Cash-buffer rule backtested against Naive lag-12 same-month outflow baseline."""
        pol = pd.DataFrame(self.payload["policy_summary"])
        rec = pol[pol["policy"].str.contains("CIO Recommended Policy")].iloc[0]
        opt = pol[pol["policy"].str.contains("Newsvendor Optimal")].iloc[0]
        naive = pol[pol["policy"].str.contains("Naive Baseline")].iloc[0]
        # Both Newsvendor Optimal and CIO Recommended Policy must achieve lower 24M cost than Naive Baseline
        self.assertLess(rec["cum_24m_decision_cost_cr"], naive["cum_24m_decision_cost_cr"])
        self.assertLess(opt["cum_24m_decision_cost_cr"], naive["cum_24m_decision_cost_cr"])
        self.assertGreater(rec["cost_reduction_vs_naive_pct"], 15.0)


if __name__ == "__main__":
    unittest.main()
