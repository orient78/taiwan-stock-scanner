import importlib
import os
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

# Execution tests do not download prices or require network packages.
for name in ("yfinance", "twstock"):
    if importlib.util.find_spec(name) is None:
        sys.modules[name] = types.ModuleType(name)
import backtest_optimizer as engine


class ExecutionTests(unittest.TestCase):
    def test_sample_covers_code_range_without_duplicates(self):
        codes = [str(i) for i in range(1100, 9900)] + ["2330"]
        selected = engine.select_symbols(codes, 120)
        self.assertEqual(len(selected), 120)
        self.assertEqual(len(set(selected)), 120)
        self.assertIn("2330", selected)
        self.assertEqual(selected[0], "1100")
        self.assertEqual(selected[-1], "9899")
        self.assertTrue(any(code >= "6000" for code in selected))

    def test_full_universe_and_small_sample_limits(self):
        codes = ["1101", "2330", "9958"]
        self.assertEqual(engine.select_symbols(codes, 0), codes)
        self.assertEqual(engine.select_symbols(codes, 100), codes)
        self.assertEqual(engine.select_symbols(codes, 1), ["2330"])
        self.assertEqual(len(engine.select_symbols(codes, 2)), 2)
        with self.assertRaises(ValueError):
            engine.select_symbols(codes, -1)
        with self.assertRaises(ValueError):
            engine.select_symbols(["1101"], 0)

    def setUp(self):
        self.env = patch.dict(os.environ, {"BUY_COST_BPS": "0", "SELL_COST_BPS": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.p = dict(institutional_2d=False, ma5_confirm=1, ma10_confirm=1, reentry=True,
                      ma_order=False, rsi_min=40, rsi_max=80,
                      vol_ratio=0, breakout_pct=None)

    def frame(self):
        d = pd.DataFrame(index=pd.date_range("2026-01-01", periods=25))
        for c, v in {"Open": 100., "Close": 100., "MA5": 90., "MA10": 90.,
                     "MA20": 90., "RSI": 60., "VOL20": 100.,
                     "HIGH20_PREV": 110., "Volume": 100.}.items():
            d[c] = v
        return d

    def test_next_open_and_final_mark(self):
        d = self.frame()
        d.iloc[1, d.columns.get_loc("Open")] = 200
        d.iloc[-1, d.columns.get_loc("Close")] = 50
        r = engine.backtest_one(d, self.p, prepared=True)
        self.assertAlmostEqual(r["return_pct"], -75)
        self.assertAlmostEqual(r["mdd_pct"], -75)
        self.assertEqual(r["closed_trades"], 0)
        self.assertEqual(r["open_trades"], 1)

    def test_below_ma5_keeps_full_position(self):
        d = self.frame()
        d.iloc[1:, d.columns.get_loc("MA5")] = 150
        d.iloc[-1, d.columns.get_loc("Close")] = 120
        r = engine.backtest_one(d, self.p, prepared=True)
        self.assertAlmostEqual(r["return_pct"], 20)
        self.assertEqual(r["trades"], 1)
        self.assertEqual(r["closed_trades"], 0)
        self.assertEqual(r["open_trades"], 1)

    def test_equal_ma10_holds_then_exits_full_next_open(self):
        d = self.frame()
        d.iloc[1, d.columns.get_loc("MA10")] = 100
        d.iloc[2, d.columns.get_loc("MA10")] = 101
        d.iloc[3, d.columns.get_loc("Open")] = 120
        d.iloc[3:, d.columns.get_loc("RSI")] = 0
        r = engine.backtest_one(d, self.p, prepared=True)
        self.assertAlmostEqual(r["return_pct"], 20)
        self.assertEqual(r["closed_trades"], 1)
        self.assertEqual(r["open_trades"], 0)

    def test_ma10_exit_ignores_missing_entry_indicators(self):
        d = self.frame()
        d.iloc[1, d.columns.get_loc("MA10")] = 110
        d.iloc[1:, d.columns.get_loc("RSI")] = np.nan
        d.iloc[2, d.columns.get_loc("Open")] = 80
        r = engine.backtest_one(d, self.p, prepared=True)
        self.assertAlmostEqual(r["return_pct"], -20)
        self.assertEqual(r["closed_trades"], 1)

    def test_ma10_exit_has_priority(self):
        d = self.frame()
        d.iloc[1, d.columns.get_loc("MA5")] = 110
        d.iloc[1, d.columns.get_loc("MA10")] = 110
        d.iloc[2:, d.columns.get_loc("RSI")] = 0
        r = engine.backtest_one(d, self.p, prepared=True)
        self.assertEqual(r["closed_trades"], 1)
        self.assertEqual(r["open_trades"], 0)

    def test_costs_reduce_return(self):
        d = self.frame()
        with patch.dict(os.environ, {"BUY_COST_BPS": "14.25"}):
            r = engine.backtest_one(d, self.p, prepared=True)
        self.assertLess(r["return_pct"], 0)

    def test_final_valuation_costs_match_buy_hold(self):
        d = self.frame()
        with patch.dict(os.environ, {"BUY_COST_BPS": "14.25", "SELL_COST_BPS": "44.25"}):
            r = engine.backtest_one(d, self.p, prepared=True)
        self.assertAlmostEqual(r["return_pct"], r["buy_hold_pct"])
        self.assertAlmostEqual(r["liquidation_return_pct"], r["buy_hold_liquidation_pct"])
        self.assertLess(r["liquidation_return_pct"], r["return_pct"])
        self.assertEqual(r["closed_trades"], 0)

    def test_revised_same_length_prices_invalidate_cache(self):
        d = self.frame()
        engine.backtest_one(d, self.p, prepared=True)
        d.iloc[1, d.columns.get_loc("Open")] = 200
        r = engine.backtest_one(d, self.p, prepared=True)
        self.assertAlmostEqual(r["return_pct"], -50)

    def test_copied_frame_does_not_reuse_stale_signal_cache(self):
        d = self.frame()
        engine.backtest_one(d, self.p, prepared=True)
        revised = d.copy()
        revised["Volume"] = 0
        r = engine.backtest_one(revised, dict(self.p, vol_ratio=1), prepared=True)
        self.assertEqual(r["trades"], 0)

    def test_final_day_exit_signal_is_not_a_completed_trade(self):
        d = self.frame()
        d.iloc[-1, d.columns.get_loc("MA10")] = 110
        r = engine.backtest_one(d, self.p, prepared=True)
        self.assertEqual(r["closed_trades"], 0)
        self.assertEqual(r["open_trades"], 1)

    def test_training_selection_is_independent_of_later_prices(self):
        d = self.frame().reindex(pd.date_range("2026-01-01", periods=280)).ffill()
        grid = [self.p, dict(self.p, rsi_min=65)]
        before = engine.select_training_params({"2330": d}, grid, "2026-01-01", "2026-05-01")
        revised = d.copy()
        revised.loc["2026-05-01":, "Close"] = 1000
        revised.loc["2026-05-01":, "RSI"] = 75
        after = engine.select_training_params({"2330": revised}, grid, "2026-01-01", "2026-05-01")
        self.assertEqual(before, after)

    def test_exclusive_end_does_not_fill_order_in_next_window(self):
        d = self.frame().reindex(pd.date_range("2026-01-01", periods=50)).ffill()
        d["RSI"] = 0
        d.loc["2026-01-25", "RSI"] = 60
        r = engine.backtest_one(d, self.p, prepared=True, end_date="2026-01-26")
        self.assertEqual(r["trades"], 0)

    def test_summary_counts_all_symbols_and_pooled_closed_trades(self):
        r = engine.backtest_one(self.frame(), self.p, prepared=True)
        details = [dict(r, return_pct=21, mdd_pct=-10, winning_trades=1, closed_trades=2),
                   dict(r, return_pct=21, mdd_pct=-25, winning_trades=2, closed_trades=3),
                   dict(r, return_pct=20, mdd_pct=-5, winning_trades=0, closed_trades=0)]
        s = engine.summarize(details, self.p)
        self.assertAlmostEqual(s["pass_rate_pct"], 200 / 3)
        self.assertAlmostEqual(s["risk_pass_rate_pct"], 100 / 3)
        self.assertAlmostEqual(s["pooled_win_rate_pct"], 60)

    def test_confirmations_cannot_relax_exit_rule(self):
        with self.assertRaises(ValueError):
            engine.backtest_one(self.frame(), dict(self.p, ma10_confirm=3), prepared=True)
        for p in engine.parameter_grid():
            self.assertEqual(p["ma10_confirm"], 1)
            self.assertNotIn("reentry", p)
            self.assertNotIn("ma5_confirm", p)

    def test_rsi_keeps_trading_dates(self):
        d = self.frame()
        d["Close"] = np.arange(100., 125.)
        d["High"] = d["Close"] + 1
        r = engine.indicators(d)
        self.assertEqual(len(r), len(d))
        self.assertEqual(r["RSI"].iloc[-1], 100)

    def test_indicator_prefix_is_causal(self):
        d = self.frame()
        d["High"] = d["Close"] + 1
        prefix = engine.indicators(d.iloc[:20])
        d.iloc[20:, d.columns.get_loc("Close")] = 1000
        pd.testing.assert_frame_equal(prefix, engine.indicators(d).iloc[:20])

    def test_missing_institutional_flag_blocks_entry(self):
        row = self.frame().iloc[0].to_dict()
        row["INSTITUTIONAL_2D"] = np.nan
        self.assertFalse(engine.entry_signal(row, dict(self.p, institutional_2d=True)))

    def test_funnel_rejections_are_disjoint_and_exclude_unfillable_last_signal(self):
        d = self.frame().iloc[:6].copy()
        d["INSTITUTIONAL_2D"] = True
        d.iloc[0, d.columns.get_loc("RSI")] = np.nan
        d.iloc[1, d.columns.get_loc("INSTITUTIONAL_2D")] = False
        d.iloc[2, d.columns.get_loc("Close")] = 80
        d.iloc[3, d.columns.get_loc("RSI")] = 20
        p = dict(self.p, institutional_2d=True)
        counts = engine.entry_diagnostics({"2330": d}, p)["totals"]
        self.assertEqual(counts["sessions"], 6)
        self.assertEqual(counts["invalid_inputs"], 1)
        self.assertEqual(counts["institutional"], 1)
        self.assertEqual(counts["moving_averages"], 1)
        self.assertEqual(counts["rsi"], 1)
        self.assertEqual(counts["passed"], 2)
        self.assertEqual(counts["passed_with_next_session"], 1)
        self.assertEqual(counts["passed_without_next_session"], 1)
        self.assertEqual(counts["sessions"], counts["invalid_inputs"] + counts["passed"] + sum(counts[k] for k in engine.ENTRY_STAGES))

    def test_funnel_window_cannot_count_next_window_execution(self):
        d = self.frame()
        counts = engine.entry_diagnostics({"2330": d}, self.p, end_date="2026-01-06")["totals"]
        self.assertEqual(counts["passed"], 5)
        self.assertEqual(counts["passed_with_next_session"], 4)
        self.assertEqual(counts["passed_without_next_session"], 1)

    def test_observed_slope_uses_no_future_prices(self):
        d = pd.DataFrame({"Close": np.arange(1., 91.), "High": np.arange(2., 92.), "Volume": 100.}, index=pd.date_range("2025-10-01", periods=90))
        before = engine.indicators(d)
        revised = d.copy()
        revised.iloc[71:, revised.columns.get_loc("Close")] = 9999
        after = engine.indicators(revised)
        pd.testing.assert_frame_equal(before.iloc[:71], after.iloc[:71])
        for n in (10, 20):
            self.assertAlmostEqual(before[f"MA{n}_CHANGE"].iloc[70], before[f"MA{n}"].iloc[70] - before[f"MA{n}"].iloc[69])

    def test_slope_and_distance_gate_and_cache_revision(self):
        d = self.frame()
        d["MA10_CHANGE"] = 1.
        d["MA20_CHANGE"] = 1.
        p = dict(self.p, rising_ma10_ma20=True, max_ma10_distance_pct=12)
        self.assertTrue(engine.entry_signal(d.iloc[0], p))
        self.assertEqual(engine.entry_rejection(d.iloc[0], dict(p, max_ma10_distance_pct=8)), "ma10_distance")
        self.assertEqual(engine.backtest_one(d, p, prepared=True)["trades"], 1)
        d["MA20_CHANGE"] = -1.
        self.assertEqual(engine.entry_rejection(d.iloc[0], p), "ma_slope")
        self.assertEqual(engine.backtest_one(d, p, prepared=True)["trades"], 0)
        d["MA20_CHANGE"] = np.nan
        self.assertEqual(engine.entry_rejection(d.iloc[0], p), "ma_slope")

    def test_comparison_preserves_baseline_and_required_gates(self):
        d = self.frame()
        d["MA10_CHANGE"] = d["MA20_CHANGE"] = 1.
        params = dict(self.p, institutional_2d=True)
        original = params.copy()
        report = engine.teaching_comparison({"2330": d}, params, "2026-05-01", "2026-08-01")
        self.assertEqual(params, original)
        for variant in report["variants"].values():
            self.assertTrue(variant["params"]["institutional_2d"])
            self.assertEqual(variant["windows"]["full_period"]["total_entries"], 0)

    def test_foreign_single_day_entry_and_revised_cache(self):
        d = self.frame()
        d['FOREIGN_NET_SHARES'] = 10.
        d['INSTITUTIONAL_2D'] = False
        p = dict(self.p, institutional_mode='foreign_big_buy', foreign_volume_pct=10)
        self.assertTrue(engine.entry_signal(d.iloc[0], p))
        self.assertEqual(engine.backtest_one(d, p, prepared=True)['trades'], 1)
        d['FOREIGN_NET_SHARES'] = 9.
        self.assertEqual(engine.entry_rejection(d.iloc[0], p), 'institutional')
        self.assertEqual(engine.backtest_one(d, p, prepared=True)['trades'], 0)


if __name__ == "__main__":
    unittest.main()


