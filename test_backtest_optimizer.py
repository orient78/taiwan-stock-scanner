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
    def setUp(self):
        self.env = patch.dict(os.environ, {"BUY_COST_BPS": "0", "SELL_COST_BPS": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.p = dict(ma5_confirm=1, ma10_confirm=1, reentry=True,
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


if __name__ == "__main__":
    unittest.main()
