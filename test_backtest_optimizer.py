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

