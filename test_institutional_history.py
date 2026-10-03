import unittest
from unittest.mock import patch

import pandas as pd
from institutional_history import FOREIGN, TRUST, attach_gate, parse_t86, load_reports
import test_backtest_optimizer as fixtures
import backtest_optimizer as engine


class InstitutionalTests(unittest.TestCase):
    def setUp(self):
        self.sessions = pd.to_datetime(["2025-12-31", "2026-01-02", "2026-01-05", "2026-01-06"])
        self.df = pd.DataFrame(index=self.sessions)
        self.reports = {str(d.date()): {"2330": [10, 20]} for d in self.sessions}

    def test_weekend_is_consecutive_and_missing_day_breaks_streak(self):
        gate = attach_gate(self.df, "2330", self.sessions, self.reports)
        self.assertEqual(gate.INSTITUTIONAL_2D.tolist(), [False, True, True, True])
        del self.reports["2026-01-02"]
        gate = attach_gate(self.df, "2330", self.sessions, self.reports)
        self.assertEqual(gate.INSTITUTIONAL_2D.tolist(), [False, False, False, True])

    def test_both_investors_must_buy_on_both_days(self):
        self.reports["2026-01-02"]["2330"] = [10, 0]
        self.reports["2026-01-06"]["2330"] = [-1, 20]
        self.assertFalse(attach_gate(self.df, "2330", self.sessions, self.reports).INSTITUTIONAL_2D.any())

    def test_suspended_stock_cannot_bridge_market_session(self):
        del self.reports["2026-01-02"]["2330"]
        df = self.df.drop(pd.Timestamp("2026-01-02"))
        gate = attach_gate(df, "2330", self.sessions, self.reports)
        self.assertFalse(gate.loc["2026-01-05", "INSTITUTIONAL_2D"])

    def test_future_reports_cannot_change_past_gate(self):
        before = attach_gate(self.df.iloc[:3], "2330", self.sessions, self.reports)
        self.reports["2026-01-06"]["2330"] = [-100, -200]
        after = attach_gate(self.df.iloc[:3], "2330", self.sessions, self.reports)
        pd.testing.assert_frame_equal(before, after)

    def test_parser_schema_dates_and_comma_values(self):
        payload = {"stat": "OK", "title": "115年01月02日 三大法人買賣超日報", "fields": [TRUST, "證券代號", FOREIGN], "data": [["2,000", "2330", "-1,000"]]}
        self.assertEqual(parse_t86(payload, pd.Timestamp("2026-01-02")), {"2330": [-1000, 2000]})
        with self.assertRaises(ValueError):
            parse_t86(payload, pd.Timestamp("2026-01-05"))
        payload["fields"] = []
        with self.assertRaises(ValueError):
            parse_t86(payload, pd.Timestamp("2026-01-02"))

    def test_history_uses_previous_session_and_records_failure(self):
        with patch("institutional_history.fetch_daily", side_effect=[{"2330": [1, 1]}, ValueError("missing"), {"2330": [1, 1]}, {"2330": [1, 1]}]):
            dates, reports, failures = load_reports(self.sessions, "2026-01-01")
        self.assertEqual(len(dates), 4)
        self.assertIn("2025-12-31", reports)
        self.assertIn("2026-01-02", failures)

    def test_gate_executes_next_open_and_does_not_block_exit(self):
        fixture = fixtures.ExecutionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        df = fixture.frame()
        p = {**fixture.p, "institutional_2d": True}
        self.assertEqual(engine.backtest_one(df, p, prepared=True)["trades"], 0)
        df["INSTITUTIONAL_2D"] = False
        df.iloc[0, df.columns.get_loc("INSTITUTIONAL_2D")] = True
        df.iloc[1, df.columns.get_loc("Open")] = 200
        df.iloc[2, df.columns.get_loc("Close")] = 80
        df.iloc[3, df.columns.get_loc("Open")] = 150
        r = engine.backtest_one(df, p, prepared=True)
        self.assertEqual(r["trades"], 1)
        self.assertEqual(r["closed_trades"], 1)
        self.assertAlmostEqual(r["return_pct"], -25)
        # Revising only the gate must invalidate the engine record cache.
        df["INSTITUTIONAL_2D"] = False
        self.assertEqual(engine.backtest_one(df, p, prepared=True)["trades"], 0)


if __name__ == "__main__":
    unittest.main()
