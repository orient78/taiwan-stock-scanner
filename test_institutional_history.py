import unittest
from institutional_history import foreign_big_buy

class ForeignBigBuyTests(unittest.TestCase):
    def test_scanner_requires_current_date_and_uses_shared_threshold(self):
        import ast
        from pathlib import Path
        node = next(n for n in ast.parse(Path('scanner.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'passes_foreign_big_buy_gate')
        scope = {'foreign_big_buy': foreign_big_buy, 'FOREIGN_BIG_BUY_PCT': 10}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'scanner.py', 'exec'), scope)
        gate = scope[node.name]
        stock = dict(institutional_available=True, date='2026-10-08', institutional_date='2026-10-08', foreign_net_shares=100, volume=1000, trust_net_shares=-50, foreign_buy_streak=1)
        self.assertTrue(gate(stock))
        self.assertFalse(gate(dict(stock, institutional_date='2026-10-07')))
        self.assertFalse(gate(dict(stock, foreign_net_shares=99)))

    def test_threshold_and_invalid_inputs(self):
        self.assertTrue(foreign_big_buy(100, 1000, 10))
        self.assertFalse(foreign_big_buy(99, 1000, 10))
        for net, volume in [(None, 1000), (-100, 1000), (100, 0), (100, float('nan'))]:
            self.assertFalse(foreign_big_buy(net, volume))

    def test_single_session_does_not_require_trust_or_previous_buy(self):
        import pandas as pd
        from institutional_history import attach_gate
        dates = pd.DatetimeIndex(['2026-01-02', '2026-01-05'])
        df = pd.DataFrame({'Volume': [1000, 1000]}, index=dates)
        r = attach_gate(df, '2330', dates, {'2026-01-02': {'2330': [-100, -50]}, '2026-01-05': {'2330': [100, -50]}})
        self.assertFalse(r.INSTITUTIONAL_2D.any())
        self.assertTrue(foreign_big_buy(r.FOREIGN_NET_SHARES.iloc[1], r.Volume.iloc[1]))
from unittest.mock import patch

import pandas as pd
from institutional_history import (FOREIGN, TRUST, attach_gate, parse_t86, load_reports,
                                   parse_market_month, filter_market_sessions, load_market_calendar)
import test_backtest_optimizer as fixtures
import backtest_optimizer as engine


class InstitutionalTests(unittest.TestCase):
    def test_official_calendar_excludes_unscheduled_closure_before_ma(self):
        payload = {"stat": "OK", "title": "115年07月市場成交資訊",
                   "fields": ["日期"], "data": [["115/07/09"], ["115/07/13"]]}
        calendar = parse_market_month(payload, "2026-07-01")
        df = pd.DataFrame({"Close": [100., 999., 110.]},
                          index=pd.to_datetime(["2026-07-09", "2026-07-10", "2026-07-13"]).tz_localize("Asia/Taipei"))
        clean = filter_market_sessions(df, calendar)
        self.assertEqual(clean.Close.tolist(), [100., 110.])
        self.assertEqual(clean.Close.rolling(2).mean().iloc[-1], 105.)
        reports = {"2026-07-09": {"2330": [1, 1]}, "2026-07-13": {"2330": [1, 1]}}
        self.assertEqual(attach_gate(clean, "2330", calendar, reports).INSTITUTIONAL_2D.tolist(), [False, True])

    def test_market_calendar_rejects_wrong_month_empty_and_duplicate_dates(self):
        payload = {"stat": "OK", "title": "115年07月市場成交資訊",
                   "fields": ["日期"], "data": [["115/07/09"]]}
        with self.assertRaises(ValueError):
            parse_market_month(payload, "2026-08-01")
        payload["data"] = []
        with self.assertRaises(ValueError):
            parse_market_month(payload, "2026-07-01")
        payload["data"] = [["115/07/09"], ["115/07/09"]]
        with self.assertRaises(ValueError):
            parse_market_month(payload, "2026-07-01")
        payload["data"] = [["115/08/01"]]
        with self.assertRaises(ValueError):
            parse_market_month(payload, "2026-07-01")

    def test_current_month_calendar_refreshes_and_request_bounds_apply(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock
        payload = {"stat": "OK", "title": "115年07月市場成交資訊",
                   "fields": ["日期"], "data": [["115/07/09"], ["115/07/13"], ["115/07/14"]]}
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(payload).encode()
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "2026-07.json").write_text(json.dumps({**payload, "data": [["115/07/09"]]}))
            with patch("institutional_history.urllib.request.urlopen", return_value=response) as request:
                dates = load_market_calendar("2026-07-10", "2026-07-13", tmp)
            request.assert_called_once()
        self.assertEqual(dates.tolist(), [pd.Timestamp("2026-07-13")])

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

