import unittest
import pandas as pd
from gap_exit import gap_exit_series


def frame(rows):
    return pd.DataFrame(rows, columns=["High", "Low", "Close"])


class GapExitTests(unittest.TestCase):
    def test_gap_fill_and_swing_break(self):
        rows = [
            (105, 100, 103), (104, 99, 102), (103, 96, 100),
            (105, 100, 104), (107, 102, 106), (109, 103, 108),
            (111, 108, 110),  # gap: bottom 109; pivot 96 confirmed
            (112, 109, 111),  # filled, still above swing
            (110, 94, 95),    # close below 96
        ]
        result = gap_exit_series(frame(rows))
        self.assertFalse(result[6]["gap_filled"])
        self.assertTrue(result[7]["gap_filled"])
        self.assertFalse(result[7]["exit_signal"])
        self.assertTrue(result[8]["exit_signal"])

    def test_no_premature_pivot_and_prefix_invariance(self):
        rows = [(10, 8, 9), (11, 7, 10), (12, 6, 11),
                (13, 9, 12), (14, 10, 13), (16, 15, 16)]
        for i in range(1, len(rows)+1):
            self.assertEqual(gap_exit_series(frame(rows))[:i],
                             gap_exit_series(frame(rows[:i])))

    def test_latest_gap_replaces_prior(self):
        rows = [(10, 8, 9), (9, 7, 8), (10, 6, 9),
                (11, 8, 10), (12, 9, 11), (14, 13, 14),
                (16, 15, 16)]
        states = gap_exit_series(frame(rows))
        self.assertEqual(states[-1]["gap_formed_index"], 6)
        self.assertEqual(states[-1]["gap_bottom"], 14)


if __name__ == "__main__":
    unittest.main()
