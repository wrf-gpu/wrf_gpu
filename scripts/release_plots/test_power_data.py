"""Sensor clocks determine energy; missed seconds must not look like savings."""
import os
from pathlib import Path
import tempfile
import unittest

from bench_plots import read_dmon


class PowerSamples(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory(dir=os.environ.get("RELEASE_PLOTS_TEST_TMP")) as tmp:
            p = Path(tmp) / "dmon.log"
            p.write_text(text)
            return read_dmon(p)

    def test_irregular_clock_intervals(self):
        out = self.parse("#Time gpu pwr\n00:00:00 0 100\n00:00:02 0 200\n00:00:05 0 100\n")
        # Independent trapezoids: 2*150 + 3*150 = 750 J.
        self.assertEqual((out["energy_j"], out["span_s"], out["mean_w"], out["max_gap_s"]), (750, 5, 150, 3))

    def test_midnight(self):
        out = self.parse("#Time gpu pwr\n23:59:58 0 100\n00:00:01 0 100\n")
        self.assertEqual((out["energy_j"], out["span_s"]), (300, 3))

    def test_missing_power_does_not_remove_elapsed_time(self):
        out = self.parse("#Time gpu pwr\n00:00:00 0 100\n00:00:01 0 -\n00:00:03 0 100\n")
        self.assertEqual((out["energy_j"], out["span_s"], out["valid_power_samples"]), (300, 3, 2))

    def test_historical_nominal_interval_is_identified(self):
        out = self.parse("#gpu pwr\n0 100\n0 200\n")
        self.assertEqual(out["energy_j"], 300)
        self.assertIn("nominal 1 s", out["integration_method"])

    def test_no_power_is_not_zero_energy(self):
        out = self.parse("#Time gpu fb\n00:00:00 0 5000\n00:00:01 0 5001\n")
        self.assertFalse(out["has_power"])
        self.assertNotIn("energy_j", out)


if __name__ == "__main__":
    unittest.main()
