"""Timing integrity checks: independently specified clocks, two receipt schemas.

Run with JAX_PLATFORMS=cpu python -m unittest discover -s scripts/release_plots
-p test_run_data.py. This never imports JAX or uses the GPU.
"""
import datetime as dt
import json
import os
from pathlib import Path
import tempfile
import unittest

from run_data import arm_rates, benchmark_endpoint, case_outputs, prod_summary, t_to_lead, production_batch_summary


class TimingReceipts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("RELEASE_PLOTS_TEST_TMP"))
        self.addCleanup(self.tmp.cleanup)
        self.arm = Path(self.tmp.name)
        self.run = {"schema": "gpuwrf-parallel-cases-v1", "start_utc": "1970-01-01T00:16:40+00:00",
                    "wall_s": 800, "cli_args": ["--hours", "72"], "peak_concurrency": 2,
                    "cases": {name: {"rc": 0, "vram_peak_mib": 6000} for name in ("a", "b")}}
        self.write(self.arm / "parallel_run.json", self.run)
        for i, name in enumerate(("a", "b")):
            case = self.arm / name
            case.mkdir()
            init = dt.datetime(2026, 2, 27, 18)
            outputs = []
            for h in range(73):
                for d in range(1, 4):
                    valid = init + dt.timedelta(hours=h, seconds=18 if d == 1 else 0)
                    outputs.append({"d": f"d{d:02}", "file": f"wrfout_d{d:02}_{valid:%Y-%m-%d_%H:%M:%S}",
                                    "t_published_s": 20 + h * 10 + d + i * 4})
            self.write(case / "receipt.json", {"schema": "gpuwrf-parallel-case-v1", "rc": 0,
                       "start_utc": dt.datetime.fromtimestamp(1000 + i * 10, dt.timezone.utc).isoformat(),
                       "cli_summary": {"effective_max_dom": 3, "effective_hours": 72}, "outputs": outputs})

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def change_case(self, name, mutate):
        path = self.arm / name / "receipt.json"
        r = json.loads(path.read_text())
        mutate(r)
        self.write(path, r)

    def production_six(self):
        for name in ("c", "d", "e", "f"):
            (self.arm / name).mkdir()
            self.write(self.arm / name / "receipt.json", json.loads((self.arm / "a/receipt.json").read_text()))
            self.run["cases"][name] = {"rc": 0}
        self.run.update(peak_concurrency=3, case_hours=432, compress_rc=0)
        self.write(self.arm / "parallel_run.json", self.run)

    def test_fifo_allowed_only_for_explicit_production_row(self):
        self.production_six()
        result = production_batch_summary(self.arm)
        self.assertEqual(result["s_per_case_h"], 800 / 432)
        with self.assertRaisesRegex(ValueError, "queued"):
            benchmark_endpoint(self.arm)

    def test_production_row_rejects_missing_frame(self):
        self.production_six()
        self.change_case("f", lambda r: r["outputs"].pop())
        with self.assertRaisesRegex(ValueError, "every frame"):
            production_batch_summary(self.arm)

    def test_production_row_rejects_failed_compression(self):
        self.production_six()
        self.run["compress_rc"] = 1
        self.write(self.arm / "parallel_run.json", self.run)
        with self.assertRaisesRegex(ValueError, "successful"):
            production_batch_summary(self.arm)

    def test_production_row_rejects_wrong_case_hours(self):
        self.production_six()
        self.run["case_hours"] = 431
        self.write(self.arm / "parallel_run.json", self.run)
        with self.assertRaisesRegex(ValueError, "denominator"):
            production_batch_summary(self.arm)

    def test_publication_offsets_and_domain_completion(self):
        # Arm starts at Unix 1000; b starts at 1010 and finishes d03 at
        # +24 h after 20 + 240 + 3 + 4 s. Compression mtimes play no role.
        wall, n, _ = t_to_lead(self.arm, 24)
        self.assertEqual((wall, n), (277, 2))
        rates = arm_rates(self.arm)
        self.assertEqual(rates["throughput_whole_run_s_per_case_h"], 800 / 144)
        self.assertEqual(rates["throughput_stepping_s_per_case_h"], 5)

    def test_legacy_and_launcher_same_clocks(self):
        for i, name in enumerate(("a", "b")):
            rows = [{"d": f"d{d:02}", "own_step": h * 100, "t_start": 20 + h * 10 + d + i * 4 - 0.5,
                     "s": 0.5} for h in range(73) for d in range(1, 4)]
            self.write(self.arm / name / "receipt.json", {"rc": 0, "t0_wall_utc": 1000 + i * 10, "outputs": rows})
        self.assertEqual(t_to_lead(self.arm, 24)[:2], (277, 2))

    def test_missing_domain_refused(self):
        self.change_case("a", lambda r: r.update(outputs=[o for o in r["outputs"] if o["d"] != "d02"]))
        with self.assertRaisesRegex(ValueError, "domains"):
            t_to_lead(self.arm, 24)

    def test_missing_late_frame_refused(self):
        self.change_case("b", lambda r: r["outputs"].pop())
        with self.assertRaisesRegex(ValueError, "missing domain output"):
            arm_rates(self.arm)

    def test_failed_case_refused(self):
        self.change_case("b", lambda r: r.update(rc=1))
        with self.assertRaisesRegex(ValueError, "failed run"):
            t_to_lead(self.arm, 24)

    def test_queued_cases_are_not_parallel_cases(self):
        self.run["peak_concurrency"] = 1
        self.write(self.arm / "parallel_run.json", self.run)
        with self.assertRaisesRegex(ValueError, "queued"):
            arm_rates(self.arm)
        with self.assertRaisesRegex(ValueError, "queued"):
            benchmark_endpoint(self.arm)

    def test_four_hour_sweep_has_four_hour_denominator(self):
        self.run["cli_args"] = ["--hours", "4"]
        self.run["wall_s"] = 100
        self.write(self.arm / "parallel_run.json", self.run)
        for name in ("a", "b"):
            self.change_case(name, lambda r: r.update(cli_summary={"effective_max_dom": 3, "effective_hours": 4},
                outputs=[o for o in r["outputs"] if o["t_published_s"] < 70]))
        result = benchmark_endpoint(self.arm)
        self.assertEqual((result["hours"], result["n_cases"], result["wall_s"]), (4, 2, 100))
        self.assertEqual(result["last_publication_wall_s"], 77)
        self.assertEqual(result["s_per_case_h"], 100 / 8)

    def test_two_hour_smoke_cannot_supply_release_throughput(self):
        self.run["cli_args"] = ["--hours", "2"]
        self.write(self.arm / "parallel_run.json", self.run)
        with self.assertRaisesRegex(ValueError, "four complete"):
            benchmark_endpoint(self.arm)

    def test_six_hour_sweep_has_six_hour_denominator(self):
        self.run["cli_args"] = ["--hours", "6"]
        self.run["wall_s"] = 120
        self.write(self.arm / "parallel_run.json", self.run)
        for name in ("a", "b"):
            self.change_case(name, lambda r: r.update(cli_summary={"effective_max_dom": 3, "effective_hours": 6},
                outputs=[o for o in r["outputs"] if o["t_published_s"] < 90]))
        result = benchmark_endpoint(self.arm)
        self.assertEqual((result["hours"], result["n_cases"], result["wall_s"]), (6, 2, 120))
        self.assertEqual(result["last_publication_wall_s"], 97)
        self.assertEqual(result["s_per_case_h"], 120 / 12)

    def test_duplicate_output_refused(self):
        self.change_case("a", lambda r: r["outputs"].append(r["outputs"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            case_outputs(self.arm / "a")

    def test_prod_endpoint_is_file_publication_not_callback_or_teardown(self):
        r = self.arm / "prod_receipt.json"
        out = self.arm / "wrfout"
        out.mkdir()
        f = out / "wrfout_d01_2026-02-28_00:00:00"
        f.touch()
        os.utime(f, (1157, 1157))  # async write finishes after callback at 1155
        self.write(r, {"t0_wall_utc": 1000, "cli_argv": ["run", "--output-dir", str(out), "--hours", "24"],
                       "outputs": [{"t_start": 153, "s": 2}]})
        bench = self.arm / "bench.json"
        self.write(bench, {"arms": {"extra24h": {"rc": 0, "t_end_s": 160, "receipt": str(r)}}})
        result = prod_summary({"bench_json": str(bench), "whole_run_arm": "extra24h", "whole_run_hours": 24})
        self.assertEqual(result["wall_s"], 157)
        self.assertEqual(result["s_per_fch"], 157 / 24)
        # The recorded receipt reproduces timing after file removal.
        f.unlink()
        replay = prod_summary({"bench_json": str(bench), "whole_run_arm": "extra24h", "whole_run_hours": 24,
                               "endpoint_receipt": result["endpoint"]})
        self.assertEqual(replay["wall_s"], 157)

    def test_wrong_forecast_denominator_refused(self):
        r = self.arm / "prod_receipt.json"
        self.write(r, {"cli_argv": ["run", "--hours", "6"]})
        bench = self.arm / "bench.json"
        self.write(bench, {"arms": {"extra6h": {"rc": 0, "t_end_s": 60, "receipt": str(r)}}})
        with self.assertRaisesRegex(ValueError, "forecast hours"):
            prod_summary({"bench_json": str(bench), "whole_run_arm": "extra6h", "whole_run_hours": 24})


if __name__ == "__main__":
    unittest.main()
