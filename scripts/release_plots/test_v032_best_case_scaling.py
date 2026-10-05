"""Dimensional/provenance controls only; fabricated inputs are not forecast evidence."""
import copy
import unittest

import v032_best_case_scaling as model


def anchors():
    geometry = {"cells_3d": 2_000_000, "cell_updates_per_forecast_hour": 400_000_000}
    return {"schema": model.SCHEMA, "status": "measured_anchors", "target_version": "0.3.2",
            "freeze_revision": "a"*40, "sources": [{"artifact": "UNIT TEST ONLY", "sha256": "0"*64}],
            "sweep": {"hours": 24, "n": 4, "steady_s_per_case_fc_h": 4,
                      "fixed_s_per_case": 24, "warm_vram_mib_per_case": 5000,
                      "warm_host_high_water_kib_per_case": 6_000_000, "geometry": geometry},
            "prod": {"steady_s_per_fc_h": 2, "geometry": geometry},
            "cpu": {"ensemble_12core_s_per_case_fc_h": 120, "prod_12core_s_per_fc_h": 90}}


class ScalingControls(unittest.TestCase):
    def test_cell_updates_weight_each_domains_timestep(self):
        proof = {"metadata": {"domains": {
            "d01": {"grid": {"mass_shape": [44,70,120]}, "namelist": {"dt_s":54}},
            "d02": {"grid": {"mass_shape": [44,117,267]}, "namelist": {"dt_s":18}},
            "d03": {"grid": {"mass_shape": [44,93,111]}, "namelist": {"dt_s":6}}}}}
        self.assertEqual(model.geometry(proof)["cell_updates_per_forecast_hour"], 572_070_400)

    def test_fixed_seconds_amortize_without_scaling_host_cost(self):
        short=model.scenarios(anchors(),hours=24)["devices"]["RTX 5090"]
        long=model.scenarios(anchors(),hours=72)["devices"]["RTX 5090"]
        self.assertAlmostEqual(short["amortized_s_per_case_fc_h"],5)
        self.assertAlmostEqual(long["amortized_s_per_case_fc_h"],4+1/3)
        self.assertEqual(short["steady_throughput_case_fc_h_per_wall_h"],long["steady_throughput_case_fc_h_per_wall_h"])
        self.assertEqual(long["scenario_label"],"I")

    def test_giant_clock_scales_steps_and_compares_equal_grid(self):
        six=model.scenarios(anchors(),giant_dt=6)["devices"]["B200"]["giant"]
        twelve=model.scenarios(anchors(),giant_dt=12)["devices"]["B200"]["giant"]
        self.assertEqual(six["cells_3d"],44*six["side"]**2)
        self.assertEqual(six["cells_3d"],twelve["cells_3d"])
        self.assertAlmostEqual(six["gpu_s_per_fc_h"],2*twelve["gpu_s_per_fc_h"])
        self.assertAlmostEqual(six["cpu_12core_s_per_fc_h"],2*twelve["cpu_12core_s_per_fc_h"])

    def test_memory_reserve_and_growth_reduce_claimed_capacity(self):
        roomy=model.scenarios(anchors(),growth=1)["devices"]["B200"]["giant"]
        guarded=model.scenarios(anchors(),growth=1.25)["devices"]["B200"]["giant"]
        self.assertGreater(roomy["side"],guarded["side"])
        self.assertLess(roomy["cells_3d"]*5000*2**20/2_000_000,192e9*.9)

    def test_unmeasured_wrong_release_and_short_windows_refused(self):
        for key,value in (("status","pending"),("target_version","0.3.1")):
            bad=anchors();bad[key]=value
            with self.assertRaises(ValueError):model.scenarios(bad)
        bad=anchors();bad["sweep"]["hours"]=6
        with self.assertRaises(ValueError):model.scenarios(bad)

    def test_nonfinite_and_missing_hashes_refused(self):
        bad=anchors();bad["sweep"]["steady_s_per_case_fc_h"]=float("nan")
        with self.assertRaises(ValueError):model.scenarios(bad)
        bad=anchors();bad["sources"]=[]
        with self.assertRaises(ValueError):model.scenarios(bad)


if __name__=="__main__":unittest.main()
