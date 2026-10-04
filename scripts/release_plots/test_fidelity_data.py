"""Independent small scorer receipts verify documentation coverage decisions."""
import unittest

import numpy as np

from fidelity_data import gate_summary, integrity_summary, approved_integrity_summary
from identity_curves import stack


class FidelityCoverage(unittest.TestCase):
    def setUp(self):
        self.report = {"summaries": {"coverage_issues": [], "comparable_field_count": 1, "tolerance_failure_count": 0},
                       "inventory": {"common": ["T2", "Times"], "cpu_only_count": 0, "incompatible_common": []},
                       "pairing": {"common_leads_h": [0, 1, 2]}, "tolerances": {"supplied": True, "field_count": 1},
                       "field_summaries": {
                           "T2": {"by_lead": [{"lead_h": h, "n": 4, "finite_gpu": 4} for h in range(3)],
                                  "tolerance_result": {"supplied": True}},
                           "Times": {"all_equal": True, "checks": [{"lead_h": h} for h in range(3)]}}}

    def test_compared_numeric_and_bounded_counts_differ(self):
        self.assertEqual(gate_summary(self.report, 2), {"fields": 2, "numeric_fields": 1, "bounded_fields": 1,
                                                       "frames": 3, "finite_gpu": True, "failures": 0})

    def test_missing_field_refused(self):
        del self.report["field_summaries"]["T2"]
        with self.assertRaisesRegex(ValueError, "every common field"):
            gate_summary(self.report, 2)

    def test_failed_hour_remains_visible_when_aggregate_passes(self):
        self.report["field_summaries"]["T2"]["by_lead"][2]["tolerance_result"] = {"pass": False}
        self.assertEqual(self.report["summaries"]["tolerance_failure_count"], 0)
        self.assertEqual(gate_summary(self.report, 2)["failures"], 1)

    def test_missing_field_lead_refused(self):
        self.report["field_summaries"]["T2"]["by_lead"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete lead coverage"):
            gate_summary(self.report, 2)

    def test_no_tolerance_for_manifest_field_refused(self):
        self.report["field_summaries"]["T2"]["tolerance_result"]["supplied"] = False
        with self.assertRaisesRegex(ValueError, "frozen tolerance manifest"):
            gate_summary(self.report, 2)

    def test_nonfinite_values_remain_visible(self):
        self.report["field_summaries"]["T2"]["by_lead"][2]["finite_gpu"] = 3
        self.assertFalse(gate_summary(self.report, 2)["finite_gpu"])

    def test_missing_metadata_lead_refused(self):
        self.report["field_summaries"]["Times"]["checks"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete metadata"):
            gate_summary(self.report, 2)

    def test_identity_does_not_hide_missing_leads_in_intersection(self):
        entries = [("a", np.array([0., 1., 2.]), np.array([.1, .2, .3]), np.array([0., 0., 0.])),
                   ("b", np.array([0., 2.]), np.array([.1, .3]), np.array([0., 0.]))]
        with self.assertRaisesRegex(ValueError, "complete hourly lead set"):
            stack(entries)


class OutputIntegrity(unittest.TestCase):
    def setUp(self):
        self.report = {"rule": "missing CPU variables + degenerate fields + WRF globals", "pass": True,
                       "domains": {d: {"frames_paired": 25, "frames_checked": ["first", "middle", "last"],
                                       "missing_variables_vs_cpu": {}, "degenerate_fields": {},
                                       "global_attrs": {"missing_required": []}, "pass": True}
                                   for d in ("d01", "d02", "d03")}}

    def test_complete_report_passes(self):
        self.assertTrue(integrity_summary(self.report)["pass"])

    def test_top_pass_cannot_hide_missing_fields(self):
        self.report["domains"]["d03"]["missing_variables_vs_cpu"] = {"file": ["Q2"]}
        self.assertFalse(integrity_summary(self.report)["pass"])

    def test_top_pass_cannot_hide_degenerate_fields_or_missing_attrs(self):
        self.report["domains"]["d02"]["degenerate_fields"] = {"ALBEDO": [{}]}
        self.report["domains"]["d01"]["global_attrs"]["missing_required"] = ["DT"]
        result = integrity_summary(self.report)
        self.assertFalse(result["pass"])
        self.assertEqual((result["degenerate_fields"], result["missing_required_attrs"]), (1, 1))

    def test_incomplete_domain_refused(self):
        del self.report["domains"]["d02"]
        with self.assertRaisesRegex(ValueError, "every WN3 domain"):
            integrity_summary(self.report)

    def full_header(self):
        self.report["rule"] += " (full CPU header, every paired frame)"
        for row in self.report["domains"].values():
            row.update(pairing_ok=True, cpu_frames=25)
            row["global_attrs"].update(frames_checked=25, missing_vs_cpu=[], by_frame={})

    def test_full_header_report_passes(self):
        self.full_header()
        result = integrity_summary(self.report, require_full_header=True)
        self.assertTrue(result["pass"])
        self.assertTrue(result["full_cpu_header"])

    def test_top_pass_cannot_hide_cpu_header_omission(self):
        self.full_header()
        self.report["domains"]["d02"]["global_attrs"]["missing_vs_cpu"] = ["TITLE"]
        result = integrity_summary(self.report)
        self.assertFalse(result["pass"])
        self.assertEqual(result["missing_cpu_attrs"], 1)

    def test_top_pass_cannot_hide_later_frame_omission(self):
        self.full_header()
        self.report["domains"]["d03"]["global_attrs"]["by_frame"] = {
            "wrfout_d03_2026-02-28_18:00:00": {"missing_vs_cpu": ["TITLE"], "missing_required": []}}
        result = integrity_summary(self.report)
        self.assertFalse(result["pass"])
        self.assertEqual(result["global_frames_failing"], 1)

    def test_incomplete_global_frame_coverage_refused(self):
        self.full_header()
        self.report["domains"]["d01"]["global_attrs"]["frames_checked"] = 24
        with self.assertRaisesRegex(ValueError, "full-header frame coverage"):
            integrity_summary(self.report)

    def test_unequal_pairing_refused_even_with_top_pass(self):
        self.full_header()
        self.report["domains"]["d02"]["cpu_frames"] = 24
        with self.assertRaisesRegex(ValueError, "full-header frame coverage"):
            integrity_summary(self.report)

    def test_final_requires_full_cpu_header_rule(self):
        with self.assertRaisesRegex(ValueError, "full CPU header"):
            integrity_summary(self.report, require_full_header=True)

    def test_swiss_domain_must_be_requested_explicitly(self):
        self.full_header()
        self.report["domains"] = {"d01": self.report["domains"]["d01"]}
        with self.assertRaisesRegex(ValueError, "every WN3 domain"):
            integrity_summary(self.report, require_full_header=True)
        self.assertTrue(integrity_summary(self.report, require_full_header=True, domains=("d01",))["pass"])

    def test_swiss_request_rejects_wrong_domain(self):
        self.full_header()
        self.report["domains"] = {"d02": self.report["domains"]["d02"]}
        with self.assertRaisesRegex(ValueError, "every requested domain"):
            integrity_summary(self.report, require_full_header=True, domains=("d01",))


class IntegrityApprovals(unittest.TestCase):
    def setUp(self):
        fixture = OutputIntegrity()
        fixture.setUp()
        fixture.full_header()
        self.report = fixture.report
        self.report.update({"gpu_dir": "/runs/20260227_18z_a1/gpu_24h", "pass": False})
        events = {"QICE": [{"file": "wrfout_d02_2026-02-28_12:00:00", "gpu_value": 0,
                             "cpu_range": [0, 1e-6]}]}
        self.report["domains"]["d02"].update({"pass": False, "degenerate_fields": events})
        self.exception = {"case": "20260227_18z_a1", "fields": {"integrity_24h": {"d02": events}}}
        self.approval = "2026-10-04T10:00:00Z"
        self.note = f"20260227_18z — RE-DELIVERY on FINAL-b; manager approval {self.approval}"

    def summary(self):
        return approved_integrity_summary(self.report, self.exception, self.approval, self.note)

    def test_exact_approved_exception_retains_raw_fail(self):
        result = self.summary()
        self.assertFalse(result["pass"])
        self.assertTrue(result["accepted_with_exception"])
        self.assertEqual(result["degenerate_fields"], 1)

    def test_structural_failure_is_never_approved(self):
        self.report["domains"]["d03"]["global_attrs"]["missing_vs_cpu"] = ["DT"]
        with self.assertRaisesRegex(ValueError, "structural"):
            self.summary()

    def test_extra_event_not_in_approval_rejected(self):
        self.report["domains"]["d03"]["degenerate_fields"] = {"CANICE": []}
        with self.assertRaisesRegex(ValueError, "every raw strict event"):
            self.summary()

    def test_exception_for_other_case_rejected(self):
        self.exception["case"] = "20260502_18z_a1"
        with self.assertRaisesRegex(ValueError, "different GPU outputs"):
            self.summary()

    def test_predecessor_delivery_cannot_approve_finalb(self):
        self.note = self.note.replace("RE-DELIVERY on FINAL-b", "FINAL 1fece48da")
        with self.assertRaisesRegex(ValueError, "FINAL-b delivery note"):
            self.summary()

    def test_explicit_finalb_release_gate_approval_retains_raw_fail(self):
        self.note = (f"20260227_18z; manager approval {self.approval}; "
                     "RELEASE GATE (WN3 part) ACCEPTED on FINAL-b")
        result = self.summary()
        self.assertFalse(result["pass"])
        self.assertTrue(result["accepted_with_exception"])

    def test_missing_approval_reference_rejected(self):
        self.approval = ""
        with self.assertRaisesRegex(ValueError, "explicit manager approval"):
            self.summary()


if __name__ == "__main__":
    unittest.main()
