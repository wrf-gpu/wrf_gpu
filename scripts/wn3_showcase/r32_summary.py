"""Summarise ONE pinned-R32-01 result like alisios' r3231_summarize (same rules): FREEZE_OK cells decide (WITHIN=PASS,
OUTSIDE=FAIL), FREEZE_OK_WITH_NOTES = notes, else report-only; full bins tau 0-23 vs provisional. Usage: r32_summary.py RESULT.json"""
import collections, json, sys
s = json.load(open(sys.argv[1]))
role = {"FREEZE_OK": "DECIDES", "FREEZE_OK_WITH_NOTES": "NOTES"}
verdict = lambda st: "PASS" if "WITHIN" in st else "FAIL" if "OUTSIDE" in st else "NO_VERDICT"  # noqa: E731
out = {"issue": s["issue"], "scope": s["scope_status"], "g2a_passed": f"{sum(x['status'] == 'passed' for x in s['g2a'])}/{len(s['g2a'])}"}
for name, want in (("freeze_ok_tau0_23", "DECIDES"), ("with_notes_tau0_23", "NOTES")):
    cells = [c for c in s["field_cells"] if role.get(c.get("freeze_recommendation", "HELD")) == want and not c["status"].startswith("PROVISIONAL_")]
    fails = [c for c in cells if verdict(c["status"]) == "FAIL"]
    out[name] = {"n": len(cells), "pass": sum(verdict(c["status"]) == "PASS" for c in cells), "fail": len(fails),
                 "fail_by_var_domain": dict(sorted(collections.Counter(f"{c['var']}/{c['domain']}" for c in fails).items())),
                 "worst": sorted(((round(c["median_abs"] / c["limit"], 2), f"{c['var']}/{c['domain']}/{c['pixel_class']}/{c['tau_lo']}-{c['tau_hi']}")
                                  for c in fails if c.get("limit")), reverse=True)[:5]}
print(json.dumps(out, indent=1))
