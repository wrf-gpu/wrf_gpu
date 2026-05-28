# Pull request

## Summary

<!-- One or two sentences describing what changes and why. -->

## Related issue

<!-- Link the issue this PR addresses, e.g. "closes #42". If none, briefly justify. -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor (no behaviour change)
- [ ] Documentation
- [ ] Tests only
- [ ] Performance improvement (profiler artifact attached)

## Validation evidence

<!-- For any change to src/gpuwrf/, describe how you validated it. -->

- [ ] `pytest -q tests/` passes locally.
- [ ] Relevant invariant tests pass (B6 savepoint, restart bitwise, repeatability, D2H = 0). Tick if not applicable.
- [ ] If the change makes a performance claim, profiler output is attached or linked.
- [ ] If the change touches dycore, physics, or runtime, the affected test files under `tests/test_m6*.py` or `tests/test_m7*.py` are listed below and shown to pass.

```text
<paste test output here>
```

## License

By submitting this PR I confirm that:

- [ ] I have the right to license my contribution under AGPL-3.0-or-later.
- [ ] My contribution does not include code copied from incompatibly-licensed sources without proper attribution.

## Checklist

- [ ] Code follows the style of surrounding files.
- [ ] New public functions have docstrings.
- [ ] Significant changes are reflected in CHANGELOG.md.
- [ ] No personal paths, secrets, or local-only references in the diff.
