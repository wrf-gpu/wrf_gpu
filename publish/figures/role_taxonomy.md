# Figure Spec: Role Taxonomy

Purpose: show how the multi-agent workflow separates responsibility for planning, implementation, validation, review, and debugging.

Canvas: two-column flow diagram. Left column is role; right column is proof output.

ASCII layout:

```text
Principal / human author
        |
        v
Manager ---------> sprint contract, file ownership, closeout decision
   |     |   +---------> Reviewer: blind review, acceptance or required fixes
   |
   +-------------> Worker: implementation, proof objects, worker-report
   |
   +-------------> Tester: adversarial tests, validation reports
   |
   +-------------> Debugger/RCA: bisection, localization, blocked verdicts
```

Role labels:
- Manager: scopes contracts, freezes interfaces, prevents overclaiming, integrates evidence.
- Worker: changes only owned files, produces proof objects, reports command output.
- Tester: validates physics/system invariants and tries to break assumptions.
- Reviewer: checks claims against files and gates acceptance.
- Debugger/RCA: localizes regressions after a red gate without broad rewrites.
- Principal: owns final scientific and publication acceptance.

Rendering notes: use solid arrows for dispatch, dashed arrows for review feedback, and attach proof-object icons to every role except the principal.
