# Fable/Mythos Heavy-Problem Lane

- For v0.14, Fable/Mythos in tmux `0:1` is a scarce high-end debug resource.
  Conserve its tokens.
- Do not use Fable/Mythos for routine polling, proof grooming, ordinary
  validation triage, or issues likely solvable by one focused GPT 5.5 sprint.
- For validation failures, first send GPT 5.5 workers to collect, localize, and
  attempt direct fixes when feasible. Escalate only the unresolved hard core to
  Fable/Mythos.
- Send extremely hard problems to Fable/Mythos as whole endpoint-defined
  assignments instead of slicing them into narrow micro-prompts.
- The manager remains responsible for sprint contracts, file and GPU locks,
  proof gates, diff review, merge/reject decisions, and the final v0.14 close.
- Before each new Fable/Mythos sprint after a completion or context-risk point,
  send
  `/compact` to tmux `0:1`, wait about two minutes for the TUI to return to a
  prompt, then send the full assignment and Enter. Use delayed repeated Enter
  presses if the TUI stages text without submitting.
- 2026-06-09 memory/FP32 lane result: closed and merged after manager review.
  MYNN BouLac column tiling is the material non-radiation memory fix; FP32 R0 is
  default-inert; FP32 R1+ waits for the fp64 one-step dynamics frontier.
