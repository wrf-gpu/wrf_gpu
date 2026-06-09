# Mythos Heavy-Problem Lane

- For v0.14, send extremely hard problems to Mythos in tmux `0:1` as whole
  endpoint-defined assignments instead of slicing them into narrow micro-prompts.
- The manager remains responsible for sprint contracts, file and GPU locks,
  proof gates, diff review, merge/reject decisions, and the final v0.14 close.
- Before each new Mythos sprint after a completion or context-risk point, send
  `/compact` to tmux `0:1`, wait about two minutes for the TUI to return to a
  prompt, then send the full assignment and Enter. Use delayed repeated Enter
  presses if the TUI stages text without submitting.
- Current Mythos target: memory/FP32 lane. Endpoint is all known memory issues
  fixed where technically safe, any newly discovered material memory issue fixed
  or exactly proven/deferred, and every memory/performance claim backed by proof.
