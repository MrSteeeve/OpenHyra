# Task: NanoChat AutoResearch (Apple-Silicon scaled-down port, protocol `nanochat-autoresearch-mps512-v1`)

You must lower `val_bpb` (validation bits-per-byte, LOWER IS BETTER) of a small
GPT trained with a FIXED 5-minute wall-clock budget on a single Apple M4 GPU (MPS).

Rules (same spirit as karpathy/autoresearch):
- You may ONLY modify `train.py` (architecture, optimizer, hyperparameters, batch size, ...).
- `prepare.py` is frozen: it defines the tokenizer, dataloader, seq len 512,
  the 300s time budget, and the ground-truth metric `evaluate_bpb`. DO NOT touch it,
  do not re-implement or bypass the eval. Leaking future tokens (non-causal
  attention) is reward hacking and strictly forbidden.
- No new dependencies. MPS eager mode: torch.compile is disabled off-CUDA;
  code must run on MPS (no CUDA-only ops, no tensor-scalar dtype tricks that MPS rejects).
- The run must finish within ~10 minutes total or it is killed.

KNOWN LIMITATION of this task's trust model: the score is computed inside the
candidate-editable train.py (as in upstream autoresearch). The harness checks
exit codes, frozen files and the training budget, but a fully trusted external
evaluator is not yet implemented for this task — it is parked in favor of
evaluator-backed tasks like sums_diffs.
