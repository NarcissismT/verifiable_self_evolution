# V0.1.2-pre-pilot-fix Status

## Completed

- Formal CLI surface: `init-formal`, `seal-capsules`, `bind-freeze`,
  `freeze-check`, capsule/contamination audits, promotion/final decisions,
  vertical-slice execution, and ledger verification.
- Five-way paper selection with 100 train, 9 dev, 12 promotion, and planned
  final sets; all assignments, cutoffs, reserves, and exclusion reasons are
  content-addressed.
- Hash-chain run ledger and contiguous promotion attempt state machine.
- Paper-level bootstrap after inner rollout/adapter aggregation; exact seed
  grid and VDS schema/range checks.
- Deterministic identifier/algorithm/phrase/code/numeric leak scans plus a
  required semantic leak review receipt.
- Training export revalidation and separate teacher-anchor, verified-success,
  and corrected-counterexample buckets.
- Frozen-config QLoRA entrypoint with exact token packing and provenance
  receipt. It refuses to run while `freeze-check` is blocked.
- No-training vertical-slice gate requiring 3-5 independent cases excluded
  from every formal split, with bound generation/execution/evaluation receipts.
- Trusted producer receipts now bind the measured runtime, network policy,
  container digest, evaluator stage, and semantic review to each slice case.
- The primary causal study is explicitly one generation and one promotion
  attempt; the three-generation setting remains only in the recursive config.
- Candidate selection enforces per-stratum reserve minima and supports a
  same-stratum replacement transition. OOD strata have machine-readable shift
  definitions.
- Aggregate contamination auditing covers every selected paper x model x probe
  x seed grid and reports reserve routing for failed papers.

## Current exit state

`configs/paper_rediscovery_v0_1.json` remains
`blocked_pending_implementation_and_artifacts`. The remaining blockers are
real pilot capsules and evaluator receipts, an independent power pilot,
model/tokenizer hashes, container digests, decoding values, segmented
loss-mask validation, human rubric, and sealed freeze bindings.

The current evidence boundary is:

```yaml
control-plane hardening: passed
real vertical-slice infrastructure: partially ready
real paper pilot: not yet run
power confirmation: not estimable from 3-5 cases
QLoRA: blocked
recursive evolution: blocked
```

## Next allowed work

1. Build 3-5 real, independent bilevel/Stackelberg engineering pilot capsules
   and their trusted evaluator in the network-disabled environment.
2. Run the vertical-slice command and inspect all positive and negative
   receipts. Do not use these cases as training or formal evaluation data, and
   do not estimate paper-level power from them.
3. Run the causal power pilot with 30 train, 9 dev, and 12 pilot-eval papers.
   This pilot is currently scoped to bilevel/Stackelberg; it cannot justify a
   broader multi-stratum final design without additional stratum coverage.
4. Use the 12 pilot-eval papers plus conservative sensitivity analysis to
   freeze the formal held-out/OOD counts.
5. Only after the one-generation verifier-filtering comparison succeeds,
   consider the separate recursive config (`paper_rediscovery_recursive_v1.json`).

Real QLoRA, three-generation recursion, and formal paper-rediscovery claims are
not authorized by this milestone.
