# V0.1.1 Protocol Hardening Status

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

## Current exit state

`configs/paper_rediscovery_v0_1.json` remains
`blocked_pending_implementation_and_artifacts`. The remaining blockers are
real pilot capsules and evaluator receipts, power simulation, model/tokenizer
hashes, container digests, decoding values, segmented loss-mask validation,
human rubric, and the sealed freeze bindings.

## Next allowed work

1. Build 3-5 real, independent bilevel/Stackelberg pilot capsules and their
   trusted evaluator in the network-disabled environment.
2. Run the vertical-slice command and inspect all positive and negative
   receipts. Do not use pilot cases as training or formal evaluation data.
3. Estimate paper-level variance and power from that pilot.
4. Freeze the causal one-generation pilot config (`paper_rediscovery_causal_pilot_v0_2.json`).
5. Only after its verifier-filtering comparison succeeds, consider the separate
   recursive config (`paper_rediscovery_recursive_v1.json`).

Real QLoRA, three-generation recursion, and formal paper-rediscovery claims are
not authorized by this milestone.
