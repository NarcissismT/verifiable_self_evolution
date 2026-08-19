# Frozen V0.1 Paper Rediscovery Study

This is the executable freeze derived from the supplied protocol response. It
is separate from the quadratic toy smoke configuration. The machine-readable
source of truth is [`configs/paper_rediscovery_v0_1.json`](configs/paper_rediscovery_v0_1.json).

## Split boundary

The five sets are disjoint:

| Set | Size | Use |
|---|---:|---|
| Train | 100 | teacher bootstrap, SFT, self-evolution |
| Dev | 9 | teacher-free diagnostics before formal freeze, 3 per ID stratum |
| Promotion | 12 | 8 ID + 4 OOD, at most three generation decisions |
| Final ID held-out | planned 24 | pending independent-pilot power confirmation |
| Final OOD | planned 16 | pending independent-pilot power confirmation |

Promotion never reads final held-out or final OOD. Final evaluation runs once
after all generations and training are complete.

## Models

Qwen2.5-7B-Instruct at revision `d1e200f` is the primary model. Qwen2.5-14B-
Instruct at revision `336cd7f` is a scale replication only. Both model
checkpoints, tokenizers, and contamination receipts are hashed before use.

## Paper time capsule

For target paper `p`, let `T_p` be the earliest timestamp among arXiv v1,
OpenReview first public time, public code first commit, and public blog/slides/
report. The public capsule cutoff is `T_p - 30 days`. Target identifiers,
algorithm names, code, results, and target text remain sealed. Ground truth is
the target v1 plus official code first public within 30 days; later versions are
diagnostic only.

## Primary endpoint and gates

The sole primary endpoint is VDS:

```text
0.35 empirical
+ 0.20 hypothesis
+ 0.20 experiment
+ 0.15 novelty
+ 0.10 calibration
```

Promotion requires VDS gain >= 0.05, 90% paper-level paired bootstrap lower
bound > 0, hard/executable pass rate >= 95%, zero fabricated results and target
leaks, no VDS-component decrease below 0.02, cost ratio <= 1.10, and at least
two of three adapter seeds passing with a passing median.

Final ID requires mean VDS gain >= 0.05 and a 95% lower bound above zero. Final
OOD uses a non-inferiority lower bound of -0.03. The outer statistical unit is
the paper; rollout seeds are averaged within paper and never reported as
independent papers.

## Training freeze

QLoRA uses NF4, double quantization, BF16 compute, rank 64, alpha 128, dropout
0.05, all-linear targets, paged AdamW 8-bit, betas `(0.9, 0.95)`, cosine
scheduler, 3% warmup, gradient clipping 1.0, 8192 sequence length, and
gradient checkpointing. Learning rate is `1e-4` for 7B and `8e-5` for 14B.

The causal pilot registers three arms: `Base`, `Teacher-SFT`, and
`Verifier-Grounded Archive`. `Linear Self-Teacher` is reserved for the separate
recursive configuration [`paper_rediscovery_recursive_v1.json`](configs/paper_rediscovery_recursive_v1.json).
Teacher access is train-only; a teacher cannot see
promotion, final held-out, final OOD, target papers, or promotion receipts.

The full values, container names, resource budget, contamination probes,
replay mixture, and decoding freeze are in the JSON configuration.

## Readiness boundary

The supplied response does not provide concrete decoding values,
container image digests, trusted-evaluator commit/digest, or full model and
tokenizer file hashes. Candidate-pool membership and the independent human
review rubric are also not yet sealed. The config therefore remains
`blocked_pending_implementation_and_artifacts`; it is a valid partial
preregistration, not launch authorization. The formal CLI, freeze-check, ledger,
and a real 3-5 capsule engineering vertical slice remain mandatory before
QLoRA. These 3-5 cases validate the execution method and cannot confirm power;
power requires the separate 12-paper pilot-eval plus conservative sensitivity
analysis.
