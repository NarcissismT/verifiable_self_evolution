# Paper Time-Capsule Protocol v1

## Objective

Measure whether a small open model can reconstruct a scientifically useful
hypothesis and discriminating experiment from information that was available
before a target paper appeared. The model is not asked to reproduce the target
paper's wording. It must produce an executable, falsifiable research proposal
whose claims survive independent verification.

## Unit of evaluation

One task is one target paper and one cutoff timestamp `t0`. The public task is
a time capsule containing only artifacts available at or before `t0`. The
target paper must first become public strictly after `t0`.

The public capsule contains:

- a target-neutral problem statement;
- immutable pre-`t0` paper, dataset, web, and software snapshots;
- artifact release/version timestamps and SHA-256 digests;
- a network-disabled execution environment lock;
- allowed tools and resource limits;
- a cryptographic commitment to the sealed target.

The sealed target contains:

- target identity and immutable paper snapshot;
- method and claim annotations;
- result tables or machine-readable expected ranges;
- an independently implemented evaluator specification;
- a random salt used by the public commitment.

The target title, identifiers, text, repository, later citations, and result
values must not occur in the public capsule.

## Eligibility gates

A paper is eligible only when all gates below pass.

1. The target appeared after both the capsule cutoff and the documented model
   knowledge cutoff.
2. A contamination audit does not elicit target-specific method names, result
   values, or distinctive phrases from the base checkpoint without capsule
   evidence.
3. The central claim can be tested within the frozen compute budget using
   pre-cutoff data or a faithful simulator.
4. At least one independent hard metric exists. Examples include analytic
   error, KKT residual, NI gap, theorem/property tests, held-out predictive
   error, calibrated confidence coverage, or reproducible task reward.
5. The target code and packages implementing the target method are absent from
   the environment. Every dependency version was public by `t0`.
6. The public problem remains meaningful without revealing the target paper's
   answer.

An unknown model cutoff is not accepted as a documented cutoff. Such a model
requires a stronger empirical contamination study and cannot support a clean
historical-rediscovery claim by default.

## Student output contract

Before execution, the student must emit one immutable JSON proposal containing:

- a falsifiable structural hypothesis;
- assumptions and at least one plausible alternative explanation;
- a predicted failure mode;
- the observation that discriminates the main hypothesis from alternatives;
- executable code;
- baseline definitions;
- generation, data, and execution seeds;
- primary and secondary metrics with direction;
- expected effect and variance used for power planning;
- a stopping rule independent of observed held-out results;
- a resource schedule for the resource-to-solution curve.

The proposal is hashed before any candidate experiment starts. All seeds in a
comparison use the same task instance and matched random streams where valid.

## Candidate generation

For each training task, the pool contains exactly one current champion, at
least one frozen legacy model, at least two declared variants, and optionally a
closed teacher under the following restrictions.

- Generation zero: the teacher may propose bootstrap candidates on train only.
- Later rounds: the teacher may repair a train state only after a frozen number
  of machine-verifiable failures establishes that it is hard.
- The teacher never receives a target paper, hidden result, held-out/OOD task,
  or promotion receipt.
- Teacher proposals carry source provenance and pass the same executor and
  verifier. They are never treated as labels merely because of their source.

Dev is teacher-free and training-free. It is used only before freezing a formal
round. Held-out and OOD are evaluator-only.

## Execution and verification

Candidate code runs in a network-disabled container with read-only capsule and
dataset mounts. The trusted evaluator, not candidate code, computes scientific
metrics from output artifacts.

The gate order is fixed:

1. **Leakage and provenance:** all artifact, task, model, code, and environment
   commitments match.
2. **Execution:** parse success, no timeout, finite outputs, resource accounting,
   and all mandatory unit/property tests pass.
3. **Scientific validity:** analytic/oracle residuals and task-specific hard
   metrics satisfy preregistered thresholds on every required seed.
4. **Discrimination:** the proposed experiment separates its stated hypothesis
   from frozen baselines with the preregistered effect direction, uncertainty,
   and power.
5. **Efficiency:** report solution quality on a common resource grid. A more
   expensive method cannot hide cost by using fewer outer iterations.

Only after these gates may the evaluator compare the result with the sealed
target. Target alignment has three separate diagnostics: claim overlap,
experiment overlap, and outcome agreement. A scientifically valid alternative
is retained even when it differs from the historical paper.

Verifier feedback released to training contains failure categories, not hidden
target values or target text.

Candidate JSON is parsed against an exact schema. Candidate/model/source IDs,
round number, parent lineage, task identity, and allowed seed registry are
assigned by the orchestrator rather than trusted from model output.

## Data routing

```text
train accepted  -> success library -> QLoRA SFT data
train rejected  -> counterexample library -> repair or preference construction
dev             -> diagnostics only
promotion       -> promotion receipts only
held-out/OOD    -> final evaluation receipts only
```

No held-out/OOD task, output, failure, score, or teacher repair is added to a
future training round. The number of allowed promotion attempts is frozen to
limit adaptive overfitting.

## Promotion and final evaluation

Champion and candidate are evaluated in the same run on the same frozen
promotion papers and seeds. Promotion contains eight ID and four OOD papers;
it is the only split that can trigger a generation upgrade. Papers, not seeds,
are the outer statistical unit. Promotion requires the frozen promotion gate
to pass:

- all mandatory hard-validity gates;
- no more than the preregistered fraction of per-paper regressions;
- positive paired mean quality delta with its confidence lower bound above the
  frozen margin;
- resource ratio below the frozen ceiling.

The gate accepts no train/dev metrics, teacher votes, or post-hoc target-paper
selection. Promotion OOD must encode a real frozen shift such as field, problem
scale, data regime, or experimental modality, not merely a new random seed.
After evolution is complete, the champion and final candidate are evaluated
once on the disjoint 24-paper ID held-out set and 16-paper OOD set. Final ID
uses a positive 95% paper-level bootstrap lower bound; final OOD uses the
pre-registered non-inferiority margin.

## First pilot scope

The first pilot should validate the causal question, not maximize dataset size.
A defensible starting point is:

- one base checkpoint and one 7B/14B size comparison;
- 20-40 train papers for bootstrap and QLoRA;
- a teacher-free dev set;
- held-out and OOD sizes chosen by a pre-run power simulation using papers as
  the resampling unit;
- one QLoRA round before considering recurrence;
- three arms: base model, teacher-bootstrap QLoRA, and verifier-filtered QLoRA;
- identical inference budget and candidate count for all arms.

This isolates whether hard verifier filtering adds value beyond ordinary
teacher distillation. Champion/legacy/variant self-play should be added only
after this first causal contrast is measured.
