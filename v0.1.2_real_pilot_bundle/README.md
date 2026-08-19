# v0.1.2 real-paper vertical-slice launch bundle

This is an administrator-side preparation kit for commit
`9bf1e2f9a25494e944a815dd9b0270ff929ffeaf` of
`NarcissismT/verifiable_self_evolution`.

It prepares three real, independent, permanently excluded pilot cases in
`bilevel_stackelberg_alignment_optimization`:

1. `realpilot_flatness_penalty` — lower-level flatness without lower-level
   strong convexity;
2. `realpilot_nonconvex_simple` — joint stationarity in nonconvex simple
   bilevel optimization;
3. `realpilot_linear_coupling` — bilevel optimization with coupled linear
   constraints.

The target papers and target code metadata live in
`admin/targets_admin.json`. That file, `sealed/`, and the hydration logs are
secret curator material. Never mount or copy them into the student/model
runtime. Only `public/<case_id>/` may be shown to a proposer.

## What this bundle does

- downloads exact arXiv v1 snapshots for target and pre-cutoff sources;
- obtains public timestamps from arXiv, OpenReview and GitHub and uses the
  earliest observed timestamp minus 30 days as the capsule cutoff;
- pins and records the oldest commit reachable from the official target-code
  repository's default branch;
- extracts searchable text and hashes every source, environment lock and
  sealed target;
- creates target commitments and pre-review public capsules compatible with
  `vse.paper_capsule`;
- gives an independent reviewer a receipt-producing workflow without
  pre-asserting `independent=true`;
- provides deterministic hidden test instances, a trusted evaluator and
  reference baseline for a real network-disabled engineering slice;
- builds the three bound stage receipts and the repository-native vertical
  slice manifest from real trusted-producer receipts.

The reusable kit does **not** ship a trust key, a fabricated semantic-review
receipt, a container digest or a model checkpoint. The checked-in `run/`
directory now contains a provisional model-generated engineering rehearsal;
those artifacts are not trusted producer receipts and do not satisfy formal
ready preflight. The formal status therefore remains
`blocked_pending_external_attestations`.

## Minimal Qwen engineering pilot

The bundle includes `scripts/run_minimal_qwen_pilot.sh`. It runs the verified
Qwen checkpoint in a network-disabled container, exposes only public capsules
to the student, then executes the trusted evaluator on all three cases and
four frozen seeds. It refuses to overwrite an existing case and writes a
provisional report under `run/`.

The checkpoint must contain `MODEL_MANIFEST.json`. The verified checkpoint is
`Qwen/Qwen2.5-7B-Instruct`, revision
`d1e200fcf95ef0d4326873ddf63e5562d5f1fdbb`, with manifest digest
`6791085ae67e1e7dd6cdff568b903a9698fefd042397c8a0d84c33869afde37e`.
Because Docker cannot always mount a JuiceFS path directly, copy the model to
a local filesystem path such as `/tmp` before running:

```bash
bash v0.1.2_real_pilot_bundle/scripts/run_minimal_qwen_pilot.sh \
  /tmp/vse-qwen7b-model \
  v0.1.2_real_pilot_bundle/run
```

This is an engineering vertical slice only. Qwen supplies the structured
`algorithm_family` and `hypothesis`; `qwen_solution.py` is a fixed,
target-neutral engineering adapter recorded separately. A passing report does
not count as paper rediscovery, causal evidence, power confirmation, QLoRA
training, or recursive evolution. Formal `preflight --phase ready` still
requires independent semantic and evaluator-custodian attestations.

## Required host tools

- Python 3.11+
- the checked-out `verifiable_self_evolution` repository at the commit above
- `pdftotext` (preferred) or Python package `pypdf`
- Docker for real stage execution
- optional `GITHUB_TOKEN` to avoid public API rate limits

## Run order

From the repository root, unpack/copy this directory as
`pilot/real_v0_1_2`, then run:

```bash
PYTHONPATH=. python pilot/real_v0_1_2/scripts/hydrate_and_seal.py \
  --bundle-root pilot/real_v0_1_2 \
  --output-root pilot/real_v0_1_2/run

PYTHONPATH=. python pilot/real_v0_1_2/scripts/preflight.py \
  --bundle-root pilot/real_v0_1_2 \
  --run-root pilot/real_v0_1_2/run --phase pre-review
```

If a rate limit or transient network error interrupts hydration, rerun the
same command with `--resume`. Completed case summaries and the private target
salt are reused; frozen files are never silently overwritten.

If OpenReview requires an interactive challenge, an administrator may export
the unmodified response of the official `/notes?id=<note-id>` API into
`<snapshot-root>/<note-id>.json` and pass `--openreview-snapshot-root`. The
script accepts no hand-entered date: it rechecks the exact title and note ID,
uses `odate`, and records the snapshot SHA-256 in provenance.

For the engineering-only vertical slice, if no browser session is available,
you may explicitly use the conservative fallback below. It excludes OpenReview
from the observed-public-time minimum and uses a 180-day cutoff lag. The output
is marked `cutoff_provisional`; it requires `--allow-provisional-cutoff` in
preflight and cannot be used for power confirmation, formal freeze, or claims
about rediscovering the target paper.

```bash
PYTHONPATH=. python pilot/real_v0_1_2/scripts/hydrate_and_seal.py \
  --bundle-root pilot/real_v0_1_2 \
  --output-root pilot/real_v0_1_2/run \
  --allow-openreview-challenge \
  --fallback-lag-days 180

PYTHONPATH=. python pilot/real_v0_1_2/scripts/preflight.py \
  --bundle-root pilot/real_v0_1_2 \
  --run-root pilot/real_v0_1_2/run \
  --phase pre-review \
  --allow-provisional-cutoff
```

Give a clean copy of each `review_packet/<case_id>/` directory to an
independent reviewer. The reviewer runs the command printed in
`REVIEW_INSTRUCTIONS.md`; the script creates the receipt and updates only the
corresponding public capsule's `semantic_leak_review_digest`.

Then build the evaluator image, record its immutable image ID, and create a
private key:

```bash
docker pull python:3.11.10-slim-bookworm
BASE_IMAGE=$(docker image inspect --format '{{index .RepoDigests 0}}' \
  python:3.11.10-slim-bookworm)
test -n "$BASE_IMAGE"
docker build --pull=false --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  -f pilot/real_v0_1_2/docker/Dockerfile \
  -t vse-real-pilot-evaluator:0.1.2 pilot/real_v0_1_2
docker image inspect --format '{{.Id}}' vse-real-pilot-evaluator:0.1.2 \
  > pilot/real_v0_1_2/run/evaluator_image_digest.txt
umask 077
openssl rand -out pilot/real_v0_1_2/run/trusted_producer.key 32
sha256sum pilot/real_v0_1_2/run/trusted_producer.key | awk '{print $1}' \
  > pilot/real_v0_1_2/run/trusted_producer_key.sha256
```

The independent evaluator custodian must then run the network-disabled
self-test and review the frozen thresholds/source:

```bash
docker run --rm --network none -e VSE_NETWORK_POLICY=none \
  vse-real-pilot-evaluator:0.1.2 self-test \
  --reference-module /opt/vse/student/reference_baseline.py
PYTHONPATH=. python pilot/real_v0_1_2/scripts/attest_evaluator.py \
  --bundle-root pilot/real_v0_1_2 \
  --run-root pilot/real_v0_1_2/run \
  --image-digest-file pilot/real_v0_1_2/run/evaluator_image_digest.txt \
  --trust-anchor-digest-file pilot/real_v0_1_2/run/trusted_producer_key.sha256 \
  --reviewer-id YOUR_STABLE_REVIEWER_ID \
  --evaluator-version real-pilot-evaluator-review-v1 \
  --decision pass --findings-file /path/to/evaluator_findings.txt \
  --attest-independent-of-target --attest-independent-of-capsule \
  --attest-independent-of-student
```

Use the frozen base model to create one proposal per public capsule in a
separate `--network none` generation container. Proposal files must satisfy
`schemas/proposal.schema.json`; they must include the measured model digest.
Do not use `student/reference_baseline.py` as the model proposal. It exists
only to verify the execution/evaluation chain before consuming GPU time.
Follow `GENERATION_AND_RUNBOOK.md`; the repository's current producer API
requires a deterministic two-pass replay to bind a generated proposal digest.

An evaluator custodian who is independent of the target authors, capsule
curator and proposer must inspect the evaluator and image, run its self-test,
and issue `evaluator_review_receipt.json` with
`scripts/attest_evaluator.py`. This receipt is required by the stage builder.

For each real proposal, invoke the repository's `produce-trusted-receipt`
command for generation. Then `scripts/run_trusted_stages.py` produces the
execution/evaluation receipts. Full commands are in
`GENERATION_AND_RUNBOOK.md`. Once producer receipts exist, run:

```bash
PYTHONPATH=. python pilot/real_v0_1_2/scripts/build_stage_receipts.py \
  --bundle-root pilot/real_v0_1_2 \
  --run-root pilot/real_v0_1_2/run \
  --trust-key pilot/real_v0_1_2/run/trusted_producer.key \
  --model-digest-file /sealed/model/model_digest.txt \
  --generation-image-digest-file /sealed/model/generation_image_digest.txt

PYTHONPATH=. python pilot/real_v0_1_2/scripts/preflight.py \
  --bundle-root pilot/real_v0_1_2 \
  --run-root pilot/real_v0_1_2/run --phase ready \
  --trust-key pilot/real_v0_1_2/run/trusted_producer.key

python -m vse.cli run-vertical-slice \
  --manifest pilot/real_v0_1_2/run/public/vertical_slice_manifest.json \
  --public-root pilot/real_v0_1_2/run/public \
  --sealed-root pilot/real_v0_1_2/run/sealed \
  --trust-key pilot/real_v0_1_2/run/trusted_producer.key \
  --output pilot/real_v0_1_2/run/vertical_slice_report.json
```

## Go/no-go rule

The real no-training vertical slice may start only when `preflight --phase
ready` exits 0. Passing this slice validates the real data/control chain. It
does not unlock QLoRA. Formal training still requires the repository's full
`freeze-check` to pass, including model/tokenizer hashes, power receipt,
container policy, candidate manifest, frozen rubric and binding/ledger
anchors.

## Permanent exclusions

`admin/pilot_exclusions.json` contains every target identifier. Copy these
records into the formal candidate-selection exclusion input. Before a formal
freeze, run `scripts/check_formal_exclusion.py` against the selected-paper
manifest. Any matching title, arXiv ID, OpenReview ID, target ID or repository
URL is a hard failure.

```bash
python pilot/real_v0_1_2/scripts/check_formal_exclusion.py \
  --selection-manifest /path/to/formal_selected_papers.json \
  --exclusions pilot/real_v0_1_2/admin/pilot_exclusions.json
```

## Scientific boundary

The current evaluator exposes structured synthetic instance parameters and
accepts a submission-reported oracle-call count. Its hard metrics validate the
capsule, execution, receipt and scoring chain only. They are not evidence of
black-box first-order oracle complexity, algorithmic novelty, or target-paper
rediscovery. A later scientific pilot must replace this API with an
evaluator-enforced oracle service and independently justified task metrics.
