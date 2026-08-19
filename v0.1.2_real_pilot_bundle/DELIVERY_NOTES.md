# Delivery notes

- Bundle version: `0.1.2-real-pilot-1`
- Repository binding: `9bf1e2f9a25494e944a815dd9b0270ff929ffeaf`
- Included real target definitions: 3
- Included pre-cutoff source definitions: 9
- Formal split membership: forbidden for all three targets
- Local validation: Python compilation, JSON parse, forbidden-stratum scan,
  repository tests, and a network-disabled evaluator reference self-test for
  all three cases in an existing container
- Hydration was attempted on 2026-08-19. arXiv retry/resume handling worked,
  but the official OpenReview API required an interactive challenge. No valid
  capsule or receipt was produced, and the empty partial run tree was removed.
- Not performed in this delivery environment: digest-pinned Docker image build,
  independent human receipts, real Qwen generation, or GPU execution

Post-delivery hardening binds every producer receipt to the generated artifact,
binds proposals to their implementation SHA-256, freezes one container digest
per stage, confines proposal artifacts to `run/proposals/`, and rejects altered
artifacts, implementations, or stage-image registries. The evaluator custodian
receipt also anchors the trusted-producer key digest used by the stage receipts.

The bundle now has an explicit engineering-only OpenReview challenge fallback:
it uses a 180-day lag, records `cutoff_provisional`, and requires an explicit
preflight flag. It is not accepted as evidence for power, formal freeze, or
paper rediscovery.

The evaluator is an engineering gate for the real no-training vertical slice;
it is not a statistical-power study and does not replace the formal trusted
evaluator/rubric receipt required by `freeze-check` before QLoRA.

Scientific limitation: the included structured-instance API and reported
oracle-call field are suitable only for control-chain validation. They do not
establish black-box oracle complexity or a paper-rediscovery result.
