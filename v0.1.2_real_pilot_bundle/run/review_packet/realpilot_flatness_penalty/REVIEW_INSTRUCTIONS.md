# Independent semantic review: realpilot_flatness_penalty

Review the public capsule and every frozen search-text artifact against the sealed target. Check title/identifier, algorithm-name, paraphrase, code, numerical-result, citation-graph and task-wording leakage. Do not approve if you authored the target, curated this capsule, or implemented the trusted evaluator. Record concrete findings in a UTF-8 text file, one finding per line. From the repository root, run:

```bash
python pilot/real_v0_1_2/scripts/review_semantic.py \
  --run-root pilot/real_v0_1_2/run --case-id realpilot_flatness_penalty \
  --reviewer-id YOUR_STABLE_REVIEWER_ID --evaluator-version YOUR_REVIEW_PROTOCOL_VERSION \
  --decision pass --findings-file /path/to/findings.txt \
  --attest-independent-of-target --attest-independent-of-capsule --attest-independent-of-evaluator
```
