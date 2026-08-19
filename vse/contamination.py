from __future__ import annotations

from dataclasses import asdict, dataclass

from .hashing import content_hash


PROBE_FAMILIES = frozenset({"bibliographic", "method", "result", "algorithm_name"})


@dataclass(frozen=True)
class ProbeObservation:
    target_id: str
    model_id: str
    model_digest: str
    probe_id: str
    probe_family: str
    seed: int
    output_digest: str
    remembered_title_or_algorithm: bool
    target_specific_contributions: int
    max_exact_phrase_words: int
    target_specific_numeric_results: int


@dataclass(frozen=True)
class ContaminationDecision:
    target_id: str
    passed: bool
    excluded_models: tuple[str, ...]
    failures: tuple[str, ...]
    observation_count: int
    audit_digest: str = ""

    def sealed(self) -> "ContaminationDecision":
        value = asdict(self)
        value["audit_digest"] = ""
        return ContaminationDecision(
            **{**asdict(self), "audit_digest": content_hash(value)}
        )


def evaluate_contamination(
    observations: tuple[ProbeObservation, ...],
    *,
    expected_models: frozenset[str],
    probes_per_paper: int = 12,
    seeds_per_probe: int = 3,
    exact_phrase_words_threshold: int = 13,
    numeric_results_threshold: int = 2,
) -> ContaminationDecision:
    if not observations:
        raise ValueError("contamination observations are required")
    if probes_per_paper % len(PROBE_FAMILIES) != 0:
        raise ValueError("probes_per_paper must divide evenly across probe families")
    target_ids = {item.target_id for item in observations}
    if len(target_ids) != 1:
        raise ValueError("one contamination audit must cover exactly one target")
    target_id = next(iter(target_ids))
    failures: list[str] = []
    excluded_models: list[str] = []
    actual_models = {item.model_id for item in observations}
    if actual_models != expected_models:
        failures.append("model_grid_mismatch")

    for model_id in sorted(expected_models):
        model_rows = [item for item in observations if item.model_id == model_id]
        digests = {item.model_digest for item in model_rows}
        if len(digests) != 1:
            failures.append(f"model_digest_mismatch:{model_id}")
        probe_ids = {item.probe_id for item in model_rows}
        if len(probe_ids) != probes_per_paper:
            failures.append(f"probe_count_mismatch:{model_id}")
        families = {item.probe_family for item in model_rows}
        if families != PROBE_FAMILIES:
            failures.append(f"probe_family_mismatch:{model_id}")
        expected_probes_per_family = probes_per_paper // len(PROBE_FAMILIES)
        for family in PROBE_FAMILIES:
            family_probe_ids = {
                item.probe_id for item in model_rows if item.probe_family == family
            }
            if len(family_probe_ids) != expected_probes_per_family:
                failures.append(f"probe_family_count_mismatch:{model_id}:{family}")
        for probe_id in probe_ids:
            rows = [item for item in model_rows if item.probe_id == probe_id]
            if len(rows) != seeds_per_probe or len({item.seed for item in rows}) != seeds_per_probe:
                failures.append(f"probe_seed_grid_mismatch:{model_id}:{probe_id}")

        title_and_contributions = any(
            item.remembered_title_or_algorithm
            and item.target_specific_contributions >= 2
            for item in model_rows
        )
        phrase_memory = any(
            item.max_exact_phrase_words >= exact_phrase_words_threshold
            for item in model_rows
        )
        numeric_memory = any(
            item.target_specific_numeric_results >= numeric_results_threshold
            for item in model_rows
        )
        if title_and_contributions or phrase_memory or numeric_memory:
            excluded_models.append(model_id)
            failures.append(f"target_specific_memory:{model_id}")

    passed = not failures
    return ContaminationDecision(
        target_id=target_id,
        passed=passed,
        excluded_models=tuple(sorted(excluded_models)),
        failures=tuple(sorted(set(failures))),
        observation_count=len(observations),
    ).sealed()
