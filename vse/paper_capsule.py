from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from .contracts import Split
from .hashing import canonical_json, content_hash, file_hash


class ArtifactKind(str, Enum):
    PAPER = "paper"
    DATASET = "dataset"
    SOFTWARE = "software"
    WEB = "web"


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must include a timezone: {value!r}")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class TemporalArtifact:
    artifact_id: str
    kind: ArtifactKind
    title: str
    version: str
    available_at_utc: str
    snapshot_path: str
    snapshot_sha256: str
    search_text_path: str
    search_text_sha256: str
    provenance_url: str

    def validate(self, root: Path, cutoff: datetime) -> list[str]:
        failures: list[str] = []
        available = parse_utc(self.available_at_utc)
        if available > cutoff:
            failures.append(
                f"artifact_after_cutoff:{self.artifact_id}:{self.available_at_utc}"
            )
        path = (root / self.snapshot_path).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            failures.append(f"artifact_outside_capsule:{self.artifact_id}")
            return failures
        if not path.is_file():
            failures.append(f"artifact_missing:{self.artifact_id}")
        elif file_hash(path) != self.snapshot_sha256:
            failures.append(f"artifact_hash_mismatch:{self.artifact_id}")
        text_path = (root / self.search_text_path).resolve()
        try:
            text_path.relative_to(root.resolve())
        except ValueError:
            failures.append(f"artifact_text_outside_capsule:{self.artifact_id}")
            return failures
        if not text_path.is_file():
            failures.append(f"artifact_text_missing:{self.artifact_id}")
        elif file_hash(text_path) != self.search_text_sha256:
            failures.append(f"artifact_text_hash_mismatch:{self.artifact_id}")
        return failures

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["kind"] = self.kind.value
        return value


@dataclass(frozen=True)
class CapsuleResearchContext:
    research_question: str
    formal_problem: str
    assumptions: tuple[str, ...]
    known_results: tuple[dict[str, Any], ...]
    known_failures_and_conflicts: tuple[dict[str, Any], ...]
    candidate_metrics: tuple[str, ...]
    available_datasets: tuple[dict[str, Any], ...]
    available_environments: tuple[dict[str, Any], ...]
    baseline_code: tuple[dict[str, Any], ...]
    compute_budget: dict[str, Any]
    claim_evidence_graph: tuple[dict[str, Any], ...]
    source_manifest: tuple[dict[str, Any], ...] = ()

    def validate(self, artifacts_by_id: dict[str, TemporalArtifact]) -> list[str]:
        failures: list[str] = []
        if not self.research_question or not self.formal_problem:
            failures.append("missing_research_question_or_formal_problem")
        if not self.assumptions:
            failures.append("missing_assumptions")
        if not self.candidate_metrics:
            failures.append("missing_candidate_metrics")
        if not self.baseline_code:
            failures.append("missing_baseline_code")
        if not self.claim_evidence_graph:
            failures.append("empty_claim_evidence_graph")
        if not self.source_manifest:
            failures.append("missing_source_manifest")
        for index, source in enumerate(self.source_manifest):
            source_id = str(source.get("artifact_id", ""))
            if source_id not in artifacts_by_id:
                failures.append(f"source_manifest_unknown_artifact:{index}:{source_id}")
                continue
            declared_sha256 = str(source.get("sha256", ""))
            if not declared_sha256:
                failures.append(f"source_manifest_missing_sha256:{index}:{source_id}")
            elif declared_sha256 != artifacts_by_id[source_id].snapshot_sha256:
                failures.append(f"source_manifest_hash_mismatch:{index}:{source_id}")
        for index, edge in enumerate(self.claim_evidence_graph):
            source_id = str(edge.get("source_id", ""))
            locator = str(edge.get("locator", ""))
            claim = str(edge.get("claim", ""))
            if source_id not in artifacts_by_id:
                failures.append(f"claim_unknown_source:{index}:{source_id}")
            if not locator:
                failures.append(f"claim_missing_locator:{index}")
            if not claim:
                failures.append(f"claim_missing_text:{index}")
        return failures

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["assumptions"] = list(self.assumptions)
        value["candidate_metrics"] = list(self.candidate_metrics)
        return value


@dataclass(frozen=True)
class PublicPaperCapsule:
    capsule_id: str
    split: Split
    field: str
    cutoff_utc: str
    public_problem: str
    research_context: CapsuleResearchContext
    artifacts: tuple[TemporalArtifact, ...]
    environment_lock_path: str
    environment_lock_sha256: str
    allowed_tools: tuple[str, ...]
    target_commitment: str
    semantic_leak_review_digest: str = ""
    schema_version: int = 1

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "capsule_id": self.capsule_id,
            "split": self.split.value,
            "field": self.field,
            "cutoff_utc": self.cutoff_utc,
            "public_problem": self.public_problem,
            "research_context": self.research_context.payload(),
            "artifacts": [artifact.payload() for artifact in self.artifacts],
            "environment_lock_path": self.environment_lock_path,
            "environment_lock_sha256": self.environment_lock_sha256,
            "allowed_tools": list(self.allowed_tools),
            "target_commitment": self.target_commitment,
            "semantic_leak_review_digest": self.semantic_leak_review_digest,
        }

    @property
    def digest(self) -> str:
        return content_hash(self.payload())

    @property
    def pre_review_digest(self) -> str:
        value = self.payload()
        value["semantic_leak_review_digest"] = ""
        return content_hash(value)


@dataclass(frozen=True)
class SealedTarget:
    capsule_id: str
    target_id: str
    title: str
    identifiers: tuple[str, ...]
    first_public_at_utc: str
    target_snapshot_path: str
    target_snapshot_sha256: str
    hidden_claims: tuple[dict[str, Any], ...]
    hidden_result_spec: dict[str, Any]
    salt: str
    algorithm_names: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()
    code_fingerprints: tuple[str, ...] = ()
    numeric_fingerprints: tuple[str, ...] = ()
    distinctive_phrases: tuple[str, ...] = ()

    @property
    def commitment(self) -> str:
        value = asdict(self)
        return content_hash(value)


@dataclass(frozen=True)
class CapsuleAudit:
    capsule_id: str
    capsule_digest: str
    passed: bool
    failures: tuple[str, ...]


def _normalized(value: str) -> str:
    return "".join(char.lower() for char in value if char.isalnum())


def audit_capsule(
    capsule: PublicPaperCapsule,
    sealed: SealedTarget,
    capsule_root: Path,
    sealed_root: Path,
) -> CapsuleAudit:
    failures: list[str] = []
    cutoff = parse_utc(capsule.cutoff_utc)
    target_release = parse_utc(sealed.first_public_at_utc)
    if capsule.capsule_id != sealed.capsule_id:
        failures.append("capsule_target_id_mismatch")
    if target_release <= cutoff:
        failures.append("target_not_strictly_after_cutoff")
    if capsule.target_commitment != sealed.commitment:
        failures.append("target_commitment_mismatch")
    if not capsule.semantic_leak_review_digest:
        failures.append("missing_semantic_leak_review")
    sealed_snapshot = (sealed_root / sealed.target_snapshot_path).resolve()
    try:
        sealed_snapshot.relative_to(sealed_root.resolve())
    except ValueError:
        failures.append("target_snapshot_outside_sealed_root")
    else:
        if not sealed_snapshot.is_file():
            failures.append("target_snapshot_missing")
        elif file_hash(sealed_snapshot) != sealed.target_snapshot_sha256:
            failures.append("target_snapshot_hash_mismatch")
    if not capsule.artifacts:
        failures.append("empty_evidence_capsule")
    artifacts_by_id: dict[str, TemporalArtifact] = {}
    for artifact in capsule.artifacts:
        if artifact.artifact_id in artifacts_by_id:
            failures.append(f"duplicate_artifact_id:{artifact.artifact_id}")
        artifacts_by_id[artifact.artifact_id] = artifact
        failures.extend(artifact.validate(capsule_root, cutoff))
    failures.extend(capsule.research_context.validate(artifacts_by_id))

    environment_lock = (capsule_root / capsule.environment_lock_path).resolve()
    try:
        environment_lock.relative_to(capsule_root.resolve())
    except ValueError:
        failures.append("environment_lock_outside_capsule")
    else:
        if not environment_lock.is_file():
            failures.append("environment_lock_missing")
        elif file_hash(environment_lock) != capsule.environment_lock_sha256:
            failures.append("environment_lock_hash_mismatch")

    public_parts = [
        capsule.public_problem,
        canonical_json(capsule.research_context.payload()),
        *[artifact.title for artifact in capsule.artifacts],
        *[artifact.provenance_url for artifact in capsule.artifacts],
    ]
    for artifact in capsule.artifacts:
        text_path = capsule_root / artifact.search_text_path
        if text_path.is_file():
            public_parts.append(text_path.read_text(encoding="utf-8", errors="ignore"))
    public_raw = "\n".join(public_parts)
    public_text = _normalized(public_raw)
    forbidden = [sealed.title, sealed.target_id, *sealed.identifiers]
    for identifier in forbidden:
        token = _normalized(identifier)
        if len(token) >= 6 and token in public_text:
            failures.append(f"target_identifier_leak:{identifier}")
        if len(token) < 6:
            continue
    for value in (*sealed.algorithm_names, *sealed.forbidden_terms):
        token = _normalized(value)
        if token and token in public_text:
            failures.append(f"target_forbidden_term_leak:{value}")
    for phrase in sealed.distinctive_phrases:
        token = _normalized(phrase)
        if token and token in public_text:
            failures.append(f"target_distinctive_phrase_leak:{content_hash(phrase)[:12]}")
    for fingerprint in sealed.code_fingerprints:
        token = _normalized(fingerprint)
        if token and token in public_text:
            failures.append(f"target_code_fingerprint_leak:{content_hash(fingerprint)[:12]}")
    compact_raw = "".join(public_raw.split())
    for fingerprint in sealed.numeric_fingerprints:
        if fingerprint and "".join(str(fingerprint).split()) in compact_raw:
            failures.append(f"target_numeric_fingerprint_leak:{content_hash(fingerprint)[:12]}")

    return CapsuleAudit(
        capsule_id=capsule.capsule_id,
        capsule_digest=capsule.digest,
        passed=not failures,
        failures=tuple(sorted(set(failures))),
    )


def capsule_from_mapping(value: dict[str, Any]) -> PublicPaperCapsule:
    artifacts = tuple(
        TemporalArtifact(
            artifact_id=item["artifact_id"],
            kind=ArtifactKind(item["kind"]),
            title=item["title"],
            version=item["version"],
            available_at_utc=item["available_at_utc"],
            snapshot_path=item["snapshot_path"],
            snapshot_sha256=item["snapshot_sha256"],
            search_text_path=item["search_text_path"],
            search_text_sha256=item["search_text_sha256"],
            provenance_url=item["provenance_url"],
        )
        for item in value["artifacts"]
    )


    context = value["research_context"]
    return PublicPaperCapsule(
        schema_version=int(value.get("schema_version", 1)),
        capsule_id=value["capsule_id"],
        split=Split(value["split"]),
        field=value["field"],
        cutoff_utc=value["cutoff_utc"],
        public_problem=value["public_problem"],
        research_context=CapsuleResearchContext(
            research_question=context["research_question"],
            formal_problem=context["formal_problem"],
            assumptions=tuple(context["assumptions"]),
            known_results=tuple(context["known_results"]),
            known_failures_and_conflicts=tuple(
                context["known_failures_and_conflicts"]
            ),
            candidate_metrics=tuple(context["candidate_metrics"]),
            available_datasets=tuple(context["available_datasets"]),
            available_environments=tuple(context["available_environments"]),
            baseline_code=tuple(context["baseline_code"]),
            compute_budget=dict(context["compute_budget"]),
            claim_evidence_graph=tuple(context["claim_evidence_graph"]),
            source_manifest=tuple(context.get("source_manifest", ())),
        ),
        artifacts=artifacts,
        environment_lock_path=value["environment_lock_path"],
        environment_lock_sha256=value["environment_lock_sha256"],
        allowed_tools=tuple(value["allowed_tools"]),
        target_commitment=value["target_commitment"],
        semantic_leak_review_digest=value.get("semantic_leak_review_digest", ""),
    )


def sealed_target_from_mapping(value: dict[str, Any]) -> SealedTarget:
    return SealedTarget(
        capsule_id=value["capsule_id"],
        target_id=value["target_id"],
        title=value["title"],
        identifiers=tuple(value["identifiers"]),
        first_public_at_utc=value["first_public_at_utc"],
        target_snapshot_path=value["target_snapshot_path"],
        target_snapshot_sha256=value["target_snapshot_sha256"],
        hidden_claims=tuple(value.get("hidden_claims", ())),
        hidden_result_spec=dict(value.get("hidden_result_spec", {})),
        salt=value["salt"],
        algorithm_names=tuple(value.get("algorithm_names", ())),
        forbidden_terms=tuple(value.get("forbidden_terms", ())),
        code_fingerprints=tuple(value.get("code_fingerprints", ())),
        numeric_fingerprints=tuple(str(item) for item in value.get("numeric_fingerprints", ())),
        distinctive_phrases=tuple(value.get("distinctive_phrases", ())),
    )


def assert_unique_commitments(capsules: Iterable[PublicPaperCapsule]) -> None:
    seen: set[str] = set()
    for capsule in capsules:
        if capsule.target_commitment in seen:
            raise ValueError("target commitment reused across capsules")
        seen.add(capsule.target_commitment)
