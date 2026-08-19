#!/usr/bin/env python3
"""Hydrate three real pilot capsules and create sealed target commitments.

Run this only on the curator host. The output is deliberately split into
public, sealed, review_packet and provenance roots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


USER_AGENT = "verifiable-self-evolution-real-pilot/0.1.2"
LAST_REQUEST_AT: dict[str, float] = {}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timezone missing: {value}")
    return parsed.astimezone(timezone.utc)


def utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def request_bytes(url: str, *, accept: str = "*/*") -> tuple[bytes, dict[str, str]]:
    headers = {"User-Agent": USER_AGENT, "Accept": accept}
    token = os.environ.get("GITHUB_TOKEN")
    if token and urlparse(url).netloc == "api.github.com":
        headers["Authorization"] = f"Bearer {token}"
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    host = urlparse(url).netloc
    minimum_interval = 3.0 if "arxiv.org" in host else 0.5
    for attempt in range(5):
        elapsed = time.monotonic() - LAST_REQUEST_AT.get(host, 0.0)
        if elapsed < minimum_interval:
            time.sleep(minimum_interval - elapsed)
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=90) as response:
                LAST_REQUEST_AT[host] = time.monotonic()
                return response.read(), dict(response.headers.items())
        except HTTPError as error:
            LAST_REQUEST_AT[host] = time.monotonic()
            retryable = error.code == 429 or 500 <= error.code < 600
            if not retryable or attempt == 4:
                raise RuntimeError(f"download failed for {url}: {error}") from error
            retry_after = error.headers.get("Retry-After", "")
            delay = float(retry_after) if retry_after.isdigit() else 0.0
            delay = max(delay, (15.0 if "arxiv.org" in host else 2.0) * (2 ** attempt))
            time.sleep(delay)
        except URLError as error:
            LAST_REQUEST_AT[host] = time.monotonic()
            if attempt == 4:
                raise RuntimeError(f"download failed for {url}: {error}") from error
            time.sleep(2.0 * (2 ** attempt))
    raise AssertionError("request retry loop exhausted")


def persistent_secret(path: Path) -> str:
    if path.is_file():
        value = path.read_text(encoding="ascii").strip()
        if len(value) != 64:
            raise ValueError(f"invalid persisted target salt: {path}")
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    value = secrets.token_hex(32)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(value + "\n")
    return value


def request_json(url: str) -> dict[str, Any] | list[Any]:
    body, _ = request_bytes(url, accept="application/json")
    value = json.loads(body)
    if not isinstance(value, (dict, list)):
        raise RuntimeError(f"non-object JSON from {url}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"refusing to replace different file: {path}")
    path.write_text(serialized, encoding="utf-8")


def download_file(url: str, path: Path) -> None:
    body, _ = request_bytes(url, accept="application/pdf,application/octet-stream")
    if not body.startswith(b"%PDF"):
        raise RuntimeError(f"expected PDF at {url}, received {body[:24]!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != body:
            raise FileExistsError(f"download changed at frozen output path: {path}")
        return
    path.write_bytes(body)


def download_blob(url: str, path: Path) -> None:
    body, _ = request_bytes(url, accept="application/octet-stream,application/gzip")
    if len(body) < 64 or not body.startswith(b"\x1f\x8b"):
        raise RuntimeError(f"downloaded blob is not a valid gzip archive: {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != body:
            raise FileExistsError(f"download changed at frozen output path: {path}")
        return
    path.write_bytes(body)


def extract_text(pdf_path: Path, text_path: Path) -> None:
    text_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vse-pdf-") as temporary:
        candidate = Path(temporary) / "paper.txt"
        if shutil.which("pdftotext"):
            result = subprocess.run(
                ["pdftotext", "-layout", str(pdf_path), str(candidate)],
                check=False,
                capture_output=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"pdftotext failed for {pdf_path}: {result.stderr!r}")
            text = candidate.read_text(encoding="utf-8", errors="replace")
        else:
            try:
                from pypdf import PdfReader  # type: ignore
            except ImportError as error:
                raise RuntimeError("install pypdf or pdftotext before hydration") from error
            reader = PdfReader(str(pdf_path))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    # pypdf may emit CRLF/CR depending on the parser path; normalize line
    # endings so resume is byte-stable across equivalent extraction runs.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text.strip()) < 1000:
        raise RuntimeError(f"extracted text is unexpectedly short: {pdf_path}")
    if text_path.exists():
        existing = text_path.read_text(encoding="utf-8")
        existing = existing.replace("\r\n", "\n").replace("\r", "\n")
        if existing != text:
            raise FileExistsError(f"text extraction changed at frozen path: {text_path}")
    text_path.write_text(text, encoding="utf-8")


def arxiv_v1_timestamp(arxiv_id: str) -> tuple[datetime, dict[str, Any]]:
    normalized_id = str(arxiv_id)
    if not normalized_id.endswith("v1"):
        normalized_id = f"{normalized_id}v1"
    url = "https://export.arxiv.org/api/query?" + urlencode({"id_list": normalized_id})
    body, _ = request_bytes(url, accept="application/atom+xml")
    root = ET.fromstring(body)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)
    if len(entries) != 1:
        raise RuntimeError(f"arXiv lookup for {arxiv_id} returned {len(entries)} entries")
    entry = entries[0]
    published = entry.findtext("atom:published", default="", namespaces=ns)
    title = " ".join(entry.findtext("atom:title", default="", namespaces=ns).split())
    if not published or not title:
        raise RuntimeError(f"arXiv metadata incomplete for {arxiv_id}")
    entry_id = entry.findtext("atom:id", default="", namespaces=ns)
    if not entry_id.rstrip("/").endswith(normalized_id):
        raise RuntimeError(f"arXiv API did not return requested v1: {normalized_id}")
    return parse_utc(published), {
        "arxiv_id": normalized_id.removesuffix("v1"),
        "version": "v1",
        "published": published,
        "title": title,
        "api_url": url,
    }


def openreview_metadata(
    target: dict[str, Any],
    *,
    snapshot_root: Path | None = None,
    allow_challenge_fallback: bool = False,
) -> tuple[datetime | None, dict[str, Any]]:
    note_id = target.get("openreview_id")
    if note_id:
        url = "https://api2.openreview.net/notes?" + urlencode({"id": note_id})
    elif target.get("openreview_title_query"):
        url = "https://api2.openreview.net/notes?" + urlencode(
            {"content.title": target["openreview_title_query"], "limit": 100}
        )
    else:
        return None, {"status": "not_declared"}
    snapshot_path = snapshot_root / f"{note_id}.json" if snapshot_root and note_id else None
    if snapshot_path is not None and snapshot_path.is_file():
        raw = json.loads(snapshot_path.read_text(encoding="utf-8"))
        snapshot_binding = {
            "metadata_source": "official_api_snapshot",
            "snapshot_sha256": file_hash(snapshot_path),
        }
    else:
        try:
            raw = request_json(url)
        except RuntimeError as error:
            challenge = "403" in str(error) and "openreview.net" in url
            if not (allow_challenge_fallback and challenge):
                raise
            return None, {
                "status": "challenge_fallback",
                "api_url": url,
                "reason": "official OpenReview API returned a browser challenge",
                "cutoff_policy": "exclude OpenReview and use conservative 180-day lag",
            }
        snapshot_binding = {"metadata_source": "live_official_api"}
    if not isinstance(raw, dict):
        raise RuntimeError("OpenReview response is not an object")
    notes = raw.get("notes", [])
    if not isinstance(notes, list):
        raise RuntimeError("OpenReview notes response malformed")
    exact: list[dict[str, Any]] = []
    expected_title = target["title"].casefold()
    for note in notes:
        content = note.get("content", {}) if isinstance(note, dict) else {}
        title_value = content.get("title", "") if isinstance(content, dict) else ""
        if isinstance(title_value, dict):
            title_value = title_value.get("value", "")
        if (
            str(title_value).strip().casefold() == expected_title
            and (not note_id or str(note.get("id", "")) == str(note_id))
        ):
            exact.append(note)
    if not exact:
        if note_id:
            raise RuntimeError(f"OpenReview note not found: {note_id}")
        return None, {"status": "unresolved_title_query", "api_url": url}
    public_times = [int(note["odate"]) for note in exact if note.get("odate") is not None]
    if not public_times:
        raise RuntimeError(f"OpenReview note lacks odate: {note_id or target['title']}")
    timestamp = datetime.fromtimestamp(min(public_times) / 1000.0, tz=timezone.utc)
    resolved_ids = sorted({str(note.get("id", "")) for note in exact if note.get("id")})
    return timestamp, {
        "status": "resolved",
        "api_url": url,
        "resolved_ids": resolved_ids,
        "odate": min(public_times),
        "first_public_at_utc": utc_text(timestamp),
        **snapshot_binding,
    }


def github_metadata(repository_url: str) -> tuple[list[datetime], dict[str, Any]]:
    match = re.fullmatch(r"https://github\.com/([^/]+)/([^/#]+?)(?:\.git)?", repository_url)
    if not match:
        raise ValueError(f"unsupported GitHub repository URL: {repository_url}")
    owner, repo = match.groups()
    api = f"https://api.github.com/repos/{quote(owner)}/{quote(repo)}"
    metadata = request_json(api)
    if not isinstance(metadata, dict):
        raise RuntimeError("GitHub metadata malformed")
    created = parse_utc(str(metadata["created_at"]))
    default_branch = str(metadata["default_branch"])
    commits: list[dict[str, Any]] = []
    page = 1
    while True:
        url = api + "/commits?" + urlencode({"sha": default_branch, "per_page": 100, "page": page})
        batch = request_json(url)
        if not isinstance(batch, list):
            raise RuntimeError("GitHub commits response malformed")
        commits.extend(item for item in batch if isinstance(item, dict))
        if len(batch) < 100:
            break
        page += 1
        if page > 100:
            raise RuntimeError("refusing to traverse more than 10,000 commits")
    if not commits:
        raise RuntimeError(f"target code repository has no commits: {repository_url}")
    roots = [item for item in commits if not item.get("parents")]
    if not roots:
        raise RuntimeError(f"could not identify root commit for {repository_url}")
    root = min(
        roots,
        key=lambda item: str(item.get("commit", {}).get("committer", {}).get("date", "9999")),
    )
    root_date_text = str(root["commit"]["committer"]["date"])
    root_date = parse_utc(root_date_text)
    return [created, root_date], {
        "repository": repository_url,
        "api_url": api,
        "created_at": utc_text(created),
        "default_branch": default_branch,
        "reachable_commit_count": len(commits),
        "root_commit": str(root["sha"]),
        "root_commit_date": utc_text(root_date),
    }


def environment_lock(case_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "case_id": case_id,
        "python": "3.11",
        "numpy": "1.26.4",
        "student_api": "solve(case_id, problem, seed, oracle_budget, hyperparameters) -> {point, oracle_calls}",
        "evaluation_seeds": [1031, 2063, 4099, 8191],
        "oracle_budget": 4000,
        "network_policy": "none",
        "filesystem_policy": "read-only except declared output directory",
        "target_visible_to_student": False,
    }


def build_context(case: dict[str, Any], artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    context = dict(case["public_context"])
    context.update(
        {
            "available_datasets": [],
            "available_environments": [
                {"environment_id": "vse_hidden_smooth_family_v1", "interface": "student_api_v1"}
            ],
            "baseline_code": [
                {
                    "baseline_id": f"precutoff_penalty_synthesis_{index + 1}",
                    "artifact_id": artifact["artifact_id"],
                    "implementation_note": "Target-neutral engineering baseline; source paper is frozen in this capsule.",
                }
                for index, artifact in enumerate(artifacts)
            ],
            "compute_budget": {"gpu_hours": 0, "cpu_hours": 2, "oracle_calls_per_seed": 4000},
            "source_manifest": [
                {"artifact_id": artifact["artifact_id"], "sha256": artifact["snapshot_sha256"], "git_commit": None}
                for artifact in artifacts
            ],
            "claim_evidence_graph": [
                {
                    "claim": result["summary"],
                    "source_id": artifacts[min(index, len(artifacts) - 1)]["artifact_id"],
                    "locator": "abstract and method sections in frozen arXiv v1 snapshot",
                }
                for index, result in enumerate(context["known_results"])
            ],
        }
    )
    return context


def public_problem(case_id: str) -> str:
    common = (
        "Design a target-neutral first-order bilevel optimization procedure for the supplied "
        "oracle family. Return one point for each of four fixed seeds within 4,000 oracle "
        "calls. The hidden evaluator checks lower-level validity, upper-level progress, "
        "finite outputs and seed reproducibility. You may use only the frozen pre-cutoff "
        "evidence in this capsule and the declared Python interface."
    )
    additions = {
        "realpilot_flatness_penalty": " Lower-level strong convexity must not be assumed.",
        "realpilot_nonconvex_simple": " Both upper and lower objectives may be nonconvex.",
        "realpilot_linear_coupling": " Coupled linear feasibility is a hard gate.",
    }
    return common + additions[case_id]


def hydrate_case(
    case: dict[str, Any],
    *,
    output_root: Path,
    lag_days: int,
    openreview_snapshot_root: Path | None = None,
    allow_openreview_challenge: bool = False,
    fallback_lag_days: int = 180,
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    target = dict(case["target"])
    public_case = output_root / "public" / case_id
    sealed_case = output_root / "sealed" / case_id
    provenance_case = output_root / "provenance" / case_id
    public_case.mkdir(parents=True, exist_ok=True)
    sealed_case.mkdir(parents=True, exist_ok=True)
    provenance_case.mkdir(parents=True, exist_ok=True)

    arxiv_time, arxiv_meta = arxiv_v1_timestamp(str(target["arxiv_id"]))
    openreview_time, openreview_meta = openreview_metadata(
        target,
        snapshot_root=openreview_snapshot_root,
        allow_challenge_fallback=allow_openreview_challenge,
    )
    cutoff_lag_days = (
        fallback_lag_days
        if openreview_meta.get("status") == "challenge_fallback"
        else lag_days
    )
    github_times, github_meta = github_metadata(str(target["code_repository"]))
    public_times = [arxiv_time, *github_times]
    if openreview_time is not None:
        public_times.append(openreview_time)
    first_public = min(public_times)
    cutoff = first_public - timedelta(days=cutoff_lag_days)

    target_pdf = sealed_case / "target.pdf"
    download_file(str(target["arxiv_v1_url"]), target_pdf)
    target_text = sealed_case / "target.txt"
    extract_text(target_pdf, target_text)
    target_code = sealed_case / "target_code.tar.gz"
    repository_api = str(github_meta["api_url"])
    try:
        download_blob(f"{repository_api}/tarball/{github_meta['root_commit']}", target_code)
    except RuntimeError as error:
        # Some GitHub API proxies reject the archive media type with 415. The
        # codeload endpoint serves the same commit-pinned archive directly.
        if "415" not in str(error):
            raise
        repository_url = str(target["code_repository"]).rstrip("/")
        download_blob(
            f"https://codeload.github.com/{repository_url.split('github.com/', 1)[1]}"
            f"/tar.gz/{github_meta['root_commit']}",
            target_code,
        )
    write_json(provenance_case / "target_timestamps.json", {
        "arxiv": arxiv_meta,
        "openreview": openreview_meta,
        "github": github_meta,
        "first_public_at_utc": utc_text(first_public),
        "cutoff_lag_days": lag_days,
        "effective_cutoff_lag_days": cutoff_lag_days,
        "cutoff_provisional": openreview_meta.get("status") == "challenge_fallback",
        "cutoff_utc": utc_text(cutoff),
    })

    artifacts: list[dict[str, Any]] = []
    for source in case["prior_sources"]:
        source = dict(source)
        actual_time, actual_meta = arxiv_v1_timestamp(str(source["arxiv_id"]))
        if actual_time > cutoff:
            raise RuntimeError(f"pre-cutoff source is after cutoff: {case_id}/{source['artifact_id']}")
        declared = source.get("available_at_utc")
        relative_pdf = Path(case_id) / "artifacts" / f"{source['artifact_id']}.pdf"
        relative_text = Path(case_id) / "artifacts" / f"{source['artifact_id']}.txt"
        pdf_path = output_root / "public" / relative_pdf
        text_path = output_root / "public" / relative_text
        download_file(str(source["url"]), pdf_path)
        extract_text(pdf_path, text_path)
        write_json(provenance_case / f"source_{source['artifact_id']}.json", {
            **actual_meta,
            "declared_available_at_utc": declared,
            "timestamp_matches_declaration": bool(
                declared and abs((parse_utc(str(declared)) - actual_time).total_seconds()) <= 1
            ),
        })
        artifacts.append({
            "artifact_id": source["artifact_id"],
            "kind": "paper",
            "title": source["title"],
            "version": "v1",
            "available_at_utc": utc_text(actual_time),
            "snapshot_path": str(relative_pdf),
            "snapshot_sha256": file_hash(pdf_path),
            "search_text_path": str(relative_text),
            "search_text_sha256": file_hash(text_path),
            "provenance_url": f"https://arxiv.org/abs/{source['arxiv_id']}v1",
        })

    lock_relative = Path(case_id) / "environment" / "environment.lock"
    lock_path = output_root / "public" / lock_relative
    write_json(lock_path, environment_lock(case_id))

    identifiers = [str(target["arxiv_id"]), str(target["proceedings_url"]), str(target["code_repository"])]
    if target.get("openreview_id"):
        identifiers.append(str(target["openreview_id"]))
    elif openreview_meta.get("resolved_ids"):
        identifiers.extend(str(value) for value in openreview_meta["resolved_ids"])
    hidden_result_spec = dict(target["hidden_result_spec"])
    hidden_result_spec["target_code_snapshot"] = {
        "path": f"{case_id}/target_code.tar.gz",
        "sha256": file_hash(target_code),
        "git_commit": github_meta["root_commit"],
        "repository": target["code_repository"],
    }
    sealed = {
        "capsule_id": case_id,
        "target_id": target["target_id"],
        "title": target["title"],
        "identifiers": identifiers,
        "first_public_at_utc": utc_text(first_public),
        "target_snapshot_path": f"{case_id}/target.pdf",
        "target_snapshot_sha256": file_hash(target_pdf),
        "hidden_claims": target["hidden_claims"],
        "hidden_result_spec": hidden_result_spec,
        "algorithm_names": target.get("algorithm_names", []),
        "forbidden_terms": target.get("forbidden_terms", []),
        "code_fingerprints": target.get("code_fingerprints", []),
        "numeric_fingerprints": target.get("numeric_fingerprints", []),
        "distinctive_phrases": target.get("distinctive_phrases", []),
        "salt": persistent_secret(provenance_case / "target_salt.txt"),
    }
    commitment = content_hash(sealed)
    write_json(sealed_case / "target.json", sealed)

    capsule = {
        "schema_version": 1,
        "capsule_id": case_id,
        "split": "dev",
        "field": "bilevel_stackelberg_alignment_optimization",
        "cutoff_utc": utc_text(cutoff),
        "public_problem": public_problem(case_id),
        "research_context": build_context(case, artifacts),
        "artifacts": artifacts,
        "environment_lock_path": str(lock_relative),
        "environment_lock_sha256": file_hash(lock_path),
        "allowed_tools": ["python", "numpy", "first_order_oracle"],
        "target_commitment": commitment,
        "semantic_leak_review_digest": "",
    }
    capsule_path = public_case / "capsule.json"
    write_json(capsule_path, capsule)

    packet = output_root / "review_packet" / case_id
    packet.mkdir(parents=True, exist_ok=True)
    write_json(packet / "pre_review_capsule.json", capsule)
    write_json(packet / "sealed_target.json", sealed)
    instructions = (
        f"# Independent semantic review: {case_id}\n\n"
        "Review the public capsule and every frozen search-text artifact against the sealed "
        "target. Check title/identifier, algorithm-name, paraphrase, code, numerical-result, "
        "citation-graph and task-wording leakage. Do not approve if you authored the target, "
        "curated this capsule, or implemented the trusted evaluator. Record concrete findings "
        "in a UTF-8 text file, one finding per line. From the repository root, run:\n\n"
        "```bash\n"
        "python pilot/real_v0_1_2/scripts/review_semantic.py \\\n"
        f"  --run-root pilot/real_v0_1_2/run --case-id {case_id} \\\n"
        "  --reviewer-id YOUR_STABLE_REVIEWER_ID --evaluator-version YOUR_REVIEW_PROTOCOL_VERSION \\\n"
        "  --decision pass --findings-file /path/to/findings.txt \\\n"
        "  --attest-independent-of-target --attest-independent-of-capsule --attest-independent-of-evaluator\n"
        "```\n"
    )
    (packet / "REVIEW_INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")
    summary = {
        "case_id": case_id,
        "first_public_at_utc": utc_text(first_public),
        "cutoff_utc": utc_text(cutoff),
        "capsule_path": str(capsule_path.relative_to(output_root)),
        "capsule_pre_review_digest": content_hash(capsule),
        "target_path": str((sealed_case / "target.json").relative_to(output_root)),
        "target_commitment": commitment,
        "root_code_commit": github_meta["root_commit"],
        "cutoff_provisional": openreview_meta.get("status") == "challenge_fallback",
        "openreview_metadata_status": openreview_meta.get("status"),
    }
    write_json(provenance_case / "hydration_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--openreview-snapshot-root", type=Path)
    parser.add_argument(
        "--allow-openreview-challenge",
        action="store_true",
        help="use a conservative 180-day cutoff when official OpenReview requires browser verification",
    )
    parser.add_argument("--fallback-lag-days", type=int, default=180)
    args = parser.parse_args()
    bundle_root = args.bundle_root.resolve()
    output_root = args.output_root.resolve()
    spec = json.loads((bundle_root / "admin" / "targets_admin.json").read_text(encoding="utf-8"))
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise SystemExit(f"output root must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for case in spec["cases"]:
        summary_path = (
            output_root / "provenance" / str(case["case_id"]) / "hydration_summary.json"
        )
        if args.resume and summary_path.is_file():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        summaries.append(
            hydrate_case(
                case,
                output_root=output_root,
                lag_days=int(spec["cutoff_lag_days"]),
                openreview_snapshot_root=(
                    args.openreview_snapshot_root.resolve()
                    if args.openreview_snapshot_root is not None
                    else None
                ),
                allow_openreview_challenge=args.allow_openreview_challenge,
                fallback_lag_days=args.fallback_lag_days,
            )
        )
    commitments = [item["target_commitment"] for item in summaries]
    if len(commitments) != len(set(commitments)):
        raise RuntimeError("target commitment reused")
    shutil.copy2(bundle_root / "admin" / "pilot_exclusions.json", output_root / "pilot_exclusions.json")
    provisional = any(item.get("cutoff_provisional", False) for item in summaries)
    write_json(output_root / "hydration_manifest.json", {
        "schema_version": 1,
        "bundle_id": spec["bundle_id"],
        "curated_at_utc": utc_text(datetime.now(timezone.utc)),
        "cases": summaries,
        "cutoff_provisional": provisional,
        "status": (
            "blocked_pending_independent_semantic_review_provisional_cutoff"
            if provisional
            else "blocked_pending_independent_semantic_review"
        ),
    })
    print(json.dumps({"status": "hydrated", "cases": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
