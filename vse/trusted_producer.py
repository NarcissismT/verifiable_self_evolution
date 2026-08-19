from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Sequence

from .hashing import canonical_json, content_hash, file_hash


TRUSTED_PRODUCER_ID = "vse-trusted-launcher-v1"


def _docker_network_values(command: Sequence[str]) -> tuple[str, ...]:
    values: list[str] = []
    for index, token in enumerate(command):
        if token.startswith("--network="):
            values.append(token.split("=", 1)[1])
        elif token == "--network":
            if index + 1 >= len(command):
                raise ValueError("docker --network requires a value")
            values.append(command[index + 1])
    return tuple(values)


@dataclass(frozen=True)
class TrustedProducerReceipt:
    stage: str
    capsule_digest: str
    proposal_digest: str
    command_digest: str
    producer_id: str
    producer_version: str
    container_digest: str
    network_policy: str
    network_policy_measured: str
    runtime_enforced: bool
    exit_code: int
    stdout_digest: str
    stderr_digest: str
    runtime_seconds: float
    execution_digest: str = ""
    signature: str = ""
    runtime_mode: str = "docker"
    artifact_digest: str = ""

    def sealed(self, trust_key: bytes) -> "TrustedProducerReceipt":
        if len(trust_key) < 32:
            raise ValueError("trusted producer key must contain at least 32 bytes")
        value = asdict(self)
        value["execution_digest"] = ""
        value["signature"] = ""
        execution_digest = content_hash(value)
        signed = {**value, "execution_digest": execution_digest}
        signature = hmac.new(
            trust_key,
            canonical_json(signed).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return TrustedProducerReceipt(
            **{
                **asdict(self),
                "execution_digest": execution_digest,
                "signature": signature,
            }
        )


def verify_trusted_receipt(value: dict, *, trust_key: bytes) -> TrustedProducerReceipt:
    receipt = TrustedProducerReceipt(
        stage=str(value["stage"]),
        capsule_digest=str(value["capsule_digest"]),
        proposal_digest=str(value.get("proposal_digest", "")),
        command_digest=str(value["command_digest"]),
        producer_id=str(value["producer_id"]),
        producer_version=str(value["producer_version"]),
        container_digest=str(value["container_digest"]),
        network_policy=str(value["network_policy"]),
        network_policy_measured=str(value["network_policy_measured"]),
        runtime_enforced=bool(value["runtime_enforced"]),
        exit_code=int(value["exit_code"]),
        stdout_digest=str(value["stdout_digest"]),
        stderr_digest=str(value["stderr_digest"]),
        runtime_seconds=float(value["runtime_seconds"]),
        execution_digest=str(value["execution_digest"]),
        signature=str(value["signature"]),
        runtime_mode=str(value.get("runtime_mode", "docker")),
        artifact_digest=str(value.get("artifact_digest", "")),
    )
    if receipt.producer_id != TRUSTED_PRODUCER_ID:
        raise ValueError("untrusted receipt producer")
    if receipt.sealed(trust_key) != receipt:
        raise ValueError("trusted producer receipt signature or digest mismatch")
    expected_network_measurement = (
        "docker_cli_network_none"
        if receipt.runtime_mode == "docker"
        else "local_test_marker"
    )
    if (
        receipt.network_policy != "none"
        or receipt.network_policy_measured != expected_network_measurement
    ):
        raise ValueError("trusted producer did not measure network=none")
    if receipt.runtime_mode not in {"docker", "local_test"}:
        raise ValueError("unknown trusted producer runtime mode")
    if not receipt.runtime_enforced:
        raise ValueError("trusted producer runtime enforcement is missing")
    if receipt.exit_code != 0:
        raise ValueError("trusted producer command failed")
    return receipt


def run_trusted_process(
    *,
    stage: str,
    capsule_digest: str,
    proposal_digest: str,
    command: Sequence[str],
    container_digest: str,
    trust_key: bytes,
    output_path: Path | None = None,
    artifact_path: Path | None = None,
    runtime_mode: str = "local_test",
) -> TrustedProducerReceipt:
    if not command:
        raise ValueError("trusted command is required")
    if not container_digest:
        raise ValueError("container digest is required")
    if artifact_path is not None and artifact_path.exists():
        raise FileExistsError(f"trusted artifact path must be new: {artifact_path}")
    if output_path is not None and output_path.exists():
        raise FileExistsError(f"trusted receipt path must be new: {output_path}")
    if runtime_mode not in {"docker", "local_test"}:
        raise ValueError("runtime_mode must be docker or local_test")
    if runtime_mode == "docker":
        image_positions = [
            index for index, token in enumerate(command) if token == container_digest
        ]
        if (
            Path(command[0]).name != "docker"
            or len(command) < 3
            or command[1] != "run"
            or len(image_positions) != 1
            or _docker_network_values(command[: image_positions[0]]) != ("none",)
        ):
            raise ValueError("docker trusted runtime requires --network none")
        if image_positions[0] < 2:
            raise ValueError("docker command is not bound to the declared container digest")
    started = time.monotonic()
    environment = dict(os.environ)
    environment["VSE_NETWORK_POLICY"] = "none"
    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=False,
        env=environment,
    )
    elapsed = time.monotonic() - started
    artifact_digest = ""
    if artifact_path is not None:
        if not artifact_path.is_file():
            raise RuntimeError(f"trusted command did not create artifact: {artifact_path}")
        artifact_digest = file_hash(artifact_path)
    receipt = TrustedProducerReceipt(
        stage=stage,
        capsule_digest=capsule_digest,
        proposal_digest=proposal_digest,
        command_digest=content_hash(list(command)),
        producer_id=TRUSTED_PRODUCER_ID,
        producer_version="1",
        container_digest=container_digest,
        network_policy="none",
        network_policy_measured=(
            "docker_cli_network_none" if runtime_mode == "docker" else "local_test_marker"
        ),
        runtime_enforced=True,
        exit_code=completed.returncode,
        stdout_digest=content_hash(completed.stdout.hex()),
        stderr_digest=content_hash(completed.stderr.hex()),
        runtime_seconds=elapsed,
        runtime_mode=runtime_mode,
        artifact_digest=artifact_digest,
    ).sealed(trust_key)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n")
    return receipt
