from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
from typing import Sequence

from .contracts import ExecutionResult, ExperimentProposal, Split, Task
from .hashing import content_hash


@dataclass(frozen=True)
class RunnerConfig:
    mode: str
    timeout_seconds: float
    memory_bytes: int = 2 * 1024**3
    file_bytes: int = 64 * 1024**2
    container_command: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.mode not in {"local_test", "container"}:
            raise ValueError("runner mode must be local_test or container")
        if self.mode == "container" and "{script}" not in self.container_command:
            raise ValueError("container command must contain a {script} placeholder")


def _limit_process(config: RunnerConfig) -> None:
    cpu_seconds = max(1, int(config.timeout_seconds) + 1)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_AS, (config.memory_bytes, config.memory_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (config.file_bytes, config.file_bytes))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


class CodeRunner:
    def __init__(self, config: RunnerConfig):
        config.validate()
        self.config = config

    def _command(self, script: Path) -> Sequence[str]:
        if self.config.mode == "local_test":
            return (sys.executable, "-I", str(script))
        return tuple(part.replace("{script}", str(script)) for part in self.config.container_command)

    def execute(
        self, task: Task, proposal: ExperimentProposal, seed: int
    ) -> ExecutionResult:
        if task.task_id != proposal.task_id:
            raise ValueError("proposal targets a different task")
        if seed not in proposal.seeds:
            raise ValueError("execution seed was not preregistered")
        if self.config.mode == "local_test" and task.split in {
            Split.PROMOTION,
            Split.HELDOUT,
            Split.OOD,
        }:
            raise PermissionError("formal evaluation requires the network-disabled container runner")

        with tempfile.TemporaryDirectory(prefix="vse-exec-") as raw_dir:
            workdir = Path(raw_dir)
            script = workdir / "candidate.py"
            script.write_text(proposal.experiment_code)
            input_value = {
                "task": task.payload(),
                "solution": proposal.solution,
                "seed": seed,
                "baselines": list(proposal.baselines),
                "proposal_digest": proposal.digest,
            }
            started = time.monotonic()
            before = resource.getrusage(resource.RUSAGE_CHILDREN)
            timed_out = False
            try:
                completed = subprocess.run(
                    self._command(script),
                    input=json.dumps(input_value),
                    text=True,
                    capture_output=True,
                    timeout=self.config.timeout_seconds,
                    cwd=workdir,
                    env={
                        "PATH": os.environ.get("PATH", ""),
                        "PYTHONHASHSEED": str(seed),
                        "VSE_NETWORK_POLICY": "disabled",
                    },
                    preexec_fn=(
                        (lambda: _limit_process(self.config))
                        if self.config.mode == "local_test"
                        else None
                    ),
                    check=False,
                )
                exit_code = completed.returncode
                stdout = completed.stdout[-256_000:]
                stderr = completed.stderr[-256_000:]
            except subprocess.TimeoutExpired as error:
                timed_out = True
                exit_code = 124
                stdout = (error.stdout or "")[-256_000:]
                stderr = (error.stderr or "")[-256_000:]
            elapsed = time.monotonic() - started
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            candidate_result: dict = {}
            if exit_code == 0 and not timed_out:
                try:
                    parsed = json.loads(stdout)
                    if not isinstance(parsed, dict):
                        raise TypeError("candidate output must be a JSON object")
                    candidate_result = parsed
                except (json.JSONDecodeError, TypeError) as error:
                    exit_code = 65
                    stderr = f"{stderr}\ninvalid_result_json:{error}".strip()
            return ExecutionResult(
                candidate_id=proposal.candidate_id,
                task_id=task.task_id,
                proposal_digest=proposal.digest,
                preregistered_resource_schedule=proposal.resource_schedule,
                seed=seed,
                exit_code=exit_code,
                timed_out=timed_out,
                runtime_seconds=elapsed,
                stdout=stdout,
                stderr=stderr,
                candidate_result=candidate_result,
                trusted_metrics={},
                trusted_evaluator_digest="",
                resource={
                    "user_cpu_seconds": max(0.0, after.ru_utime - before.ru_utime),
                    "system_cpu_seconds": max(0.0, after.ru_stime - before.ru_stime),
                    "max_rss_kib": float(after.ru_maxrss),
                },
            )


def execution_digest(execution: ExecutionResult) -> str:
    from dataclasses import asdict

    return content_hash(asdict(execution))
