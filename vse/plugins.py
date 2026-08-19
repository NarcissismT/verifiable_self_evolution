from __future__ import annotations

from typing import Protocol

from .contracts import ExecutionResult, Task
from .verifier import DeclarativeHardVerifier


class TrustedTaskPlugin(Protocol):
    """Boundary between untrusted candidate artifacts and scientific metrics."""

    plugin_id: str
    evaluator_digest: str

    def score_execution(
        self, task: Task, execution: ExecutionResult
    ) -> ExecutionResult:
        """Independently recompute metrics without trusting candidate claims."""

    def verifier(self) -> DeclarativeHardVerifier:
        """Return the frozen hard verifier for this task family."""

