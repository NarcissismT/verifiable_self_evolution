from __future__ import annotations

from typing import Any

from .contracts import VerificationReport


def redacted_feedback(report: VerificationReport) -> dict[str, Any]:
    """Return machine feedback safe for student/teacher repair prompts.

    Numeric metric values, target alignment metrics, execution outputs, and
    sealed reference content are intentionally absent.
    """

    gate_status = {}
    for metric in report.metrics:
        if metric.name.startswith("hidden_target_"):
            continue
        gate_status[metric.name] = metric.passes()
    return {
        "task_id": report.task_id,
        "accepted": report.accepted,
        "failure_categories": list(report.hard_failures),
        "gate_status": gate_status,
        "verification_digest": report.report_digest,
    }

