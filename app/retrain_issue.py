import logging
from datetime import datetime

from app.config import LOW_CONFIDENCE_LIMIT
from app.issue import create_github_issue

logger = logging.getLogger(__name__)

_state = {
    "low_confidence_count": 0,
    "samples": [],
    "issue_created": False,
}


def update_issue_state(text: str, label: str, score: float, threshold: float):
    """Track low-confidence predictions and create a retraining issue."""
    if score < threshold:
        _state["low_confidence_count"] += 1
        _state["samples"].append({
            "text": text,
            "label": label,
            "score": round(float(score), 4),
            "time": datetime.now().isoformat(timespec="seconds"),
        })

    if (
        _state["low_confidence_count"] >= LOW_CONFIDENCE_LIMIT
        and not _state["issue_created"]
    ):
        create_drift_issue()
        _state["issue_created"] = True

    return _state


def create_drift_issue():
    samples = _state["samples"][-5:]
    title = "[MLOps] Drift suspected (low confidence accumulation)"
    body = (
        "## Drift Detection Report\n"
        "Low-confidence predictions accumulated.\n\n"
        f"- count: {_state['low_confidence_count']}\n"
        f"- threshold: {LOW_CONFIDENCE_LIMIT}\n\n"
        "## Recent Samples\n"
    )

    for sample in samples:
        body += f"- ({sample['score']}) {sample['text']}\n"

    body += (
        "\n## Action\n"
        "- Please review data\n"
        "- Decide whether retraining is needed\n"
    )

    create_github_issue(title, body, logger)
