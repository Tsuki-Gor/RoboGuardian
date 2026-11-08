"""Planner-facing whitelist and byte/image budgets; complete records stay in Memory."""

from collections.abc import Mapping

from .models import Budget, Context, Observation, ReceiptState, _ADMISSION_CODES
from .serialization import json_data


class ContextBudgetError(ValueError):
    """Necessary planner constraints cannot fit the configured byte budget."""


def _context_diagnostics(memory, current, scope):
    """Project only current control feedback, never free-form provider text."""
    diagnostics = []
    decision = memory._latest_decision()
    if decision is not None and decision["kind"] == "admission_rejected":
        details = decision["payload"]["details"]
        code = details.get("code")
        if not isinstance(code, str) or not code or code not in _ADMISSION_CODES:
            code = "admission_rejected"
        item = {"scope": "diagnostic", "stage": "admission", "code": code}
        for key in ("invocation_id", "goal_id", "planning_observation_id"):
            if isinstance(details.get(key), str):
                item[key] = details[key]
        status = details.get("status")
        if isinstance(status, str) and status in ("needs_plan", "needs_observation", "awaiting_terminal", "unresolved", "budget_exhausted"):
            item["status"] = status
        required = details.get("required_facts", {})
        if isinstance(required, Mapping) and required:
            item["required_facts"] = required
            item["current_validity"] = memory.record_validity(current, required)
        diagnostics.append(item)
    if scope is not None:
        receipt = scope["receipt"]
        state = receipt.state if receipt else ReceiptState.UNKNOWN
        code = "effect_rejected" if state is ReceiptState.REJECTED else "effect_unknown"
        if state is ReceiptState.COMMITTED:
            code = "awaiting_terminal"
        item = {"scope": "diagnostic", "stage": "runtime" if receipt is None or state is ReceiptState.COMMITTED else "verification",
                "code": code if receipt else "execution_unconfirmed",
                "invocation_id": scope["action"].invocation_id, "goal_id": scope["action"].goal_id,
                "attempt_id": scope["attempt_id"], "receipt_id": receipt.receipt_id if receipt else None,
                "channels": {}, "partial_effect": False, "invalid_evidence": False, "callback_error": False}
        if receipt is not None:
            report = memory._receipt_report(receipt.receipt_id)
            if report is not None:
                item["channels"] = {key: value for key, value in report["channels"].items()
                                    if key in ("semantic", "pre", "post", "effect") and value in ("pass", "fail", "indeterminate")}
                item["partial_effect"] = report["partial_effect"] is True
                item["invalid_evidence"] = bool(report["diagnostics"].get("invalid_evidence"))
                item["callback_error"] = bool(report["diagnostics"].get("callback_errors"))
        diagnostics.append(item)
    return diagnostics


def build_context(memory, current: Observation, budget: Budget, required_facts: Mapping) -> Context:
    """Keep constraints; greedily include optional records in a stable order.

    Observation.state, runtime diagnostics, free-form audit strings, and
    unrelated observed facts never enter this planner-facing whitelist.
    """
    validity = memory.record_validity(current, required_facts)
    relations = memory.relations
    historical = {row["name"]: {"value": row["value"], "receipt_id": row["receipt_id"],
                                "scope": "historical", "source": "committed"} for row in memory._fact_records()}
    pending = memory.pending_attempt()
    active = memory.active_recovery()
    scope = pending or active
    recovery = None if scope is None else {
        "invocation_id": scope["action"].invocation_id, "attempt_id": scope["attempt_id"],
        "root": scope["recovery_root"], "skill": scope["action"].skill,
        "required_facts": scope["action"].required_facts,
        "observations_used": scope["observations"], "recovery_dispatches_used": scope["recovery_dispatches"],
        "terminal": scope["runtime_terminal"], "unresolved": scope["unresolved"] is not None,
        "receipt_state": scope["receipt"].state.value if scope["receipt"] else "unknown",
        "dispatch_blocked": pending is not None}
    progress = memory.progress
    data = {"episode_id": memory.episode_id, "state_version": memory.state_version,
            "progress": {"committed_count": len(progress), "scope": "historical", "steps": progress},
            "required_facts": json_data(required_facts), "current_validity": validity,
            "observation": {"observation_id": current.observation_id, "source_id": current.source_id,
                            "sequence": current.sequence, "scene_id": current.scene_id,
                            "entity_ids": current.entity_ids},
            "verified_facts": {name: value for name, value in historical.items() if name in required_facts},
            "relations": {"objects": [], "skills": [], "dynamic_objects": [], "skill_history": [],
                          "task_constraints": relations["task_constraints"]},
            "recovery": recovery,
            "limits": {"observations": budget.max_observations, "recovery_dispatches": budget.max_recovery_dispatches,
                       "plan_steps": budget.max_plan_steps, "plan_steps_used": memory.plan_steps},
            "diagnostics": _context_diagnostics(memory, current, scope),
            "evidence": []}
    images = []
    def fits():
        return Context(data, tuple(images)).byte_size <= budget.context_bytes
    if not fits():
        raise ContextBudgetError("necessary context exceeds the UTF-8 byte budget")
    for image in current.images[:budget.max_images]:
        images.append(image)
        if not fits():
            images.pop()
    for name, value in historical.items():
        if name not in data["verified_facts"]:
            data["verified_facts"][name] = value
            if not fits():
                del data["verified_facts"][name]
    for name in ("objects", "skills", "dynamic_objects", "skill_history"):
        for relation in reversed(relations[name]):
            data["relations"][name].append(relation)
            if not fits():
                data["relations"][name].pop()
    for receipt in memory._receipts():
        evidence = {"receipt_id": receipt.receipt_id, "state": receipt.state.value,
                    "scope": "receipt_evidence", "refs": receipt.evidence_refs}
        data["evidence"].append(evidence)
        if not fits():
            data["evidence"].pop()
    return Context(data, tuple(images))

