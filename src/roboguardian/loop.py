"""Shared episode loop; controllers supply execution and context policy."""

from dataclasses import dataclass

from .memory import BudgetExceeded, ContextBudgetError
from .types import Action, Signal, StepResult


@dataclass(frozen=True)
class RunResult:
    """steps is the episode's persisted total number of planner proposals."""
    status: str
    steps: int
    reason: str = ""
    last_step: StepResult | None = None


def run_loop(controller, instruction, planner, goal_check, *,
             success_status="verified_success",
             success_reason="fresh observation confirms final goal constraints"):
    """Run using the controller's pending, observation, context and dispatch policies.

    The controller supplies pending(), reconcile(pending), observe(),
    context(observation) -> (active, context), consume_plan(),
    execute(proposal, observation, active), error(provider, exc), and steps().
    execute may return None or a result with recovery_status/recovery_reason.
    A controller without final verification must choose its own success status.
    """
    last = None

    def result(status, reason):
        return RunResult(status, controller.steps(), reason, last)

    while True:
        pending = controller.pending()
        if pending is not None:
            last = controller.reconcile(pending)
            if last.recovery_status in ("unresolved", "awaiting_terminal"):
                return result("unresolved", last.recovery_reason)
            if controller.pending() is not None:
                continue
        try:
            current = controller.observe()
            active, context = controller.context(current)
            goal = goal_check(current, context)
            if not isinstance(goal, Signal):
                raise TypeError("goal_check must return Signal")
        except ContextBudgetError as exc:
            return result("budget_exhausted", str(exc))
        except Exception as exc:
            return result("failed", controller.error("goal observation/check", exc))
        if goal is Signal.PASS:
            return result(success_status, success_reason)
        try:
            controller.consume_plan()
            _, context = controller.context(current)
        except (BudgetExceeded, ContextBudgetError) as exc:
            return result("budget_exhausted", str(exc))
        try:
            proposal = planner.propose(instruction, context)
            if proposal is not None and not isinstance(proposal, Action):
                raise TypeError("planner must return Action or None")
        except Exception as exc:
            return result("failed", controller.error("planner", exc))
        if proposal is None:
            return result("needs_plan", "planner supplied no action and final goal is not confirmed")
        last = controller.execute(proposal, current, active)
        if last is not None and last.recovery_status == "budget_exhausted":
            return result("budget_exhausted", last.recovery_reason)
