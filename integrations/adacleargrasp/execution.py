"""Ada execution transport and the ungoverned experimental controller."""

import math
from threading import RLock
from urllib.error import URLError
from uuid import uuid4

from roboguardian import Observation, RuntimeResult, Signal
from roboguardian.loop import run_loop
from roboguardian.types import json_data
from integrations._transport import LoopbackTransport


class AdaExecution:
    """Serialize native operations and isolate experimental delivery faults."""

    def __init__(self, adapter, *, fault="none", timeout=300.0, response_delay=None):
        if fault not in ("none", "false_success", "response_loss"):
            raise ValueError("unknown execution fault")
        if response_delay is None:
            response_delay = timeout + 0.2
        if not math.isfinite(response_delay) or response_delay < 0:
            raise ValueError("response_delay must be finite and nonnegative")
        if fault == "response_loss" and response_delay <= timeout:
            raise ValueError("response_delay must exceed the transport timeout")
        self.adapter, self.fault, self.response_delay = adapter, fault, response_delay
        self.events, self.injected = [], False
        self._lock = RLock()
        self.transport = LoopbackTransport(self._dispatch, timeout=timeout)

    def _eligible(self, action):
        return self.adapter.validate_action(action)

    def _dispatch(self, payload):
        from roboguardian import Action

        action = Action(**payload["action"])
        attempt = payload["attempt_id"]
        with self._lock:
            before_feedback = self.adapter.native_feedback
            before_execution = self.adapter.last_execution
            try:
                if payload["false_success"] and self._eligible(action):
                    self.adapter.bind_before(action, attempt)
                    feedback = {"ok": True, "error_code": "none", "message": "ok", "advice": ""}
                    result = RuntimeResult("native_returned", True, action.invocation_id, attempt,
                                           {"native_result": feedback})
                    self.events.append({"event": "fault", "fault": "false_success",
                                        "invocation_id": action.invocation_id, "attempt_id": attempt})
                else:
                    count = len(self.adapter.dispatches)
                    result = self.adapter.execute(action, attempt)
                    for dispatch in self.adapter.dispatches[count:]:
                        self.events.append({"event": "physical_dispatch", **dispatch})
                return json_data(result)
            finally:
                self.adapter.native_feedback = before_feedback
                self.adapter.last_execution = before_execution

    def _result(self, record, action, attempt):
        if not record or record["status"] != "completed":
            status = record["status"] if record else "unknown"
            return RuntimeResult(status, None, action.invocation_id, attempt)
        result = RuntimeResult(**record["result"])
        if result.invocation_id != action.invocation_id or result.attempt_id != attempt:
            raise ValueError("runtime result identity mismatch")
        feedback = result.diagnostics.get("native_result")
        if feedback is not None:
            with self._lock:
                self.adapter.native_feedback = dict(feedback)
                self.adapter.last_execution = {"invocation_id": action.invocation_id,
                                               "attempt_id": attempt, **feedback}
        return result

    def execute(self, action, attempt):
        delay, false_success = 0, False
        with self._lock:
            eligible = (not self.injected and self.fault != "none"
                        and action.skill in (("grasp",) if self.fault == "false_success" else ("push", "pull", "grasp"))
                        and self._eligible(action))
            if eligible:
                self.injected = True
                if self.fault == "false_success":
                    false_success = True
                else:
                    delay = self.response_delay
                    self.events.append({"event": "fault", "fault": "response_loss",
                                        "invocation_id": action.invocation_id, "attempt_id": attempt})
        try:
            record = self.transport.execute(action.invocation_id, attempt,
                                            {"action": json_data(action), "attempt_id": attempt,
                                             "false_success": false_success}, delay)
        except TimeoutError:
            return RuntimeResult("timeout", None, action.invocation_id, attempt)
        except (OSError, URLError) as exc:
            return RuntimeResult("transport_error", None, action.invocation_id, attempt,
                                 {"error": f"{type(exc).__name__}: {exc}"})
        return self._result(record, action, attempt)

    def status(self, action, attempt):
        self.events.append({"event": "status_query", "invocation_id": action.invocation_id, "attempt_id": attempt})
        try:
            return self._result(self.transport.query(action.invocation_id, attempt), action, attempt)
        except TimeoutError:
            return RuntimeResult("timeout", None, action.invocation_id, attempt)
        except (OSError, URLError) as exc:
            return RuntimeResult("transport_error", None, action.invocation_id, attempt,
                                 {"error": f"{type(exc).__name__}: {exc}"})

    def observe(self, action):
        with self._lock:
            return self.adapter.observe(action)

    def close(self):
        self.transport.close()


class BaseController:
    """Use the shared loop with cumulative history and no effect governance."""

    def __init__(self, memory, budget, runtime):
        self.memory, self.budget, self.runtime = memory, budget, runtime
        self.grasp_ok = False

    def pending(self):
        return None

    def observe(self):
        observation = self.runtime.observe(None)
        if not isinstance(observation, Observation):
            raise TypeError("observer must return Observation")
        self.memory.record_observation(observation)
        return observation

    def context(self, observation):
        return None, self.memory.context(observation, self.budget, {})

    def consume_plan(self):
        self.memory.consume_plan_step(self.budget.max_plan_steps)

    def execute(self, proposal, observation, active):
        result = self.runtime.execute(proposal, uuid4().hex)
        self._record(proposal, result)
        if result.status in ("timeout", "transport_error"):
            result = self.runtime.execute(proposal, uuid4().hex)
            self._record(proposal, result)
        self.grasp_ok = (proposal.skill == "grasp" and result.terminal is True
                         and result.diagnostics.get("native_result", {}).get("ok") is True)

    def _record(self, action, result):
        self.memory.record_diagnostic("base_execution", {"action": json_data(action), "runtime_result": json_data(result)})

    def error(self, provider, exc):
        reason = f"{provider}: {type(exc).__name__}: {str(exc)!a}"
        self.memory.record_diagnostic("provider_error", {"provider": provider, "reason": reason})
        return reason

    def steps(self):
        return self.memory.plan_steps

    def run(self, instruction, planner):
        self.memory.acquire_lease()
        try:
            with self.memory.operation():
                return run_loop(self, instruction, planner,
                                lambda observation, context: Signal.PASS if self.grasp_ok else Signal.FAIL,
                                success_status="completed", success_reason="software reported grasp success")
        finally:
            self.memory.release_lease()
