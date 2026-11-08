"""Model planning and visual assessment."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import time
from uuid import uuid4

from roboguardian import Action, Signal
from roboguardian.types import json_data


SYSTEM_PROMPT = """
You are the manipulation planner inside EffectAgent. Select exactly one allowed
tool on every turn to clear only the clutter that blocks the target and then
grasp the target.

The current image is the primary scene observation. Use exact object names from
the supplied inventory. A software return value is not proof of a physical
effect: use effect_status, committed, next_hint, and receipt-linked governed
memory when deciding what is true. Do not blindly repeat an action after a
rejected, unknown, ambiguous, stuck, or no-progress result.

move_to approaches a named object in XY. lift and initarm provide safe
transitions. lower reaches the working height. push and pull act on the named
object prepared by move_to and lower. grasp invokes the fixed grasp policy.
Call done only after a grasp effect has been independently verified and
committed. The runtime may reject an unsafe or inconsistent proposal; choose a
new action from the returned evidence instead of assuming a replacement ran.
""".strip()

TASK_PROMPT = """
Goal: grasp the simulator object named target. Inspect the current image and
feedback, clear obstructing clutter only when needed, and choose one next tool.
Clutter near the target or its approach corridor is more likely to interfere
than distant clutter. There is no mandatory action sequence beyond each skill's
physical preconditions.
""".strip()


def tool_specs() -> list[dict]:
    empty = {"type": "object", "properties": {}, "additionalProperties": False}
    displacement = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "side": {
                "type": "string",
                "enum": ["left", "center", "right", "middle"],
            },
            "dist_m": {"type": "number", "exclusiveMinimum": 0},
        },
        "required": ["name", "side", "dist_m"],
        "additionalProperties": False,
    }
    move = {
        "type": "object",
        "properties": {"name": {"type": "string", "minLength": 1}},
        "required": ["name"],
        "additionalProperties": False,
    }
    definitions = (
        ("move_to", "Approach the exact named object in XY.", move),
        ("lift", "Raise the end effector to a safer transition height.", empty),
        ("lower", "Lower onto the object most recently prepared by move_to.", empty),
        (
            "push",
            "Push the prepared named object away from the robot-side table edge.",
            displacement,
        ),
        (
            "pull",
            "Pull the prepared named object toward the robot-side table edge.",
            displacement,
        ),
        ("initarm", "Reset the arm posture at a safe height.", empty),
        ("inithand", "Reset the hand to its open posture.", empty),
        ("grasp", "Run the fixed AdaClearGrasp PPO grasp skill.", empty),
        ("done", "Finish only after a grasp effect is verified and committed.", empty),
    )
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
        for name, description, parameters in definitions
    ]



def write_json(path, value, secret=""):
    text = json.dumps(json_data(value), ensure_ascii=False, indent=2, allow_nan=False)
    path.write_text(text.replace(secret, "[REDACTED]") if secret else text, encoding="utf-8")


class ChatModel:
    """Record each planner/assessor attempt, including requests and responses."""
    def __init__(self, client, model, output, *, temperature, max_tokens, image_detail):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("an explicit model name is required")
        self.client, self.model = client, model
        self.output = Path(output) / "model_requests"
        self.output.mkdir(exist_ok=True)
        self.temperature, self.max_tokens, self.image_detail = temperature, max_tokens, image_detail
        self.secret = str(getattr(client, "api_key", "") or "")
        self.calls = 0

    def request(self, role, prompt, images):
        content = [{"type": "text", "text": prompt}]
        for path in images:
            encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded, "detail": self.image_detail}})
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": "Follow the given robot governance task. Return only the requested JSON object."},
                             {"role": "user", "content": content}]}
        def parse(response):
            text = response.choices[0].message.content
            if not isinstance(text, str):
                raise ValueError("model returned no textual JSON content")
            text = text.strip()
            if text.startswith("```") and text.endswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError("model output must be one JSON object")
            return value
        return self._send(role, body, parse)

    def plan(self, messages):
        """Send a strict tool request; only provider and JSON failures are retried."""
        encoded_messages = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                content = [({"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(Path(item["path"]).read_bytes()).decode("ascii"),
                    "detail": self.image_detail}} if item.get("type") == "image" else item) for item in content]
            encoded_messages.append({**message, "content": content})
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                "messages": encoded_messages, "tools": tool_specs(), "tool_choice": "required"}
        def parse(response):
            calls = response.choices[0].message.tool_calls or []
            if len(calls) != 1:
                raise ValueError("expected exactly one tool call")
            arguments = json.loads(calls[0].function.arguments)
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be an object")
            return {"action": calls[0].function.name, "args": arguments,
                    "call_id": getattr(calls[0], "id", None) or "call_" + uuid4().hex,
                    "content": getattr(response.choices[0].message, "content", None),
                    "arguments_json": calls[0].function.arguments}
        return self._send("planner", body, parse)

    def _send(self, role, body, parse):
        try:
            from openai import APIConnectionError, APITimeoutError, RateLimitError, InternalServerError
            provider_errors = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError, TimeoutError, ConnectionError)
        except ImportError:
            provider_errors = (TimeoutError, ConnectionError)
        for attempt in range(3):
            self.calls += 1
            prefix = self.output / f"{self.calls:04d}-{role}"
            write_json(prefix.with_suffix(".request.json"), body, self.secret)
            started = time.monotonic()
            try:
                response = self.client.chat.completions.create(**body)
                write_json(prefix.with_suffix(".response.json"), {"elapsed_seconds": time.monotonic() - started,
                           "response": response.model_dump(mode="json")}, self.secret)
                return parse(response)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                if self.secret:
                    reason = reason.replace(self.secret, "[REDACTED]")
                write_json(prefix.with_suffix(".error.json"), {"error": reason, "elapsed_seconds": time.monotonic() - started})
                if not isinstance(exc, (*provider_errors, json.JSONDecodeError)):
                    raise ValueError(reason) from None
                if attempt == 2:
                    raise RuntimeError(reason) from None

    def assess(self, question, observation):
        images = observation.images[:1]
        evidence = "Image 1 is the current observation. "
        if observation.state.get("visual_reference_before") and len(observation.images) > 1:
            images = observation.images[:2]
            evidence = ("Image 1 is the current observation AFTER the action. "
                        "Image 2 is the original observation BEFORE the same action, provided only as a visual reference. "
                        "Compare the target position relative to the table and the other objects; distinguish the named target from nearby clutter. ")
        value = self.request("semantic", question + "\n" + evidence + "If occluded or ambiguous, abstain. "
                             'Return {"judgment":"pass|fail|indeterminate","reason":"brief visual evidence"}.', images)
        if not isinstance(value.get("reason"), str) or not value["reason"].strip():
            raise ValueError("semantic assessment requires an evidence reason")
        return Signal(value.get("judgment"))


class AdaPlanner:
    def __init__(self, model, adapter, *, mode="roboguardian", memory=None):
        if mode not in ("base", "roboguardian-window", "roboguardian"):
            raise ValueError("invalid Ada planner mode")
        self.model, self.adapter, self.mode, self.memory = model, adapter, mode, memory
        self.history = []

    def propose(self, instruction, context):
        events = self.memory.events() if self.memory is not None else []
        if self.history:
            previous = self.history[-1]
            feedback = self._feedback(previous["action"], context, events)
            previous["messages"].append({"role": "tool", "tool_call_id": previous["call_id"],
                                         "content": json.dumps(feedback, ensure_ascii=False)})
        else:
            feedback = {"status": "no_previous_action"}
        operation = {key: getattr(self.adapter, key) for key in ("bound_subject", "lowered", "transition_required")
                     if hasattr(self.adapter, key)}
        if hasattr(self.adapter, "stuck_transitions"):
            operation["stuck_transitions"] = sorted(self.adapter.stuck_transitions)
        control = {key: context.data[key] for key in ("recovery", "limits", "diagnostics", "current_validity", "required_facts")
                   if key in context.data}
        turn = {"role": "user", "content": [{"type": "image", "path": path} for path in context.images[:1]] + [
            {"type": "text", "text": "Latest tool feedback: " + json.dumps(feedback, ensure_ascii=False) +
             "\nOperation phase: " + json.dumps(operation, ensure_ascii=False) +
             "\nControl: " + json.dumps(json_data(control), ensure_ascii=False)}]}
        history = self.history if self.mode == "base" else self.history[-2:]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content":
            f"{TASK_PROMPT}\nTask: {instruction}\nTarget category: {self.adapter.target_label}. "
            "Use 'target' for the target. Available object names: " +
            ", ".join(getattr(self.adapter, "objects", (self.adapter.target_id,)))}]
        if self.mode == "roboguardian":
            summary = {key: context.data[key] for key in ("progress", "verified_facts", "relations", "evidence") if key in context.data}
            messages.append({"role": "user", "content": "Bounded structured memory: " + json.dumps(json_data(summary), ensure_ascii=False)})
            messages.extend(self._related_evidence(events, history, turn))
        messages.extend(message for group in history for message in group["messages"])
        messages.append(turn)
        if hasattr(self.model, "plan"):
            value = self.model.plan(messages)
        else:
            images = tuple(item["path"] for message in messages if isinstance(message.get("content"), list)
                           for item in message["content"] if item["type"] == "image")
            value = self.model.request("planner", json.dumps(messages, ensure_ascii=False), images)
        skill, params = value.get("action"), value.get("args")
        if skill == "done" and params == {}:
            return None
        if not isinstance(skill, str) or not skill.strip() or not isinstance(params, dict):
            raise ValueError("tool name must be nonempty and arguments must be an object")
        name = params.get("name", "target")
        entity_id = name if isinstance(name, str) and name.strip() and name != "target" else self.adapter.target_id
        action = Action(uuid4().hex, skill, self.adapter.embodiment, skill, params,
                        self.adapter.scene_id, (entity_id,), context.data["observation"]["observation_id"],
                        context.data["state_version"], {"target_present": True}, self.adapter.config_version)
        call_id = value.get("call_id") or "call_" + uuid4().hex
        assistant = {"role": "assistant", "content": value.get("content"), "tool_calls": [{
            "id": call_id, "type": "function", "function": {"name": skill,
            "arguments": value.get("arguments_json") or json.dumps(params, ensure_ascii=False)}}]}
        self.history = history + [{"messages": [turn, assistant], "action": action, "call_id": call_id}]
        return action

    def _feedback(self, action, context, events):
        execution = getattr(self.adapter, "last_execution", None) or {}
        matching = execution.get("invocation_id") == action.invocation_id
        feedback = {"raw_return": {key: execution[key] for key in ("ok", "error_code", "message", "advice")
                                   if matching and key in execution},
                    "invocation_id": action.invocation_id}
        if matching:
            feedback["attempt_id"] = execution.get("attempt_id")
        if self.mode != "base":
            receipt = next((event["payload"]["receipt"] for event in reversed(events)
                            if event["kind"] == "receipt"
                            and event["payload"]["receipt"]["action"]["invocation_id"] == action.invocation_id
                            and matching and event["payload"]["receipt"]["attempt_id"] == execution.get("attempt_id")), None)
            rejection = next((item for item in context.data.get("diagnostics", [])
                              if item.get("stage") == "admission"
                              and item.get("invocation_id") == action.invocation_id), None)
            state = receipt["state"] if receipt else "rejected" if rejection else "not_confirmed"
            feedback.update(effect_status=state, committed=state == "committed", next_hint="continue" if state == "committed" else "adapt")
            if receipt:
                feedback["receipt_id"] = receipt["receipt_id"]
            if rejection:
                feedback["rejection_code"] = rejection["code"]
            feedback["diagnostics"] = json_data(context.data.get("diagnostics", []))
        return feedback

    def _related_evidence(self, events, history, current):
        # ponytail: scan the episode log (at most 60 planning rounds); index by receipt if episodes grow.
        observations = {event["payload"]["observation_id"]: event["payload"].get("images", [])[:1]
                        for event in events if event["kind"] == "observation"}
        receipts = [event["payload"]["receipt"] for event in events if event["kind"] == "receipt"]
        subject = getattr(self.adapter, "bound_subject", None) or self.adapter.target_id
        recent_messages = [message for group in history for message in group["messages"]] + [current]
        recent_paths = {item["path"] for message in recent_messages if isinstance(message.get("content"), list)
                        for item in message["content"] if item["type"] == "image"}
        selected = []
        for receipt in reversed(receipts):
            action = receipt["action"]
            if subject not in action["entity_ids"] or action["skill"] not in ("push", "pull", "grasp"):
                continue
            for ref in receipt["evidence_refs"]:
                for path in observations.get(ref, []):
                    if path not in recent_paths:
                        selected.append({"role": "user", "content": [
                            {"type": "text", "text": "Receipt-linked historical evidence: " + receipt["receipt_id"] +
                             "; state: " + receipt["state"] + "; observation: " + ref},
                            {"type": "image", "path": path}]})
                        recent_paths.add(path)
                        if len(selected) == 2:
                            return selected
        return selected
