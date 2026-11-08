"""Trusted task declarations and frozen task-level semantic questions."""

import json
from dataclasses import dataclass
from pathlib import Path
import random
import re
import time
from urllib.request import Request, urlopen

from integrations._semantic import write_json


@dataclass(frozen=True)
class PairSpec:
    task_id: int
    initial_state: int
    seed: int
    fault: str


def choose_pair(task_ids, initial_state_counts, seed=None, initial_state=None, fault="none", rng=None):
    rng = rng or random.SystemRandom()
    ids = tuple(task_ids)
    if (not ids or any(type(i) is not int or not 0 <= i <= 9 for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("task ids must be unique integers from 0 to 9")
    if fault not in ("none", "stale_chunk", "dispatch_timeout_retry", "action_dropout"):
        raise ValueError("unknown fault")
    task_id = rng.choice(ids)
    count = initial_state_counts.get(task_id)
    if type(count) is not int or count < 1:
        raise ValueError("selected task must have initial states")
    if initial_state is None:
        initial_state = rng.randrange(count)
    if type(initial_state) is not int or not 0 <= initial_state < count:
        raise ValueError("initial state is outside the selected task")
    if seed is None:
        seed = rng.randrange(2**32)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be a 32-bit unsigned integer")
    return PairSpec(task_id, initial_state, seed, fault)


def entity_ids_from_bddl(path):
    text = re.sub(r";[^\n]*", "", Path(path).read_text(encoding="utf-8"))

    def section(name):
        match = re.search(r"\(:" + name + r"\s+([^()]*)\)", text, re.I)
        if not match:
            raise ValueError(f"missing or malformed BDDL {name} declaration")
        return match.group(1).split()

    declared = set()
    for name in ("objects", "fixtures"):
        tokens = section(name)
        names = []
        while tokens:
            token = tokens.pop(0)
            if token == "-":
                if not names or not tokens:
                    raise ValueError(f"malformed BDDL {name} declaration")
                declared.update(names)
                names = []
                tokens.pop(0)
            else:
                names.append(token)
        if names:
            raise ValueError(f"malformed BDDL {name} declaration")
    entities = tuple(section("obj_of_interest"))
    if not entities or len(set(entities)) != len(entities) or not set(entities) <= declared:
        raise ValueError("BDDL objects of interest must be unique declared entities")
    return entities


def generate_question(client, task_description, entity_ids):
    if not isinstance(task_description, str) or not task_description.strip():
        raise ValueError("task description is required")
    entities = tuple(entity_ids)
    if not entities or any(not isinstance(x, str) or not x.strip() for x in entities) or len(set(entities)) != len(entities):
        raise ValueError("unique entity ids are required")
    body = {"model": client.model, "temperature": client.temperature, "max_tokens": client.max_tokens,
            "messages": [{"role": "system", "content": "Create one visual task-completion question. Do not add requirements or facts. Return only JSON."},
                         {"role": "user", "content": json.dumps({"task": task_description, "allowed_entity_ids": entities,
                          "response_schema": {"question": "question containing the exact task text", "entity_ids": ["relevant allowed ids"]}})}]}
    client.calls += 1
    prefix = client.output / f"{client.calls:04d}-question"
    write_json(prefix.with_suffix(".request.json"), body, client.secret)
    started = time.monotonic()
    try:
        request = Request(client.endpoint, data=json.dumps(body, allow_nan=False).encode(),
                          headers={"Content-Type": "application/json", "Authorization": "Bearer " + client.secret}, method="POST")
        with urlopen(request, timeout=client.timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
        write_json(prefix.with_suffix(".response.json"), {"elapsed_seconds": time.monotonic() - started, "response": value}, client.secret)
        text = value["choices"][0]["message"]["content"].strip()
        if text.startswith("```") and text.endswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(text)
        if (not isinstance(result, dict) or set(result) != {"question", "entity_ids"}
                or not isinstance(result["question"], str) or task_description not in result["question"]
                or not isinstance(result["entity_ids"], list) or not result["entity_ids"]
                or any(not isinstance(x, str) for x in result["entity_ids"])
                or len(set(result["entity_ids"])) != len(result["entity_ids"])
                or not set(result["entity_ids"]) <= set(entities)):
            raise ValueError("invalid task question or unknown entity ids")
        return result["question"]
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}".replace(client.secret, "[REDACTED]")
        write_json(prefix.with_suffix(".error.json"), {"error": reason, "elapsed_seconds": time.monotonic() - started})
        raise RuntimeError(reason) from None
