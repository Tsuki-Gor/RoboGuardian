"""SQLite commit gate, complete records, and episode ownership.

Disk-store Guardian leases use POSIX ``flock`` on a sibling lock file. They
cover a Guardian's lifetime, including external runtime calls, without holding
a SQLite transaction open. This is a local-filesystem, serial-episode contract;
it is not a distributed lock or a physical exactly-once guarantee.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import uuid

from .types import (
    Action, Assessment, Budget, Context, EffectReceipt, EvidenceReport, Judgment,
    Observation, ReceiptState, RuntimeResult, action_fingerprint, canonical_json,
    json_data,
)

from .context import ContextBudgetError, build_context


class InvocationConflict(ValueError):
    """An invocation or recovery identity was reused inconsistently."""


class EpisodeBusy(RuntimeError):
    """Another Guardian, operation, or unresolved dispatch owns this episode."""


class BudgetExceeded(RuntimeError):
    """A persisted observation, planning, or recovery budget is exhausted."""


def _receipt(data):
    if data is None:
        return None
    data = json.loads(data)
    data["action"] = Action(**data["action"])
    data["state"] = ReceiptState(data["state"])
    assessment = data["assessment"]
    assessment["judgment"] = Judgment(assessment["judgment"])
    data["assessment"] = Assessment(**assessment)
    data["runtime_result"] = RuntimeResult(**data["runtime_result"])
    return EffectReceipt(**data)


def _limit(value):
    if type(value) is not int or value < 0:
        raise ValueError("budget limit must be a nonnegative integer")


def _relations(values):
    if not isinstance(values, (list, tuple)) or any(not isinstance(item, Mapping) for item in values):
        raise TypeError("relations must be a sequence of JSON mappings")
    return json_data(values)


class Memory:
    """One episode's trusted history, recovery state, and complete audit.

    ``verified_updates`` is a mapping of fact names to values, produced by a
    trusted skill adapter. Optional ``dynamic_relations`` are similarly trusted
    assertions covered by the supplied committed receipt, never inferred from
    action parameters. Read methods return detached values.
    """

    def __init__(self, path, episode_id: str):
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id must be a nonempty string")
        episode_id.encode("utf-8")
        self.path = str(path) if str(path) == ":memory:" else str(Path(path).resolve())
        self.episode_id = episode_id
        self._lease = None
        self._operation = threading.Lock()
        self._db = sqlite3.connect(self.path)
        self._db.row_factory = sqlite3.Row
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode TEXT PRIMARY KEY, version INTEGER NOT NULL DEFAULT 0,
                plan_steps INTEGER NOT NULL DEFAULT 0, active_root TEXT,
                static_relations TEXT NOT NULL DEFAULT '{}');
            CREATE TABLE IF NOT EXISTS attempts (
                episode TEXT NOT NULL, invocation TEXT NOT NULL, attempt TEXT NOT NULL UNIQUE,
                fingerprint TEXT NOT NULL, action TEXT NOT NULL, before_snapshot TEXT NOT NULL,
                root TEXT NOT NULL, observations INTEGER NOT NULL DEFAULT 0,
                runtime TEXT, terminal INTEGER, receipt TEXT, unresolved TEXT,
                PRIMARY KEY (episode, invocation));
            CREATE TABLE IF NOT EXISTS facts (
                episode TEXT NOT NULL, name TEXT NOT NULL, value TEXT NOT NULL,
                receipt TEXT NOT NULL, PRIMARY KEY (episode, name));
            CREATE TABLE IF NOT EXISTS progress (
                episode TEXT NOT NULL, invocation TEXT NOT NULL, goal TEXT NOT NULL,
                receipt TEXT NOT NULL, dynamic_relations TEXT NOT NULL,
                skill_relation TEXT NOT NULL, PRIMARY KEY (episode, invocation));
            CREATE TABLE IF NOT EXISTS observations (
                episode TEXT NOT NULL, observation TEXT NOT NULL, source TEXT NOT NULL,
                sequence INTEGER NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY (episode, observation));
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, episode TEXT NOT NULL,
                kind TEXT NOT NULL, payload TEXT NOT NULL);
        """)
        with self._db:
            self._db.execute("INSERT OR IGNORE INTO episodes(episode) VALUES (?)", (episode_id,))

    def _event(self, kind, payload):
        self._db.execute("INSERT INTO events(episode,kind,payload) VALUES (?,?,?)",
                         (self.episode_id, kind, canonical_json(payload)))

    @property
    def state_version(self) -> int:
        return self._db.execute("SELECT version FROM episodes WHERE episode=?",
                                (self.episode_id,)).fetchone()[0]

    @property
    def plan_steps(self) -> int:
        return self._db.execute("SELECT plan_steps FROM episodes WHERE episode=?",
                                (self.episode_id,)).fetchone()[0]

    @property
    def facts(self) -> dict:
        return {row["name"]: json.loads(row["value"]) for row in self._db.execute(
            "SELECT name,value FROM facts WHERE episode=? ORDER BY name", (self.episode_id,))}

    @property
    def progress(self) -> list:
        return [{"invocation_id": row["invocation"], "goal_id": row["goal"],
                 "receipt_id": row["receipt"]} for row in self._db.execute(
                     "SELECT * FROM progress WHERE episode=? ORDER BY rowid", (self.episode_id,))]

    @property
    def relations(self) -> dict:
        static = json.loads(self._db.execute("SELECT static_relations FROM episodes WHERE episode=?",
                                            (self.episode_id,)).fetchone()[0])
        result = {name: static.get(name, []) for name in ("objects", "skills", "task_constraints")}
        result["dynamic_objects"] = []
        result["skill_history"] = []
        for row in self._db.execute("SELECT * FROM progress WHERE episode=? ORDER BY rowid", (self.episode_id,)):
            result["dynamic_objects"].extend(json.loads(row["dynamic_relations"]))
            result["skill_history"].append(json.loads(row["skill_relation"]))
        return result

    def acquire_lease(self) -> None:
        if self._lease is not None:
            raise EpisodeBusy("this Memory already has an active Guardian")
        if self.path == ":memory:":
            self._lease = True
            return
        import fcntl  # POSIX only; no silent fallback to an unsafe disk lease.
        digest = hashlib.sha256(self.episode_id.encode("utf-8")).hexdigest()
        handle = open(self.path + "." + digest + ".lock", "a+b")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise EpisodeBusy("episode already has an active Guardian") from exc
        except BaseException:
            handle.close()
            raise
        self._lease = handle

    def release_lease(self) -> None:
        if self._operation.locked():
            raise EpisodeBusy("cannot release lease during an active operation")
        if self._lease not in (None, True):
            self._lease.close()
        self._lease = None

    @contextmanager
    def operation(self):
        if self._lease is None:
            raise EpisodeBusy("Guardian must acquire the episode lease first")
        if not self._operation.acquire(blocking=False):
            raise EpisodeBusy("Guardian operation is not reentrant")
        try:
            yield
        finally:
            self._operation.release()

    def _attempt(self, row):
        if row is None:
            return None
        count = self._db.execute("SELECT COUNT(*) FROM attempts WHERE episode=? AND root=?",
                                 (self.episode_id, row["root"])).fetchone()[0]
        return {"action": Action(**json.loads(row["action"])),
                "before": Observation(**json.loads(row["before_snapshot"])),
                "attempt_id": row["attempt"], "recovery_root": row["root"],
                "observations": row["observations"], "recovery_dispatches": count - 1,
                "runtime_result": RuntimeResult(**json.loads(row["runtime"])) if row["runtime"] else None,
                "runtime_terminal": bool(row["terminal"]) if row["terminal"] is not None else None,
                "receipt": _receipt(row["receipt"]), "unresolved": row["unresolved"]}

    def get_invocation(self, invocation_id: str) -> dict | None:
        return self._attempt(self._db.execute("SELECT * FROM attempts WHERE episode=? AND invocation=?",
                                             (self.episode_id, invocation_id)).fetchone())

    def pending_attempt(self) -> dict | None:
        for row in self._db.execute("SELECT * FROM attempts WHERE episode=? ORDER BY rowid", (self.episode_id,)):
            receipt = _receipt(row["receipt"])
            if row["terminal"] != 1 or receipt is None or receipt.state is ReceiptState.UNKNOWN:
                return self._attempt(row)
        return None

    def active_recovery(self) -> dict | None:
        return self._attempt(self._db.execute(
            "SELECT * FROM attempts WHERE episode=? AND root=(SELECT active_root FROM episodes "
            "WHERE episode=?) ORDER BY rowid DESC LIMIT 1", (self.episode_id, self.episode_id)).fetchone())

    def begin_attempt(self, action: Action, before: Observation, *, recovery_root=None,
                      max_recovery_dispatches=None) -> str:
        fingerprint = action_fingerprint(action)
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            existing = self._db.execute("SELECT * FROM attempts WHERE episode=? AND invocation=?",
                                        (self.episode_id, action.invocation_id)).fetchone()
            if existing:
                if existing["fingerprint"] != fingerprint:
                    raise InvocationConflict("invocation ID has different action payload or configuration")
                return existing["attempt"]
            if self.pending_attempt() is not None:
                raise EpisodeBusy("an existing dispatch still needs reconciliation or terminal confirmation")
            active = self.active_recovery()
            root = active["recovery_root"] if active else action.invocation_id
            if recovery_root is not None and (active is None or recovery_root != root):
                raise InvocationConflict("recovery root must be the core-managed active root")
            if active:
                if max_recovery_dispatches is None:
                    raise BudgetExceeded("recovery requires an explicit dispatch budget")
                _limit(max_recovery_dispatches)
                if active["recovery_dispatches"] >= max_recovery_dispatches:
                    raise BudgetExceeded("recovery dispatch budget exhausted")
            if action.expected_state_version != self.state_version:
                raise InvocationConflict("action expected state version is outdated")
            if action.based_on_observation != before.observation_id:
                raise ValueError("action is not based on this before observation")
            for row in self._db.execute("SELECT before_snapshot FROM attempts WHERE episode=?", (self.episode_id,)):
                prior = json.loads(row[0])
                if (prior["observation_id"] == before.observation_id or
                        (prior["source_id"] == before.source_id and prior["sequence"] >= before.sequence)):
                    raise ValueError("each physical attempt requires a fresh unused before observation")
            self._record_observation(before)
            latest = self.latest_observation()
            if latest is not None and latest != before:
                raise ValueError("before must be the latest accepted observation")
            attempt_id = uuid.uuid4().hex
            self._db.execute("INSERT INTO attempts(episode,invocation,attempt,fingerprint,action,before_snapshot,root) "
                             "VALUES (?,?,?,?,?,?,?)", (self.episode_id, action.invocation_id, attempt_id,
                                                       fingerprint, canonical_json(action), canonical_json(before), root))
            self._db.execute("UPDATE episodes SET active_root=? WHERE episode=?", (root, self.episode_id))
            self._event("dispatch_intent", {"action": action, "before": before, "attempt_id": attempt_id,
                                             "recovery_root": root})
        return attempt_id

    def _record_observation(self, observation):
        payload = canonical_json(observation)
        previous = self._db.execute("SELECT payload FROM observations WHERE episode=? AND observation=?",
                                    (self.episode_id, observation.observation_id)).fetchone()
        if previous:
            if previous[0] != payload:
                raise ValueError("observation ID has different payload")
            return
        maximum = self._db.execute("SELECT MAX(sequence) FROM observations WHERE episode=? AND source=?",
                                   (self.episode_id, observation.source_id)).fetchone()[0]
        if maximum is not None and observation.sequence <= maximum:
            raise ValueError("observation sequence must strictly increase within its source")
        self._db.execute("INSERT INTO observations VALUES (?,?,?,?,?)",
                         (self.episode_id, observation.observation_id, observation.source_id, observation.sequence, payload))
        self._event("observation", observation)

    def record_observation(self, observation: Observation) -> None:
        with self._db:
            self._record_observation(observation)

    def latest_observation(self) -> Observation | None:
        row = self._db.execute("SELECT payload FROM observations WHERE episode=? ORDER BY rowid DESC LIMIT 1",
                               (self.episode_id,)).fetchone()
        return Observation(**json.loads(row[0])) if row else None

    def consume_observation(self, invocation_id: str, limit: int) -> int:
        _limit(limit)
        with self._db:
            changed = self._db.execute("UPDATE attempts SET observations=observations+1 "
                                       "WHERE episode=? AND invocation=? AND observations<?",
                                       (self.episode_id, invocation_id, limit)).rowcount
            if not changed:
                if self.get_invocation(invocation_id) is None:
                    raise KeyError(invocation_id)
                raise BudgetExceeded("observation budget exhausted")
            count = self.get_invocation(invocation_id)["observations"]
            self._event("observation_budget", {"invocation_id": invocation_id, "used": count, "limit": limit})
        return count

    def consume_plan_step(self, limit: int) -> int:
        _limit(limit)
        with self._db:
            changed = self._db.execute("UPDATE episodes SET plan_steps=plan_steps+1 WHERE episode=? AND plan_steps<?",
                                       (self.episode_id, limit)).rowcount
            if not changed:
                raise BudgetExceeded("planning step budget exhausted")
            count = self.plan_steps
            self._event("planning_budget", {"used": count, "limit": limit})
        return count

    def set_runtime_result(self, attempt_id: str, result: RuntimeResult) -> None:
        row = self._db.execute("SELECT * FROM attempts WHERE episode=? AND attempt=?",
                               (self.episode_id, attempt_id)).fetchone()
        if row is None or result.attempt_id != attempt_id or result.invocation_id != row["invocation"]:
            raise InvocationConflict("runtime result does not belong to this attempt")
        with self._db:
            # A confirmed terminal attempt cannot become running through a stale status.
            terminal = 1 if row["terminal"] == 1 else result.terminal
            self._db.execute("UPDATE attempts SET runtime=?,terminal=? WHERE episode=? AND attempt=?",
                             (canonical_json(result), terminal, self.episode_id, attempt_id))
            self._event("runtime_status", result)

    def set_unresolved(self, invocation_id: str, reason: str | None) -> None:
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise ValueError("unresolved reason must be nonempty text or None")
        with self._db:
            changed = self._db.execute("UPDATE attempts SET unresolved=? WHERE episode=? AND invocation=?",
                                       (reason, self.episode_id, invocation_id)).rowcount
            if not changed:
                raise KeyError(invocation_id)
            self._event("recovery_status", {"invocation_id": invocation_id, "unresolved": reason})

    def record_receipt(self, action: Action, attempt_id: str, receipt: EffectReceipt,
                       report: EvidenceReport, verified_updates: Mapping, *, dynamic_relations=()) -> None:
        if not isinstance(verified_updates, Mapping):
            raise TypeError("verified_updates must be a mapping of fact names to JSON values")
        updates = json_data(verified_updates)
        relations = _relations(dynamic_relations)
        payload = {"receipt": receipt, "report": report, "verified_updates": updates,
                   "dynamic_relations": relations}
        canonical_json(payload)
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            row = self._db.execute("SELECT * FROM attempts WHERE episode=? AND invocation=?",
                                   (self.episode_id, action.invocation_id)).fetchone()
            if (row is None or row["attempt"] != attempt_id or row["fingerprint"] != action_fingerprint(action)
                    or action_fingerprint(receipt.action) != row["fingerprint"] or receipt.attempt_id != attempt_id
                    or report.invocation_id != action.invocation_id or report.attempt_id != attempt_id):
                raise InvocationConflict("receipt/report does not belong to the persisted action and attempt")
            original_before = json.loads(row["before_snapshot"])["observation_id"]
            if report.before_observation_id not in (None, original_before):
                raise InvocationConflict("reconciliation must retain the original before observation")
            previous = _receipt(row["receipt"])
            if previous is not None and previous.state is ReceiptState.COMMITTED:
                self._event("receipt_replay", payload)
                return
            latest = self._db.execute("SELECT invocation FROM attempts WHERE episode=? AND root=? ORDER BY rowid DESC LIMIT 1",
                                      (self.episode_id, row["root"])).fetchone()[0]
            if latest != action.invocation_id:
                self._event("superseded_receipt", payload)
                return
            terminal = 1 if row["terminal"] == 1 else receipt.runtime_result.terminal
            unresolved = row["unresolved"] if receipt.state is ReceiptState.UNKNOWN else None
            self._db.execute("UPDATE attempts SET receipt=?,runtime=?,terminal=?,unresolved=? "
                             "WHERE episode=? AND invocation=?",
                             (canonical_json(receipt), canonical_json(receipt.runtime_result), terminal, unresolved,
                              self.episode_id, action.invocation_id))
            if receipt.state is ReceiptState.COMMITTED:
                for name, value in updates.items():
                    self._db.execute("INSERT INTO facts VALUES (?,?,?,?) ON CONFLICT(episode,name) "
                                     "DO UPDATE SET value=excluded.value,receipt=excluded.receipt",
                                     (self.episode_id, name, canonical_json(value), receipt.receipt_id))
                objects = [{"relation": relation, "source": "committed", "scope": "historical",
                            "receipt_id": receipt.receipt_id} for relation in relations]
                skill = {"skill": action.skill, "embodiment": action.embodiment,
                         "config_version": action.skill_config_version, "goal_id": action.goal_id,
                         "entity_ids": action.entity_ids, "source": "receipt_provenance",
                         "receipt_id": receipt.receipt_id}
                self._db.execute("INSERT INTO progress VALUES (?,?,?,?,?,?)",
                                 (self.episode_id, action.invocation_id, action.goal_id, receipt.receipt_id,
                                  canonical_json(objects), canonical_json(skill)))
                self._db.execute("UPDATE episodes SET version=version+1,active_root=NULL WHERE episode=?",
                                 (self.episode_id,))
            self._event("receipt", payload)

    def record_diagnostic(self, kind: str, details: Mapping) -> None:
        if not isinstance(kind, str) or not kind.strip() or not isinstance(details, Mapping):
            raise ValueError("diagnostic requires a kind and a details mapping")
        with self._db:
            self._event(kind, {"scope": "diagnostic", "details": details})

    def set_relations(self, *, object_relations=(), skill_relations=(), task_constraints=()) -> None:
        data = {name: [{"relation": item, "source": "static_configuration",
                       "scope": "unverified_configuration" if name == "objects" else "configuration"}
                      for item in _relations(values)]
                for name, values in (("objects", object_relations), ("skills", skill_relations),
                                     ("task_constraints", task_constraints))}
        payload = canonical_json(data)
        with self._db:
            self._db.execute("UPDATE episodes SET static_relations=? WHERE episode=?", (payload, self.episode_id))
            self._event("static_relations", data)

    def record_validity(self, current: Observation, required_facts: Mapping) -> dict:
        if not isinstance(required_facts, Mapping):
            raise TypeError("required_facts must be a mapping")
        validity = {}
        for name, expected in json_data(required_facts).items():
            present = name in current.facts
            status = ("match" if canonical_json(current.facts[name]) == canonical_json(expected) else "conflict") if present else "missing"
            validity[name] = {"status": status, "source": "observation_assertion",
                              "observation_id": current.observation_id}
            if present:
                validity[name]["observed"] = json_data(current.facts[name])
        with self._db:
            self._event("current_validity", {"observation_id": current.observation_id, "facts": validity})
        return validity

    def _fact_records(self):
        return [{"name": row["name"], "value": json.loads(row["value"]), "receipt_id": row["receipt"]}
                for row in self._db.execute("SELECT * FROM facts WHERE episode=? ORDER BY name", (self.episode_id,))]

    def _latest_decision(self):
        row = self._db.execute(
            "SELECT kind,payload FROM events WHERE episode=? "
            "AND kind IN ('admission_rejected','dispatch_intent') ORDER BY id DESC LIMIT 1",
            (self.episode_id,)).fetchone()
        return {"kind": row["kind"], "payload": json.loads(row["payload"])} if row is not None else None

    def _receipt_report(self, receipt_id):
        for row in self._db.execute("SELECT payload FROM events WHERE episode=? AND kind='receipt' ORDER BY id DESC",
                                    (self.episode_id,)):
            payload = json.loads(row[0])
            if payload["receipt"]["receipt_id"] == receipt_id:
                return payload["report"]
        return None

    def _receipts(self):
        for row in self._db.execute("SELECT receipt FROM attempts WHERE episode=? AND receipt IS NOT NULL ORDER BY rowid DESC",
                                    (self.episode_id,)):
            yield _receipt(row[0])

    def context(self, current: Observation, budget: Budget, required_facts: Mapping) -> Context:
        """Build the bounded planner projection without trimming stored history."""
        return build_context(self, current, budget, required_facts)

    def events(self) -> list:
        return [{"id": row["id"], "kind": row["kind"], "payload": json.loads(row["payload"])}
                for row in self._db.execute("SELECT * FROM events WHERE episode=? ORDER BY id", (self.episode_id,))]

    def close(self) -> None:
        self.release_lease()
        self._db.close()
