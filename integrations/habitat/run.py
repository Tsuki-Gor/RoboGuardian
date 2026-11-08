"""Run, inspect, resume or cancel a single native LaViRA evaluation."""

import argparse
import gzip
import json
import math
from pathlib import Path
from uuid import uuid4

from roboguardian import Budget, Guardian, Memory, ReceiptState, Signal, Verifier
from roboguardian.types import json_data
from .adapter import HabitatAdapter, _skill_specs
from .native import read_env
from .planner import HabitatPlanner
from .process import HabitatProcess
from .tasks import write_exact_subset

NATIVE_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "LaViRA"


def _write_json(path, value):
    Path(path).write_text(json.dumps(json_data(value), ensure_ascii=False, indent=2, allow_nan=False),
                          encoding="utf-8")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("run", help="start a new macro evaluation")
    for name in ("worker-python", "assets-root", "dataset", "ground-truth", "scene-root", "env-file", "hf-home"):
        launch.add_argument("--" + name, type=Path, required=True)
    launch.add_argument("--episode-id", action="append", required=True)
    launch.add_argument("--gpu", type=int, default=0)
    launch.add_argument("--seed", type=int, default=0)
    launch.add_argument("--max-native-steps", type=int, default=300)
    launch.add_argument("--budget-seconds", type=float, default=14400)
    launch.add_argument("--model-timeout", type=float, default=120)
    launch.add_argument("--retry-interval", type=float, default=1)
    launch.add_argument("--la-model", default="openai/gpt-4o-2024-11-20")
    launch.add_argument("--va-model", default="qwen/qwen3.5-27b")
    launch.add_argument("--no-video", dest="record_video", action="store_false")
    status = commands.add_parser("status", help="query without restarting or changing the worker")
    resume = commands.add_parser("resume", help="continue the same paused job, or reconcile its result")
    resume.add_argument("--budget-seconds", type=float, default=14400)
    cancel = commands.add_parser("cancel", help="cancel the original job and release its resources")
    for command in (launch, status, resume, cancel):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--request-timeout", type=float, default=10)
    return result


def _positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")


def _prepare(args):
    for value, name in ((args.budget_seconds, "budget"), (args.model_timeout, "model timeout"),
                        (args.retry_interval, "retry interval")):
        _positive(value, name)
    if args.gpu < 0 or args.seed < 0 or not 1 <= args.max_native_steps <= 300:
        raise ValueError("GPU, seed or native step budget is invalid")
    for name in ("worker_python", "dataset", "ground_truth", "env_file"):
        if not getattr(args, name).is_file():
            raise FileNotFoundError(name + ": " + str(getattr(args, name)))
    if not (NATIVE_ROOT / "vlnce_baselines/lavira_main.py").is_file():
        raise FileNotFoundError("initialize third_party/LaViRA with git submodule update --init")
    values = read_env(args.env_file)
    if values["LA_MODEL_NAME"] != args.la_model or values["VA_MODEL_NAME"] != args.va_model:
        raise ValueError("explicit model names must match the environment file")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("output directory must be empty: " + str(output))
    selected = write_exact_subset(args.dataset, args.episode_id, output_path=output / "episodes.json.gz")
    with gzip.open(output / "episodes.json.gz", "rt", encoding="utf-8") as stream:
        if any(type(item["episode_id"]) is not int for item in json.load(stream)["episodes"]):
            raise ValueError("native R2R requires integer episode_id values in the dataset")
    with gzip.open(args.ground_truth, "rt", encoding="utf-8") as stream:
        truth = json.load(stream)
    for episode in selected:
        if episode["episode_id"] not in truth:
            raise ValueError("ground truth missing episode " + episode["episode_id"])
        scene_id = episode["scene_id"].removeprefix("data/scene_datasets/")
        scene = args.scene_root.resolve() / scene_id
        if not scene.is_file():
            raise FileNotFoundError("scene asset missing: " + str(scene))
    config = {
        "job_id": uuid4().hex, "episode_ids": [item["episode_id"] for item in selected],
        "episodes": selected, "native_root": str(NATIVE_ROOT), "output_dir": str(output),
        "worker_python": str(args.worker_python.resolve()), "assets_root": str(args.assets_root.resolve()),
        "dataset_path": str(output / "episodes.json.gz"),
        "ground_truth_path": str(args.ground_truth.resolve()), "scene_root": str(args.scene_root.resolve()),
        "env_file": str(args.env_file.resolve()), "hf_home": str(args.hf_home.resolve()),
        "gpu": args.gpu, "seed": args.seed, "max_native_steps": args.max_native_steps,
        "total_budget_seconds": args.budget_seconds, "model_timeout": args.model_timeout,
        "retry_interval": args.retry_interval, "record_video": args.record_video,
        "la_model": args.la_model, "va_model": args.va_model,
    }
    _write_json(output / "config.json", config)
    _write_json(output / "manifest.json", {
        "scope": "native_navigation_evaluation", "episodes": selected,
        "models": {"LA": {"name": args.la_model, "temperature": .7, "max_tokens": 8192},
                   "VA": {"name": args.va_model, "temperature": 0, "max_tokens": 4096, "reasoning": "disabled"}},
        "configuration": config, "receipt_certifies": "evaluation_complete, not navigation success",
    })
    return config


def _drive(config, pipe, command, budget_seconds=None):
    output = Path(config["output_dir"])
    memory = Memory(output / "guardian.sqlite", config["job_id"])
    previous = memory.latest_observation()
    adapter = HabitatAdapter(pipe, config["job_id"], config["episode_ids"],
                             sequence=previous.sequence if previous else 0)
    pending = memory.pending_attempt()
    if pending:
        adapter._attempts[pending["action"].invocation_id] = pending["attempt_id"]
    # Each explicit resume authorizes one more reconciliation, not another physical attempt.
    budget = Budget(max_observations=(pending["observations"] + 1 if pending else 1),
                    max_recovery_dispatches=0, max_plan_steps=1, context_bytes=65536, max_images=0)
    result, step = None, None
    try:
        with Guardian(memory, _skill_specs(adapter), adapter, adapter,
                      Verifier(lambda q, o: Signal.INDETERMINATE), budget) as guardian:
            if command == "run":
                memory.set_relations(task_constraints=({"goal": "Complete native evaluation for the selected episodes."},))
                before = adapter.observe()
                action = HabitatPlanner(adapter).propose("", memory.context(before, budget, {}))
                step = guardian.execute(action, observation=before)
            else:
                state = pipe.request("status")
                if pending:
                    action = pending["action"]
                    if (state["invocation_id"], state["attempt_id"]) != (action.invocation_id, pending["attempt_id"]):
                        raise ValueError("saved Guardian attempt and worker identity do not match")
                if command == "resume":
                    if pending is None:
                        raise ValueError("no pending Guardian attempt to resume")
                    if state["state"] == "paused":
                        pipe.request("resume", budget_seconds=budget_seconds)
                    memory.record_diagnostic("manual_reconciliation", {"budget_seconds": budget_seconds})
                elif command == "cancel" and state["state"] in ("running", "paused"):
                    pipe.request("cancel")
                    pipe.wait(include_paused=True)
                if pending:
                    step = guardian.reconcile(pending["action"].invocation_id)
            try:
                state = pipe.request("status")
            except Exception as exc:
                state = {"state": "unreachable", "error": type(exc).__name__}
            receipt = step.receipt if step else None
            result = {"worker_state": state["state"],
                      "evaluation_complete": bool(receipt and receipt.state is ReceiptState.COMMITTED),
                      "navigation_results": state.get("metrics"), "governance": step,
                      "invocation_id": state.get("invocation_id"), "attempt_id": state.get("attempt_id")}
            _write_json(output / "result.json", result)
            _write_json(output / "audit.json", {"episode_id": memory.episode_id, "events": memory.events()})
            if (state["state"] == "ready" or
                    state["state"] in ("completed", "failed", "cancelled")
                    and receipt is not None and receipt.state is not ReceiptState.UNKNOWN):
                try:
                    pipe.close()
                except Exception:
                    pass  # Evaluation and its persisted result are already terminal.
        return json_data(result)
    finally:
        memory.close()


def run(args):
    _positive(args.request_timeout, "request timeout")
    if args.command == "run":
        config = _prepare(args)
        pipe = HabitatProcess(args.output, python=config["worker_python"],
                              config=args.output.resolve() / "config.json", request_timeout=args.request_timeout)
        return _drive(config, pipe, "run")
    if args.command == "resume":
        _positive(args.budget_seconds, "additional budget")
    output = args.output.resolve()
    config = json.loads((output / "config.json").read_text())
    if Path(config["output_dir"]).resolve() != output:
        raise ValueError("configuration belongs to another output directory")
    pipe = HabitatProcess(output, request_timeout=args.request_timeout)
    if args.command == "status":
        try:
            return {"source": "live_worker", **pipe.request("status")}
        except Exception as exc:
            saved = output / "result.json"
            return {"source": "saved_result", "connection_error": type(exc).__name__,
                    "result": json.loads(saved.read_text()) if saved.exists() else None}
    return _drive(config, pipe, args.command, getattr(args, "budget_seconds", None))


def main(argv=None):
    result = run(parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("evaluation_complete") or result.get("source") else 2


if __name__ == "__main__":
    raise SystemExit(main())
