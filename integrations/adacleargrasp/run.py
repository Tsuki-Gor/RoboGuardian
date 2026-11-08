"""Run the self-owned Ada agent loop against pinned native sources.

Optional native/model dependencies are imported only by this integration.
The core package stays independent of them. No native planner or baseline code
is imported; model proposals pass through Guardian before native skill calls."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

from roboguardian import Budget, Guardian, Memory, Signal, Verifier
from roboguardian.types import canonical_json, json_data

from .adapter import final_native_summary, AdaAdapter
from .planner import write_json, ChatModel, AdaPlanner


@contextmanager
def pci_renderer(render_device):
    """Scope the installed ManiSkill PCI-parser compatibility fix to gym.make."""
    if not render_device.startswith("pci:"):
        yield
        return
    native = importlib.import_module("mani_skill.envs.sapien_env")
    sapien = importlib.import_module("sapien")
    original = native.parse_sim_and_render_backend
    def parse(sim_backend, render_backend):
        if render_backend != render_device:
            return original(sim_backend, render_backend)
        backend = original(sim_backend, "none")
        backend.render_device = sapien.Device(render_device)
        backend.render_backend = "sapien_cuda"
        return backend
    native.parse_sim_and_render_backend = parse
    try:
        yield
    finally:
        native.parse_sim_and_render_backend = original


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--assets-root", type=Path, required=True, help="directory containing scenes/ and models/")
    result.add_argument("--env-file", type=Path, help="optional explicit credentials file; environment takes precedence")
    result.add_argument("--model", default=os.environ.get("ROBOGUARDIAN_MODEL", "qwen/qwen3.5-27b"))
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--scene", action="append", help="category/clutter/id, or an explicit scene JSON path; repeat for a batch")
    result.add_argument("--episodes", type=int, default=1)
    result.add_argument("--seed", type=int, help="first episode seed; otherwise selected at the start of this experiment")
    result.add_argument("--modes", nargs="+", choices=("base", "roboguardian-window", "roboguardian"), default=["roboguardian"])
    result.add_argument("--fault", choices=("none", "false_success", "response_loss"), default="none")
    result.add_argument("--checkpoint", type=Path, help="default: assets-root/models/ppo/PickClutterYCB-XArm7-v1/ppo_grasp.zip")
    result.add_argument("--render-device", required=True)
    result.add_argument("--record-video", action="store_true")
    result.add_argument("--instruction", default="Clear obstructing clutter as needed, then grasp and lift the target.")
    result.add_argument("--temperature", type=float, default=0.2)
    result.add_argument("--max-tokens", type=int, default=800)
    result.add_argument("--image-detail", choices=("low", "high", "auto"), default="low")
    result.add_argument("--request-timeout", type=float, default=120)
    result.add_argument("--skill-timeout", type=float, default=120, help="native HTTP request timeout in seconds")
    result.add_argument("--response-delay", type=float, help="response-loss delivery delay; must exceed skill-timeout")
    result.add_argument("--initarm-tolerance", type=float, default=0.05)
    result.add_argument("--height-threshold", type=float, default=0.15)
    result.add_argument("--relation-threshold", type=float, default=0.18)
    result.add_argument("--min-contacts", type=int, default=2)
    result.add_argument("--max-plan-steps", type=int, default=60)
    result.add_argument("--max-observations", type=int, default=2)
    result.add_argument("--max-recovery-dispatches", type=int, default=8)
    result.add_argument("--context-bytes", type=int, default=24000)
    return result


NATIVE_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "AdaClearGrasp"
NATIVE_REVISION = "2d9561a0995c38cd3305fce707ebf7a9090252e0"
CATEGORIES = ("apple", "ball", "can", "cube", "lego", "mug", "pear")
MODES = ("base", "roboguardian-window", "roboguardian")


def select_pairs(args):
    """Choose once, then give every mode the same scene, seed and fault."""
    import random
    assets = args.assets_root.expanduser().resolve()
    if args.episodes < 1 or not args.modes or len(set(args.modes)) != len(args.modes):
        raise ValueError("episodes must be positive and modes unique")
    if args.seed is not None and not 0 <= args.seed < 2 ** 31:
        raise ValueError("seed must lie in [0, 2**31)")
    rng = random.Random(args.seed)
    scenes = []
    if args.scene:
        for value in args.scene:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = assets / "scenes" / path
            if path.suffix != ".json":
                path = path.with_suffix(".json")
            scenes.append(path.resolve())
    else:
        scenes = [path for category in CATEGORIES for clutter in (2, 4, 6)
                  for path in sorted((assets / "scenes" / category / str(clutter)).glob("*.json"))]
        if not scenes:
            raise FileNotFoundError("no native scenes found under assets-root/scenes")
        scenes = [rng.choice(scenes)]
    pairs = []
    for scene in scenes:
        data = json.loads(scene.read_text())
        if not isinstance(data.get("target"), dict) or not isinstance(data.get("clutter"), list):
            raise ValueError(f"scene must contain target and clutter: {scene}")
        for trial in range(args.episodes):
            seed = args.seed if not pairs and args.seed is not None else rng.randrange(2 ** 31)
            pairs.append({"pair_id": f"pair-{len(pairs):04d}", "scene": str(scene),
                          "scene_definition": data,
                          "target_label": str(data["target"]["model_id"]), "trial": trial,
                          "seed": seed, "fault": args.fault})
    return pairs


def validate_native():
    import subprocess
    if not (NATIVE_ROOT / "env" / "sim" / "pick_clutter_xarm.py").is_file():
        raise RuntimeError("initialize native sources with python scripts/setup_adacleargrasp.py")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=NATIVE_ROOT, text=True).strip()
    if revision != NATIVE_REVISION:
        raise RuntimeError("Ada submodule is not at the pinned revision")
    patch = NATIVE_ROOT.parent / "patches" / "adacleargrasp-push-pose.patch"
    check = subprocess.run(["git", "apply", "--reverse", "--check", str(patch)], cwd=NATIVE_ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if check.returncode:
        raise RuntimeError("apply the recorded push correction with python scripts/setup_adacleargrasp.py")
    # The fixed source is the public commit plus exactly the reviewed native patch.
    expected = patch.read_text().split("diff --git a/tests/", 1)[0]
    actual = subprocess.check_output(["git", "-c", "diff.algorithm=myers", "-c", "diff.indentHeuristic=true",
        "diff", "--no-ext-diff", "--no-textconv", "--no-color", "--unified=3", "--inter-hunk-context=0",
        "--abbrev=7", "--src-prefix=a/", "--dst-prefix=b/", "HEAD", "--"], cwd=NATIVE_ROOT, text=True)
    extra = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=NATIVE_ROOT, text=True).splitlines()
    if actual != expected or set(extra) != {"tests/test_push_pose.py"}:
        raise RuntimeError("Ada native sources differ from the pinned commit and recorded push patch")
    return revision


def semantic_assessor(model):
    # Operation-stage receipts certify a software return, not an object effect.
    def assess(question, observation):
        skill = observation.state.get("visual_reference_before", {}).get("skill")
        if skill in ("initarm", "inithand", "lift", "lower"):
            return Signal.INDETERMINATE
        return model.assess(question, observation)
    return assess


def run_episode(args, pair, mode, output, checkpoint, secret, endpoint):
    from .execution import AdaExecution, BaseController
    from openai import OpenAI
    output.mkdir()
    env = adapter = memory = runtime = model = None
    run_result = None
    summary = {"pair": pair, "mode": mode, "execution_complete": False,
               "guardian_verified_success": False, "native_task_success": None}
    budget = Budget(args.max_observations, args.max_recovery_dispatches, args.max_plan_steps, args.context_bytes, 1)
    config = {"model": args.model, "temperature": args.temperature, "max_tokens": args.max_tokens,
              "image_detail": args.image_detail, "height_threshold": args.height_threshold,
              "initarm_tolerance": args.initarm_tolerance,
              "relation_threshold": args.relation_threshold, "min_contacts": args.min_contacts,
              "budget": json_data(budget), "native_revision": NATIVE_REVISION}
    config_version = "ada-v2-" + hashlib.sha256(canonical_json(config).encode()).hexdigest()[:16]
    manifest = {"configuration": config, "pair": pair, "mode": mode, "checkpoint": str(checkpoint),
                "native_root": str(NATIVE_ROOT), "render_device": args.render_device,
                "skill_timeout": args.skill_timeout, "response_delay": args.response_delay,
                "created_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "manifest.json", manifest)
    previous_cwd = Path.cwd()
    started = time.monotonic()
    try:
        sys.path.insert(0, str(NATIVE_ROOT))
        os.chdir(NATIVE_ROOT)
        with (output / "native.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
            import random
            import numpy as np
            import torch
            import gymnasium as gym
            import env.sim.pick_clutter_xarm  # noqa: F401
            from exec.skills.init import InitSkill
            from exec.skills.grasp import GraspConfig, GraspSkill
            from exec.skills.ee_move import EEMoveSkill
            from exec.skills.clear import PushSkill, PullSkill

            random.seed(pair["seed"])
            np.random.seed(pair["seed"])
            torch.manual_seed(pair["seed"])
            with pci_renderer(args.render_device):
                env = gym.make("PickClutterYCB-XArm7-v1", num_envs=1, mode="plan", control_mode="pd_ee_delta_pose",
                               render_mode="rgb_array", obs_mode="rgb", only_target_object=False, reconfiguration_freq=0,
                               scene_json=pair["scene"], sim_backend="cpu", render_backend=args.render_device)
            if args.record_video:
                from mani_skill.utils.wrappers.record import RecordEpisode
                env = RecordEpisode(env, output_dir=str(output / "videos"), save_trajectory=False,
                                    save_video=True, video_fps=30)
            env.reset(seed=pair["seed"])
            adapter = AdaAdapter(env, InitSkill(env),
                GraspSkill(env, GraspConfig(model_zip=str(checkpoint), device="cpu", sleep=False, interp_sleep=False)),
                output, scene_id=f"{pair['pair_id']}:{Path(pair['scene']).stem}:seed{pair['seed']}",
                target_label=pair["target_label"], config_version=config_version,
                initarm_tolerance=args.initarm_tolerance, height_threshold=args.height_threshold,
                relation_threshold=args.relation_threshold, min_contacts=args.min_contacts,
                mover=EEMoveSkill(env), pusher=PushSkill(env), puller=PullSkill(env))
            model = ChatModel(OpenAI(api_key=secret, base_url=endpoint, timeout=args.request_timeout, max_retries=0),
                args.model, output, temperature=args.temperature, max_tokens=args.max_tokens, image_detail=args.image_detail)
            runtime = AdaExecution(adapter, fault=pair["fault"], timeout=args.skill_timeout, response_delay=args.response_delay)
            memory = Memory(output / "memory.sqlite", pair["pair_id"] + "-" + mode)
            memory.set_relations(task_constraints=({"goal": "grasp and lift target", "physical_rule": "two_of_three",
                "height_m_at_least": args.height_threshold, "tcp_distance_m_at_most": args.relation_threshold,
                "minimum_finger_contacts": args.min_contacts, "semantic_required": True},))
            planner = AdaPlanner(model, adapter, mode=mode, memory=memory)
            def goal_check(observation, context):
                if memory.active_recovery() is not None:
                    return Signal.FAIL
                physical = adapter.grasp_signal(observation)
                if physical is not Signal.PASS:
                    return physical
                return model.assess(f"Is the target {adapter.target_label} currently held by the robot hand and lifted clear of the tabletop?", observation)
            if mode == "base":
                run_result = BaseController(memory, budget, runtime).run(args.instruction, planner)
            else:
                with Guardian(memory, adapter.skills(), runtime, runtime, Verifier(semantic_assessor(model)), budget) as guardian:
                    run_result = guardian.run(args.instruction, planner, goal_check)
            # Join physical calls before reading the final scene or releasing native resources.
            runtime.close()
            final = runtime.observe(None)
            summary["final_physical_judgment"] = adapter.grasp_signal(final).value
            summary["final_semantic_judgment"] = model.assess(
                f"Is the target {adapter.target_label} currently held by the robot hand and lifted clear of the tabletop?", final).value
            summary["final_observation"] = json_data(final)
            summary["execution_complete"] = True
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}".replace(secret, "[REDACTED]")
    finally:
        def persist(name, value):
            try:
                write_json(output / name, value, secret)
            except Exception as exc:
                summary["execution_complete"] = False
                summary["guardian_verified_success"] = False
                summary.setdefault("artifact_errors", {})[name] = type(exc).__name__
        if runtime is not None:
            try:
                runtime.close()
            except Exception as exc:
                summary["execution_complete"] = False
                summary["transport_close_error"] = type(exc).__name__
            persist("execution.json", runtime.events)
            persist("transport.json", runtime.transport.events)
            summary["fault_triggered"] = any(e["event"] == "fault" for e in runtime.events)
        summary.update(final_native_summary(env))
        if adapter is not None:
            dispatches = adapter.dispatches
            counts = {}
            for item in dispatches:
                counts[item["invocation_id"]] = counts.get(item["invocation_id"], 0) + 1
            summary.update(physical_dispatches=len(dispatches), duplicate_dispatches=sum(n - 1 for n in counts.values()),
                           observations=adapter.sequence)
            persist("dispatches.json", dispatches)
        if memory is not None:
            events = memory.events()
            recovery = any(e["kind"] == "observation_budget" or
                (e["kind"] == "dispatch_intent" and e["payload"]["recovery_root"] != e["payload"]["action"]["invocation_id"])
                for e in events)
            recovery = recovery or bool(runtime and any(e["event"] == "status_query" for e in runtime.events))
            summary.update(recovery_triggered=recovery,
                recovery_success=summary["native_task_success"] if recovery else None,
                verified_facts=memory.facts, verified_progress=memory.progress,
                unresolved=memory.pending_attempt() is not None, active_recovery=memory.active_recovery() is not None)
            persist("audit.json", events)
            memory.close()
        summary["run_result"] = json_data(run_result) if run_result is not None else None
        summary["model_calls"] = model.calls if model is not None else 0
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                summary["execution_complete"] = False
                summary["environment_close_error"] = type(exc).__name__
        summary["guardian_verified_success"] = bool(mode != "base" and run_result
            and summary["execution_complete"] and run_result.status == "verified_success"
            and not summary.get("unresolved") and not summary.get("active_recovery")
            and summary.get("final_physical_judgment") == "pass" and summary.get("final_semantic_judgment") == "pass")
        os.chdir(previous_cwd)
        if sys.path and sys.path[0] == str(NATIVE_ROOT):
            sys.path.pop(0)
        summary["elapsed_seconds"] = time.monotonic() - started
        persist("summary.json", summary)
    return summary


def main(argv=None):
    args = parser().parse_args(argv)
    import math
    from dotenv import load_dotenv
    if args.env_file is not None:
        if not args.env_file.expanduser().is_file():
            raise FileNotFoundError("the explicit credentials file does not exist")
        load_dotenv(args.env_file.expanduser(), override=False)
    secret, endpoint = os.environ.get("OPENAI_API_KEY"), os.environ.get("OPENAI_BASE_URL")
    if not secret or not endpoint:
        raise ValueError("OPENAI_API_KEY and OPENAI_BASE_URL are required")
    for value in (args.request_timeout, args.skill_timeout):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("timeouts must be finite and positive")
    if args.response_delay is not None and (not math.isfinite(args.response_delay) or args.response_delay < 0
            or args.fault == "response_loss" and args.response_delay <= args.skill_timeout):
        raise ValueError("response-loss delay must exceed skill-timeout")
    Budget(args.max_observations, args.max_recovery_dispatches, args.max_plan_steps, args.context_bytes, 1)
    native_revision = validate_native()
    pairs = select_pairs(args)
    assets = args.assets_root.expanduser().resolve()
    checkpoint = (args.checkpoint.expanduser().resolve() if args.checkpoint else
                  assets / "models" / "ppo" / "PickClutterYCB-XArm7-v1" / "ppo_grasp.zip")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PPO checkpoint missing: {checkpoint}")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("use a new output directory")
    output.mkdir(parents=True)
    for pair in pairs:
        directory = output / pair["pair_id"]
        directory.mkdir()
        snapshot = directory / "scene.json"
        write_json(snapshot, pair.pop("scene_definition"))
        pair["source_scene"], pair["scene"] = pair["scene"], str(snapshot)
    import subprocess
    root = Path(__file__).resolve().parents[2]
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "--",
                                        "src", "integrations", "scripts"], cwd=root, text=True)
    if untracked.strip():
        raise RuntimeError("stage new source files before running so source.diff captures the executed code")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    source_diff = subprocess.check_output(["git", "diff", "HEAD", "--", "src", "integrations", "scripts"], cwd=root, text=True)
    (output / "source.diff").write_text(source_diff)
    experiment = {"modes": args.modes, "pairs": pairs, "native_revision": native_revision,
                  "checkpoint": str(checkpoint), "model": args.model,
                  "source_commit": source_commit, "source_dirty": bool(source_diff),
                  "native_patch": "third_party/patches/adacleargrasp-push-pose.patch"}
    # Path arguments are serialized explicitly.
    experiment["configuration"] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(output / "experiment.json", experiment)
    results = []
    for pair in pairs:
        directory = output / pair["pair_id"]
        for mode in args.modes:
            result = run_episode(args, pair, mode, directory / mode, checkpoint, secret, endpoint)
            results.append(result)
            write_json(output / "results.json", results, secret)
            print(json.dumps({"pair": pair["pair_id"], "mode": mode, "complete": result["execution_complete"],
                              "official_success": result["native_task_success"], "error": result.get("error")}, ensure_ascii=False), flush=True)
    return 0 if all(r["execution_complete"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
