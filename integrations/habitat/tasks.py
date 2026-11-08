"""Select Habitat episodes and check the original per-episode results."""

import gzip
import json
import math
from pathlib import Path


def _ids(values, label):
    ids = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
            raise ValueError(f"{label} must contain non-empty string or integer IDs")
        ids.append(str(value).strip())
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"{label} must be non-empty and unique")
    return ids


def _read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique_object)


def write_exact_subset(dataset_path, episode_ids, *, output_path):
    """Write the selected original episodes and return their navigation metadata."""
    requested = _ids(episode_ids, "requested episode IDs")
    dataset = _read_json(dataset_path)
    if not isinstance(dataset, dict) or not isinstance(dataset.get("episodes"), list):
        raise ValueError("dataset must contain an episodes list")
    episodes = dataset["episodes"]
    if not all(isinstance(episode, dict) for episode in episodes):
        raise ValueError("dataset episodes must be objects")
    source_ids = _ids((episode.get("episode_id") for episode in episodes), "dataset episode IDs")
    by_id = dict(zip(source_ids, episodes))
    missing = sorted(set(requested) - set(by_id))
    if missing:
        raise ValueError(f"requested episode IDs missing from dataset: {missing}")

    selected = []
    for episode_id in requested:
        episode = by_id[episode_id]
        instruction = episode.get("instruction")
        if isinstance(instruction, dict):
            instruction = instruction.get("instruction_text")
        scene_id = episode.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError(f"episode {episode_id} must contain a scene_id")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"episode {episode_id} must contain an instruction")
        selected.append({"episode_id": episode_id, "scene_id": scene_id, "instruction": instruction})

    target = Path(output_path)
    if not target.name.endswith(".json.gz"):
        raise ValueError("subset output must end in .json.gz")
    payload = json.dumps({**dataset, "episodes": [by_id[item] for item in requested]},
                         ensure_ascii=False).encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb") as stream:
            stream.write(payload)
    return selected


def load_metrics_exact(metric_paths, episode_ids):
    """Load native results only when their IDs exactly match the requested set."""
    metrics = {}
    for path in metric_paths:
        value = _read_json(path)
        if not isinstance(value, dict):
            raise ValueError(f"metric file must contain an object: {path}")
        for episode_id, item in value.items():
            if episode_id in metrics:
                raise ValueError(f"duplicate metric episode ID: {episode_id}")
            metrics[episode_id] = item
    return validate_metrics_exact(metrics, episode_ids)


def validate_metrics_exact(metrics, episode_ids):
    """Validate an in-memory native result mapping and return it unchanged."""
    expected = _ids(episode_ids, "expected metric episode IDs")
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be an object")
    actual = set(_ids(metrics, "metric episode IDs")) if metrics else set()
    wanted = set(expected)
    if actual != wanted:
        raise ValueError("metric episode IDs do not exactly match expected IDs: "
                         f"missing={sorted(wanted - actual)} extra={sorted(actual - wanted)}")
    numeric = {"success", "oracle_success", "spl", "ndtw", "sdtw",
               "distance_to_goal", "path_length", "steps_taken"}
    for episode_id, item in metrics.items():
        if not isinstance(item, dict) or not numeric <= item.keys():
            raise ValueError(f"metric for episode {episode_id} must contain all native metrics")
        for name in numeric:
            number = item[name]
            if (isinstance(number, bool) or not isinstance(number, (int, float))
                    or (isinstance(number, float) and not math.isfinite(number))):
                raise ValueError(f"metric {name} for episode {episode_id} must be finite numeric")
    return metrics
