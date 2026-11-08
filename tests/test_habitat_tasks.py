"""Offline checks for Habitat dataset selection and native metrics."""

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from integrations.habitat.tasks import load_metrics_exact, validate_metrics_exact, write_exact_subset

METRICS = {"success": 0, "oracle_success": 1, "spl": 0, "ndtw": 0.2,
           "sdtw": 0, "distance_to_goal": 4.2, "path_length": 3.5, "steps_taken": 10}


class HabitatTaskTests(unittest.TestCase):
    def test_subset_preserves_dataset_and_requested_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source.json.gz", root / "subset.json.gz"
            episodes = [
                {"episode_id": 11, "scene_id": "mp3d/a/a.glb",
                 "instruction": {"instruction_text": "Go left. ", "instruction_tokens": [1]}},
                {"episode_id": 7, "scene_id": "mp3d/b/b.glb", "instruction": "Go right."},
            ]
            with gzip.open(source, "wt", encoding="utf-8") as stream:
                json.dump({"episodes": episodes, "instruction_vocab": {"words": ["go"]}}, stream)
            selected = write_exact_subset(source, ["7", "11"], output_path=target)
            self.assertEqual(selected, [
                {"episode_id": "7", "scene_id": "mp3d/b/b.glb", "instruction": "Go right."},
                {"episode_id": "11", "scene_id": "mp3d/a/a.glb", "instruction": "Go left. "},
            ])
            with gzip.open(target, "rt", encoding="utf-8") as stream:
                self.assertEqual(json.load(stream), {"episodes": episodes[::-1],
                                                    "instruction_vocab": {"words": ["go"]}})

    def test_metrics_preserve_failure_and_native_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "rank0.json", root / "rank1.json"
            first.write_text(json.dumps({"11": METRICS}))
            second.write_text(json.dumps({"7": {**METRICS, "success": 1, "spl": 0.7}}))
            self.assertEqual(load_metrics_exact([first, second], ["11", "7"]), {
                "11": METRICS,
                "7": {**METRICS, "success": 1, "spl": 0.7},
            })

    def test_selection_rejects_invalid_ids_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source.json.gz", root / "subset.json.gz"
            episodes = [{"episode_id": 11, "scene_id": "scene",
                         "instruction": {"instruction_text": "Go."}}]
            def write_source(value):
                with gzip.open(source, "wt", encoding="utf-8") as stream:
                    json.dump({"episodes": value}, stream)
            write_source(episodes)
            for ids in ([], [11, "11"], ["missing"]):
                with self.subTest(ids=ids), self.assertRaises(ValueError):
                    write_exact_subset(source, ids, output_path=target)
            write_exact_subset(source, [11], output_path=target)
            existing = target.read_bytes()
            with self.assertRaises(FileExistsError):
                write_exact_subset(source, [11], output_path=target)
            self.assertEqual(target.read_bytes(), existing)
            write_source(episodes * 2)
            with self.assertRaises(ValueError):
                write_exact_subset(source, [11], output_path=root / "other.json.gz")

    def test_metrics_reject_incomplete_duplicate_extra_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first.json", root / "second.json"
            first.write_text('{"11": {"success": 0}}')
            second.write_text('{"7": {"success": 1}}')
            with self.assertRaisesRegex(ValueError, "missing=.*7"):
                load_metrics_exact([first], [11, 7])
            with self.assertRaisesRegex(ValueError, "extra=.*7"):
                load_metrics_exact([first, second], [11])
            with self.assertRaisesRegex(ValueError, "duplicate metric episode ID"):
                load_metrics_exact([first, first], [11])
            first.write_text('{"11":{"success":0},"11":{"success":1}}')
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_metrics_exact([first], [11])
            for bad in (True, None, float("nan"), float("inf"), "0"):
                first.write_text(json.dumps({"11": {**METRICS, "success": bad}}))
                with self.subTest(value=bad), self.assertRaises(ValueError):
                    load_metrics_exact([first], [11])
            first.write_text('{"11": {}}')
            with self.assertRaises(ValueError):
                load_metrics_exact([first], [11])

    def test_worker_metrics_use_the_same_validation(self):
        metrics = {"11": {**METRICS, "ndtw": 0.5}, "7": {**METRICS, "success": 1}}
        self.assertIs(validate_metrics_exact(metrics, [11, 7]), metrics)
        self.assertEqual(validate_metrics_exact({11: METRICS}, ["11"]), {11: METRICS})
        for value, expected in (({"11": METRICS}, [11, 7]),
                                ({"11": METRICS}, [7]),
                                ({"11": {**METRICS, "success": True}}, [11]),
                                ({"11": {}}, [11]),
                                ({"11": {**METRICS, "spl": float("nan")}}, [11]),
                                ({11: METRICS, "11": METRICS}, [11]),
                                ([], [11])):
            with self.subTest(value=value, expected=expected), self.assertRaises(ValueError):
                validate_metrics_exact(value, expected)

    def test_every_native_metric_is_required(self):
        for field in METRICS:
            partial = {key: value for key, value in METRICS.items() if key != field}
            with self.subTest(missing=field), self.assertRaises(ValueError):
                validate_metrics_exact({"11": partial}, [11])
            for value in (False, float("nan"), float("inf")):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate_metrics_exact({"11": {**METRICS, field: value}}, [11])
