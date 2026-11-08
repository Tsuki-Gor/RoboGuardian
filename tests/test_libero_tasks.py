import json
import random
from pathlib import Path
from unittest.mock import patch

import os
import tempfile
import unittest

from integrations._semantic import SemanticClient
from integrations.libero.tasks import choose_pair, entity_ids_from_bddl, generate_question


class TaskTests(unittest.TestCase):
    def test_pair_selection_and_bounds(self):
        pair = choose_pair([0, 9], {0: 3, 9: 3}, seed=7, fault="action_dropout", rng=random.Random(3))
        assert pair.task_id in (0, 9) and 0 <= pair.initial_state < 3 and pair.seed == 7
        with self.assertRaises(ValueError):
            choose_pair([0], {0: 2}, initial_state=2)


    def test_entities_and_question(self):
        directory = tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"])
        self.addCleanup(directory.cleanup)
        tmp_path = Path(directory.name)
        bddl = tmp_path / "task.bddl"
        bddl.write_text("(define (:fixtures table_1 - table) (:objects cup_1 - cup) "
                        "(:obj_of_interest cup_1 table_1) (:init (On cup_1 table_1)))")
        entities = entity_ids_from_bddl(bddl)
        assert entities == ("cup_1", "table_1")
        client = SemanticClient("https://example.invalid/v1", "test-secret", "qwen/qwen3.5-27b", tmp_path)
        task = "put the cup on the table"

        class Response:
            def __init__(self, value):
                self.value = value

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self):
                return json.dumps({"choices": [{"message": {"content": json.dumps(self.value)}}]}).encode()

        answer = {"question": f"Is the task complete: {task}?", "entity_ids": ["cup_1", "table_1"]}
        with patch("integrations.libero.tasks.urlopen", return_value=Response(answer)):
            assert generate_question(client, task, entities) == answer["question"]
        assert client.calls == 1
        request = (Path(tmp_path) / "model_requests/0001-question.request.json").read_text()
        assert "test-secret" not in request
        for value in ({**answer, "entity_ids": ["unknown"]}, {"question": "unknown", "entity_ids": []}):
            with patch("integrations.libero.tasks.urlopen", return_value=Response(value)):
                with self.assertRaises(RuntimeError):
                    generate_question(client, task, entities)
        with patch("integrations.libero.tasks.urlopen", side_effect=TimeoutError("test-secret")):
            with self.assertRaisesRegex(RuntimeError, "REDACTED"):
                generate_question(client, task, entities)
        bddl.write_text(bddl.read_text().replace("cup_1 table_1)", "unknown table_1)", 1))
        with self.assertRaises(ValueError):
            entity_ids_from_bddl(bddl)
