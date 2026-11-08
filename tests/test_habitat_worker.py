import json
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from integrations.habitat.worker import Worker, make_server


class WorkerHTTPTest(unittest.TestCase):
    def test_pause_resume_keeps_single_execution_and_identity(self):
        visits = []
        def run(control):
            visits.append('entered')
            time.sleep(0.08)
            control.checkpoint()
            visits.append('continued')
            return {'11': {'success': 0.0}}

        worker = Worker(run, 0.03)
        server = make_server(worker, 'secret')
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = 'http://127.0.0.1:%d' % server.server_port
        def call(path, body=None, token='secret'):
            data = None if body is None else json.dumps(body).encode()
            request = Request(url + path, data=data, headers={'Authorization': 'Bearer ' + token,
                              'Content-Type': 'application/json'})
            with urlopen(request, timeout=2) as response:
                return json.load(response)
        try:
            with self.assertRaises(HTTPError):
                call('/status', token='wrong')
            self.assertEqual(call('/status')['state'], 'ready')
            identity = {'invocation_id': 'one', 'attempt_id': 'a'}
            call('/start', identity)
            deadline = time.monotonic() + 2
            while call('/status')['state'] != 'paused':
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            self.assertEqual(visits, ['entered'])
            self.assertEqual(call('/start', identity)['state'], 'paused')
            with self.assertRaises(HTTPError):
                call('/start', {'invocation_id': 'two', 'attempt_id': 'b'})
            call('/resume', {'budget_seconds': 1})
            while call('/status')['state'] != 'completed':
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            self.assertEqual(visits, ['entered', 'continued'])
            self.assertEqual(call('/status')['metrics'], {'11': {'success': 0.0}})
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_cancel_paused_task_releases_its_resources(self):
        released = threading.Event()
        def run(control):
            try:
                time.sleep(0.06)
                control.checkpoint()
            finally:
                released.set()
        worker = Worker(run, 0.02)
        worker.start('one', 'a')
        deadline = time.monotonic() + 2
        while worker.status()['state'] != 'paused':
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.01)
        worker.cancel()
        self.assertTrue(released.wait(1))
        while worker.status()['state'] != 'cancelled':
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.01)

    def test_final_step_info_and_cancel_close_environment(self):
        from io import StringIO
        from types import SimpleNamespace
        from unittest.mock import patch
        from integrations.habitat.native import EnvProxy
        ready = threading.Event()
        class Environment:
            closed = False
            def reset(self): return [{'rgb': None}]
            def current_episodes(self): return [SimpleNamespace(episode_id=11)]
            def step(self, actions):
                return [({'rgb': None}, 0, True,
                         {'position': {'position': [[1, 2, 3]], 'distance': [2]}})]
            def close(self): self.closed = True
        env = Environment()
        observed = []
        def run(control):
            with patch('pathlib.Path.open', return_value=StringIO()):
                proxy = EnvProxy(env, control, '.', False)
            try:
                proxy.reset()
                proxy.step([{'action': 0}])
                observed.append(proxy.metric(lambda infos: infos[0]['position']['distance'][-1],
                                             [{'position': {'distance': [9]}}]))
                ready.set()
                time.sleep(0.08)
                proxy.step([{'action': 1}])
            finally:
                proxy.close()
        worker = Worker(run, 0.04)
        worker.start('one', 'a')
        self.assertTrue(ready.wait(1))
        deadline = time.monotonic() + 2
        while worker.status()['state'] != 'paused':
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.01)
        worker.cancel()
        while worker.status()['state'] != 'cancelled':
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.01)
        self.assertEqual(observed, [2])
        self.assertTrue(env.closed)

    def test_http_timeout_does_not_kill_or_restart_task(self):
        gate = threading.Event()
        visits = []
        def run(control):
            visits.append('started')
            gate.wait(2)
            control.checkpoint()
            return {'11': {'success': 1.0}}
        class SlowReplyWorker(Worker):
            def start(self, *identity):
                result = super().start(*identity)
                time.sleep(0.1)
                return result
        worker = SlowReplyWorker(run, 2)
        server = make_server(worker, 'secret')
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = 'http://127.0.0.1:%d' % server.server_port
        headers = {'Authorization': 'Bearer secret', 'Content-Type': 'application/json'}
        try:
            payload = json.dumps({'invocation_id': 'one', 'attempt_id': 'a'}).encode()
            with self.assertRaises(TimeoutError):
                urlopen(Request(base + '/start', data=payload, headers=headers), timeout=0.02)
            gate.set()
            deadline = time.monotonic() + 2
            while True:
                with urlopen(Request(base + '/status', headers=headers), timeout=1) as response:
                    status = json.load(response)
                if status['state'] == 'completed':
                    break
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            self.assertEqual(visits, ['started'])
        finally:
            gate.set()
            server.shutdown()
            server.server_close()
            thread.join()

class ModelRetryTest(unittest.TestCase):
    def test_invalid_json_and_box_are_retried_without_fallback(self):
        from types import SimpleNamespace
        from integrations.habitat.native import RetryingGenerate
        class Control:
            def checkpoint(self): return 10
            def wait(self, seconds): pass
        class Client:
            def __init__(self):
                self.chat = SimpleNamespace(completions=self)
                self.responses = iter([None, 'broken', '{"action":"STOP"}',
                    '{"action":"STOP","bbox_2d":[1,2,4,5],"target":"door",'
                    '"reasoning":"there","progress":"close"}'])
            def with_options(self, **kwargs): return self
            def create(self, **kwargs):
                content = next(self.responses)
                if content is None:
                    raise OSError('offline')
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=None)
        client = Client()
        native = SimpleNamespace(la_client=client, va_client=client, la_model_name='la',
            va_model_name='va', va_base_url='', stats={'Vision Action Model': {'calls': 0}})
        generate = RetryingGenerate(native, Control(), None, 1, 0.01, 10, 10)
        result = json.loads(generate(messages=[], use_la=False))
        self.assertEqual(result['bbox_2d'], [1, 2, 4, 5])

    def test_qwen3_normalized_box_is_passed_to_native_as_pixels(self):
        from types import SimpleNamespace
        from integrations.habitat.native import RetryingGenerate
        native = SimpleNamespace(la_client=None, va_client=None)
        generate = RetryingGenerate(native, None, None, 1, 1, 640, 480)
        response = '{"action":"STOP","bbox_2d":[0,500,1000,1000],"target":"door","reasoning":"close","progress":"near"}'
        result = json.loads(generate.validate(response, False, 'Qwen3.5-27B'))
        self.assertEqual(result['bbox_2d'], [0, 240, 640, 480])


if __name__ == '__main__':
    unittest.main()
