import threading
import time
import unittest

from integrations.libero.transport import LoopbackTransport, delayed_policy_result


class LoopbackTransportTest(unittest.TestCase):
    def test_real_timeout_query_and_repeat(self):
        calls = []
        with LoopbackTransport(lambda action: calls.append(action) or len(calls), timeout=0.05) as transport:
            first = transport.execute("logical", "first", [1])
            self.assertEqual(first["result"], 1)
            with self.assertRaises(TimeoutError):
                transport.execute("logical", "second", [1], response_delay=0.2)
            second = transport.query("logical", "second")
            self.assertEqual(second["status"], "completed")
            self.assertEqual(second["result"], 2)
            self.assertLessEqual(second["started_at"], second["completed_at"])
            self.assertEqual(calls, [[1], [1]])
            self.assertIsNone(transport.query("logical", "missing"))
            self.assertNotIn("response_delay", second)
            self.assertEqual([event["event"] for event in transport.events],
                             ["dispatch", "start", "complete", "dispatch", "start", "complete", "timeout"])

    def test_callbacks_are_serial_and_shutdown_reaps_threads(self):
        active = 0
        overlap = []
        lock = threading.Lock()

        def step(value):
            nonlocal active
            with lock:
                active += 1
                overlap.append(active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return value

        transport = LoopbackTransport(step, timeout=1)
        workers = [threading.Thread(target=transport.execute, args=("id", str(i), i)) for i in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        transport.close()
        self.assertEqual(overlap, [1, 1])
        self.assertFalse(transport._thread.is_alive())

    def test_policy_lateness_keeps_original_provenance(self):
        response = {"request_id": "r1", "observation_id": "o1", "version": 3}
        received, late = delayed_policy_result(response, 0.02, time.monotonic() + 0.005)
        self.assertIs(received, response)
        self.assertTrue(late)
        self.assertEqual(received, {"request_id": "r1", "observation_id": "o1", "version": 3})
        self.assertFalse(delayed_policy_result(response, 0, time.monotonic() + 1)[1])


if __name__ == "__main__":
    unittest.main()


class ShutdownTests(unittest.TestCase):
    def test_close_joins_running_callback_after_client_timeout(self):
        started, finished = threading.Event(), threading.Event()
        def slow_step(value):
            started.set()
            time.sleep(.15)
            finished.set()
            return value
        transport = LoopbackTransport(slow_step, timeout=.02)
        with self.assertRaises(TimeoutError):
            transport.execute('id', 'attempt', {})
        self.assertTrue(started.is_set())
        transport.close()
        self.assertTrue(finished.is_set())
        transport.close()
