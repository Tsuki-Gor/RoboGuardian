"""Local HTTP transport for serial native calls and genuine client timeouts."""

import json
import math
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import URLError
from urllib.request import Request, urlopen


def delayed_policy_result(response, delay, deadline):
    """Delay delivery without changing the response or its provenance."""
    if not math.isfinite(delay) or delay < 0 or not math.isfinite(deadline):
        raise ValueError("delay and deadline must be finite, with delay nonnegative")
    time.sleep(delay)
    return response, time.monotonic() > deadline


class LoopbackTransport:
    """Execute callbacks serially; response delivery may outlive the client request."""

    def __init__(self, callback, timeout=1.0):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.callback, self.timeout = callback, timeout
        self.events, self._records = [], {}
        self._lock, self._step_lock = threading.Lock(), threading.Lock()
        self._closing = threading.Event()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_POST(self):
                try:
                    length = int(self.headers["Content-Length"])
                    request = json.loads(self.rfile.read(length))
                    if self.path == "/execute":
                        result = owner._execute(request)
                    elif self.path == "/query":
                        with owner._lock:
                            record = owner._records.get((request["logical_id"], request["attempt_id"]))
                            result = dict(record) if record is not None else None
                    else:
                        self.send_error(404)
                        return
                    body = json.dumps(result, allow_nan=False).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # The client's real timeout closes the socket.
                except Exception as exc:
                    try:
                        self.send_error(400, str(exc))
                    except (BrokenPipeError, ConnectionResetError):
                        pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        # Handler completion owns the simulator resources; closing must join it.
        self._server.daemon_threads = False
        self._thread = threading.Thread(target=self._server.serve_forever)
        self._thread.start()

    def _event(self, event, key):
        self.events.append({"event": event, "logical_id": key[0], "attempt_id": key[1],
                            "at": time.monotonic()})

    def _execute(self, request):
        key = (request["logical_id"], request["attempt_id"])
        delay = request["response_delay"]
        if not all(isinstance(value, str) and value for value in key):
            raise ValueError("logical_id and attempt_id must be nonempty strings")
        if not isinstance(delay, (int, float)) or not math.isfinite(delay) or delay < 0:
            raise ValueError("response_delay must be finite and nonnegative")
        with self._lock:
            if key in self._records:
                raise ValueError("attempt already exists")
            self._records[key] = {"status": "queued", "started_at": None, "completed_at": None}
        with self._step_lock:
            with self._lock:
                record = self._records[key]
                record.update(status="running", started_at=time.monotonic())
                self._event("start", key)
            try:
                result = self.callback(request["payload"])
                # Verify serializability before recording success.
                json.dumps(result, allow_nan=False)
                outcome = {"status": "completed", "result": result}
            except Exception as exc:
                outcome = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            with self._lock:
                record.update(outcome, completed_at=time.monotonic())
                self._event("complete", key)
                response = dict(record)
        self._closing.wait(delay)
        return response

    def _request(self, path, payload):
        body = json.dumps(payload, allow_nan=False).encode()
        request = Request(f"http://127.0.0.1:{self._server.server_port}/{path}", body,
                          {"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except (TimeoutError, socket.timeout) as exc:
            raise TimeoutError(f"{path} request timed out") from exc
        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise TimeoutError(f"{path} request timed out") from exc
            raise

    def execute(self, logical_id, attempt_id, payload, response_delay=0):
        key = (logical_id, attempt_id)
        with self._lock:
            self._event("dispatch", key)
        try:
            return self._request("execute", {"logical_id": logical_id, "attempt_id": attempt_id,
                                             "payload": payload, "response_delay": response_delay})
        except TimeoutError:
            with self._lock:
                self._event("timeout", key)
            raise

    def query(self, logical_id, attempt_id):
        return self._request("query", {"logical_id": logical_id, "attempt_id": attempt_id})

    def close(self):
        if self._closing.is_set():
            return
        self._closing.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
