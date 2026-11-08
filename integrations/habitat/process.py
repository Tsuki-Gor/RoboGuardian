"""Local control client; a lost response never terminates or restarts execution."""

import json
from pathlib import Path
import subprocess
import time
from urllib.request import Request, ProxyHandler, build_opener
from urllib.parse import urlsplit


class HabitatProcess:
    def __init__(self, output, *, python=None, config=None, request_timeout=10, startup_timeout=30):
        self.output = Path(output)
        self.python, self.config = python, config
        self.request_timeout, self.startup_timeout = request_timeout, startup_timeout
        self.process = None
        self.connection = None
        self.http = build_opener(ProxyHandler({}))

    def _connect(self):
        if self.connection is not None:
            return
        connection = self.output / "worker.json"
        if not connection.exists():
            if self.python is None or self.config is None or self.process is not None:
                raise RuntimeError("worker connection unavailable; the evaluation will not be replayed")
            worker = Path(__file__).with_name("worker.py")
            with (self.output / "native-worker.log").open("ab") as log:
                self.process = subprocess.Popen(
                    [str(self.python), "-B", str(worker), "--config-json", str(self.config)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
            deadline = time.monotonic() + self.startup_timeout
            while not connection.exists():
                if self.process.poll() is not None:
                    raise RuntimeError("worker startup failed; see native-worker.log")
                if time.monotonic() >= deadline:
                    raise TimeoutError("worker startup pending; process preserved, inspect the output directory")
                time.sleep(.05)
        value = json.loads(connection.read_text())
        url = urlsplit(value["url"])
        if (url.scheme != "http" or url.hostname != "127.0.0.1" or not url.port
                or url.username or url.password or url.path or url.query or url.fragment
                or not isinstance(value.get("token"), str) or not value["token"]):
            raise ValueError("invalid local worker connection")
        self.connection = value

    def request(self, operation, **payload):
        if operation not in ("status", "start", "resume", "cancel", "close"):
            raise ValueError("unknown worker operation")
        self._connect()
        request = Request(self.connection["url"] + "/" + operation,
                          data=None if operation == "status" else json.dumps(payload, allow_nan=False).encode(),
                          headers={"Authorization": "Bearer " + self.connection["token"],
                                   "Content-Type": "application/json"})
        # No retry of mutating requests: query the original identity after a lost reply.
        with self.http.open(request, timeout=self.request_timeout) as response:
            value = json.load(response)
        if not isinstance(value, dict):
            raise ValueError("invalid worker status")
        return value

    def wait(self, *, include_paused=False):
        while True:
            value = self.request("status")
            if value.get("state") not in (("running", "paused") if include_paused else ("running",)):
                return value
            time.sleep(.2)

    def close(self):
        state = self.request("status")
        if state.get("state") in ("ready", "completed", "failed", "cancelled"):
            return self.request("close")
        return state
