"""Authenticated local control for one resumable native Habitat evaluation (Python 3.9)."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import secrets
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Cancelled(BaseException):
    pass


class Worker:
    def __init__(self, runner, budget_seconds):
        if not math.isfinite(budget_seconds) or budget_seconds <= 0:
            raise ValueError('total budget must be positive')
        self.runner = runner
        self.budget = budget_seconds
        self.deadline = None
        self.condition = threading.Condition()
        self.data = {'state': 'ready', 'sequence': 0, 'invocation_id': None,
                     'attempt_id': None, 'metrics': None}
        self.cancelled = False
        self.closing = False
        self.secrets = ()

    def update(self, **values):
        with self.condition:
            self.data.update(values)
            self.data['sequence'] += 1
            self.condition.notify_all()

    def status(self):
        with self.condition:
            return copy.deepcopy(self.data)

    def checkpoint(self):
        with self.condition:
            while not self.cancelled and time.monotonic() >= self.deadline:
                if self.data['state'] != 'paused':
                    self.update(state='paused')
                self.condition.wait()
            if self.cancelled:
                raise Cancelled()
            return max(0.001, self.deadline - time.monotonic())

    def wait(self, seconds):
        until = time.monotonic() + seconds
        while True:
            remaining = self.checkpoint()
            delay = min(until - time.monotonic(), remaining)
            if delay <= 0:
                return
            with self.condition:
                if self.cancelled:
                    raise Cancelled()
                self.condition.wait(delay)

    def start(self, invocation_id, attempt_id):
        if not all(isinstance(x, str) and x for x in (invocation_id, attempt_id)):
            raise ValueError('invocation_id and attempt_id must be nonempty strings')
        with self.condition:
            if self.closing:
                raise ValueError('worker is closing')
            existing = (self.data['invocation_id'], self.data['attempt_id'])
            if existing != (None, None):
                if existing != (invocation_id, attempt_id):
                    raise ValueError('worker already bound to another invocation')
                return self.status()
            self.deadline = time.monotonic() + self.budget
            self.update(state='running', invocation_id=invocation_id, attempt_id=attempt_id)
            threading.Thread(target=self._run, daemon=True).start()
            return self.status()

    def _run(self):
        try:
            metrics = self.runner(self)
            with self.condition:
                if self.cancelled:
                    raise Cancelled()
                self.update(state='completed', metrics=metrics)
        except Cancelled:
            self.update(state='cancelled')
        except Exception as exc:
            diagnostic = traceback.format_exc()
            for secret in self.secrets:
                if secret:
                    diagnostic = diagnostic.replace(secret, '[REDACTED]')
            print(diagnostic, flush=True)
            self.update(state='failed', error=type(exc).__name__)

    def resume(self, seconds):
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('budget_seconds must be positive')
        with self.condition:
            if self.data['state'] != 'paused' or self.cancelled:
                raise ValueError('worker is not paused or is cancelling')
            self.deadline = time.monotonic() + seconds
            self.update(state='running')
            return self.status()

    def cancel(self):
        with self.condition:
            if self.data['state'] not in ('running', 'paused'):
                raise ValueError('worker is not running or paused')
            self.cancelled = True
            self.condition.notify_all()
            return self.status()

    def close(self):
        with self.condition:
            if self.data['state'] not in ('ready', 'completed', 'failed', 'cancelled'):
                raise ValueError('worker is active')
            self.closing = True
            return self.status()


def make_server(worker, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def reply(self, status, value):
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def authorized(self):
            return secrets.compare_digest(self.headers.get('Authorization', '').encode('utf-8'), ('Bearer ' + token).encode('ascii'))

        def do_GET(self):
            if not self.authorized():
                return self.reply(401, {'error': 'unauthorized'})
            if self.path != '/status':
                return self.reply(404, {'error': 'not found'})
            self.reply(200, worker.status())

        def do_POST(self):
            if not self.authorized():
                return self.reply(401, {'error': 'unauthorized'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length > 65536 or length < 0:
                    raise ValueError('invalid body length')
                body = json.loads(self.rfile.read(length) or b'{}')
                if not isinstance(body, dict):
                    raise ValueError('body must be an object')
                if self.path == '/start':
                    result = worker.start(body.get('invocation_id'), body.get('attempt_id'))
                elif self.path == '/resume':
                    result = worker.resume(body.get('budget_seconds'))
                elif self.path == '/cancel':
                    result = worker.cancel()
                elif self.path == '/close':
                    result = worker.close()
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                else:
                    return self.reply(404, {'error': 'not found'})
                self.reply(200, result)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self.reply(400, {'error': str(exc)})
    return ThreadingHTTPServer(('127.0.0.1', 0), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-json', required=True)
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config_json).read_text(encoding='utf-8'))
    # Script execution must not put integrations/habitat ahead of Habitat's package.
    import sys
    worker_dir = Path(__file__).resolve().parent
    sys.path[:] = [entry for entry in sys.path if Path(entry).resolve() not in (worker_dir, worker_dir.parent)]
    sys.path.insert(0, str(worker_dir))
    from native import run_native
    worker = Worker(lambda control: run_native(config, control), float(config['total_budget_seconds']))
    token = secrets.token_urlsafe(32)
    server = make_server(worker, token)
    connection = {'pid': os.getpid(), 'url': 'http://127.0.0.1:%d' % server.server_port, 'token': token}
    path = Path(config['output_dir']) / 'worker.json'
    temporary = path.with_name('.worker-' + secrets.token_hex(8))
    fd = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(connection, stream)
        os.link(str(temporary), str(path))
    finally:
        temporary.unlink()
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
