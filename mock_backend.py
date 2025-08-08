import json
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Reduce default noise, but keep our own prints
        return

    def do_GET(self):
        length = int(self.headers.get('Content-Length') or 0)
        data = self.rfile.read(length) if length else b''
        print(f"[MOCK GET] {self.path} headers={dict(self.headers)} body={data.decode('utf-8', 'ignore')}")
        # Minimal OK payload
        response = {"ok": True}
        # For conversation info endpoints, return minimal expected fields
        if self.path.startswith('/api/sdk/conversations/') and '/history' not in self.path and self.path.count('/') >= 4:
            response = {
                "agent_id": "agent_test",
                "start_time": "2025-01-01T00:00:00",
                "run_id": self.path.rsplit('/', 1)[-1],
                "user_id": None,
                "is_ended": False
            }
        self._send_json(response, 200)

    def do_POST(self):
        length = int(self.headers.get('Content-Length') or 0)
        data = self.rfile.read(length) if length else b''
        try:
            parsed = json.loads(data.decode('utf-8')) if data else {}
        except Exception:
            parsed = {"_raw": data.decode('utf-8', 'ignore')}
        print(f"[MOCK POST] {self.path} headers={dict(self.headers)} json={parsed}")
        self._send_json({"ok": True}, 200)

if __name__ == '__main__':
    addr = ('127.0.0.1', 8080)
    print(f"Mock backend listening on http://{addr[0]}:{addr[1]}")
    HTTPServer(addr, Handler).serve_forever() 