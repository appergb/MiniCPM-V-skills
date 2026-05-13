# tests/integration/test_lifecycle.py
import subprocess
import time
from minicpm_v_local.server import manager
from minicpm_v_local.server.state import read_state
from minicpm_v_local import paths

# 用一个简单的 python HTTP server 模拟 backend
FAKE_SERVER = """
import http.server, socketserver, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200); self.end_headers(); return
        self.send_response(404); self.end_headers()
    def log_message(self, *a): pass
port = int(sys.argv[1])
with socketserver.TCPServer(('127.0.0.1', port), H) as httpd:
    httpd.serve_forever()
"""

class FakeBackend:
    tag = "cpu"; quant = "Q4_K_M"
    def artifact_id(self): return "fake/model"
    def launch_cmd(self, model_dir, port):
        return ["python", "-c", FAKE_SERVER, str(port)]
    def install_check(self): return True, ""
    def health_path(self): return "/health"

def test_warm_path_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "state_file", lambda: tmp_path / "state.json")
    monkeypatch.setattr(paths, "run_dir", lambda: tmp_path)
    monkeypatch.setattr(paths, "log_dir", lambda: tmp_path)
    b = FakeBackend()
    try:
        s1 = manager.ensure_warm(b, tmp_path, port_range=(9700, 9710),
                                  health_timeout=10, ttl_seconds=60,
                                  max_lifetime=600, keep=False, isolation_mode="none")
        s2 = manager.ensure_warm(b, tmp_path, port_range=(9700, 9710),
                                  health_timeout=10, ttl_seconds=60,
                                  max_lifetime=600, keep=False, isolation_mode="none")
        assert s1.server_pid == s2.server_pid  # warm path
    finally:
        manager.stop()
