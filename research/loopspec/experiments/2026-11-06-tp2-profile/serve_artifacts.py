"""Temporary authenticated TLS stream of explicitly selected experiment assets.

Launch on X5, fetch on the selected A host, stop after verified transfer.
No directory traversal, listing, upload, or request-controlled commands.
"""
import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import ssl
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bind', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--client-ip', required=True)
    p.add_argument('--certificate', required=True)
    p.add_argument('--key', required=True)
    p.add_argument('--token-file', type=Path, required=True)
    p.add_argument('--models', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--container', required=True)
    a = p.parse_args()
    token = a.token_file.read_text().strip()
    assert len(token) >= 32
    routes = {
        '/environment.tar.gz': ['docker', 'exec', a.container, 'tar', '-czf', '-',
                               '-C', '/opt/sglang/lib/python3.12/site-packages', '.'],
        '/target.tar': ['tar', '-cf', '-', '-C', a.models, 'Ouro-2.6B-Thinking'],
        '/draft.tar': ['tar', '-cf', '-', '-C', a.models, 'Ouro-2.6B-Thinking-DFlash'],
        '/dataset.jsonl': ['cat', a.dataset],
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.client_address[0] != a.client_ip or not hmac.compare_digest(
                    self.headers.get('Authorization', ''), 'Bearer ' + token):
                self.send_error(403)
                return
            if self.path not in routes:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.end_headers()
            with subprocess.Popen(routes[self.path], stdout=subprocess.PIPE) as proc:
                try:
                    shutil.copyfileobj(proc.stdout, self.wfile, 1024 * 1024)
                except (BrokenPipeError, ConnectionResetError):
                    proc.terminate()
                finally:
                    proc.stdout.close()
                code = proc.wait()
            print(self.path, 'exit', code, flush=True)

    server = ThreadingHTTPServer((a.bind, a.port), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(a.certificate, a.key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    print('artifact server ready', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
