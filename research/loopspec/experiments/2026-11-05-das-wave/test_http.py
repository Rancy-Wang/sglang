"""Exercise actual HTTP response formats, including failure propagation."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import unittest
from urllib.error import HTTPError

from verify_server import http


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/flush_cache?timeout=15':
            self.send_response(200); self.end_headers()
            self.wfile.write(b'Cache flushed.\nPlease check backend logs for more details.')
        else:
            self.send_response(400); self.end_headers(); self.wfile.write(b'busy')

    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"ready":true}')

    def log_message(self, *args): pass


class HTTPContract(unittest.TestCase):
    def test_response_formats(self):
        with ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
            thread = Thread(target=server.serve_forever); thread.start()
            try:
                url = f'http://127.0.0.1:{server.server_port}'
                self.assertEqual(http(url, '/server_info'), {'ready': True})
                self.assertTrue(http(url, '/flush_cache?timeout=15', {}).startswith('Cache flushed.'))
                with self.assertRaises(HTTPError): http(url, '/failure', {})
            finally:
                server.shutdown(); thread.join()


if __name__ == '__main__': unittest.main()
