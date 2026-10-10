import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import zlib

body = (bytes(range(251)) * (3 * 1024 * 1024 // 251 + 1))[:3 * 1024 * 1024]

class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def do_GET(self):
        name = self.path.rsplit('/', 1)[-1]
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        if name.startswith('crc-'):
            checksum = zlib.crc32(body) if name.endswith('-ok') else 0
            self.send_header('x-amz-checksum-crc32', base64.b64encode(checksum.to_bytes(4, 'big')).decode())
        elif name.startswith('sha-'):
            checksum = hashlib.sha256(body).digest() if name.endswith('-ok') else bytes(32)
            self.send_header('x-amz-checksum-sha256', base64.b64encode(checksum).decode())
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(fmt % args, flush=True)

ThreadingHTTPServer(('127.0.0.1', 18099), Handler).serve_forever()
