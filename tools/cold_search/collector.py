from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from google.protobuf.json_format import MessageToDict
from pathlib import Path
import json, threading, gzip
from paths import workspace as w; lock=threading.Lock()
class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        data=self.rfile.read(int(self.headers['Content-Length']))
        marker=w/'active-trace-label'
        if marker.exists():
            label=marker.read_text().strip()
            if self.headers.get('Content-Encoding')=='gzip': data=gzip.decompress(data)
            req=ExportTraceServiceRequest(); req.ParseFromString(data)
            with lock, gzip.open(w/'results'/f'{label}-traces.jsonl.gz','at',compresslevel=1) as f:
                f.write(json.dumps(MessageToDict(req))+'\n')
        self.send_response(200); self.send_header('Content-Type','application/x-protobuf'); self.end_headers()
    def log_message(self,*args): pass
ThreadingHTTPServer(('127.0.0.1',4318),Handler).serve_forever()
