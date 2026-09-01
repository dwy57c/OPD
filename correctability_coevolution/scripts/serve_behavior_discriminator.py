#!/usr/bin/env python3
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Lock

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from coevo.hinter_training.behavior_discriminator import score_texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve scalar behavior-discriminator scores")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
    ).to(args.device)
    model.eval()
    model_lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                self._send(404, {"error": "not found"})
                return
            self._send(200, {"status": "ok", "model": args.model})

        def do_POST(self):
            if self.path != "/score":
                self._send(404, {"error": "not found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(size))
                inputs = payload["inputs"]
                if not isinstance(inputs, list) or not all(
                    isinstance(value, str) for value in inputs
                ):
                    raise ValueError("inputs must be a list of strings")
                if not 1 <= len(inputs) <= args.max_batch_size:
                    raise ValueError("request exceeds discriminator batch limit")
                with model_lock:
                    scores = score_texts(
                        model, tokenizer, inputs, max_length=args.max_length
                    )
                self._send(200, {"scores": scores})
            except Exception as error:
                self._send(
                    400,
                    {"error": f"{type(error).__name__}: {error}"},
                )

        def log_message(self, _format, *_args):
            return

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
