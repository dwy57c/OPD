#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import time

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("urls", nargs="+")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--model")
    parser.add_argument("--health-path", default="/v1/models")
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()
    session = requests.Session()
    # Every managed inference endpoint is local.  Inherited HTTP(S)_PROXY
    # variables can otherwise route loopback health checks through an external
    # proxy even when the service is listening successfully.
    session.trust_env = False
    deadline = time.time() + args.timeout
    pending = set(args.urls)
    while pending and time.time() < deadline:
        if args.pid is not None:
            try:
                os.kill(args.pid, 0)
            except ProcessLookupError as error:
                log_tail = ""
                if args.log_file and args.log_file.exists():
                    log_tail = "\n" + "".join(
                        args.log_file.read_text(errors="replace").splitlines(True)[-40:]
                    )
                raise RuntimeError(
                    f"server process {args.pid} exited{log_tail}"
                ) from error
        for url in list(pending):
            try:
                endpoint = url.rstrip("/") + "/" + args.health_path.lstrip("/")
                response = session.get(endpoint, timeout=5)
                if response.ok:
                    detail = args.health_path
                    if args.model is not None:
                        model_id = response.json()["data"][0]["id"]
                        if model_id != args.model:
                            continue
                        detail = model_id
                    print(f"ready {url}: {detail}")
                    pending.remove(url)
            except (requests.RequestException, KeyError, IndexError, ValueError):
                pass
        if pending:
            time.sleep(5)
    if pending:
        raise TimeoutError(
            f"servers not ready at {args.health_path}: {sorted(pending)}"
        )


if __name__ == "__main__":
    main()
