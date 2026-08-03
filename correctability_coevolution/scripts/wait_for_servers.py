#!/usr/bin/env python3
import argparse
import time

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("urls", nargs="+")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    deadline = time.time() + args.timeout
    pending = set(args.urls)
    while pending and time.time() < deadline:
        for url in list(pending):
            try:
                response = requests.get(url.rstrip("/") + "/v1/models", timeout=5)
                if response.ok:
                    print(f"ready {url}: {response.json()['data'][0]['id']}")
                    pending.remove(url)
            except requests.RequestException:
                pass
        if pending:
            time.sleep(5)
    if pending:
        raise TimeoutError(f"servers not ready: {sorted(pending)}")


if __name__ == "__main__":
    main()

