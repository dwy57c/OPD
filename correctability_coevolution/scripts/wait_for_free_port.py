#!/usr/bin/env python3
import argparse
import socket
import time


def port_is_bindable(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        try:
            handle.bind(("", port))
        except OSError:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout
    while not port_is_bindable(args.port):
        if time.monotonic() >= deadline:
            raise SystemExit(
                f"port {args.port} did not become bindable within {args.timeout:g}s"
            )
        time.sleep(0.1)


if __name__ == "__main__":
    main()
