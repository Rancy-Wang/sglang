"""Launch the standard server, exercise its standard API, then stop our group."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from verify_ouro import http_request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30126)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "sglang.launch_server",
               "--model-path", "/home/ruoxi/models/Ouro-2.6B",
               "--host", "127.0.0.1", "--port", str(args.port),
               "--dtype", "bfloat16", "--attention-backend", "triton",
               "--max-running-requests", "1", "--page-size", "1",
               "--disable-overlap-schedule", "--disable-radix-cache",
               "--chunked-prefill-size", "-1", "--disable-cuda-graph",
               "--context-length", "2048", "--mem-fraction-static", "0.7"]
    if args.depths:
        command += ["--speculative-algorithm", "loopspec", "--speculative-draft-depth"]
        command += list(map(str, args.depths))
    (args.output / "command.json").write_text(json.dumps(command))
    with (args.output / "server.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            url = f"http://127.0.0.1:{args.port}"
            deadline = time.monotonic() + 240
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server failed with exit {process.returncode}")
                try:
                    with urllib.request.urlopen(url + "/health", timeout=1):
                        break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise TimeoutError("Server did not become healthy")
                    time.sleep(1)
            for index in range(2):
                result = http_request(url)
                (args.output / f"response-{index}.json").write_text(json.dumps(result, indent=2))
                print(json.dumps(result), flush=True)
        finally:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == "__main__":
    main()
