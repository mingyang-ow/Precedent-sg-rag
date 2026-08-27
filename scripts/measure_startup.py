from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _request(url: str) -> tuple[int, dict[str, object]]:
    with urllib.request.urlopen(url, timeout=1) as response:
        return response.status, json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure local API cold-start readiness.")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    environment = dict(os.environ)
    environment.pop("OPENAI_API_KEY", None)
    environment["PRECEDENT_RETRIEVAL_ARTIFACTS"] = str(args.artifacts.resolve())
    command = (
        sys.executable,
        "-m",
        "uvicorn",
        "sg_legal_rag.api.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--workers",
        "1",
        "--no-access-log",
    )
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    health_ms: float | None = None
    ready_ms: float | None = None
    readiness: dict[str, object] | None = None
    try:
        deadline = started + args.timeout
        while time.perf_counter() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"API exited during startup with status {process.returncode}")
            try:
                health_status, _ = _request(f"http://127.0.0.1:{args.port}/health")
            except (OSError, urllib.error.URLError):
                time.sleep(0.01)
                continue
            if health_status == 200:
                health_ms = (time.perf_counter() - started) * 1000
                break
        if health_ms is None:
            raise TimeoutError("API health startup timed out")
        ready_status, readiness = _request(f"http://127.0.0.1:{args.port}/ready")
        ready_ms = (time.perf_counter() - started) * 1000
        if ready_status != 200 or readiness.get("retrieval") is not True:
            raise RuntimeError("API did not become retrieval-ready")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    print(
        json.dumps(
            {
                "generation_configured": readiness.get("generation_configured")
                if readiness
                else None,
                "health_ms": round(health_ms or 0.0, 3),
                "maximum_rss_kib": usage.ru_maxrss,
                "provider_calls": 0,
                "ready_ms": round(ready_ms or 0.0, 3),
                "readiness_status": readiness.get("status") if readiness else None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
