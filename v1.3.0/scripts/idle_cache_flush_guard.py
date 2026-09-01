#!/usr/bin/env python3
"""Idle high-HBM cache flush guard for Qwen3.8 SGLang services.

This is deliberately external to SGLang.  It never patches scheduler/model code.
It calls the official /flush_cache endpoint only when /v1/loads reports
running=0, waiting=0, used_tokens=0 and one of two idle/high-HBM policies fires:
  * emergency: continuously idle for --emergency-idle-seconds and HBM >=
    --emergency-mem-threshold-pct;
  * normal: continuously idle for --idle-seconds and HBM >= --mem-threshold-pct.

The official endpoint is still the final safety gate: if a request races with the
flush, SGLang rejects it.  Failures are logged and serving is left untouched.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import requests


@dataclass
class LoadState:
    running: int
    waiting: int
    used_tokens: int

    @property
    def fully_idle(self) -> bool:
        return self.running == 0 and self.waiting == 0 and self.used_tokens == 0


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{now_iso()}] {msg}", flush=True)


def read_loads(session: requests.Session, base: str, timeout: float) -> LoadState:
    r = session.get(base + "/v1/loads?include=core", timeout=timeout)
    r.raise_for_status()
    doc = r.json()
    agg = doc.get("aggregate") or {}
    return LoadState(
        running=int(agg.get("total_running_reqs") or 0),
        waiting=int(agg.get("total_waiting_reqs") or 0),
        used_tokens=int(agg.get("total_used_tokens") or 0),
    )


def read_mem_pct(gpu_id: int) -> int:
    cp = subprocess.run(
        ["/usr/local/hyhal/bin/hy-smi", "-d", str(gpu_id), "--showmemuse"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=10,
    )
    pat = re.compile(rf"HCU\[{gpu_id}\].*HCU memory use \(%\):\s*([0-9]+(?:\.[0-9]+)?)")
    for line in cp.stdout.splitlines():
        m = pat.search(line)
        if m:
            return int(round(float(m.group(1))))
    raise RuntimeError(f"cannot parse HCU[{gpu_id}] memory use")


def flush(session: requests.Session, base: str, timeout: float) -> tuple[int, str]:
    r = session.post(base + "/flush_cache", timeout=timeout)
    return r.status_code, r.text.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument(
        "--gpu-id",
        type=int,
        action="append",
        required=True,
        help="physical HCU id; repeat for TP2/TP4 multi-GPU services",
    )
    ap.add_argument("--idle-seconds", type=float, default=600.0)
    ap.add_argument("--poll-seconds", type=float, default=5.0)
    ap.add_argument("--mem-threshold-pct", type=int, default=97)
    ap.add_argument("--emergency-idle-seconds", type=float, default=15.0)
    ap.add_argument("--emergency-mem-threshold-pct", type=int, default=99)
    ap.add_argument("--http-timeout", type=float, default=5.0)
    ap.add_argument("--flush-timeout", type=float, default=30.0)
    ap.add_argument("--max-cycles", type=int, default=0,
                    help="0 means forever; positive is useful for validation")
    args = ap.parse_args()

    if args.idle_seconds < 1:
        raise SystemExit("--idle-seconds must be >= 1")
    if args.poll_seconds < 0.5:
        raise SystemExit("--poll-seconds must be >= 0.5")
    if not (1 <= args.mem_threshold_pct <= 100):
        raise SystemExit("--mem-threshold-pct must be 1..100")
    if args.emergency_idle_seconds < 1:
        raise SystemExit("--emergency-idle-seconds must be >= 1")
    if not (1 <= args.emergency_mem_threshold_pct <= 100):
        raise SystemExit("--emergency-mem-threshold-pct must be 1..100")
    if args.emergency_mem_threshold_pct < args.mem_threshold_pct:
        raise SystemExit("emergency HBM threshold must be >= normal HBM threshold")
    if args.emergency_idle_seconds > args.idle_seconds:
        raise SystemExit("emergency idle seconds must be <= normal idle seconds")

    base = f"http://127.0.0.1:{args.port}"
    s = requests.Session()
    idle_since: float | None = None
    flushed_this_idle = False
    cycles = 0

    log(
        "guard start "
        f"port={args.port} gpus={args.gpu_id} idle_s={args.idle_seconds:g} "
        f"mem_threshold={args.mem_threshold_pct}% emergency_idle_s={args.emergency_idle_seconds:g} "
        f"emergency_mem_threshold={args.emergency_mem_threshold_pct}% poll_s={args.poll_seconds:g}"
    )

    while True:
        cycles += 1
        if args.max_cycles and cycles > args.max_cycles:
            log("max cycles reached; exit")
            return 0

        try:
            state = read_loads(s, base, args.http_timeout)
        except Exception as exc:
            idle_since = None
            flushed_this_idle = False
            log(f"loads unavailable: {type(exc).__name__}: {exc}")
            time.sleep(args.poll_seconds)
            continue

        if not state.fully_idle:
            if idle_since is not None or flushed_this_idle:
                log(
                    "activity observed; re-arm "
                    f"running={state.running} waiting={state.waiting} used={state.used_tokens}"
                )
            idle_since = None
            flushed_this_idle = False
            time.sleep(args.poll_seconds)
            continue

        now = time.monotonic()
        if idle_since is None:
            idle_since = now
            log("fully idle; grace timer started")

        if flushed_this_idle:
            time.sleep(args.poll_seconds)
            continue

        idle_elapsed = now - idle_since
        if idle_elapsed < args.emergency_idle_seconds:
            time.sleep(args.poll_seconds)
            continue

        try:
            mem_by_gpu = {gpu: read_mem_pct(gpu) for gpu in args.gpu_id}
            mem_pct = max(mem_by_gpu.values())
        except Exception as exc:
            log(f"memory telemetry unavailable: {type(exc).__name__}: {exc}")
            time.sleep(args.poll_seconds)
            continue

        emergency_due = (
            idle_elapsed >= args.emergency_idle_seconds
            and mem_pct >= args.emergency_mem_threshold_pct
        )
        normal_due = idle_elapsed >= args.idle_seconds and mem_pct >= args.mem_threshold_pct
        if not emergency_due and not normal_due:
            if idle_elapsed >= args.idle_seconds:
                flushed_this_idle = True
                log(
                    f"normal idle grace reached but HBM={mem_by_gpu} max={mem_pct}% "
                    f"< threshold; no flush"
                )
            time.sleep(args.poll_seconds)
            continue

        flush_reason = "emergency" if emergency_due else "normal"

        # Recheck immediately before the mutating endpoint to minimize the race.
        try:
            before = read_loads(s, base, args.http_timeout)
            if not before.fully_idle:
                log(
                    "flush cancelled by activity race "
                    f"running={before.running} waiting={before.waiting} used={before.used_tokens}"
                )
                idle_since = None
                flushed_this_idle = False
                time.sleep(args.poll_seconds)
                continue
            status, text = flush(s, base, args.flush_timeout)
            flushed_this_idle = True
            time.sleep(1.0)
            try:
                after_by_gpu = {gpu: read_mem_pct(gpu) for gpu in args.gpu_id}
            except Exception:
                after_by_gpu = {}
            log(
                f"flush attempted reason={flush_reason} status={status} "
                f"HBM={mem_by_gpu}->{after_by_gpu or '?'} "
                f"response={json.dumps(text[:240])}"
            )
        except Exception as exc:
            # Fail open: never affect serving if housekeeping itself fails.
            flushed_this_idle = True
            log(f"flush housekeeping failed-open: {type(exc).__name__}: {exc}")

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
