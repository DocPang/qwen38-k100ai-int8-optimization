#!/usr/bin/env python3
"""Launch ``sglang.launch_server`` only if the configured sitecustomize succeeds.

CPython's automatic ``sitecustomize`` import is intentionally fail-open: an
exception is printed as ``Error in sitecustomize`` and interpreter startup
continues. That behavior is unsafe for this research stack because every
runtime patch uses sitecustomize as a correctness/performance contract; a
missing required environment variable can otherwise silently start stock
SGLang and produce invalid benchmark evidence.

The automatic import has already happened before this file runs. If it failed,
``sitecustomize`` is not left successfully imported, so this explicit import
retries it and lets the exception terminate the process. If it succeeded, the
module is cached and this is a no-op. Only then do we execute the standard
``sglang.launch_server`` module in the same interpreter so all monkeypatches
remain active.
"""
from __future__ import annotations

import runpy
import sys
import traceback

def _require_sitecustomize() -> None:
    try:
        import sitecustomize  # noqa: F401
    except BaseException:
        print(
            "FATAL: required sitecustomize runtime patch failed; refusing to launch SGLang",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        raise


# SGLang uses multiprocessing with the ``spawn`` start method. Spawned children
# re-execute this file as ``__mp_main__`` while Python is bootstrapping.  They
# must NOT launch another server, but they must still convert CPython's normally
# fail-open automatic sitecustomize import into a fatal patch contract.
_require_sitecustomize()


def main() -> None:
    runpy.run_module("sglang.launch_server", run_name="__main__", alter_sys=False)


if __name__ == "__main__":
    main()
