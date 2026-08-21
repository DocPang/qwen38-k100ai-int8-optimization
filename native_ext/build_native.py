#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
from pathlib import Path

from torch.utils.cpp_extension import load

SRC = Path(os.environ.get("NATIVE_SRC", "/src"))
OUT = Path(os.environ.get("NATIVE_OUT", "/out"))
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = [
    ("k100_int8_gemv_v7_sglang", "k100_int8_gemv_v7.hip"),
    ("k100_int8_gemv_generic_v2_sglang", "k100_int8_gemv_generic_v2.hip"),
    ("k100_int8_gemv_deep_v4_sglang", "k100_int8_gemv_deep_v4.hip"),
    ("k100_int8_gemv_tp4_row_ldsx_v1_sglang", "k100_int8_gemv_tp4_row_ldsx_v1.hip"),
]

for module_name, source_name in TARGETS:
    source = SRC / source_name
    if not source.is_file():
        raise SystemExit(f"missing source: {source}")

    build_dir = Path("/tmp") / f"q38-build-{module_name}"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    print(f"[native] build {module_name} <- {source_name}", flush=True)
    module = load(
        name=module_name,
        sources=[str(source)],
        build_directory=str(build_dir),
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=["-O3", "-std=c++20", "-fno-gpu-rdc"],
        with_cuda=True,
        verbose=True,
    )

    destination = OUT / f"{module_name}.so"
    shutil.copy2(module.__file__, destination)
    print(f"[native] wrote {destination}", flush=True)

expected = {f"{name}.so" for name, _ in TARGETS}
actual = {p.name for p in OUT.glob("*_sglang.so")}
if actual != expected:
    raise SystemExit(f"native output mismatch: expected={sorted(expected)}, actual={sorted(actual)}")

print("[native] all four gfx928 user-space extensions built successfully", flush=True)
