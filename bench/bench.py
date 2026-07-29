"""Benchmark mojo-snappy against the upstream python-snappy bindings."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time

import numpy as np
import snappy as upstream

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import mojo_snappy as snappy  # noqa: E402


def best_time(function, repetitions=7):
    function()
    best = float("inf")
    result = None
    for _ in range(repetitions):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - start)
    return best, result


def machine():
    model = "unknown CPU"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return (
        f"{model}; {platform.system()} {platform.machine()}; "
        f"Python {platform.python_version()}"
    )


def main():
    size = 8 * 1024 * 1024
    record = (
        b'{"time":"2026-07-29T12:00:00Z","level":"info",'
        b'"service":"compressor","message":"request completed","status":200}\n'
    )
    text = (record * (size // len(record) + 1))[:size]
    random_data = np.random.default_rng(7).integers(
        0, 256, size=size, dtype=np.uint8
    ).tobytes()
    small = text[:4096]

    upstream_text = upstream.compress(text)
    upstream_random = upstream.compress(random_data)
    upstream_small = upstream.compress(small)

    cases = [
        (
            "compress, repetitive 8 MiB",
            lambda: snappy.compress(text),
            lambda: upstream.compress(text),
            text,
        ),
        (
            "decompress, repetitive 8 MiB",
            lambda: snappy.decompress(upstream_text),
            lambda: upstream.decompress(upstream_text),
            text,
        ),
        (
            "compress, random 8 MiB",
            lambda: snappy.compress(random_data),
            lambda: upstream.compress(random_data),
            random_data,
        ),
        (
            "decompress, random 8 MiB",
            lambda: snappy.decompress(upstream_random),
            lambda: upstream.decompress(upstream_random),
            random_data,
        ),
        (
            "compress, repetitive 4 KiB",
            lambda: snappy.compress(small),
            lambda: upstream.compress(small),
            small,
        ),
        (
            "decompress, repetitive 4 KiB",
            lambda: snappy.decompress(upstream_small),
            lambda: upstream.decompress(upstream_small),
            small,
        ),
    ]

    print(f"Machine: {machine()}")
    mojo_version = subprocess.run(
        ["mojo", "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"Mojo: {mojo_version}")
    print(f"Upstream: python-snappy {upstream.__version__}")
    print()
    print("| case | mojo-snappy | python-snappy | relative |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_function, upstream_function, source in cases:
        mojo_seconds, mojo_result = best_time(mojo_function)
        upstream_seconds, upstream_result = best_time(upstream_function)
        if name.startswith("compress"):
            if upstream.decompress(mojo_result) != source:
                raise AssertionError(f"Mojo benchmark stream is invalid for {name}")
            if snappy.decompress(upstream_result) != source:
                raise AssertionError(f"upstream benchmark stream is invalid for {name}")
        elif mojo_result != source or upstream_result != source:
            raise AssertionError(f"benchmark outputs differ for {name}")
        relative = upstream_seconds / mojo_seconds
        print(
            f"| {name} | {mojo_seconds * 1000:.3f} ms | "
            f"{upstream_seconds * 1000:.3f} ms | {relative:.2f}x |"
        )


if __name__ == "__main__":
    main()
