# mojo-snappy

Snappy block compression implemented in
[Mojo](https://www.modular.com/mojo) and exposed to Python through a small
`ctypes` layer. It produces the standard raw Snappy block format and mirrors
the covered one-shot names and signatures from
[`python-snappy`](https://github.com/intake/python-snappy).

```python
import mojo_snappy as snappy

source = b"Mojo compresses this data. " * 1000
compressed = snappy.compress(source)
assert snappy.decompress(compressed) == source
```

Mojo-produced blocks decode with `python-snappy`, and blocks produced by
`python-snappy` decode with Mojo. The implementation is independent: it does
not link Google Snappy, cramjam, or another compression library.

## Coverage

| `python-snappy` API | status |
| --- | --- |
| `compress(data, encoding="utf-8")` | covered |
| `uncompress(data, decoding=None)` | covered |
| `decompress` alias | covered |
| `isValidCompressed(data)` | covered |
| `UncompressError` | covered |
| contiguous buffer-protocol inputs | covered, including `memoryview` and NumPy arrays |

This repository intentionally covers raw Snappy blocks only. The framed-stream
classes, Hadoop stream classes, `stream_compress`, `stream_decompress`, file
format detection, and the command-line interface are not implemented. Snappy
framing adds chunk headers and checksums around the block codec and is outside
the stated scope.

The decoder accepts literal tags and all three copy forms from the
[official block format](https://github.com/google/snappy/blob/main/format_description.txt).
It rejects truncated length prefixes and tags, zero or out-of-range offsets,
output overruns, trailing commands, and declared-size mismatches.

## Install and run

The repository pins its Mojo nightly and Python environment:

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

`pixi run build` compiles the single Mojo source file to
`dist/libmojo-snappy.so`. The pixi environment sets `PYTHONPATH=python`, so the
usage example runs directly from the checkout.

## Performance

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux
x86-64, Python 3.13.14, Mojo `1.1.0.dev2026081105`, and upstream
`python-snappy` 0.7.3. Each result is the best of seven timed runs after one
warmup. Decode cases use the exact same upstream-produced compressed block.
Relative is upstream time divided by mojo-snappy time, so values above 1 mean
Mojo was faster.

| case | mojo-snappy | python-snappy | relative |
| --- | ---: | ---: | ---: |
| compress, repetitive 8 MiB | 0.960 ms | 1.492 ms | 1.56x |
| decompress, repetitive 8 MiB | 0.957 ms | 2.086 ms | 2.18x |
| compress, random 8 MiB | 1.906 ms | 1.987 ms | 1.04x |
| decompress, random 8 MiB | 0.907 ms | 1.637 ms | 1.80x |
| compress, repetitive 4 KiB | 0.005 ms | 0.005 ms | 1.04x |
| decompress, repetitive 4 KiB | 0.004 ms | 0.004 ms | 0.96x |

Mojo wins five of these six measured cases. The 4 KiB decode is within
measurement noise of parity; fixed Python-to-FFI call overhead dominates such
short work.

No multithreaded or GPU path is included. Decode commands depend on previously
produced bytes. Encoding fragments are independent, but a measured parallel
prototype required fixed-slot temporary output and a compaction copy; that
regressed incompressible input enough to outweigh its compressible-data gain.
Snappy hashing and copying are memory-bound and below the roughly two
flops-per-byte threshold where GPU transfer and launch overhead can pay off, so
a GPU path is not justified.

## How it works

`src/snappy.mojo` is one compilation unit containing a greedy Snappy encoder,
a bounds-checked decoder, and three C ABI exports. The encoder writes the
uncompressed-size varint, divides input into 64 KiB fragments, and uses a
16,384-entry hash table to find four-byte LZ77 matches. Matches become standard
one- or two-byte-offset copy commands; incompressible ranges become literal
commands. Four-way-unrolled SIMD copies literals and non-overlapping match
ranges, SIMD extends matches, and scalar tails handle remainders and short
overlapping matches.

Python owns every allocation. A contiguous source buffer crosses the ABI as an
integer address and is reconstructed in Mojo as
`UnsafePointer[UInt8, AnyOrigin[mut=True]]`. Compression writes into the
standard Snappy maximum-output allocation. Large results resize that allocation
in place instead of copying it into a second `bytes` object; small results use
the lower-overhead normal Python allocation path.
Decompression reads the declared size first, allocates one exact Python
`bytes` object, and Mojo fills that storage directly. The hash table is
thread-local and uses generation-tagged entries, avoiding a full table clear
for every fragment while keeping consecutive calls isolated. Concurrent calls
do not share mutable scratch memory. Mojo does not retain Python pointers or
allocate heap memory.

## License

MIT
