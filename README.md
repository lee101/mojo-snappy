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
x86-64, Python 3.13.14, Mojo `1.0.0b3.dev2026072406`, and upstream
`python-snappy` 0.7.3. Each result is the best of seven timed runs after one
warmup. Decode cases use the exact same upstream-produced compressed block.
Relative is upstream time divided by mojo-snappy time, so values above 1 mean
Mojo was faster.

| case | mojo-snappy | python-snappy | relative |
| --- | ---: | ---: | ---: |
| compress, repetitive 8 MiB | 0.971 ms | 1.846 ms | 1.90x |
| decompress, repetitive 8 MiB | 1.433 ms | 2.134 ms | 1.49x |
| compress, random 8 MiB | 2.686 ms | 2.541 ms | 0.95x |
| decompress, random 8 MiB | 1.681 ms | 1.930 ms | 1.15x |
| compress, repetitive 4 KiB | 0.005 ms | 0.006 ms | 1.06x |
| decompress, repetitive 4 KiB | 0.004 ms | 0.004 ms | 0.98x |

Mojo wins four of these six measured cases. Upstream is slightly faster for
random-data compression and the 4 KiB decode. Fixed Python-to-FFI call overhead
remains significant for such short work.

No multithreaded or GPU path is included. Decode commands depend on previously
produced bytes, while parallel encoding would require per-fragment temporary
outputs followed by another compaction copy. Snappy hashing and copying are
memory-bound and below the arithmetic-intensity threshold where GPU transfer
and launch overhead can pay off, so a GPU path is not justified.

## How it works

`src/snappy.mojo` is one compilation unit containing a greedy Snappy encoder,
a bounds-checked decoder, and three C ABI exports. The encoder writes the
uncompressed-size varint, divides input into 64 KiB fragments, and uses a
16,384-entry hash table to find four-byte LZ77 matches. Matches become standard
one- or two-byte-offset copy commands; incompressible ranges become literal
commands. SIMD extends matches and copies literals and non-overlapping match
ranges, with scalar copying for short overlapping matches.

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
