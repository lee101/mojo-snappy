import inspect
import os
import ctypes

import numpy as np
import pytest
import snappy as upstream

import mojo_snappy as snappy
from mojo_snappy._lib import buffer_address, lib


def patterned(size=300_000):
    record = (
        b'{"time":"2026-07-29T12:00:00Z","level":"info",'
        b'"service":"codec","status":200}\n'
    )
    return (record * (size // len(record) + 1))[:size]


def test_public_signatures_match_upstream_block_api():
    assert inspect.signature(snappy.compress) == inspect.signature(upstream.compress)
    assert inspect.signature(snappy.uncompress) == inspect.signature(upstream.uncompress)
    assert snappy.decompress is snappy.uncompress


@pytest.mark.parametrize(
    ("encoded", "decoded"),
    [
        (b"\x00", b""),
        (b"\x05\x10Hello", b"Hello"),
        (b"\x05\x00a\x01\x01", b"aaaaa"),
        (b"\x41\x00a\xfe\x01\x00", b"a" * 65),
        (b"\x02\x00a\x03\x01\x00\x00\x00", b"aa"),
    ],
)
def test_published_format_sequences(encoded, decoded):
    assert snappy.decompress(encoded) == decoded
    assert upstream.decompress(encoded) == decoded


@pytest.mark.parametrize(
    "source",
    [
        b"",
        b"a",
        b"hello world",
        b"a" * 100_000,
        bytes(range(256)) * 1000,
        patterned(),
        os.urandom(300_000),
    ],
)
def test_mojo_blocks_decode_upstream(source):
    encoded = snappy.compress(source)
    assert upstream.decompress(encoded) == source


@pytest.mark.parametrize(
    "source",
    [
        b"",
        b"small input",
        b"abc" * 100_000,
        bytes(range(251)) * 1700,
        patterned(1_000_000),
        os.urandom(250_000),
    ],
)
def test_upstream_blocks_decode_mojo(source):
    encoded = upstream.compress(source)
    assert snappy.decompress(encoded) == source


@pytest.mark.parametrize(
    "size",
    [0, 1, 3, 4, 11, 12, 59, 60, 61, 255, 256, 4096, 65_535, 65_536, 65_537],
)
def test_boundary_sizes_cross_compatible(size):
    source = patterned(size)
    ours = snappy.compress(source)
    theirs = upstream.compress(source)
    assert upstream.decompress(ours) == source
    assert snappy.decompress(theirs) == source


@pytest.mark.parametrize("size", [31, 32, 33, 63, 64, 65, 95, 96, 97])
def test_simd_copy_tails_cross_compatible(size):
    source = bytes((i * 131 + 17) & 255 for i in range(size))
    ours = snappy.compress(source)
    theirs = upstream.compress(source)
    assert upstream.decompress(ours) == source
    assert snappy.decompress(theirs) == source


def test_hash_table_generations_isolate_consecutive_calls():
    sources = [
        patterned(4096),
        os.urandom(4096),
        b"\0" * 4096,
        patterned(65_537),
    ]
    for _ in range(4):
        for source in sources:
            assert upstream.decompress(snappy.compress(source)) == source


def test_randomized_cross_compatibility():
    rng = np.random.default_rng(20260729)
    for _ in range(75):
        size = int(rng.integers(0, 150_000))
        source = rng.integers(0, 32, size=size, dtype=np.uint8).tobytes()
        assert upstream.decompress(snappy.compress(source)) == source
        assert snappy.decompress(upstream.compress(source)) == source


def test_str_encoding_and_decoding_match_upstream():
    text = "Mojo says héllo"
    encoded = snappy.compress(text, encoding="utf-8")
    assert upstream.decompress(encoded).decode("utf-8") == text
    assert snappy.decompress(encoded, decoding="utf-8") == text
    with pytest.raises(snappy.UncompressError, match="only possible"):
        snappy.decompress(encoded.decode("latin1"))


def test_memoryview_and_numpy_buffers():
    source = np.arange(200_000, dtype=np.uint32)
    expected = source.tobytes()
    address, size, keepalive = buffer_address(source)
    assert address == source.ctypes.data
    assert size == source.nbytes
    encoded = snappy.compress(memoryview(source))
    _ = keepalive
    assert upstream.decompress(encoded) == expected
    assert snappy.decompress(memoryview(upstream.compress(expected))) == expected


def test_noncontiguous_buffer_is_rejected_before_ffi():
    source = np.arange(100, dtype=np.uint8)[::2]
    with pytest.raises(BufferError, match="contiguous"):
        snappy.compress(source)


def test_multibyte_numpy_dtype_uses_nbytes_without_narrowing():
    source = np.array([0x0102, 0xA0B0, 0xFF00], dtype=np.uint16)
    encoded = snappy.compress(source)
    assert upstream.decompress(encoded) == source.tobytes()


def test_c_abi_rejects_null_pointers_and_negative_lengths():
    library = lib()
    byte = ctypes.c_uint8()
    address = ctypes.addressof(byte)
    table = (ctypes.c_uint64 * ((1 << 14) + 1))()
    assert library.msn_uncompressed_length(0, 0) < 0
    assert library.msn_uncompressed_length(address, -1) < 0
    assert library.msn_decompress(0, 0, address, 0) < 0
    assert library.msn_decompress(address, -1, address, 0) < 0
    assert library.msn_compress(address, 0, 0, 1, ctypes.addressof(table)) < 0
    assert library.msn_compress(address, 0, address, -1, ctypes.addressof(table)) < 0


@pytest.mark.parametrize(
    "encoded",
    [
        b"",
        b"\x80",
        b"\x80\x80\x80\x80\x10",
        b"\x05",
        b"\x05\x10Hell",
        b"\x05\x00a\x01",
        b"\x05\x00a\x01\x00",
        b"\x05\x00a\x01\x02",
        b"\x01\x00a\x00x",
        b"\x02\x00a",
    ],
)
def test_malformed_blocks_raise_and_validate_false(encoded):
    with pytest.raises(snappy.UncompressError):
        snappy.decompress(encoded)
    assert not snappy.isValidCompressed(encoded)


def test_validation_matches_upstream():
    valid = [upstream.compress(b""), upstream.compress(patterned()), snappy.compress(b"abc" * 50)]
    invalid = [b"", b"not compressed", b"\xff\xff\xff\xff\xff"]
    for encoded in valid + invalid:
        assert snappy.isValidCompressed(encoded) == upstream.isValidCompressed(encoded)


def test_compressor_uses_matches_and_64k_fragments():
    source = patterned(2_000_000)
    encoded = snappy.compress(source)
    assert len(encoded) < len(source) // 8
    assert upstream.decompress(encoded) == source


def test_exception_type_is_upstream_compatible():
    assert issubclass(snappy.UncompressError, Exception)
    with pytest.raises(snappy.UncompressError):
        snappy.uncompress(b"bad")
