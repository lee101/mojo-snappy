"""Snappy block compression powered by Mojo."""

from __future__ import annotations

from ._lib import LibraryError, compress_raw, decompress_raw

__all__ = [
    "UncompressError",
    "compress",
    "decompress",
    "isValidCompressed",
    "uncompress",
]
__version__ = "0.1.0"


class UncompressError(Exception):
    pass


def compress(data, encoding="utf-8"):
    if isinstance(data, str):
        data = data.encode(encoding)
    return compress_raw(data)


def uncompress(data, decoding=None):
    if isinstance(data, str):
        raise UncompressError("It's only possible to uncompress bytes")
    try:
        result = decompress_raw(data)
    except LibraryError as error:
        raise UncompressError from error
    if decoding:
        return result.decode(decoding)
    return result


decompress = uncompress


def isValidCompressed(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    try:
        decompress(data)
    except UncompressError:
        return False
    return True
