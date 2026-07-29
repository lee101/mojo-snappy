"""ctypes bridge to the Mojo Snappy block kernels."""

from __future__ import annotations

import ctypes
import os
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB_PATH = os.path.join(ROOT, "dist", "libmojo-snappy.so")

I = ctypes.c_int64
_I64_MAX = (1 << 63) - 1
_SNAPPY_MAX_INPUT = (1 << 32) - 1
_RAW_OUTPUT_THRESHOLD = 64 * 1024

_new_bytes = ctypes.pythonapi.PyBytes_FromStringAndSize
_new_bytes.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t]
_new_bytes.restype = ctypes.py_object
_bytes_data = ctypes.pythonapi.PyBytes_AsString
_bytes_data.argtypes = [ctypes.py_object]
_bytes_data.restype = ctypes.c_void_p

_raw_api = ctypes.PyDLL(None)
_new_bytes_raw = _raw_api.PyBytes_FromStringAndSize
_new_bytes_raw.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t]
_new_bytes_raw.restype = ctypes.c_void_p
_bytes_data_raw = _raw_api.PyBytes_AsString
_bytes_data_raw.argtypes = [ctypes.c_void_p]
_bytes_data_raw.restype = ctypes.c_void_p
_resize_bytes_raw = _raw_api._PyBytes_Resize
_resize_bytes_raw.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_ssize_t,
]
_resize_bytes_raw.restype = ctypes.c_int
_decref_raw = _raw_api.Py_DecRef
_decref_raw.argtypes = [ctypes.c_void_p]
_decref_raw.restype = None


class _PyBuffer(ctypes.Structure):
    _fields_ = [
        ("buf", ctypes.c_void_p),
        ("obj", ctypes.py_object),
        ("len", ctypes.c_ssize_t),
        ("itemsize", ctypes.c_ssize_t),
        ("readonly", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("format", ctypes.c_char_p),
        ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
        ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
        ("suboffsets", ctypes.POINTER(ctypes.c_ssize_t)),
        ("internal", ctypes.c_void_p),
    ]


_get_buffer = ctypes.pythonapi.PyObject_GetBuffer
_get_buffer.argtypes = [ctypes.py_object, ctypes.POINTER(_PyBuffer), ctypes.c_int]
_get_buffer.restype = ctypes.c_int
_release_buffer = ctypes.pythonapi.PyBuffer_Release
_release_buffer.argtypes = [ctypes.POINTER(_PyBuffer)]
_release_buffer.restype = None


class _BufferExport:
    __slots__ = ("buffer", "exported")

    def __init__(self, data):
        self.buffer = _PyBuffer()
        self.exported = False
        _get_buffer(data, ctypes.byref(self.buffer), 0)
        self.exported = True

    def __del__(self):
        if self.exported:
            _release_buffer(ctypes.byref(self.buffer))


_SIGNATURES = {
    "msn_compress": ([I, I, I, I, I], I),
    "msn_uncompressed_length": ([I, I], I),
    "msn_decompress": ([I, I, I, I], I),
}


class LibraryError(RuntimeError):
    pass


_library: ctypes.CDLL | None = None
_thread_state = threading.local()


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB_PATH):
            raise LibraryError("shared library is missing; run `pixi run build`")
        _library = ctypes.CDLL(LIB_PATH)
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def writable_bytes(size: int) -> tuple[bytes, int]:
    if not 0 <= size <= _I64_MAX:
        raise OverflowError("buffer size is outside the native ABI range")
    data = _new_bytes(None, size)
    address = int(_bytes_data(data))
    if not address:
        raise MemoryError("Python returned a null bytes buffer")
    return data, address


def _allocate_bytes(size: int) -> tuple[int, int]:
    if not 0 <= size <= _I64_MAX:
        raise OverflowError("buffer size is outside the native ABI range")
    owner = int(_new_bytes_raw(None, size))
    if not owner:
        raise MemoryError("Python returned a null bytes object")
    address = int(_bytes_data_raw(owner))
    if not address:
        _decref_raw(owner)
        raise MemoryError("Python returned a null bytes buffer")
    return owner, address


def _finish_bytes(owner: int, size: int | None = None) -> bytes:
    pointer = ctypes.c_void_p(owner)
    if size is not None:
        status = _resize_bytes_raw(ctypes.byref(pointer), size)
        if status != 0 or not pointer.value:
            # _PyBytes_Resize releases the original reference and clears the
            # pointer on failure.
            raise MemoryError("Python could not resize the bytes object")
    result = ctypes.cast(pointer, ctypes.py_object).value
    _decref_raw(pointer)
    return result


def bytes_address(data: bytes) -> tuple[int, object]:
    address = _bytes_data(data)
    if not address:
        raise BufferError("bytes object returned a null pointer")
    return int(address), data


def buffer_address(data) -> tuple[int, int, object]:
    if isinstance(data, bytes):
        storage = data or b"\0"
        address, keepalive = bytes_address(storage)
        return address, len(data), keepalive
    try:
        view = memoryview(data)
    except TypeError as exc:
        raise TypeError("a bytes-like object is required") from exc
    if not view.c_contiguous:
        raise BufferError("source buffer is not C-contiguous")
    view = view.cast("B")
    if not view:
        storage = b"\0"
        address, keepalive = bytes_address(storage)
        return address, 0, (view, keepalive)
    keepalive = _BufferExport(view)
    address = int(keepalive.buffer.buf)
    size = int(keepalive.buffer.len)
    if not address:
        raise BufferError("buffer exporter returned a null pointer")
    if not 0 <= size <= _I64_MAX:
        raise OverflowError("buffer size is outside the native ABI range")
    return address, size, (view, keepalive)


def _hash_table():
    table = getattr(_thread_state, "hash_table", None)
    if table is None:
        table = (ctypes.c_uint64 * ((1 << 14) + 1))()
        _thread_state.hash_table = table
    return table


def compress_raw(data) -> bytes:
    source_address, source_size, source_keepalive = buffer_address(data)
    if source_size > _SNAPPY_MAX_INPUT:
        raise OverflowError("Snappy block input exceeds the format limit")
    library = lib()
    capacity = 32 + source_size + source_size // 6
    owner = 0
    destination = b""
    if source_size >= _RAW_OUTPUT_THRESHOLD:
        owner, destination_address = _allocate_bytes(max(capacity, 1))
    else:
        destination, destination_address = writable_bytes(max(capacity, 1))
    table = _hash_table()
    result = library.msn_compress(
        source_address,
        source_size,
        destination_address,
        capacity,
        ctypes.addressof(table),
    )
    _ = source_keepalive
    if result < 0:
        if owner:
            _decref_raw(owner)
        raise LibraryError(f"compression failed with error {result}")
    if owner:
        return _finish_bytes(owner, result)
    return destination[:result]


def decompress_raw(data) -> bytes:
    source_address, source_size, source_keepalive = buffer_address(data)
    if isinstance(data, bytes):
        if source_size == 0:
            expected = -1
        else:
            byte = data[0]
            expected = byte & 127
            if byte >= 128:
                if source_size < 2:
                    expected = -1
                else:
                    byte = data[1]
                    expected |= (byte & 127) << 7
                    if byte >= 128:
                        if source_size < 3:
                            expected = -1
                        else:
                            byte = data[2]
                            expected |= (byte & 127) << 14
                            if byte >= 128:
                                if source_size < 4:
                                    expected = -1
                                else:
                                    byte = data[3]
                                    expected |= (byte & 127) << 21
                                    if byte >= 128:
                                        if source_size < 5 or data[4] > 15:
                                            expected = -1
                                        else:
                                            expected |= data[4] << 28
    else:
        expected = int(lib().msn_uncompressed_length(source_address, source_size))
    if expected < 0:
        raise LibraryError("invalid Snappy uncompressed-length prefix")
    if expected < _RAW_OUTPUT_THRESHOLD:
        destination = bytes(max(expected, 1))
        destination_address = int(_bytes_data(destination))
    else:
        destination, destination_address = writable_bytes(expected)
    result = lib().msn_decompress(
        source_address,
        source_size,
        destination_address,
        expected,
    )
    _ = source_keepalive
    if result < 0:
        raise LibraryError(f"decompression failed with error {result}")
    if result == 0:
        return b""
    if result != expected:
        raise LibraryError("decompressor produced an unexpected size")
    return destination
