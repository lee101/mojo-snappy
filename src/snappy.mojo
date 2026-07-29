"""Snappy block compression and decompression exposed through a small C ABI."""

from std.sys.info import simd_width_of as simdwidthof

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime U64Ptr = UnsafePointer[UInt64, AnyOrigin[mut=True]]
comptime HASH_BITS = 14
comptime HASH_SIZE = 1 << HASH_BITS


@always_inline
def copy_bytes(dst: BPtr, dst_pos: Int, src: BPtr, src_pos: Int, size: Int):
    comptime BYTE_W = simdwidthof[DType.float64]() * 8
    var i = 0
    while i + BYTE_W <= size:
        var values = src.load[width=BYTE_W, alignment=1](src_pos + i)
        dst.store[alignment=1](dst_pos + i, values)
        i += BYTE_W
    while i < size:
        dst[dst_pos + i] = src[src_pos + i]
        i += 1


@always_inline
def read32(src: BPtr, i: Int) -> UInt32:
    return (
        UInt32(src[i])
        | (UInt32(src[i + 1]) << 8)
        | (UInt32(src[i + 2]) << 16)
        | (UInt32(src[i + 3]) << 24)
    )


@always_inline
def hash_sequence(src: BPtr, i: Int) -> Int:
    return Int((read32(src, i) * UInt32(0x1E35A7BD)) >> (32 - HASH_BITS))


def emit_varint(dst: BPtr, pos: Int, value: Int, capacity: Int) -> Int:
    var op = pos
    var remaining = value
    while remaining >= 128:
        if op >= capacity:
            return -2
        dst[op] = UInt8((remaining & 127) | 128)
        op += 1
        remaining >>= 7
    if op >= capacity:
        return -2
    dst[op] = UInt8(remaining)
    return op + 1


def emit_literal(
    dst: BPtr,
    pos: Int,
    capacity: Int,
    src: BPtr,
    src_pos: Int,
    size: Int,
) -> Int:
    if size == 0:
        return pos

    var header_size = 1
    var encoded_size = size - 1
    if size >= 61:
        var bytes_needed = 1
        if encoded_size >= 1 << 8:
            bytes_needed = 2
        if encoded_size >= 1 << 16:
            bytes_needed = 3
        if encoded_size >= 1 << 24:
            bytes_needed = 4
        header_size += bytes_needed

    if pos + header_size + size > capacity:
        return -2

    var op = pos
    if size < 61:
        dst[op] = UInt8(encoded_size << 2)
        op += 1
    else:
        var bytes_needed = header_size - 1
        dst[op] = UInt8((59 + bytes_needed) << 2)
        op += 1
        for j in range(bytes_needed):
            dst[op] = UInt8((encoded_size >> (8 * j)) & 255)
            op += 1

    copy_bytes(dst, op, src, src_pos, size)
    return op + size


def emit_copy(dst: BPtr, pos: Int, capacity: Int, offset: Int, size: Int) -> Int:
    var op = pos
    var remaining = size
    while remaining > 0:
        var chunk = remaining
        if chunk > 64:
            chunk = 64

        if offset < 2048 and chunk >= 4 and chunk <= 11:
            if op + 2 > capacity:
                return -2
            dst[op] = UInt8(
                1 | ((chunk - 4) << 2) | ((offset >> 8) << 5)
            )
            dst[op + 1] = UInt8(offset & 255)
            op += 2
        else:
            if op + 3 > capacity:
                return -2
            dst[op] = UInt8(2 | ((chunk - 1) << 2))
            dst[op + 1] = UInt8(offset & 255)
            dst[op + 2] = UInt8((offset >> 8) & 255)
            op += 3
        remaining -= chunk
    return op


def compress_fragment(
    src: BPtr,
    fragment_start: Int,
    fragment_end: Int,
    dst: BPtr,
    pos: Int,
    capacity: Int,
    table: U64Ptr,
    generation: UInt64,
) -> Int:
    var write_pos = pos
    var anchor = fragment_start
    var ip = fragment_start
    if ip + 4 <= fragment_end:
        table[hash_sequence(src, ip)] = (generation << 32) | UInt64(ip)
        ip += 1

    var search_attempts = 32
    while ip + 4 <= fragment_end:
        var h = hash_sequence(src, ip)
        var entry = table[h]
        var candidate = Int(entry & UInt64(0xFFFFFFFF))
        table[h] = (generation << 32) | UInt64(ip)

        var matched = False
        if (
            entry >> 32 == generation
            and candidate >= fragment_start
            and ip - candidate <= 65535
        ):
            matched = read32(src, candidate) == read32(src, ip)

        if not matched:
            var skip = search_attempts >> 5
            search_attempts += 1
            ip += skip
            continue

        var op = emit_literal(
            dst, write_pos, capacity, src, anchor, ip - anchor
        )
        if op < 0:
            return op

        var match_size = 4
        comptime BYTE_W = simdwidthof[DType.float64]() * 8
        while ip + match_size + BYTE_W <= fragment_end:
            var candidate_values = src.load[width=BYTE_W, alignment=1](
                candidate + match_size
            )
            var input_values = src.load[width=BYTE_W, alignment=1](
                ip + match_size
            )
            if candidate_values != input_values:
                break
            match_size += BYTE_W
        while (
            ip + match_size < fragment_end
            and src[candidate + match_size] == src[ip + match_size]
        ):
            match_size += 1

        op = emit_copy(dst, op, capacity, ip - candidate, match_size)
        if op < 0:
            return op

        ip += match_size
        anchor = ip
        write_pos = op
        search_attempts = 32

        if ip >= fragment_end:
            break
        if ip >= fragment_start + 1 and ip + 3 < fragment_end:
            table[hash_sequence(src, ip - 1)] = (
                (generation << 32) | UInt64(ip - 1)
            )

    return emit_literal(
        dst, write_pos, capacity, src, anchor, fragment_end - anchor
    )


def compress_block(
    src: BPtr,
    src_size: Int,
    dst: BPtr,
    dst_capacity: Int,
    table: U64Ptr,
) -> Int:
    var op = emit_varint(dst, 0, src_size, dst_capacity)
    if op < 0:
        return op

    var generation = table[HASH_SIZE] + 1
    if generation > UInt64(0xFFFFFFFF):
        comptime W = simdwidthof[DType.float64]()
        var zeros = SIMD[DType.uint64, W](0)
        var i = 0
        while i + W <= HASH_SIZE:
            table.store(i, zeros)
            i += W
        while i < HASH_SIZE:
            table[i] = 0
            i += 1
        generation = 1
    table[HASH_SIZE] = generation

    var fragment_start = 0
    while fragment_start < src_size:
        var fragment_end = fragment_start + 65536
        if fragment_end > src_size:
            fragment_end = src_size
        op = compress_fragment(
            src,
            fragment_start,
            fragment_end,
            dst,
            op,
            dst_capacity,
            table,
            generation,
        )
        if op < 0:
            return op
        fragment_start = fragment_end
    return op


def encoded_length(src: BPtr, src_size: Int) -> Int:
    var value = 0
    var shift = 0
    var ip = 0
    while ip < src_size and ip < 5:
        var byte = Int(src[ip])
        if ip == 4 and byte > 15:
            return -1
        value |= (byte & 127) << shift
        ip += 1
        if byte < 128:
            return value
        shift += 7
    return -1


def decompress_block(
    src: BPtr,
    src_size: Int,
    dst: BPtr,
    dst_capacity: Int,
) -> Int:
    var expected = 0
    var shift = 0
    var ip = 0
    var found_length = False
    while ip < src_size and ip < 5:
        var byte = Int(src[ip])
        if ip == 4 and byte > 15:
            return -1
        expected |= (byte & 127) << shift
        ip += 1
        if byte < 128:
            found_length = True
            break
        shift += 7

    if not found_length:
        return -1
    if expected != dst_capacity:
        return -2

    var op = 0
    while ip < src_size:
        var tag = Int(src[ip])
        ip += 1
        var kind = tag & 3

        if kind == 0:
            var literal_size = (tag >> 2) + 1
            if literal_size >= 61:
                var bytes_needed = literal_size - 60
                if ip + bytes_needed > src_size:
                    return -1
                var encoded_size = 0
                for j in range(bytes_needed):
                    encoded_size |= Int(src[ip + j]) << (8 * j)
                ip += bytes_needed
                literal_size = encoded_size + 1
            if ip + literal_size > src_size:
                return -1
            if op + literal_size > dst_capacity:
                return -2
            copy_bytes(dst, op, src, ip, literal_size)
            ip += literal_size
            op += literal_size
            continue

        var offset: Int
        var match_size: Int
        if kind == 1:
            if ip >= src_size:
                return -1
            match_size = 4 + ((tag >> 2) & 7)
            offset = ((tag & 0xE0) << 3) | Int(src[ip])
            ip += 1
        elif kind == 2:
            if ip + 2 > src_size:
                return -1
            match_size = 1 + (tag >> 2)
            offset = Int(src[ip]) | (Int(src[ip + 1]) << 8)
            ip += 2
        else:
            if ip + 4 > src_size:
                return -1
            match_size = 1 + (tag >> 2)
            offset = (
                Int(src[ip])
                | (Int(src[ip + 1]) << 8)
                | (Int(src[ip + 2]) << 16)
                | (Int(src[ip + 3]) << 24)
            )
            ip += 4

        if offset == 0 or offset > op:
            return -3
        if op + match_size > dst_capacity:
            return -2

        var match_pos = op - offset
        comptime BYTE_W = simdwidthof[DType.float64]() * 8
        if offset >= BYTE_W:
            copy_bytes(dst, op, dst, match_pos, match_size)
        else:
            for j in range(match_size):
                dst[op + j] = dst[match_pos + j]
        op += match_size

    if op != expected:
        return -2
    return op


@export("msn_compress")
def msn_compress(
    src_addr: Int,
    src_size: Int,
    dst_addr: Int,
    dst_capacity: Int,
    table_addr: Int,
) abi("C") -> Int:
    if (
        src_addr == 0
        or dst_addr == 0
        or table_addr == 0
        or src_size < 0
        or dst_capacity < 0
    ):
        return -4
    return compress_block(
        BPtr(unsafe_from_address=src_addr),
        src_size,
        BPtr(unsafe_from_address=dst_addr),
        dst_capacity,
        U64Ptr(unsafe_from_address=table_addr),
    )


@export("msn_uncompressed_length")
def msn_uncompressed_length(src_addr: Int, src_size: Int) abi("C") -> Int:
    if src_addr == 0 or src_size < 0:
        return -4
    return encoded_length(BPtr(unsafe_from_address=src_addr), src_size)


@export("msn_decompress")
def msn_decompress(
    src_addr: Int,
    src_size: Int,
    dst_addr: Int,
    dst_capacity: Int,
) abi("C") -> Int:
    if (
        src_addr == 0
        or dst_addr == 0
        or src_size < 0
        or dst_capacity < 0
    ):
        return -4
    return decompress_block(
        BPtr(unsafe_from_address=src_addr),
        src_size,
        BPtr(unsafe_from_address=dst_addr),
        dst_capacity,
    )
