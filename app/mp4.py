"""Move an MP4's `moov` box in front of `mdat` ("faststart"), without ffmpeg.

VHS_VideoCombine writes moov at the end. A browser then cannot start playing
until it has fetched the tail of the file, which looks like a video that loads
forever. Moving moov forward means shifting every chunk offset by its size.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path


def _boxes(buf: bytes, start: int = 0, end: int | None = None):
    end = len(buf) if end is None else end
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack(">I4s", buf[pos:pos + 8])
        hdr = 8
        if size == 1:
            size, = struct.unpack(">Q", buf[pos + 8:pos + 16])
            hdr = 16
        elif size == 0:
            size = end - pos
        if size < hdr:
            raise ValueError("corrupt box")
        yield kind, pos, hdr, size
        pos += size


def _shift(moov: bytearray, start: int, end: int, delta: int) -> None:
    for kind, pos, hdr, size in _boxes(moov, start, end):
        if kind in (b"trak", b"mdia", b"minf", b"stbl", b"edts"):
            _shift(moov, pos + hdr, pos + size, delta)
        elif kind in (b"stco", b"co64"):
            n, = struct.unpack(">I", moov[pos + hdr + 4:pos + hdr + 8])
            fmt, width = (">I", 4) if kind == b"stco" else (">Q", 8)
            for i in range(n):
                o = pos + hdr + 8 + i * width
                v, = struct.unpack(fmt, moov[o:o + width])
                moov[o:o + width] = struct.pack(fmt, v + delta)


def faststart(path: Path) -> bool:
    """Rewrite `path` in place with moov first. Returns True if it changed."""
    buf = path.read_bytes()
    top = list(_boxes(buf))
    kinds = [k for k, *_ in top]
    if b"moov" not in kinds or b"mdat" not in kinds or kinds.index(b"moov") < kinds.index(b"mdat"):
        return False
    _, mpos, _, msize = next(t for t in top if t[0] == b"moov")
    moov = bytearray(buf[mpos:mpos + msize])
    _shift(moov, 8, len(moov), msize)
    rest = [buf[p:p + s] for k, p, _, s in top if k != b"moov"]
    # moov goes right after ftyp (first box), before mdat.
    out = b"".join([rest[0], bytes(moov), *rest[1:]])
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(out)
    tmp.replace(path)
    return True


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(arg, "fixed" if faststart(Path(arg)) else "already ok")
