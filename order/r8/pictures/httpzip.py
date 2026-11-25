"""Read members of a remote zip (or of a byte-split zip in several parts) with HTTP Range requests,
without downloading the whole archive.  usage: HttpFile(urls) -> seekable file for zipfile.ZipFile."""
import io
import time
import urllib.request

UA = {"User-Agent": "layout-benchmark/1.0 (research test)"}


def _open(url, headers, timeout=120):
    req = urllib.request.Request(url, headers={**UA, **headers})
    return urllib.request.urlopen(req, timeout=timeout)


def size_of(url):
    for k in range(6):
        try:
            with _open(url, {"Range": "bytes=0-0"}) as r:
                cr = r.headers.get("Content-Range")
                if cr and "/" in cr:
                    return int(cr.rsplit("/", 1)[1])
                return int(r.headers["Content-Length"])
        except Exception:  # noqa: BLE001
            if k == 5:
                raise
            time.sleep(3 * (k + 1))


class HttpFile(io.RawIOBase):
    """Seekable read-only view over one URL or the concatenation of several URLs (split archives)."""

    def __init__(self, urls, block=1 << 20):
        self.urls = [urls] if isinstance(urls, str) else list(urls)
        self.sizes = [size_of(u) for u in self.urls]
        self.total = sum(self.sizes)
        self.pos = 0
        self.block = block
        self.cache = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.total + off
        return self.pos

    def _fetch(self, start, end):
        """bytes [start, end) of the virtual concatenation"""
        out = bytearray()
        base = 0
        for u, s in zip(self.urls, self.sizes):
            a, b = max(start, base), min(end, base + s)
            if a < b:
                for k in range(8):
                    try:
                        with _open(u, {"Range": f"bytes={a - base}-{b - base - 1}"}) as r:
                            data = r.read()
                        if len(data) != b - a:
                            raise IOError(f"short read {len(data)} of {b - a}")
                        out += data
                        break
                    except Exception:  # noqa: BLE001
                        if k == 7:
                            raise
                        time.sleep(3 * (k + 1))
            base += s
        return bytes(out)

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.total - self.pos
        n = max(0, min(n, self.total - self.pos))
        if n == 0:
            return b""
        if n > 4 * self.block:
            data = self._fetch(self.pos, self.pos + n)
            self.pos += n
            return data
        out = bytearray()
        while len(out) < n:
            bi = self.pos // self.block
            if bi not in self.cache:
                if len(self.cache) > 64:
                    self.cache.clear()
                s = bi * self.block
                self.cache[bi] = self._fetch(s, min(self.total, s + self.block))
            blk = self.cache[bi]
            o = self.pos - bi * self.block
            take = min(n - len(out), len(blk) - o)
            out += blk[o:o + take]
            self.pos += take
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def hf_url(repo, path, kind="datasets"):
    return f"https://huggingface.co/{kind}/{repo}/resolve/main/{path}"
