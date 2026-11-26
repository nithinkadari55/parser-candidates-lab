"""Round 9 probes (counts and sizes only, never table text).
usage: probe9.py tableseq ZIP   -> parameter count and bytes by dtype of the TableSeq checkpoint
       probe9.py access         -> HTTP status of the html9 sources from this runner
       probe9.py lore DIR       -> files and top-level keys of the LORE checkpoints"""
import os, sys, json, zipfile, collections, urllib.request


def tensors(obj, prefix=""):
    import torch
    if isinstance(obj, torch.Tensor):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items(): yield from tensors(v, prefix + "/" + str(k))


def tableseq(z):
    import torch
    d = os.path.dirname(z); zipfile.ZipFile(z).extractall(d)
    p = os.path.join(d, "checkpoints", "ptn.pt")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    print("tableseq top keys", list(ck.keys())[:20] if isinstance(ck, dict) else type(ck).__name__)
    by = collections.Counter(); n = collections.Counter(); groups = collections.Counter()
    for k, t in tensors(ck):
        by[str(t.dtype)] += t.numel() * t.element_size(); n[str(t.dtype)] += t.numel(); groups[k.split("/")[1] if "/" in k else k] += t.numel() * t.element_size()
    print("tableseq bytes by dtype", {k: round(v / 1e6, 1) for k, v in by.items()}, "params", dict(n))
    print("tableseq bytes by top-level key (MB)", {k: round(v / 1e6, 1) for k, v in groups.items()})


def lore(d):
    import torch
    for root, _, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f); print("lore file", os.path.relpath(p, d), round(os.path.getsize(p) / 1e6, 2), "MB")
            if f.endswith((".pth", ".pt")):
                ck = torch.load(p, map_location="cpu", weights_only=False)
                keys = list(ck.keys())[:8] if isinstance(ck, dict) else type(ck).__name__
                sd = ck.get("state_dict", ck) if isinstance(ck, dict) else {}
                print("  keys", keys, "state_dict tensors", sum(1 for _ in tensors(sd)), "first", [k for k, _ in list(tensors(sd))[:5]])


def access():
    urls = ["https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?pagesize=2",
            "https://www.federalregister.gov/api/v1/documents.json?per_page=2",
            "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679",
            "https://www.courtlistener.com/api/rest/v4/search/?q=table"]
    for u in urls:
        try:
            r = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "table-bench/1.0"}), timeout=60)
            print("access", r.status, len(r.read()), u)
        except Exception as e:
            print("access FAIL", repr(e)[:120], u)


if __name__ == "__main__":
    {"tableseq": lambda: tableseq(sys.argv[2]), "access": access, "lore": lambda: lore(sys.argv[2])}[sys.argv[1]]()
