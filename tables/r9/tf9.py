"""Round 9 arm 1: TableFormer through docling-rs 2.84.0, exactly tables/r8/tf8.py (same settings, same matching and
crop rule) over the round-8 and round-9 sets. Writes $TB_WORK/out/r9/tf/tf_<set>_s<k>.jsonl.
usage: tf9.py SHARD NSHARDS [SETS]"""
import os, sys, resource, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common9 as C9
import tf8

if __name__ == "__main__":
    tf8.R8OUT = C9.R9OUT
    sets = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else ",".join(C9.ALL_SETS9)
    sys.argv = sys.argv[:3] + [sets]
    try:
        tf8.main()
    finally:
        d = os.path.join(C9.R9OUT, "tf"); os.makedirs(d, exist_ok=True)
        json.dump(dict(peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), open(os.path.join(d, f"rss_s{sys.argv[1]}.json"), "w"))
