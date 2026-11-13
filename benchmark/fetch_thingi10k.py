"""Pick a stratified sample of Thingi10K models, write them as STL and make suites/thingi10k.json.

    pip install thingi10k                       # (not a cutmesh dependency)
    python benchmark/fetch_thingi10k.py         # 20 models per tier, seed 0
    python benchmark/fetch_thingi10k.py -n 50 --seed 1 --max-vertices 100000

The first call of thingi10k.init() downloads the whole dataset (npz variant) into the
package's cache (or --cache DIR); later calls reuse it.  Every model keeps its own licence
(recorded in the suite); the STLs go to benchmark/data/thingi10k/, which git ignores.

Tiers (by the dataset's own metadata):
    clean   closed, one component, no self-intersections     expect "valid"
    multi   closed, 2-10 components, no self-intersections   expect "valid"
    selfint closed, self-intersecting                        expect "reject"
    open    not closed                                       expect "reject"
"""
import argparse
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_synthetic import write_binary  # noqa: E402

TIERS = {
    "clean": (dict(closed=True, num_components=1, self_intersecting=False), "valid"),
    "multi": (dict(closed=True, num_components=(2, 10), self_intersecting=False), "valid"),
    "selfint": (dict(closed=True, self_intersecting=True), "reject"),
    "open": (dict(closed=False), "reject"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=20, help="models per tier")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-vertices", type=int, default=200000)
    ap.add_argument("--min-vertices", type=int, default=100)
    ap.add_argument("--tiers", default=",".join(TIERS))
    ap.add_argument("--cache", help="thingi10k cache directory")
    args = ap.parse_args(argv)
    try:
        import thingi10k
    except ImportError:
        print("pip install thingi10k  (then run this again)")
        return 1
    kw = {"cache_dir": args.cache} if args.cache else {}
    thingi10k.init(**kw)
    out = os.path.join(HERE, "data", "thingi10k")
    os.makedirs(out, exist_ok=True)
    rng = random.Random(args.seed)
    cases = []
    for tier in args.tiers.split(","):
        filt, expect = TIERS[tier]
        entries = list(thingi10k.dataset(num_vertices=(args.min_vertices, args.max_vertices), **filt))
        entries.sort(key=lambda e: int(e["file_id"]))
        pick = rng.sample(entries, min(args.n, len(entries)))
        print(f"{tier}: {len(entries)} candidates, {len(pick)} picked")
        for e in pick:
            fid = int(e["file_id"])
            V, F = thingi10k.load_file(e["file_path"])
            V, F = np.asarray(V, float), np.asarray(F, int)
            if F.ndim != 2 or F.shape[1] != 3:
                print(f"  skip {fid}: faces are not triangles")
                continue
            path = os.path.join(out, f"{fid}.stl")
            write_binary(path, V[F], name=f"thingi_{fid}")
            cases.append({"name": f"{tier}_{fid}", "stl": f"../data/thingi10k/{fid}.stl", "expect": expect,
                          "note": f"Thingi10K {fid}, {len(V)} vertices, {len(F)} faces",
                          "license": e.get("license"), "author": e.get("author")})
    suite = {"name": "thingi10k",
             "description": f"Thingi10K sample: {args.n} per tier, seed {args.seed}, "
                            f"{args.min_vertices}-{args.max_vertices} vertices. Regenerate with fetch_thingi10k.py.",
             "defaults": {"cellsAcross": 24, "timeout": 900,
                          "meshDict": {"surfaceRefinement": {"default": 2}}},
             "cases": cases}
    p = os.path.join(HERE, "suites", "thingi10k.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(suite, f, indent=1)
    print(f"{len(cases)} cases -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
