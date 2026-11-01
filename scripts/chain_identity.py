"""Per-stage content hashes of runchain passes, to show two code versions give the same run per seed.

Wraps the stage functions (tier-1b sampling, charges, digis, reco waveforms, clusters) without changing them and
hashes every returned table column (dtype, shape, bytes). `compare` reports the first stage that differs per shard.

    python3 scripts/chain_identity.py run --inputs-root ROOT --out A.json --shards 22 24 ... [--jax ...]
    python3 scripts/chain_identity.py compare A.json B.json
"""
import argparse
import functools
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

STAGES = ("tier1b", "charges", "digis", "reco_waveforms", "clusters")


def digest(obj, h=None):
    h = h or hashlib.sha256()
    if isinstance(obj, pd.DataFrame):
        for c in obj.columns:
            digest(obj[c].to_numpy(), h.update(str(c).encode()) or h)
    elif isinstance(obj, dict):
        for k in sorted(obj):
            h.update(str(k).encode())
            digest(obj[k], h)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            digest(v, h)
    elif isinstance(obj, np.ndarray) or np.isscalar(obj):
        a = np.ascontiguousarray(obj)
        if a.dtype.hasobject:
            a = a.astype(str)
        h.update(f"{a.dtype.str}{a.shape}".encode())
        h.update(a.tobytes())
    elif obj is None:
        h.update(b"None")
    else:
        try:
            digest(np.asarray(obj), h)
        except Exception:
            h.update(repr(type(obj)).encode())
    return h


def run(a):
    from crvfastsim import chain, tier1b
    from crvfastsim.analyses import runchain
    root = Path(a.inputs_root)
    record, current = {}, {}

    def wrap(obj, name, stage, pick):
        f = getattr(obj, name)

        @functools.wraps(f)
        def call(*args, **kw):
            out = f(*args, **kw)
            current.setdefault(stage, []).append(digest(pick(out)).hexdigest())
            return out
        return patch.object(obj, name, call)

    patches = [wrap(tier1b, "sample", "tier1b", lambda o: o[:2]), wrap(chain, "charges", "charges", lambda o: o),
               wrap(chain, "digis", "digis", lambda o: o), wrap(chain, "reco_waveforms", "reco_waveforms", lambda o: o),
               wrap(chain, "clusters", "clusters", lambda o: o[0])]
    if a.tier1b_backend == "jax":
        from crvfastsim import tier1b_jax
        patches.append(wrap(tier1b_jax.DeviceTier1b, "sample", "tier1b", lambda o: o[:2]))
    for p in patches:
        p.start()
    try:
        for s in a.shards:
            current.clear()
            res = runchain.run([str(root / f"gen_even/gen.001400_{s:08d}.root")], str(Path(a.out).parent / f"pass_{s}"),
                               calib=str(root / "data/geometry/crvcalib_sim_best_v1_5_run1400.csv"),
                               deviation=str(root / "data/geometry/crvphoton_sim_best_v1_5_run1400.csv"),
                               gdml=str(root / "data/geometry/mu2e_extracted_v04.gdml"), jax=a.jax, seed=a.seed,
                               max_events=a.max_events, tier1b_set="strict_v4", tier1b_dir=str(root / "tier1b_strict_v2"),
                               transport_backend=a.transport_backend, tier1b_backend=a.tier1b_backend)
            record[str(s)] = {"stages": dict(current), "generated_events": res["generated_events"],
                              "clusters_per_generated": res["clusters_per_generated"]}
            print(json.dumps({"shard": s, **record[str(s)]}), flush=True)
    finally:
        for p in patches:
            p.stop()
    with open(a.out, "x") as f:
        json.dump({"args": vars(a), "shards": record}, f, indent=1)


def compare(a):
    x, y = (json.loads(Path(p).read_text())["shards"] for p in (a.a, a.b))
    bad = {}
    for s in sorted(set(x) | set(y), key=int):
        if s not in x or s not in y:
            bad[s] = "missing"
            continue
        for st in STAGES:
            if x[s]["stages"].get(st) != y[s]["stages"].get(st):
                bad[s] = st
                break
    print(json.dumps({"shards": len(set(x) & set(y)), "identical": not bad, "first_difference": bad}, indent=1))
    return 0 if not bad else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--inputs-root", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--shards", nargs="+", type=int, default=[22])
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--max-events", type=int)
    r.add_argument("--jax", action="store_true")
    r.add_argument("--transport-backend", choices=("numpy", "jax"), default="numpy")
    r.add_argument("--tier1b-backend", choices=("numpy", "jax"), default="numpy")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = p.parse_args()
    sys.exit(run(a) if a.cmd == "run" else compare(a))


if __name__ == "__main__":
    main()
