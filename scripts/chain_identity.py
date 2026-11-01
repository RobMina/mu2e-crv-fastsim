"""Per-stage content hashes of runchain passes, to show two code versions give the same run per seed.

Wraps the stage functions (tier-1b sampling, charges, digis, reco waveforms, clusters) without changing them and
hashes every returned table column (dtype, shape, bytes). `compare` reports the first stage that differs per shard.

    python3 scripts/chain_identity.py run --inputs-root ROOT --out A.json --shards 22 24 ... [--jax ...]
    python3 scripts/chain_identity.py compare A.json B.json
    python3 scripts/chain_identity.py clusters DIR_A DIR_B      (lists saved with run --save-clusters)
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

    saved = {}

    def keep_clusters(o):
        if a.save_clusters:
            saved.setdefault("clusters", []).append({k: np.asarray(v) for k, v in o[0][0].items()
                                                     if k.startswith("cl_")} if o[0] else {})
        return o[0]

    patches = [wrap(tier1b, "sample", "tier1b", lambda o: o[:2]), wrap(chain, "charges", "charges", lambda o: o),
               wrap(chain, "digis", "digis", lambda o: o), wrap(chain, "reco_waveforms", "reco_waveforms", lambda o: o),
               wrap(chain, "clusters", "clusters", keep_clusters)]
    if a.tier1b_backend == "jax":
        from crvfastsim import tier1b_jax
        patches.append(wrap(tier1b_jax.DeviceTier1b, "sample", "tier1b", lambda o: o[:2]))
    for p in patches:
        p.start()
    try:
        for s in a.shards:
            current.clear()
            saved.clear()
            res = runchain.run([str(root / f"gen_even/gen.001400_{s:08d}.root")], str(Path(a.out).parent / f"pass_{s}"),
                               calib=str(root / "data/geometry/crvcalib_sim_best_v1_5_run1400.csv"),
                               deviation=str(root / "data/geometry/crvphoton_sim_best_v1_5_run1400.csv"),
                               gdml=str(root / "data/geometry/mu2e_extracted_v04.gdml"), jax=a.jax, seed=a.seed,
                               max_events=a.max_events, tier1b_set="strict_v4", tier1b_dir=str(root / "tier1b_strict_v2"),
                               transport_backend=a.transport_backend, tier1b_backend=a.tier1b_backend)
            record[str(s)] = {"stages": dict(current), "generated_events": res["generated_events"],
                              "clusters_per_generated": res["clusters_per_generated"]}
            print(json.dumps({"shard": s, **record[str(s)]}), flush=True)
            if a.save_clusters:
                Path(a.save_clusters).mkdir(parents=True, exist_ok=True)
                np.savez(Path(a.save_clusters) / f"clusters_{s}.npz", **saved["clusters"][0])
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


def clusters(a):
    """Cluster lists of two runs (--save-clusters dirs). Shards are aligned per (event, sector type) cell with the same
    number of clusters in both runs (in their stored order); reported: cells/clusters kept and dropped, integer
    mismatches, and per float field the max and RMS |difference| over aligned clusters (cl_pos per component)."""
    tot = {"aligned": 0, "dropped_a": 0, "dropped_b": 0, "integer_mismatch": 0}
    diffs = {}
    same_all = True
    for f in sorted(Path(a.a).glob("clusters_*.npz")):
        x, y = np.load(f), np.load(Path(a.b) / f.name)
        kx = x["cl_event"].astype(np.int64) * 10 + x["cl_type"].astype(np.int64)
        ky = y["cl_event"].astype(np.int64) * 10 + y["cl_type"].astype(np.int64)
        ux, cx = np.unique(kx, return_counts=True)
        uy, cy = np.unique(ky, return_counts=True)
        common, ix, iy = np.intersect1d(ux, uy, return_indices=True)
        good = set(common[cx[ix] == cy[iy]].tolist())
        mx = np.array([k in good for k in kx], bool)
        my = np.array([k in good for k in ky], bool)
        ox, oy = np.flatnonzero(mx)[np.argsort(kx[mx], kind="stable")], np.flatnonzero(my)[np.argsort(ky[my], kind="stable")]
        same_all &= len(kx) == len(ky) and mx.all() and my.all()
        tot["aligned"] += len(ox)
        tot["dropped_a"] += int((~mx).sum())
        tot["dropped_b"] += int((~my).sum())
        for k in x.files:
            if x[k].dtype.kind in "iub":
                tot["integer_mismatch"] += int((x[k][ox] != y[k][oy]).sum())
            else:
                d = np.abs(x[k][ox] - y[k][oy]).reshape(len(ox), -1)
                for c in range(d.shape[1]):
                    diffs.setdefault(f"{k}[{c}]" if d.shape[1] > 1 else k, []).append(d[:, c])
    stats = {k: {"max": float(np.max(np.concatenate(v), initial=0)),
                 "rms": float(np.sqrt(np.mean(np.concatenate(v) ** 2))) if sum(map(len, v)) else 0.0,
                 "changed": int((np.concatenate(v) > 0).sum())} for k, v in diffs.items()}
    same = same_all and tot["integer_mismatch"] == 0
    print(json.dumps({"same_lists": same, **tot, "float": stats}, indent=1))
    return 0 if same else 1


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
    r.add_argument("--save-clusters", help="also write each shard's cluster list (cl_* arrays) to this directory")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    k = sub.add_parser("clusters")
    k.add_argument("a")
    k.add_argument("b")
    a = p.parse_args()
    sys.exit(run(a) if a.cmd == "run" else compare(a) if a.cmd == "compare" else clusters(a))


if __name__ == "__main__":
    main()
