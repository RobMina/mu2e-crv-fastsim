"""Host memory of a loaded tier-1b asset set, per library entry and column (pandas deep usage, NumPy nbytes).

    python3 scripts/library_memory.py --counters C --library-dir D [--tier1b-set strict_v4] [--out OUT.json]
"""
import argparse
import json

import numpy as np
import pandas as pd

from crvfastsim import geometry, tier1b_assets


def sizes(obj, prefix, out):
    if isinstance(obj, pd.DataFrame):
        mem = obj.memory_usage(deep=True, index=True)
        out[prefix] = {"bytes": int(mem.sum()), "rows": len(obj),
                       "columns": {str(c): {"bytes": int(mem[c]), "dtype": str(obj[c].dtype)} for c in obj.columns}}
    elif isinstance(obj, pd.Series):
        out[prefix] = {"bytes": int(obj.memory_usage(deep=True)), "rows": len(obj), "dtype": str(obj.dtype)}
    elif isinstance(obj, np.ndarray):
        out[prefix] = {"bytes": int(obj.nbytes), "shape": list(obj.shape), "dtype": str(obj.dtype)}
    elif isinstance(obj, dict):
        for k, v in obj.items():
            sizes(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            sizes(v, f"{prefix}[{i}]", out)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--counters", required=True)
    p.add_argument("--library-dir", required=True)
    p.add_argument("--tier1b-set", default="strict_v4")
    p.add_argument("--out")
    a = p.parse_args()
    geom = geometry.from_offline_config(a.counters)
    lib = tier1b_assets.load(tier1b_assets.resolve(a.tier1b_set, directory=a.library_dir), geom)
    out = sizes(lib, "", {})
    total = sum(v["bytes"] for v in out.values())
    for k, v in sorted(out.items(), key=lambda kv: -kv[1]["bytes"]):
        print(f"{k:40s} {v['bytes'] / 2 ** 20:9.1f} MiB  rows {v.get('rows', v.get('shape'))}")
        for c, x in sorted(v.get("columns", {}).items(), key=lambda kv: -kv[1]["bytes"])[:60]:
            print(f"    {c:30s} {x['bytes'] / 2 ** 20:8.1f} MiB {x['dtype']}")
    print(f"total {total / 2 ** 30:.3f} GiB")
    if a.out:
        with open(a.out, "x") as f:
            json.dump({"total_bytes": total, "entries": out}, f, indent=1)


if __name__ == "__main__":
    main()
