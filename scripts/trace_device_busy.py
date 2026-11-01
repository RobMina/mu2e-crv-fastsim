"""Device busy time from a JAX profiler trace (perfetto JSON written by profile_prepared_chain.py --trace-pass).

Union of the device-side events per device process, overall and inside each annotated host stage (stage:<name>),
against the stage wall time: the fraction of a stage the accelerator is idle is host work or transfer waits.

    python3 scripts/trace_device_busy.py TRACE.json.gz [--out OUT.json]
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def union_length(iv):
    """Total length of the union of [start, end) intervals."""
    if not iv:
        return 0.0
    iv = np.asarray(sorted(iv), float)
    total, lo, hi = 0.0, iv[0, 0], iv[0, 1]
    for a, b in iv[1:]:
        if a > hi:
            total += hi - lo
            lo, hi = a, b
        else:
            hi = max(hi, b)
    return total + hi - lo


def clip(iv, a, b):
    return [(max(s, a), min(e, b)) for s, e in iv if e > a and s < b]


def analyse(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        d = json.load(f)
    events = d["traceEvents"] if isinstance(d, dict) else d
    names = {e["pid"]: e["args"]["name"] for e in events if e.get("ph") == "M" and e.get("name") == "process_name"}
    device_pids = {p for p, n in names.items() if "/device:" in n and "CPU" not in n.upper().split("/DEVICE:")[-1]}
    dev = [(e["ts"], e["ts"] + e["dur"]) for e in events
           if e.get("ph") == "X" and e.get("pid") in device_pids and e.get("dur")]
    stages = [(e["name"], e["ts"], e["ts"] + e["dur"]) for e in events
              if e.get("ph") == "X" and str(e.get("name", "")).startswith("stage:") and e.get("dur")]
    out = {"device_processes": {str(p): names[p] for p in device_pids}, "device_events": len(dev)}
    if stages:
        lo, hi = min(s for _, s, _ in stages), max(e for _, _, e in stages)
    else:
        lo, hi = min(s for s, _ in dev), max(e for _, e in dev)
    out["span_s"] = (hi - lo) * 1e-6
    out["device_busy_s"] = union_length(clip(dev, lo, hi)) * 1e-6
    per = {}
    for name, a, b in stages:
        r = per.setdefault(name, {"wall_s": 0.0, "device_busy_s": 0.0, "calls": 0})
        r["wall_s"] += (b - a) * 1e-6
        r["device_busy_s"] += union_length(clip(dev, a, b)) * 1e-6
        r["calls"] += 1
    out["stages"] = per
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("--out")
    a = p.parse_args()
    res = analyse(a.trace)
    text = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
