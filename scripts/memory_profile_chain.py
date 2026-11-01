"""Host and device memory per stage of one runchain pass (no change to the chain: stage functions are wrapped).

Per stage call: resident set before/after, the peak resident set during the call (VmHWM after resetting it through
/proc/self/clear_refs, Linux), and with JAX the device bytes in use before/after and the allocator peak so far.

    python3 scripts/memory_profile_chain.py --inputs-root ROOT --outdir DIR [--shards 22] [--max-events 50000] \
        [--jax --transport-backend jax --tier1b-backend jax]
"""
import argparse
import functools
import json
import time
from pathlib import Path
from unittest.mock import patch

from crvfastsim import chain, tier1b, tier1b_assets, transport
from crvfastsim.analyses import runchain

KIB = 1024


def _status(field):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * KIB
    return None


def _reset_peak():
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def _device():
    try:
        import jax
        stats = jax.devices()[0].memory_stats() or {}
        return {k: stats.get(k) for k in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")}
    except Exception:
        return None


_OPEN = []


def _fold():
    """Fold the current VmHWM into every open stage before the next reset (nested stages keep true peaks)."""
    hwm = _status("VmHWM")
    for frame in _OPEN:
        frame["peak"] = max(frame["peak"] or 0, hwm or 0)


def wrap(records, function, name):
    @functools.wraps(function)
    def call(*a, **kw):
        before, dev_before = _status("VmRSS"), _device()
        _fold()
        frame = {"peak": None}
        _OPEN.append(frame)
        reset = _reset_peak()
        start = time.perf_counter()
        try:
            return function(*a, **kw)
        finally:
            _fold()
            _OPEN.pop()
            records.append({"stage": name, "seconds": time.perf_counter() - start, "rss_before": before,
                            "rss_after": _status("VmRSS"), "anon_after": _status("RssAnon"),
                            "file_after": _status("RssFile"), "peak_during": frame["peak"] if reset else None,
                            "device_before": dev_before, "device_after": _device()})
    return call


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs-root", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--shards", nargs="+", type=int, default=[22])
    p.add_argument("--max-events", type=int, default=50000)
    p.add_argument("--jax", action="store_true")
    p.add_argument("--transport-backend", choices=("numpy", "jax"), default="numpy")
    p.add_argument("--tier1b-backend", choices=("numpy", "jax"), default="numpy")
    p.add_argument("--passes", type=int, default=1, help="runchain passes in this process (setup repeated each pass)")
    p.add_argument("--deep", nargs="*", default=[], help="also wrap every function defined in these modules "
                   "(e.g. crvfastsim.tier1b); summarised per function as the largest peak above its entry RSS")
    a = p.parse_args()
    root, out = Path(a.inputs_root), Path(a.outdir)
    out.mkdir(parents=True, exist_ok=False)
    records = []
    targets = [(tier1b_assets, "load"), (tier1b, "sample"), (transport, "transport"), (chain, "charges"),
               (chain, "digis"), (chain, "reco_waveforms"), (chain, "clusters"), (runchain, "_load_setup")]
    if a.tier1b_backend == "jax":
        from crvfastsim import tier1b_jax
        targets += [(tier1b_jax.DeviceTier1b, "__init__"), (tier1b_jax.DeviceTier1b, "sample")]
    if a.transport_backend == "jax":
        from crvfastsim import transport_jax
        targets += [(transport_jax.DeviceTransport, "__init__"), (transport_jax, "transport")]
    import importlib
    import inspect
    deep = []
    for name in a.deep:
        mod = importlib.import_module(name)
        deep += [(mod, f) for f, v in vars(mod).items()
                 if inspect.isfunction(v) and v.__module__ == name and (mod, f) not in targets]
    deep_names = {f"{m.__name__}.{f}" for m, f in deep}
    from contextlib import ExitStack
    with ExitStack() as stack:
        for obj, name in targets + deep:
            stack.enter_context(patch.object(obj, name, wrap(records, getattr(obj, name), f"{obj.__name__}.{name}")))
        passes = []
        for i in range(a.passes):
            records.append({"stage": f"pass {i} start", "rss_before": _status("VmRSS"), "device_before": _device()})
            start = time.perf_counter()
            res = runchain.run([str(root / f"gen_even/gen.001400_{s:08d}.root") for s in a.shards],
                               str(out / f"pass_{i}"), calib=str(root / "data/geometry/crvcalib_sim_best_v1_5_run1400.csv"),
                               deviation=str(root / "data/geometry/crvphoton_sim_best_v1_5_run1400.csv"),
                               gdml=str(root / "data/geometry/mu2e_extracted_v04.gdml"), jax=a.jax,
                               max_events=a.max_events, tier1b_set="strict_v4", tier1b_dir=str(root / "tier1b_strict_v2"),
                               transport_backend=a.transport_backend, tier1b_backend=a.tier1b_backend)
            passes.append({"seconds": time.perf_counter() - start, "generated_events": res["generated_events"],
                           "clusters_per_generated": res["clusters_per_generated"]})
    summary = {}
    for r in records:
        if r.get("peak_during") is None:
            continue
        x = summary.setdefault(r["stage"], {"calls": 0, "seconds": 0.0, "max_peak_above_entry": 0, "max_peak": 0})
        x["calls"] += 1
        x["seconds"] += r["seconds"]
        x["max_peak_above_entry"] = max(x["max_peak_above_entry"], r["peak_during"] - r["rss_before"])
        x["max_peak"] = max(x["max_peak"], r["peak_during"])
    result = {"args": vars(a), "passes": passes, "summary": summary, "records": records, "process_peak": _status("VmHWM"),
              "device_final": _device()}
    (out / "memory.json").write_text(json.dumps(result, indent=1))
    for k, x in sorted(summary.items(), key=lambda kv: -kv[1]["max_peak_above_entry"])[:40]:
        print(f"{k:55s} calls {x['calls']:5d} {x['seconds']:8.2f} s  peak above entry "
              f"{x['max_peak_above_entry'] / 2 ** 30:6.3f} GiB  peak {x['max_peak'] / 2 ** 30:6.3f} GiB")
    for r in records:
        if r["stage"] in deep_names:
            continue
        g = lambda k: None if r.get(k) is None else round(r[k] / 2 ** 30, 3)
        dv = r.get("device_after") or {}
        print(f"{r['stage']:40s} rss {g('rss_before')} -> {g('rss_after')} GiB (anon {g('anon_after')}, file "
              f"{g('file_after')}), peak {g('peak_during')} GiB, "
              f"device {None if not dv.get('bytes_in_use') else round(dv['bytes_in_use'] / 2 ** 30, 3)} GiB "
              f"(peak {None if not dv.get('peak_bytes_in_use') else round(dv['peak_bytes_in_use'] / 2 ** 30, 3)})")


if __name__ == "__main__":
    main()
