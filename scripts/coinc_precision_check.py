"""Coincidence clusters from saved pulses (chain_identity run --detail --save-clusters DIR writes pulses_<shard>.npz),
with this process's device precision and dispatch settings, so the coincidence kernel alone can be compared across
MU2EFS_DEVICE_FLOAT and CRVFASTSIM_COMPACT_DISPATCH (compare outputs with chain_identity.py clusters).

    python3 scripts/coinc_precision_check.py PULSE_DIR OUT_DIR --calib crvcalib.csv [--deviation crvphoton.csv]
"""
import argparse
from pathlib import Path

import numpy as np

from crvfastsim import chain, runconfig
from crvfastsim.reco import coinc as C, coinc_jax as CJ


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pulses")
    p.add_argument("out")
    p.add_argument("--calib", required=True)
    p.add_argument("--deviation")
    a = p.parse_args()
    rc = runconfig.get()
    geom, ccfg = rc.geometry(), rc.counters_cfg()
    G = C.sector_geom(geom, ccfg)
    cond = chain.Conditions.from_csv(a.calib, a.deviation)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    for f in sorted(Path(a.pulses).glob("pulses_*.npz")):
        d = np.load(f)
        o = np.argsort(d["ev"], kind="stable")
        cols = [d[k][o] for k in ("ev", "idx", "bar", "sipm", "time", "pe", "seq")]
        r = CJ.finish(CJ.find_async(*cols, G, cond.status, C.FinderConfig(), with_pos=True))
        np.savez(out / f.name.replace("pulses_", "clusters_"), **{k: np.asarray(v) for k, v in r.items()
                                                                    if k.startswith("cl_")})
        print(f.name, len(r["cl_event"]), "clusters", flush=True)


if __name__ == "__main__":
    main()
