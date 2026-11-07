"""Sequential generator-only extraction from independently selected, already-online ancestors."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locations", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--fcl-dir", required=True)
    parser.add_argument("--area", required=True)
    parser.add_argument("--reuse", action="append", default=[], metavar="NAME=ROOT")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--shards", nargs="+", type=int, help="disjoint source shards assigned to this one-core job")
    args = parser.parse_args()
    if os.environ.get("MUSE_WORK_DIR") != args.area:
        raise RuntimeError("runtime area does not match requested existing build")
    os.environ["FHICL_FILE_PATH"] = args.fcl_dir + ":" + os.environ.get("FHICL_FILE_PATH", "")
    with Path(args.locations).open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    reuse = dict(value.split("=", 1) for value in args.reuse)
    if any(not Path(path).is_file() for path in reuse.values()):
        raise ValueError("a reused generator dump is missing")
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    shards = sorted({int(r["shard"]) for r in rows})
    if args.max_shards is not None:
        shards = shards[:args.max_shards]
    if args.shards is not None:
        shards = [shard for shard in shards if shard in args.shards]
    list_name = "completed_dumps.txt" if args.shards is None else "completed_dumps_" + "_".join(map(str, shards)) + ".txt"
    outputs = []
    for shard in shards:
        selected = [r for r in rows if int(r["shard"]) == shard]
        if any(r["locality"] not in ("ONLINE", "ONLINE_AND_NEARLINE") for r in selected):
            raise ValueError("an input is not already online; no automatic tape recall")
        outputs.extend(reuse[r["name"]] for r in selected if r["name"] in reuse)
        paths = [r["path"] for r in selected if r["name"] not in reuse]
        dest = out / f"generators_{shard:03d}.root"
        log = out / f"generators_{shard:03d}.log"
        config = out / f"generators_{shard:03d}.fcl"
        config.write_text('#include "dump_c4_generators.fcl"\nsource.fileNames: ' + json.dumps(paths) + "\n")
        if paths and not (dest.exists() and log.exists() and "Art has completed and will exit with status 0." in log.read_text()):
            start = time.perf_counter()
            print(f"shard {shard}: reading {len(paths)} files", flush=True)
            with log.open("w") as stream:
                result = subprocess.run(["mu2e", "-c", str(config), "-T", str(dest)], stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"shard {shard} exited {result.returncode}; see {log}")
            print(f"shard {shard}: completed in {time.perf_counter() - start:.3f} s", flush=True)
        if paths:
            outputs.append(str(dest))
        (out / list_name).write_text("\n".join(dict.fromkeys(outputs)) + "\n")
        if (out / "stop_after_current").exists():
            print(f"stopped after completed shard {shard}", flush=True)
            return
    print(f"completed {len(shards)} development shards", flush=True)


if __name__ == "__main__":
    main()
