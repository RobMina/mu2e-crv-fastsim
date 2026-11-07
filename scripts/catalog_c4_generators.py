"""Resolve preserved generator ancestors of development ntuples with read-only MetaCat calls."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess


def fetch(client, lookups, path):
    """Cache a bounded batch of file records, including retired intermediate ancestors."""
    if path.exists():
        return json.loads(path.read_text())
    rows = []
    for start in range(0, len(lookups), 200):
        batch = list(client.get_files(lookups[start:start + 200], with_metadata=True, with_provenance=True))
        if any(row is None for row in batch) or len(batch) != len(lookups[start:start + 200]):
            raise ValueError("catalogue lookup did not return every requested file")
        rows.extend(batch)
        print(f"{path.name}: {len(rows)}/{len(lookups)}", flush=True)
    path.write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def parent_ids(records):
    return sorted({str(p["fid"] if isinstance(p, dict) else p) for row in records for p in row.get("parents", [])})


def resolve(client, sources, out):
    """Follow actual ntuple->reco->digi->DTS links, not filename or kinematic matching."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ntuples = fetch(client, [{"did": "mu2e:" + Path(s["source"]).name} for s in sources], out / "ntuples.json")
    mcs = fetch(client, [{"fid": fid} for fid in parent_ids(ntuples)], out / "mcs.json")
    digis = fetch(client, [{"fid": fid} for fid in parent_ids(mcs)], out / "digis.json")
    dts = fetch(client, [{"fid": fid} for fid in parent_ids(digis)], out / "dts.json")
    records = {str(row["fid"]): row for row in [*ntuples, *mcs, *digis, *dts]}
    by_name = {row["name"]: row for row in ntuples}
    rows = []
    for source in sources:
        ntuple = by_name[Path(source["source"]).name]
        reco = [records[k] for k in parent_ids([ntuple])]
        digi = [records[k] for k in parent_ids(reco)]
        for fid in parent_ids(digi):
            row = records[fid]
            if not row["name"].startswith("dts.mu2e.CosmicSignal."):
                raise ValueError("unexpected ancestor tier: " + row["name"])
            rows.append(dict(shard=source["shard"], fid=fid, namespace=row["namespace"], name=row["name"],
                             size=row["size"], retired=row.get("retired", False),
                             subruns=row["metadata"].get("rs.runs", []),
                             events=row["metadata"].get("rse.nevent"), parents=row.get("parents", [])))
    (out / "ancestors.json").write_text(json.dumps(rows, indent=2) + "\n")
    summary = dict(source_ntuples=len(ntuples), reco_files=len(mcs), digi_files=len(digis),
                   distinct_dts_files=len(dts), shard_dts_links=len(rows),
                   input_bytes=sum(r["size"] for r in dts),
                   scope="development-source catalogue ancestry, independent of veto outcome")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def locations(plan, out):
    """Read known-file locality only; do not recall tape or list the namespace."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with Path(plan).open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    pathfile = out / "paths.txt"
    if pathfile.exists():
        paths = pathfile.read_text().splitlines()
    else:
        dids = "\n".join(r["namespace"] + ":" + r["name"] for r in rows) + "\n"
        result = subprocess.run(["mdh", "print-url", "-s", "path", "-l", "tape", "-"],
                                input=dids, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError("mdh print-url failed: " + result.stderr + result.stdout)
        paths = result.stdout.splitlines()
        pathfile.write_text(result.stdout)
    if len(paths) != len(rows):
        raise ValueError("mdh did not return one path per input file")
    known = {}
    dest = out / "locations.tsv"
    if dest.exists():
        with dest.open() as stream:
            known = {r["fid"]: r for r in csv.DictReader(stream, delimiter="\t")}
    with dest.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["fid", "shard", "name", "path", "locality"], delimiter="\t")
        writer.writeheader()
        for i, (row, path) in enumerate(zip(rows, paths)):
            if Path(path).name != row["name"]:
                raise ValueError("mdh path order does not match requested filenames")
            saved = known.get(row["fid"])
            if saved is None:
                pseudo = Path(path).parent / (".(get)(" + Path(path).name + ")(locality)")
                try:
                    locality = pseudo.read_text().strip()
                except OSError as exc:
                    locality = type(exc).__name__
                saved = dict(fid=row["fid"], shard=row["shard"], name=row["name"], path=path, locality=locality)
            writer.writerow(saved)
            if (i + 1) % 100 == 0:
                stream.flush()
                print(f"locality {i + 1}/{len(rows)}", flush=True)
    print(f"locality complete: {len(rows)} files", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", help="primary_summary.json from cached selection recovery")
    parser.add_argument("--locations", help="files.tsv from the outcome-independent input plan")
    parser.add_argument("--split-at", type=int, default=22)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    if args.locations:
        locations(args.locations, args.outdir)
    elif args.sources:
        from metacat.webapi import MetaCatClient
        sources = [s for s in json.loads(Path(args.sources).read_text()) if s["shard"] < args.split_at]
        resolve(MetaCatClient(), sources, args.outdir)
    else:
        parser.error("specify --sources or --locations")


if __name__ == "__main__":
    main()
