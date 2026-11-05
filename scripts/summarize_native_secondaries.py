"""Summarize disjoint shard diagnostics per sampling seed; no E2E or physics acceptance verdict."""
import argparse
import json
from pathlib import Path

import numpy as np

from crvfastsim.tier1b_assets import sha256


def summarize(records):
    if not records:
        raise ValueError("no secondary diagnostics")
    groups = {}
    for record in records:
        seed = record["seed"]
        report = record["generator"]
        path = report["path"]
        group = groups.setdefault(seed, [])
        if any(r["generator"]["path"] == path for r in group):
            raise ValueError("duplicate generator shard within seed")
        if record["scope"] != "truth_primary_conditioned_secondary_diagnostic_not_transport_or_E2E_validation":
            raise ValueError("unsupported diagnostic scope")
        if record["tier1b"] != records[0]["tier1b"] or record["run_config"] != records[0]["run_config"]:
            raise ValueError("mixed native library/configuration")
        if set(record["paired"]) != set(records[0]["paired"]):
            raise ValueError("inconsistent sector/origin coverage")
        group.append(record)
    result = {}
    for seed, rows in sorted(groups.items()):
        n = np.asarray([r["generated"] for r in rows], dtype=np.float64)
        if np.any(n <= 0):
            raise ValueError("empty generator shard")
        paired = {}
        for label in sorted(rows[0]["paired"]):
            paired[label] = {}
            for metric in ("energy", "steps"):
                values = [r["paired"][label][metric] for r in rows]
                reference = np.asarray([v["reference_sum"] for v in values])
                candidate = np.asarray([v["candidate_sum"] for v in values])
                delta = candidate - reference
                se = np.asarray([v["paired_se"] for v in values], dtype=float)
                paired[label][metric] = {"reference_sum": float(reference.sum()),
                    "candidate_sum": float(candidate.sum()),
                    "ratio": float(candidate.sum() / reference.sum()) if reference.sum() else None,
                    "delta_per_generated": float(delta.sum() / n.sum()),
                    "conditional_event_se": float(np.sqrt(np.sum((n * se)**2)) / n.sum()),
                    "equal_exposure_shard_se": float((delta / n).std(ddof=1) / np.sqrt(len(rows)))
                    if len(rows) > 1 and np.all(n == n[0]) else None}
        result[str(seed)] = {"shards": len(rows), "generated": int(n.sum()), "paired": paired,
                            "desc_clamped": sum(r["sample_counts"]["desc_clamped"] for r in rows)}
    return {"run_config": records[0]["run_config"], "tier1b": records[0]["tier1b"], "seeds": result,
            "scope": "conditional_truth_primary_secondary_diagnostics_not_E2E_admission",
            "limitations": ["Sampling seeds are separate repeated arms, never additional exposure.",
                            "Event SE conditions on the frozen library; shard SE is descriptive at equal exposure.",
                            "No finite-library/systematic uncertainty, shape acceptance or physics pass gate."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    paths = [Path(p).resolve() for p in args.inputs]
    before = {str(p): sha256(p) for p in paths}
    if len(before) != len(paths):
        raise ValueError("duplicate input path")
    result = summarize([json.loads(p.read_text()) for p in paths])
    if {str(p): sha256(p) for p in paths} != before:
        raise ValueError("input drift during summary")
    result["input_sha256"] = before
    with out.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
