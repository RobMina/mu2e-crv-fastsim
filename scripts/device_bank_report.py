"""Summary of a device tier-1b audit run with a NumPy floor arm (valhist.device_tier1b --floor-seed): per shape the
stephist statistics (Poisson "deck", exact clustered, paired) for device vs NumPy and for the reseeded NumPy floor,
paired step yields per sector group, and the fixed-uniform replay outcome.

    python3 scripts/device_bank_report.py SUMMARY.json [--shapes bar t t2 lt posx]
"""
import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("summary")
    p.add_argument("--shapes", nargs="+", default=["bar", "t", "t2", "lt", "posx", "lE", "E"])
    a = p.parse_args()
    d = json.load(open(a.summary))
    c, f = d["comparison"], d["floor_comparison"]
    print("precision", d.get("precision"))
    for k in a.shapes:
        x, y = c[k], f[k]
        print(f"{k:6s} device: deck {x['deck']:.3f} exact {x['exact']:.3f} paired {x.get('paired', float('nan')):.3f}"
              f" | floor: deck {y['deck']:.3f} exact {y['exact']:.3f} paired {y.get('paired', float('nan')):.3f}")
    for g in ("_total/all", "_total/EX", "_total/T1", "_total/T2", "_total/M"):
        x, y = c[g], f[g]
        print(f"{g:11s} device {100 * x['rel']:+.4f} +- {100 * x['rel_se_paired']:.4f} % | reseed {100 * y['rel']:+.4f} +- "
              f"{100 * y['rel_se_paired']:.4f} % (paired); steps numpy {x['n_ref']:.0f} device {x['n_cand']:.0f} "
              f"reseed {y['n_cand']:.0f}")
    rr = [r["replay"] for r in d["records"]]
    if all("max_abs" in r for r in rr):
        print("replay strict: all", len(rr), "shards, max abs", max(r["max_abs"] for r in rr))
    else:
        rep = [r["report"] for r in rr if "report" in r]
        print("replay report:", len(rep), "shards; steps host", sum(r["steps"]["rows_host"] for r in rep), "device",
              sum(r["steps"]["rows_device"] for r in rep), "; shards with a changed draw count",
              sum("error" in r for r in rr), "; metadata exact on", sum(bool(r["metadata"].get("exact")) for r in d["records"]))
    mom = d.get("moments") or {}
    for arm, m in mom.items():
        print("moments", arm, {k: v for k, v in m.items() if not isinstance(v, (list, dict))})


if __name__ == "__main__":
    main()
