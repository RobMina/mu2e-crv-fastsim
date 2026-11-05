"""Load and record an explicit native bundle without transport, response or physics admission."""
import argparse
import json
from pathlib import Path

from crvfastsim import runconfig, tier1b_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    rc = runconfig.get(args.run_config)
    spec = tier1b_assets.native_bundle(args.directory, rc)
    lib = tier1b_assets.load(spec, rc.geometry())
    result = {"run_config": rc.name, "tier1b": spec.describe(),
              "anchor": {name: lib["anchor"][name] for name in ("n_roots", "n_multi", "forward_edep_share")},
              "status": "bundle loaded; integrity only, physics validation pending"}
    with out.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
