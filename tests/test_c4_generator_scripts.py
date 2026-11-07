"""Read-only catalogue traversal and sequential extraction plumbing."""
import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def load(name):
    path = Path(__file__).parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_catalog_follows_retired_intermediates_and_caches(tmp_path):
    module = load("catalog_c4_generators")
    rows = [dict(fid="1", name="source.root", namespace="mu2e", parents=[dict(fid="2")]),
            dict(fid="2", name="reco.art", namespace="mu2e", parents=["3"], retired=True),
            dict(fid="3", name="digi.art", namespace="mu2e", parents=[dict(fid="4")], retired=True),
            dict(fid="4", name="dts.mu2e.CosmicSignal.config.id.art", namespace="mu2e", parents=[],
                 size=100, retired=False, metadata={"rs.runs": [1000002], "rse.nevent": 4})]
    class Client:
        calls = 0
        def get_files(self, lookups, **options):
            assert options == dict(with_metadata=True, with_provenance=True)
            self.calls += 1
            return [next(row for row in rows if row["fid"] == lookup.get("fid") or
                         "mu2e:" + row["name"] == lookup.get("did")) for lookup in lookups]
    client = Client()
    sources = [dict(shard=0, source="source.root")]
    module.resolve(client, sources, tmp_path)
    assert client.calls == 4
    module.resolve(client, sources, tmp_path)
    assert client.calls == 4
    ancestors = json.loads((tmp_path / "ancestors.json").read_text())
    assert ancestors[0]["fid"] == "4" and ancestors[0]["shard"] == 0


def test_existing_dump_is_reused_without_repeating_mu2e(tmp_path, monkeypatch):
    module = load("dump_c4_generator_files")
    area, out = tmp_path / "area", tmp_path / "out"
    out.mkdir()
    locations = tmp_path / "locations.tsv"
    with locations.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["shard", "name", "path", "locality"], delimiter="\t")
        writer.writeheader()
        writer.writerow(dict(shard=0, name="a.art", path="/not-read/a.art", locality="ONLINE"))
        writer.writerow(dict(shard=1, name="b.art", path="/not-read/b.art", locality="NEARLINE"))
    (out / "generators_000.root").write_bytes(b"existing")
    (out / "generators_000.log").write_text("Art has completed and will exit with status 0.\n")
    monkeypatch.setenv("MUSE_WORK_DIR", str(area))
    monkeypatch.setattr(sys, "argv", ["dump", "--area", str(area), "--locations", str(locations),
                                    "--outdir", str(out), "--fcl-dir", str(tmp_path), "--shards", "0"])
    def forbidden(*args, **kwargs):
        raise AssertionError("must not repeat extraction")
    monkeypatch.setattr(module.subprocess, "run", forbidden)
    module.main()
    assert str(out / "generators_000.root") in (out / "completed_dumps_0.txt").read_text()
    monkeypatch.setattr(sys, "argv", sys.argv[:-1] + ["1"])
    with pytest.raises(ValueError, match="not already online"):
        module.main()
