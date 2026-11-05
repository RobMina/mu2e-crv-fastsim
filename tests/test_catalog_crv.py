"""Catalog inventory uses only read methods and exposes stale cached counts."""
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location("catalog_crv", Path(__file__).parents[1] / "scripts/catalog_crv.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Client:
    def get_dataset(self, did, exact_file_count):
        assert exact_file_count is False
        return {"file_count": 2}

    def get_dataset_files(self, did, with_metadata):
        assert with_metadata is False
        return iter([{"namespace": "test", "name": "one", "size": 12}])

    def get_file(self, did, with_metadata):
        assert did == "test:one" and with_metadata is True
        return {"metadata": {"gen.count": 10}}


def test_census_reads_lazy_membership_and_reports_count_disagreement():
    result = module.census(Client(), ["test:dataset"])
    assert result["read_only"] is True
    record = result["datasets"][0]
    assert record["files"] == 1 and record["bytes"] == 12
    assert record["cached_count_matches_enumeration"] is False
    assert record["file_names"] == ["test:one"]
    assert record["sample"]["metadata"]["gen.count"] == 10
