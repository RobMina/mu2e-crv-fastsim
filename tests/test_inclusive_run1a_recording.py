"""Synthetic ROOT/config and logs only: no art, builds, transport or remote access."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import uproot

from scripts import audit_inclusive_run1a_recording as audit

ROOT = Path(__file__).resolve().parents[1]
HASH = 17297756735122933474
DONE = "Art has completed and will exit with status 0.\n"
GEOMETRY = f"Geometry file: /offline/geom.txt\nGeometry lines: 1729  hash: {HASH}\n"


def row(**updates):
    return {"schema_version": 1, "input_lines": 1729, "input_hash": HASH,
            "steps_limit": 0, "steps_limit_explicit": 0, "allow_replacement": 1,
            "steps_limit_default": 0, "input_file": "/offline/geom.txt",
            "requested_input_file": audit.EXPECTED, "resolved_text": "int other = 1;\n", **updates}


def write_root(path, values=None, types=None, n=1, update=False):
    values, types = values or row(), types or audit.SCHEMA
    with (uproot.update(path) if update else uproot.recreate(path)) as root:
        tree = root.mktree("recording/config", types)
        if n: tree.extend({k: [v] * n if types[k] == "string" else np.full(n, v, dtype=types[k]) for k, v in values.items()})


@pytest.fixture
def evidence(tmp_path):
    paths = [tmp_path / name for name in ("export.root", "transport.log", "exporter.log", "audit.json")]
    write_root(paths[0])
    paths[1].write_text(GEOMETRY + DONE)
    paths[2].write_text(DONE)
    return paths


def test_clean_exact_hash_bindings_and_flags(evidence):
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence[:3]]
    result = audit.run(*evidence)
    assert result["source_derived_cap_gate_passed"] and not result["diagnostics"]
    assert result["original"]["input_hash"] == result["reconstructed"]["input_hash"] == HASH
    assert all(result[flag] is False for flag in audit.FLAGS)
    assert result["bindings_before"] == result["bindings_after"]
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence[:3]]
    assert json.loads(evidence[3].read_text())["original"]["input_hash"] == HASH


@pytest.mark.parametrize("updates", [{"input_lines": 1730}, {"input_hash": HASH + 1},
    {"input_file": "/wrong"}, {"requested_input_file": "wrong"}, {"steps_limit_default": 1},
    {"allow_replacement": 0}, {"steps_limit": 2}, {"steps_limit": -1},
    {"resolved_text": "int g4.stepsSizeLimit = 0;"}, {"steps_limit_explicit": 1},
    {"steps_limit_explicit": 1, "resolved_text": "int g4.stepsSizeLimit = -1;"},
    {"steps_limit_explicit": 1, "resolved_text": "// int g4.stepsSizeLimit = 0;"},
    {"steps_limit_explicit": 1, "resolved_text": "int g4.stepsSizeLimit = 0;\nint g4.stepsSizeLimit = 0;"}])
def test_wellformed_mismatch_is_diagnostic(evidence, updates):
    write_root(evidence[0], row(**updates))
    result = audit.run(*evidence)
    assert not result["source_derived_cap_gate_passed"] and result["diagnostics"]
    assert evidence[3].exists()


@pytest.mark.parametrize("limit", [0, -1, -2147483648, 12])
def test_explicit_active_integer(evidence, limit):
    write_root(evidence[0], row(steps_limit=limit, steps_limit_explicit=1,
                               resolved_text=f"int g4.stepsSizeLimit = {limit};\n"))
    assert audit.run(*evidence)["source_derived_cap_gate_passed"] == (limit <= 0)


@pytest.mark.parametrize("column,dtype", [("schema_version", "int32"), ("input_hash", "float64"),
    ("input_lines", "int64"), ("steps_limit", "int64"), ("steps_limit_explicit", "uint32"),
    ("allow_replacement", "float64"), ("steps_limit_default", "uint32"), ("resolved_text", "int32")])
def test_strict_types(evidence, column, dtype):
    values, types = row(), dict(audit.SCHEMA, **{column: dtype})
    if column == "resolved_text":
        values[column] = 0
    write_root(evidence[0], values, types)
    with pytest.raises(ValueError):
        audit.run(*evidence)
    assert not evidence[3].exists()


@pytest.mark.parametrize("kind", ["rows", "zero_rows", "schema", "flag", "missing_column", "duplicate_file", "duplicate_counts", "float", "overflow",
                                  "incomplete_original", "incomplete_exporter", "malformed_duplicate"])
def test_malformed_refused(evidence, kind):
    if kind in ("rows", "zero_rows", "schema", "flag", "missing_column"):
        values = row(schema_version=2 if kind == "schema" else 1, steps_limit_explicit=2 if kind == "flag" else 0)
        types = dict(audit.SCHEMA)
        if kind == "missing_column":
            del values["resolved_text"], types["resolved_text"]
        write_root(evidence[0], values, types, n={"rows": 2, "zero_rows": 0}.get(kind, 1))
    else:
        suffix = {"duplicate_file": "Geometry file: /extra\n", "duplicate_counts": f"Geometry lines: 1 hash: {HASH}\n",
                  "float": "", "overflow": "", "malformed_duplicate": "Geometry lines: bad\n"}.get(kind, "")
        text = GEOMETRY + DONE + suffix
        if kind == "float":
            text = text.replace(str(HASH), "1.7297756735122933e19")
        if kind == "overflow":
            text = text.replace(str(HASH), str(2**64))
        evidence[1].write_text(text.replace(DONE, "") if kind == "incomplete_original" else text)
        if kind == "incomplete_exporter":
            evidence[2].write_text("unfinished\n")
    with pytest.raises(ValueError):
        audit.run(*evidence)
    assert not evidence[3].exists()


def test_highest_cycle_only(evidence):
    write_root(evidence[0], row(input_lines=99), update=True)
    result = audit.run(*evidence)
    assert result["selected_tree_cycle"].endswith(";2") and not result["source_derived_cap_gate_passed"]


@pytest.mark.parametrize("kind", ["empty", "symlink", "directory", "alias", "existing", "out_alias", "out_symlink", "input_drift", "source_drift"])
def test_files_and_drift(evidence, monkeypatch, kind):
    if kind in ("empty", "symlink", "directory", "alias"):
        evidence[2].unlink()
        if kind == "empty": evidence[2].touch()
        if kind == "directory": evidence[2].mkdir()
        if kind == "symlink": evidence[2].symlink_to(evidence[1])
        if kind == "alias": os.link(evidence[1], evidence[2])
    if kind == "existing": evidence[3].write_text("keep")
    if kind == "out_alias": os.link(evidence[1], evidence[3])
    if kind == "out_symlink": evidence[3].symlink_to(evidence[3].parent / "absent")
    if kind.endswith("drift"):
        original = audit._sha256
        calls = {}
        target = evidence[1] if kind == "input_drift" else Path(audit.__file__).absolute()
        def changing(path):
            calls[path] = calls.get(path, 0) + 1
            return "changed" if path == target and calls[path] > 1 else original(path)
        monkeypatch.setattr(audit, "_sha256", changing)
    with pytest.raises((ValueError, FileExistsError)):
        audit.run(*evidence)


def test_cli_exclusive(evidence):
    command = [sys.executable, str(ROOT / "scripts/audit_inclusive_run1a_recording.py")]
    for flag, path in zip(("export-root", "transport-log", "exporter-log", "out"), evidence):
        command += ["--" + flag, str(path)]
    assert subprocess.run(command, capture_output=True).returncode == 0
    before = evidence[3].read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert evidence[3].read_bytes() == before
