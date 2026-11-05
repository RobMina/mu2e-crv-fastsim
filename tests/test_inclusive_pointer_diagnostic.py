"""Source-only diagnostic scope checks; no art readback or conversion is executed."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def test_raw_pointer_diagnostic_is_readback_only_and_process_explicit():
    text = (ROOT / "art/CrvFastSim/fcl/inspect_inclusive_capture_pointers.fcl").read_text()
    assert '#include "CrvFastSim/fcl/audit_inclusive_run1a.fcl"' in text
    assert "process_name : R3InclusivePointerDiagnostic" in text
    assert "physics.inspect : [ status, pointers, productList ]" in text
    assert 'inputTags : [ "g4run:CRV:R3Inclusive" ]' in text
    assert 'inputTags : [ "g4run::R3Inclusive" ]' in text
    assert "module_type : PointerCheck" in text
    assert "skipDereference : false" in text
    assert re.search(r"physics\.analyzers\.pointers\s*:\s*\{\s*module_type\s*:\s*PointerCheck\s*verbose\s*:\s*1", text)
    assert "verbose : 2" in text
    assert "primaryOnly : false" in text
    assert "pCut : -1" in text and "emPcut : -1" in text
    for name in ("SimParticle", "SimParticlePtr", "StepPointMC", "MCTracjectory", "StrawDigiMC",
                 "CaloHitMC", "CaloShowerStep", "CrvDigiMC"):
        assert f"skip{name} : []" in text
    assert "producers" not in text and "SelectEvents" not in text
    assert "ref_inclusive" not in text and "CrvSteps" not in text
    assert all(ord(c) < 128 for c in text)


def test_parent_independently_requires_status_product_and_has_no_transport():
    text = (ROOT / "art/CrvFastSim/fcl/audit_inclusive_run1a.fcl").read_text()
    assert "module_type : RootInput" in text
    assert 'statusG4Printer : { verbose : 1 inputTags : [ "g4run::R3Inclusive" ] }' in text
    assert "PrintEvent : true" in text
    assert "module_type : DataProductDump" in text
    assert "producers" not in text and "filters" not in text and "SelectEvents" not in text
