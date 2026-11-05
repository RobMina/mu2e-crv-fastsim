"""Per-sector saved-pulse attribution never substitutes sector type or loses zero-response events."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.check_named_sector_response import pulse_summary, compare_sector


def test_actual_bar_sectors_keep_event_zeros_and_separate_yields():
    universe = pd.DataFrame(dict(run=[1400, 1400], subrun=[102, 102], event=[1, 2]))
    pulses = pd.concat([universe.iloc[:1]] * 2, ignore_index=True).assign(bar=[0, 1], seq=0, pe=[20., 40.])
    geom = SimpleNamespace(nbars=2, sector=np.array([0, 1]), sector_names=['T1', 'T2'])
    result = pulse_summary(pulses, universe, geom)
    assert result['T1']['event_pe'].tolist() == [20., 0.]
    assert result['T2']['event_pe'].tolist() == [40., 0.]
    comparison = compare_sector(result['T1'], result['T2'])
    assert comparison['event_pe_ratio'] == 2 and comparison['event_pe_ratio_se'] == 0
    assert comparison['diagnostic_rules']['pe_median_2percent'] is None
    with pytest.raises(ValueError, match='outside saved event'):
        pulse_summary(pulses.assign(run=1430), universe, geom)


def test_sector_attribute_does_not_use_historical_shared_sector_type():
    universe = pd.DataFrame(dict(run=[1] * 200, subrun=[2] * 200, event=np.arange(200)))
    pulses = universe.assign(bar=0, seq=0, pe=30., sectortype=1)
    geom = SimpleNamespace(nbars=2, sector=np.array([0, 1]), sector_names=['T1', 'T2'])
    result = pulse_summary(pulses, universe, geom)
    assert result['T2']['pulses'] == 0 and result['T1']['pulses'] == 200
    comparison = compare_sector(result['T1'], result['T1'])
    assert comparison['event_pe_ratio'] == 1 and comparison['event_pe_ratio_se'] == 0
    assert comparison['diagnostic_rules']['pe_median_2percent']
    assert comparison['diagnostic_rules']['spectrum_3percent'] is None


def test_empty_candidate_supported_spectrum_serializes_failure():
    import json
    from crvfastsim.validation import v2compare
    reference = dict(event_pe=np.ones(3000), pulses=6000, underflow=0, overflow=0,
                     spectrum=np.full(len(v2compare.PE_EDGES) - 1, 2, dtype=int))
    candidate = dict(event_pe=np.zeros(3000), pulses=0, underflow=0, overflow=0,
                     spectrum=np.zeros(len(v2compare.PE_EDGES) - 1, dtype=int))
    result = compare_sector(reference, candidate)
    assert result['spectrum']['chi2_ndf'] is None and result['spectrum']['nonfinite_statistics'] == ['chi2_ndf']
    assert not result['diagnostic_rules']['spectrum_3percent']
    json.dumps(result, allow_nan=False)
