"""Noise diagnostics retain support limits, strict population identity and unique digi joins."""
import json
import numpy as np
import pandas as pd
import pytest

from crvfastsim.reco.waveform import WaveformConfig
from scripts.check_named_waveform_noise import (channel_summary, require_same_charge_population,
                                                residual_summary, unique_keys, clock_ticks, validate_nonoise)


def test_empty_channel_support_is_not_a_pass():
    result = channel_summary(np.array([0]), np.zeros((1, 12)), np.array([0]), np.zeros((1, 12)), 2)
    assert result['channels_tested'] == 0 and result['all_channels_pass'] is None
    assert result['status'] == 'unsupported_no_eligible_channels'
    json.dumps(result, allow_nan=False)


def test_noise_channel_threshold_retains_exact_rule():
    channels = np.repeat(np.arange(2), [200, 199])
    adc = np.full((len(channels), 12), 100)
    result = channel_summary(channels, adc, channels, adc, 2)
    assert result['channels_tested'] == 1 and result['all_channels_pass']
    assert result['per_channel'][0]['channel'] == 0 and result['min_p'] == 1.
    failed = channel_summary(channels, adc + 10, channels, adc, 2)
    assert failed['channels_below_threshold'] == 1 and not failed['all_channels_pass']


def test_noise_residual_uses_common_digis_and_excludes_saturation():
    keys = np.array([[0, 1, 2], [0, 1, 3]])
    base = np.full((2, 12), 100, dtype=np.int16)
    base[0, 0] = 4095
    noise = np.tile(np.array([-1, 0, 1] * 4), (2, 1))
    result = residual_summary(keys, base + noise, keys[:1], (base + noise)[:1], keys, base, WaveformConfig())
    assert result['digis_in_all_three'] == 1 and result['samples'] == 11
    assert result['ks']['p'] == 1 and result['ks_nonrejection'] and result['physics_pass'] is None
    json.dumps(result, allow_nan=False)


def test_noise_residual_empty_join_is_unsupported():
    key = np.array([[0, 1, 2]])
    adc = np.full((1, 12), 100, dtype=np.int16)
    result = residual_summary(key, adc, key + 10, adc, key, adc, WaveformConfig())
    assert result['samples'] == 0 and result['ks_nonrejection'] is None
    with pytest.raises(ValueError, match='duplicate waveform'):
        unique_keys(np.concatenate([key, key]))


def population():
    return dict(events=pd.DataFrame(dict(run=[1400], subrun=[102], event=[1])),
                qkeys=np.array([[1400, 102, 1, 3]]), n=np.array([1]), time=np.array([1.]),
                charge=np.array([2.]), phase=np.array([3.]))


@pytest.mark.parametrize('field', ['events', 'charge', 'phase'])
def test_noise_free_reference_requires_same_population_charges_and_phase(field):
    ref, base = population(), population()
    require_same_charge_population(ref, base)
    if field == 'events':
        base[field].loc[0, 'run'] = 1401
    else:
        base[field][0] += 1.
    with pytest.raises(ValueError, match='noise-free'):
        require_same_charge_population(ref, base)


def test_noise_free_charge_identity_preserves_signed_zero():
    ref, base = population(), population()
    ref['charge'][0], base['charge'][0] = 0., -0.
    with pytest.raises(ValueError, match='saved charges differ'):
        require_same_charge_population(ref, base)


def test_off_grid_clocks_are_not_rounded_into_join_identity():
    assert clock_ticks(np.array([15.5]), np.array([3.])).tolist() == [1]
    with pytest.raises(ValueError, match='sample grid'):
        clock_ticks(np.array([18.5]), np.array([3.]))


def test_sparse_residual_nonrejection_is_not_noise_agreement():
    key = np.array([[0, 1, 2]])
    base = np.full((1, 12), 4095, dtype=np.int16)
    base[0, 0] = 100
    fast = base.copy()
    fast[0, 0] += 10
    result = residual_summary(key, fast, key, base, key, base, WaveformConfig())
    assert result['samples'] == 1 and result['ks_nonrejection']
    assert result['physics_pass'] is None and 'pass_metric' not in result


def test_noise_free_baseline_requires_deterministic_sample_identity(monkeypatch):
    from scripts import check_named_waveform_noise as module
    base = dict(n=np.array([1]), time=np.array([1.]), charge=np.array([1.]), phase=np.array([1.]),
                qe=np.array([0]), qc=np.array([0]), key=np.array([[0, 0, 1]]),
                adc=np.full((1, 12), 100), mapped=np.array([True]))
    output = dict(chan=np.array([0]), starttdc=np.array([1]), adc=np.full((1, 12), 100))
    monkeypatch.setattr(module.waveform, 'synthesize', lambda *a, **kw: output)
    assert validate_nonoise(base, WaveformConfig())['sample_identical']
    output['adc'][0, 0] += 1
    with pytest.raises(ValueError, match='deterministic'):
        validate_nonoise(base, WaveformConfig())
