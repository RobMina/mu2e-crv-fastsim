"""Opt-in device refit of flagged waveforms: the same pulses as the NumPy full-length recovery."""
from dataclasses import replace

import numpy as np
import pytest

from crvfastsim.reco import pulsefit as PF

from test_chain_reco_contract import numpy_pulses, waveforms


@pytest.mark.parametrize('peaks,length', [((18,), 24), ((4, 20), 36), ((11,), 24), (tuple(range(4, 124, 24)), 128)])
def test_device_refit_matches_numpy_recovery(monkeypatch, peaks, length):
    pytest.importorskip('jax')
    from crvfastsim.reco import chain_jax
    monkeypatch.setenv('CRVFASTSIM_DEVICE_REFIT', '1')
    W = waveforms(peaks, length, (0, 1))
    rows = np.array([1, 0])
    ref = numpy_pulses(W)
    out = chain_jax._pulse_dispatch(W, rows, replace(PF.RecoConfig(), max_pulses=3), 4, 12)
    counts = {}
    wf, seq, pe, tm, nhost = chain_jax._pulse_collect(out, W, rows, PF.RecoConfig(), counts)
    assert nhost + counts['device_refit'] == len(rows)
    np.testing.assert_array_equal(wf, rows[ref['wf']])
    np.testing.assert_array_equal(seq, ref['seq'])
    np.testing.assert_allclose(pe, ref['pe'], rtol=1e-9)        # device fit converges to round-off of the host fit
    np.testing.assert_allclose(tm, ref['time'], atol=1e-6)


def test_device_refit_off_by_default(monkeypatch):
    pytest.importorskip('jax')
    from crvfastsim.reco import chain_jax
    monkeypatch.delenv('CRVFASTSIM_DEVICE_REFIT', raising=False)
    W = waveforms((18,), 24, (0, 1))
    rows = np.array([1, 0])
    out = chain_jax._pulse_dispatch(W, rows, replace(PF.RecoConfig(), max_pulses=3), 4, 12)
    counts = {}
    *_, nhost = chain_jax._pulse_collect(out, W, rows, PF.RecoConfig(), counts)
    assert nhost == len(rows) and counts['device_refit'] == 0
