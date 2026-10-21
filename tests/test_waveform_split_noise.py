"""The split noise path (oneAPI) gives the same waveforms and ADCs as the fused kernel on the same key."""
import numpy as np

import jax


def test_waveform_split_noise_equals_fused():
    from crvfastsim.reco import repro, waveform, waveform_jax as WJ
    cfg = waveform.WaveformConfig()
    row, t, q, tdc0 = repro._charges(300, 5)
    key = jax.random.PRNGKey(3)
    outs = []
    for split in (False, True):
        WJ.SPLIT_NOISE = split
        try:
            outs.append(WJ.synthesize(row, t, q, tdc0, cfg, rng_key=key))
        finally:
            WJ.SPLIT_NOISE = None
    a, b = outs
    assert len(a["adc"]) > 1000
    for k in ("chan", "starttdc", "adc"):
        assert np.array_equal(a[k], b[k]), k
    assert np.max(np.abs(a["volt"] - b["volt"])) < 1e-12
