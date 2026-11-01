"""Second device pass for the waveforms the pipelined pulse kernel flags (input longer than its window, all its rounds
used): full length and as many rounds as the NumPy recovery allows, instead of the host refit. Opt-in
(CRVFASTSIM_DEVICE_REFIT=1); waveforms whose fit window still overflows the kernel's fixed window stay on NumPy.

The NumPy recovery (pulsefit.reconstruct_full) bounds the rounds by max(max_pulses, longest waveform); this pass uses
at least that many, so no waveform stops earlier than there. Shapes are bucketed to powers of two (rows, samples,
rounds) to bound recompilation.
"""
import dataclasses

import numpy as np

from . import pulsefit_jax as PJ


def _bucket(n, lo):
    return max(lo, 1 << max(int(n) - 1, 0).bit_length())


def refit(W, g, cfg_np, min_rows=256, min_len=16):
    """Pulses of waveform rows g (indices into W) on the device. Returns (wf, seq, pe, time, residual): wf indexes g,
    in (wf, seq) order; residual is a bool mask over g of rows to redo on the host (fit window overflow)."""
    import jax.numpy as jnp
    n = W["n"][g]
    if not len(g):
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0), np.zeros(0), np.zeros(0, bool)
    L = _bucket(int(n.max()), min_len)
    P = _bucket(max(cfg_np.max_pulses, int(n.max())), cfg_np.max_pulses)
    R = _bucket(len(g), min_rows)
    pad = lambda x, v: np.concatenate([x, np.full((R - len(x),) + x.shape[1:], v, x.dtype)])
    adc = np.zeros((len(g), L), W["adc"].dtype)
    width = min(L, W["adc"].shape[1])
    adc[:, :width] = W["adc"][g, :width]
    cfg = dataclasses.replace(cfg_np, max_pulses=P)
    d = PJ.reconstruct_dense(jnp.asarray(pad(adc, 0)), jnp.asarray(pad(n, 0)), jnp.asarray(pad(W["start"][g], 0)),
                             jnp.asarray(pad(W["ped"][g].astype(float), 0.0)),
                             jnp.asarray(pad(W["area"][g].astype(float), 1.0)),
                             jnp.asarray(pad(W["height"][g].astype(float), 1.0)), cfg=cfg)
    k = len(g)
    found = np.asarray(d["found"])[:k]
    residual = np.asarray(d["window_overflow"])[:k] | np.asarray(d["rounds_exhausted"])[:k]
    found = found & ~residual[:, None]
    wf, seq = np.nonzero(found)
    pe = np.asarray(d["pe"], dtype=np.float64)[:k][wf, seq]
    tm = np.asarray(d["time"], dtype=np.float64)[:k][wf, seq]
    return wf, seq, pe, tm, residual
