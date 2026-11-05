"""Paired per-shard histogram variance and shared-sign-flip family limits; numerical evidence, not physics admission."""
import numpy as np
from scipy.special import gammaincc
from scipy.stats import gamma

METHOD = "paired_batch_variance_student_mean_bias_shared_sign_flip_gamma_tail_family"


def batch_variance(values):
    values = np.asarray(values, dtype=float)
    n = values.shape[-2]
    if n < 2:
        raise ValueError("need at least two shards")
    return n / (n - 1.) * np.sum((values - values.mean(axis=-2, keepdims=True))**2, axis=-2)


def statistic(delta):
    delta = np.asarray(delta, dtype=float)
    n = delta.shape[-2]
    if n <= 3:
        raise ValueError("Student mean correction requires at least four shards")
    return (n - 3.) / (n - 1.) * np.mean(delta.sum(axis=-2)**2 / np.maximum(batch_variance(delta), 1e-12), axis=-1)


def compare(reference, candidate, nflip=20000, seed=20261101, alpha=.01):
    """X1-style raw-count statistic; identical signs across histograms preserve family correlations."""
    if set(reference) != set(candidate) or not reference:
        raise ValueError("require the same nonempty histogram family in both arms")
    if type(nflip) is not int or nflip < 100 or not 0 < alpha < 1:
        raise ValueError("invalid sign-flip count or family alpha")
    n = None
    per, null, observed = {}, {}, {}
    signs = None
    for name in sorted(reference):
        a, b = np.asarray(reference[name], float), np.asarray(candidate[name], float)
        if (a.ndim != 2 or a.shape != b.shape or a.shape[0] <= 3 or a.shape[1] == 0 or
                not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a < 0) or np.any(b < 0)):
            raise ValueError(f"{name}: need matching nonnegative shard/bin counts, at least four shards")
        if n is None:
            n = a.shape[0]
            signs = np.random.default_rng(seed).integers(0, 2, size=(nflip, n), dtype=np.int8).astype(float) * 2 - 1
        if n != a.shape[0]:
            raise ValueError("histograms disagree on shard count")
        delta = b - a
        variance = batch_variance(delta)
        keep = np.sum(delta**2, axis=0) > 0
        status = ("both_empty" if not a.sum() and not b.sum() else "one_empty" if not a.sum() or not b.sum()
                  else "identical" if not keep.any() else "ok")
        row = dict(status=status, reference_total=float(a.sum()), candidate_total=float(b.sum()),
                   difference=(b.sum(0) - a.sum(0)).tolist(), paired_variance=variance.tolist(),
                   reference_variance=batch_variance(a).tolist(), candidate_variance=batch_variance(b).tolist(),
                   used_bins=int(keep.sum()), reference_counts=a.sum(0).tolist(), candidate_counts=b.sum(0).tolist())
        total_a, total_b = a.sum(axis=1), b.sum(axis=1)
        row['total_difference_se'] = float(np.sqrt(batch_variance((total_b - total_a)[:, None])[0]))
        if total_a.sum() > 0:
            ratio = total_b.sum() / total_a.sum()
            row['yield_ratio'] = float(ratio)
            row['yield_ratio_se'] = float(np.sqrt(batch_variance((total_b - ratio * total_a)[:, None])[0]) / total_a.sum())
        else:
            row.update(yield_ratio=None, yield_ratio_se=None)
        per[name] = row
        if status != 'ok':
            row.update(flag=None if status == 'one_empty' else False, threshold=None, p_family=None,
                       descriptive_one_empty=status == 'one_empty',
                       calibrated_status='unsupported_one_empty' if status == 'one_empty' else 'not_tested')
            continue
        x = delta[:, keep]
        observed[name] = row['chi2_ndf'] = float(statistic(x))
        s2 = np.sum(x*x, axis=0)
        t = np.empty(nflip)
        for first in range(0, nflip, 500):
            d = signs[first:first + 500] @ x
            v = n / (n - 1.) * (s2 - d*d / n)
            t[first:first + 500] = (n - 3.) / (n - 1.) * np.mean(d*d / np.maximum(v, 1e-12), axis=-1)
        null[name] = t
        row.update(null_mean=float(t.mean()), null_sd=float(t.std()))
    fits = {name: (t.mean()**2 / t.var(), t.var() / t.mean()) for name, t in null.items()
            if t.mean() > 0 and t.var() > 0}
    family = dict(alpha=alpha, nflip=nflip, seed=seed, shards=n, method=METHOD, fitted_histograms=len(fits),
                  pass_rule="observed statistic exceeds shared-sign-flip family-wise threshold",
                  limitations="Small-shard tail calibration and finite-library/systematic uncertainty remain limitations.")
    if fits:
        minp = np.stack([gammaincc(a, null[name] / scale) for name, (a, scale) in fits.items()]).min(0)
        pstar = float(np.quantile(minp, alpha))
        family.update(per_histogram_tail=pstar)
        for name, (a, scale) in fits.items():
            limit = float(gamma.isf(pstar, a, scale=scale))
            pv = float(gammaincc(a, observed[name] / scale))
            per[name].update(threshold=limit if np.isfinite(limit) else None,
                             flag=bool(observed[name] > limit) if np.isfinite(limit) else None,
                             p_family=float((1 + np.sum(minp <= pv)) / (nflip + 1)),
                             p_flip=float((1 + np.sum(null[name] >= observed[name])) / (nflip + 1)))
    for name in null.keys() - fits.keys():
        per[name].update(status='unsupported_constant_null', threshold=None, flag=None, p_family=None)
    return dict(method=METHOD, per_histogram=per, family=family, physics_admitted=False)
