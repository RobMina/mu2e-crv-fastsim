"""Named-only, pure donor-local diagnostics for incomplete native construction.

This module is NOT wired into tier1b, defaults, builds or admission. The existing
single-primary construction guard remains necessary and unchanged. Splitting
crossings and restricting descendant candidates does not implement the missing
full-stream near-miss reconstruction, certify physics, or establish provenance.
All particle identities here are exact stage-local signed-int64 tuples.
"""
import numpy as np
import pandas as pd

from .tables import KEY

__all__ = ["crossings", "attach_descendants", "require_full_nearmiss_bridge"]

_ENTRY = ["t0", "x0", "y0", "z0", "px0", "py0", "pz0"]
_EXIT = ["x1", "y1", "z1"]
_POS = ["x0", "y0", "z0"] + _EXIT
_XCOLS = KEY + ["bar", "sp"] + _ENTRY + ["len", "edep", "p"] + _EXIT + ["prim_key", "xid"]
_MATCH = ["xid", "selected_primary", "xbar", "score"]


def _required(frame, columns, label):
    missing = [c for c in columns if c not in frame]
    if missing:
        raise ValueError(f"{label}: missing columns {missing}; exact ancestry is required")


def _integers(frame, columns, label):
    """Copy and normalize integer columns without ever passing through floats."""
    out = frame.copy(deep=True)
    for c in columns:
        values = out[c]
        if values.isna().any():
            raise ValueError(f"{label}.{c}: missing integer identity/ancestry")
        if len(values) and (not pd.api.types.is_integer_dtype(values.dtype)
                            or pd.api.types.is_bool_dtype(values.dtype)):
            raise ValueError(f"{label}.{c}: requires exact integer identities")
        if len(values) and (values.min() < np.iinfo(np.int64).min
                            or values.max() > np.iinfo(np.int64).max):
            raise ValueError(f"{label}.{c}: outside signed int64")
        out[c] = values.astype(np.int64)
    return out


def _numeric(frame, columns, label):
    for c in columns:
        if len(frame) and (not pd.api.types.is_numeric_dtype(frame[c].dtype)
                           or pd.api.types.is_bool_dtype(frame[c].dtype)
                           or pd.api.types.is_complex_dtype(frame[c].dtype)):
            raise ValueError(f"{label}.{c}: requires finite real numeric values for scoring/construction")
        if not np.isfinite(frame[c].to_numpy(dtype=np.float64, na_value=np.nan)).all():
            raise ValueError(f"{label}.{c}: requires finite real numeric values for scoring/construction")


def _empty_crossings():
    ints = set(KEY + ["bar", "sp", "prim_key", "xid"])
    return pd.DataFrame({c: pd.Series(dtype=np.int64 if c in ints else np.float64) for c in _XCOLS})


def crossings(primary_steps):
    """Aggregate exact KEY+sp+bar tuples, retaining own prim_key=sp and local xid.

    Input must already contain only actual primary-muon steps. If prim_key is
    supplied it must equal sp. Stable KEY/bar/t0 sorting preserves historical
    single-muon crossing order; equal-time steps retain their input order. Entry,
    momentum and t0 come from the first physical row, exit from the last, with
    summed length/edep and first-entry |p|. No input frames are mutated.
    """
    if primary_steps.empty:
        return _empty_crossings()
    _required(primary_steps, KEY + ["sp", "bar"] + _ENTRY + _EXIT + ["len", "edep"], "primary_steps")
    identity = KEY + ["sp", "bar"] + (["prim_key"] if "prim_key" in primary_steps else [])
    s = _integers(primary_steps, identity, "primary_steps")
    if (s.sp < 0).any() or (s.bar < 0).any():
        raise ValueError("primary_steps: sp and bar must be nonnegative")
    if "prim_key" in s and not s.prim_key.eq(s.sp).all():
        raise ValueError("primary_steps: own prim_key must equal sp")
    _numeric(s, _ENTRY + _EXIT + ["len", "edep"], "primary_steps")
    s = s.sort_values(KEY + ["bar", "t0"], kind="stable")
    group = KEY + ["sp", "bar"]
    first = s.drop_duplicates(group, keep="first")[group + _ENTRY]
    last = s.drop_duplicates(group, keep="last")[group + _EXIT]
    sums = s.groupby(group, sort=False)[["len", "edep"]].sum().reset_index()
    x = first.merge(sums, on=group, sort=False, validate="one_to_one").merge(
        last, on=group, sort=False, validate="one_to_one")
    momentum = x[["px0", "py0", "pz0"]].to_numpy(dtype=np.float64)
    x["p"] = np.sqrt(np.sum(momentum ** 2, axis=1))
    _numeric(x, ["p", "len", "edep"], "crossings")
    x["prim_key"] = x.sp.to_numpy(dtype=np.int64)
    x["xid"] = np.arange(len(x), dtype=np.int64)
    return x[_XCOLS].reset_index(drop=True)


def attach_descendants(classified_steps, crossings):
    """Return (attached, unattached) mu_descendant rows, in original row order.

    did is a fresh stage-local row id. Candidates join ONLY KEY+prim_key to
    KEY+sp; unattached is the exact own-primary anti-join, never an event mask.
    Among own-ancestor crossings, score is -1 for the same bar, otherwise the
    descendant midpoint's distance to crossing entry. Ties follow crossing input
    row order (not xid magnitude). Attached rows add xid, selected_primary, xbar
    and score. Missing ancestry or invalid/nonfinite scoring data refuses; an
    explicit integer unknown ancestor (-1) has no crossing and stays unattached.
    These frames are diagnostic/stage-local, not library or admission evidence.
    """
    if classified_steps.empty:
        d = classified_steps.copy(deep=True).reset_index(drop=True)
    else:
        _required(classified_steps, ["cls"], "classified_steps")
        d = classified_steps.loc[classified_steps.cls.eq("mu_descendant")].copy().reset_index(drop=True)
    if not d.empty:
        _required(d, KEY + ["sp", "prim_key", "bar"] + _POS, "descendants")
        d = _integers(d, KEY + ["sp", "prim_key", "bar"], "descendants")
        _numeric(d, _POS, "descendants")
    d["did"] = np.arange(len(d), dtype=np.int64)
    if crossings.empty:
        x = _empty_crossings()
    else:
        _required(crossings, KEY + ["sp", "bar", "xid", "x0", "y0", "z0"], "crossings")
        x = _integers(crossings, KEY + ["sp", "bar", "xid"], "crossings")
        if x.xid.duplicated().any():
            raise ValueError("crossings: duplicate crossing ids (xid)")
        if (x.sp < 0).any() or (x.bar < 0).any() or (x.xid < 0).any():
            raise ValueError("crossings: sp, bar and xid must be nonnegative")
        if x.duplicated(KEY + ["sp", "bar"]).any():
            raise ValueError("crossings: duplicate exact primary/bar crossing")
        if "prim_key" in x:
            x = _integers(x, ["prim_key"], "crossings")
            if not x.prim_key.eq(x.sp).all():
                raise ValueError("crossings: own prim_key must equal sp")
        _numeric(x, ["x0", "y0", "z0"], "crossings")
    empty_attached = d.iloc[:0].copy()
    for c in _MATCH:
        empty_attached[c] = pd.Series(dtype=np.float64 if c == "score" else np.int64)
    if d.empty or x.empty:
        return empty_attached, d.copy()
    on = KEY + ["prim_key"]
    ancestors = x[KEY + ["sp"]].rename(columns={"sp": "prim_key"}).drop_duplicates()
    own = pd.MultiIndex.from_frame(d[on]).isin(pd.MultiIndex.from_frame(ancestors[on]))
    unattached = d.loc[~own].copy().reset_index(drop=True)
    xc = x[KEY + ["sp", "xid", "bar", "x0", "y0", "z0"]].rename(columns={
        "sp": "prim_key", "bar": "xbar", "x0": "ex", "y0": "ey", "z0": "ez"})
    xc["_crossing_order"] = np.arange(len(xc), dtype=np.int64)
    m = d.loc[own, on + ["did", "bar"] + _POS].merge(xc, on=on, sort=False)
    mid = 0.5 * (m[["x0", "y0", "z0"]].to_numpy(dtype=np.float64)
                 + m[_EXIT].to_numpy(dtype=np.float64))
    dist = np.linalg.norm(mid - m[["ex", "ey", "ez"]].to_numpy(dtype=np.float64), axis=1)
    m["score"] = np.where(m.bar.to_numpy() == m.xbar.to_numpy(), -1., dist)
    _numeric(m, ["score"], "descendant candidates")
    best = m.sort_values(["did", "score", "_crossing_order"], kind="stable").drop_duplicates("did")
    best = best[["did", "xid", "prim_key", "xbar", "score"]].rename(columns={"prim_key": "selected_primary"})
    attached = d.loc[own].merge(best, on="did", sort=False, validate="one_to_one")
    return attached.sort_values("did", kind="stable").reset_index(drop=True), unattached


def require_full_nearmiss_bridge():
    """Always refuse: no authoritative donor/full-generator source mapping exists."""
    raise NotImplementedError(
        "Incomplete native construction: the full-stream near-miss bridge needs an authoritative "
        "source mapping from donor primary SimParticles to full-generator primaries. Compression "
        "may copy/reindex generator collections; proximity, event-only masks, ProductID equality "
        "or index joins cannot supply that mapping. No bridge schema/receipt is invented; "
        "the existing construction guard remains required.")
