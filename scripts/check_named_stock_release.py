"""Compare retained stock-release event timing and calibration tables; not a yield-only or physics certificate."""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from crvfastsim import response_receipt, tables, tier1b_assets
from crvfastsim.validation import event_shapes, reco_detail
from scripts.check_native_response import canonical_keys


EVENT_FIELDS = tables.KEY + ['spill', 'evlen', 'pbtime']
CALIB_FIELDS = ['run', 'channel', 'pedestal', 'pulsearea', 'pulseheight', 'timeoffset', 'status', 'roc', 'feb', 'febch']


def exact_table(reference, candidate, keys, fields):
    if reference.duplicated(keys).any() or candidate.duplicated(keys).any():
        raise ValueError('duplicate stock evidence key')
    a = reference.sort_values(keys).reset_index(drop=True)
    b = candidate.sort_values(keys).reset_index(drop=True)
    if len(a) != len(b):
        raise ValueError('stock evidence row count differs')
    for field in fields:
        x, y = a[field].to_numpy(), b[field].to_numpy()
        if x.dtype.kind not in 'iuf' or y.dtype != x.dtype or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('stock evidence dtype/nonfinite mismatch: ' + field)
        if x.tobytes() != y.tobytes():
            raise ValueError('stock evidence bytes differ: ' + field)
    return dict(rows=len(a), fields=fields, identity='exact_dtype_sorted_values_and_bytes')


def evidence(path):
    result, cycles = {}, {}
    with uproot.open(path) as root:
        for name, fields in [('events', EVENT_FIELDS), ('calib', CALIB_FIELDS)]:
            tree, cycles[name] = event_shapes._highest(root, 'crvdump/' + name)
            arrays = tree.arrays(fields, library='ak')
            result[name] = pd.DataFrame({key: ak.to_numpy(arrays[key]) for key in fields})
    canonical_keys(result['events'])
    return result, cycles


def validate_calibration(calib, events):
    for name in ('run', 'channel', 'status', 'roc', 'feb', 'febch'):
        values = calib[name].to_numpy()
        if values.dtype.kind not in 'iu':
            raise ValueError('stock calibration integer dtype: ' + name)
    runs = events.run.unique()
    if len(runs) != 1 or not np.all(calib.run == runs[0]):
        raise ValueError('stock calibration run differs from event run')
    if len(calib) != 5120 or not np.array_equal(np.sort(calib.channel), np.arange(5120)):
        raise ValueError('require all 5120 Run1A channels')
    if (calib.status < 0).any():
        raise ValueError('negative stock calibration status')
    hardware = calib[['roc', 'feb', 'febch']].to_numpy()
    absent = np.all(hardware == -1, axis=1)
    mapped = hardware[~absent]
    if (np.any(mapped < [1, 1, 0]) or np.any(mapped > [18, 24, 63])
            or len(np.unique(mapped, axis=0)) != len(mapped)):
        raise ValueError('invalid stock calibration hardware')


def require_population(path, expected_count, expected_digest):
    if type(expected_count) is not int or expected_count <= 0:
        raise ValueError('positive expected stock event count required')
    identity = response_receipt.event_identity(path)
    if identity['count'] != expected_count or identity['sha256'] != expected_digest:
        raise ValueError('stock event population differs from independent pinned census')
    return identity


def run(reference, candidate, out, expected_count, expected_digest):
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    reference, candidate = str(Path(reference).resolve()), str(Path(candidate).resolve())
    if reference == candidate:
        raise ValueError('stock release arms must be distinct')
    paths = [reference, candidate, __file__, response_receipt.__file__, reco_detail.__file__, event_shapes.__file__, canonical_keys.__code__.co_filename]
    before = {str(Path(path).resolve()): tier1b_assets.sha256(path) for path in paths}
    population = require_population(reference, expected_count, expected_digest)
    require_population(candidate, expected_count, expected_digest)
    a, ca = evidence(reference)
    b, cb = evidence(candidate)
    validate_calibration(a['calib'], a['events'])
    validate_calibration(b['calib'], b['events'])
    event_identity = exact_table(a['events'], b['events'], tables.KEY, EVENT_FIELDS)
    calibration_identity = exact_table(a['calib'], b['calib'], ['run', 'channel'], CALIB_FIELDS)
    frames_a, _, ma = reco_detail._read(reference)
    frames_b, _, mb = reco_detail._read(candidate)
    if any(tier1b_assets.sha256(path) != value for path, value in before.items()):
        raise ValueError('stock release evidence drift')
    result = dict(scope='saved_stock_release_timing_calibration_and_collection_integrity_only',
                  events=event_identity, pinned_population=population, calibration=calibration_identity,
                  highest_cycles=dict(reference={**ca, **ma}, candidate={**cb, **mb}),
                  counts={label: {key: len(frame[key]) for key in ('events', 'digis', 'pulses', 'clusters')}
                          for label, frame in [('reference', frames_a), ('candidate', frames_b)]},
                  sources_sha256=before, physics_admitted=False,
                  limitations=['Exact persisted event timing/calibration does not establish identical response implementations.',
                               'A release change may also change SiPM, ADC, pulse fit and finder; this is not yield-only attribution.',
                               'No inclusive exposure, transport, geometry execution, DB IOV or full named physics certification.'])
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    with out.open('x') as stream:
        stream.write(payload)
    print(json.dumps(result['counts']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--expected-events', type=int, required=True)
    parser.add_argument('--expected-event-sha256', required=True)
    args = parser.parse_args()
    run(args.reference, args.candidate, args.out, args.expected_events, args.expected_event_sha256)
