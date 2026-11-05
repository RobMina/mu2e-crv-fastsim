"""Read-only native records vs matching accepted stock response, retaining the certified generated exposure."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crvfastsim import genassoc, native_record, tables, tier1b_assets
from crvfastsim.analyses.e2e_bench import fingerprint
from crvfastsim.validation import reco_detail


def arrays(path):
    path = Path(path).resolve()
    meta = json.loads(Path(str(path) + '.json').read_text())
    if meta['schema'] != native_record.SCHEMA or meta['scope'] != native_record.SCOPE:
        raise ValueError('not a native diagnostic record')
    if meta['output_sha256'] != tier1b_assets.sha256(path):
        raise ValueError('native record hash mismatch')
    with np.load(path, allow_pickle=False) as data:
        out = dict(data)
    if fingerprint(out) != meta['array_fingerprint']:
        raise ValueError('native record array fingerprint mismatch')
    return out, meta


def canonical_keys(frame):
    values = frame[tables.KEY].to_numpy()
    if values.dtype.kind not in 'iu' or len(np.unique(values, axis=0)) != len(values):
        raise ValueError('event census must contain unique integer triples')
    return values[np.lexsort(values[:, ::-1].T)]


def run(candidate, generator, donor, stock, out):
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    before = {str(Path(path).resolve()): tier1b_assets.sha256(path) for path in
              [candidate, str(candidate) + '.json', generator, donor, *stock]}
    data, meta = arrays(candidate)
    gen = genassoc.validate_generator_stream(generator)
    genassoc.validate_donor(donor)
    if gen.get('schema_version') != 2 or gen.get('stream_role') != 'full generator stream':
        raise ValueError('need strict full generator exposure')
    universe = genassoc.generator_event_keys(generator)
    if not np.array_equal(canonical_keys(universe), canonical_keys(pd.DataFrame(data['generated_keys'], columns=tables.KEY))):
        raise ValueError('candidate generated universe differs')
    frames, evidence = [], []
    for path in stock:
        frame, _, metadata = reco_detail._read(path)
        frames.append(frame)
        evidence.append(metadata)
    accepted = pd.concat([frame['events'] for frame in frames], ignore_index=True)
    donor_events = tables.read(donor, 'events', tables.KEY)
    if not np.array_equal(canonical_keys(accepted), canonical_keys(donor_events)):
        raise ValueError('stock response does not cover exact accepted donor population')
    if len(accepted.merge(universe, on=tables.KEY, how='inner')) != len(accepted):
        raise ValueError('accepted stock events outside generated exposure')
    counts = {name: int(accepted[column].sum()) for name, column in
              [('digis', 'ndigi'), ('pulses', 'npulse'), ('clusters', 'ncluster')]}
    for name in counts:
        if counts[name] != sum(len(frame[name]) for frame in frames):
            raise ValueError('stock table/census count mismatch')
    result = dict(scope='native_chain_vs_selected_stock_response_not_physics_certification', generated=len(universe),
                  accepted_stock_events=len(accepted), stock_counts=counts, candidate_counts=meta['counts'],
                  ratios={name: meta['counts'][name] / counts[name] if counts[name] else None for name in counts},
                  stock_highest_cycles=evidence, sources_sha256=before,
                  limitations=['Rejected G4 events have no saved response; zero-response assumption needs selection audit.',
                               'Historical finder and shared-lookup ownership differ from named defaults.',
                               'A single shard cannot supply per-shard shape acceptance or finite-library uncertainty.'],
                  physics_admitted=False)
    if {path: tier1b_assets.sha256(path) for path in before} != before:
        raise ValueError('inputs changed during comparison')
    with out.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--generator', required=True)
    parser.add_argument('--donor', required=True)
    parser.add_argument('--stock', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args.candidate, args.generator, args.donor, args.stock, args.out)
