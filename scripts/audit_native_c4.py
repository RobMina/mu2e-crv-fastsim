"""Read-only cached C4 rule replay and exact event-overlap audit; never substitute a field-off population."""
import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd

from crvfastsim import genassoc, tables, tier1b_assets
from crvfastsim.veto import VetoCard, fraction, vetoed
from crvfastsim.veto.apply import gate


def overlap(selected, universe):
    keys = selected[tables.KEY].drop_duplicates()
    count = len(keys.merge(universe, on=tables.KEY, how='inner', validate='one_to_one'))
    return dict(selected_events=len(keys), generator_events=len(universe), shared_events=count,
                selected_runs=sorted(map(int, keys.run.unique())), generator_runs=sorted(map(int, universe.run.unique())),
                exact_population=bool(count == len(keys) == len(universe)))


def run(selection, baseline, generators, out):
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    before = {str(Path(path).resolve()): tier1b_assets.sha256(path) for path in [selection, baseline, *generators]}
    saved = json.loads(Path(baseline).read_text())
    card_data = dict(saved['card'])
    expected_sha = card_data.pop('sha')
    card = VetoCard.from_dict(card_data)
    if card.sha() != expected_sha:
        raise ValueError('cached rule identity mismatch')
    data = ak.from_parquet(selection)
    event = pd.DataFrame({key: ak.to_numpy(data[key]) for key in tables.KEY})
    if event.duplicated(tables.KEY).any():
        raise ValueError('cached selected events not unique')
    p, t, coincs = data.p_mid, data.t_mid, data.coincs
    passed = gate(coincs, card)
    veto = vetoed(t, coincs, card)
    gate_tracks = ak.broadcast_arrays(passed[:, None], t)[0]
    use = (p > 85.) & (p < 200.)
    selected = int(ak.sum(use))
    frac = fraction(ak.to_numpy(ak.flatten(veto[use & gate_tracks])))
    if selected != saved['selected'] or frac['n'] != saved['vetoed']['n'] or frac['k'] != saved['vetoed']['k']:
        raise ValueError('cached C4 count no longer reproduces baseline')
    universes = []
    for path in generators:
        report = genassoc.validate_generator_stream(path)
        if report.get('schema_version') != 2 or report.get('stream_role') != 'full generator stream':
            raise ValueError('strict full generator census required')
        universes.append(genassoc.generator_event_keys(path))
    universe = pd.concat(universes, ignore_index=True)
    if universe.duplicated(tables.KEY).any():
        raise ValueError('generator populations overlap')
    result = dict(scope='cached_stock_C4_replay_and_identity_audit_not_native_C4_closure', card=saved['card'],
                  selected=selected, vetoed=frac, population=overlap(event, universe), sources_sha256=before,
                  native_C4_admitted=False, limitations=['Field/selection/timing provenance is not supplied by an event overlap.',
                  'Disjoint cached track and field-off reference populations cannot be compared as the same C4 analysis.'])
    if any(tier1b_assets.sha256(path) != digest for path, digest in before.items()):
        raise ValueError('C4 audit input drift')
    with out.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--generators', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    a = parser.parse_args()
    run(a.selection, a.baseline, a.generators, a.out)
