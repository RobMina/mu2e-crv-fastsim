"""Compact saved-only summaries of native diagnostic results; no new simulation or admission."""
import argparse
import json
from pathlib import Path

from crvfastsim import tier1b_assets


def compact(record):
    if 'groups' in record:
        return {key: record[key] for key in ('stage', 'steps', 'generated', 'metrics_pass', 'groups')}
    if 'results' in record:
        families = {}
        for label, result in {**record['results'], **{'selected/' + k: v for k, v in record.get('selected_event_diagnostic', {}).items()},
                              'sampling_seed_contrast': record.get('sampling_seed_contrast')}.items():
            if result is None:
                continue
            per = result['per_histogram']
            families[label] = dict(family=result['family'], flagged=[name for name, value in per.items() if value.get('flag') is True],
                                   unsupported=[name for name, value in per.items() if value.get('flag') is None],
                                   histograms={name: {key: value[key] for key in ('status', 'chi2_ndf', 'threshold', 'p_family',
                                                'yield_ratio', 'yield_ratio_se') if key in value} for name, value in per.items()})
        return dict(shards=record['shards'], generated_per_seed=record['generated_per_seed'], families=families)
    if 'population' in record:
        return {key: record[key] for key in ('selected', 'vetoed', 'population', 'native_C4_admitted')}
    if 'finding' in record:
        return {key: record[key] for key in ('events', 'noise', 'finding')}
    return record


def run(inputs, out):
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    before = {str(Path(path).resolve()): tier1b_assets.sha256(path) for path in inputs}
    result = dict(sources_sha256=before, summaries={path: compact(json.loads(Path(path).read_text())) for path in before},
                  physics_admitted=False)
    if any(tier1b_assets.sha256(path) != value for path, value in before.items()):
        raise ValueError('saved validation input drift')
    with out.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('--out', required=True)
    a = parser.parse_args()
    run(a.inputs, a.out)
