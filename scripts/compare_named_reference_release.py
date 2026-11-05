"""Reassess frozen native histograms against explicit stock-release replacements; no baseline rewrite or admission."""
import argparse
import copy
import json
from pathlib import Path

from crvfastsim import response_receipt, tier1b_assets
from scripts import check_named_stock_release, compare_native_histograms


def replacement_pairs(legacy, replacements):
    expected = [path for pair in legacy['pairs'] for path in pair['stock']]
    if len(set(expected)) != len(expected) or set(replacements) != set(expected):
        raise ValueError('replacement map must cover each old stock file exactly once')
    new = [str(Path(replacements[path]).resolve()) for path in expected]
    if len(set(new)) != len(new) or set(new) & {str(Path(path).resolve()) for path in expected}:
        raise ValueError('replacement stock paths alias each other or old baselines')
    result = copy.deepcopy(legacy)
    for pair in result['pairs']:
        pair['stock'] = [str(Path(replacements[path]).resolve()) for path in pair['stock']]
    return result


def run(config, outdir):
    config, outdir = Path(config).resolve(), Path(outdir)
    if outdir.exists() or outdir.is_symlink():
        raise FileExistsError(outdir)
    config_hash = tier1b_assets.sha256(config)
    spec = json.loads(config.read_text())
    legacy_path = Path(spec['legacy_pairs']['path']).resolve()
    legacy_hash = tier1b_assets.sha256(legacy_path)
    if legacy_hash != spec['legacy_pairs']['sha256']:
        raise ValueError('legacy paired configuration hash differs')
    legacy = json.loads(legacy_path.read_text())
    old_paths = {path: str(Path(path).resolve()) for path in spec['replacement_stock']}
    replacements = {path: str(Path(value).resolve()) for path, value in spec['replacement_stock'].items()}
    pairs = replacement_pairs(legacy, replacements)
    paths = [config, legacy_path, __file__, check_named_stock_release.__file__, compare_native_histograms.__file__]
    paths += list(old_paths.values()) + list(replacements.values())
    before = {str(Path(path).resolve()): tier1b_assets.sha256(path) for path in paths}
    before[str(config)], before[str(legacy_path)] = config_hash, legacy_hash
    outdir.mkdir()
    identities = []
    for index, (original, new) in enumerate(replacements.items()):
        old = old_paths[original]
        identity = response_receipt.event_identity(old)
        output = outdir / f'release_{index:02d}.json'
        check_named_stock_release.run(old, new, output, identity['count'], identity['sha256'])
        identities.append(dict(reference=old, candidate=new, events=identity, report_sha256=tier1b_assets.sha256(output)))
    pair_path = outdir / 'pairs.json'
    with pair_path.open('x') as stream:
        stream.write(json.dumps(pairs, indent=2) + '\n')
    compare_native_histograms.run(pair_path, outdir / 'paired.json')
    if any(tier1b_assets.sha256(path) != value for path, value in before.items()):
        raise ValueError('stock replacement input/source drift')
    result = dict(scope='fixed_native_records_with_verified_stock_release_replacements_not_physics_admission',
                  legacy_pairs_sha256=legacy_hash, replacement_pairs_sha256=tier1b_assets.sha256(pair_path),
                  paired_output_sha256=tier1b_assets.sha256(outdir / 'paired.json'), identities=identities,
                  sources_sha256=before, physics_admitted=False,
                  limitations=['Old baselines and frozen native outputs remain unchanged; no response correction is fitted.',
                               'Old stock censuses pin replacements; strict generator/donor population admission is retained.',
                               'Persisted timing/calibration equality is not full release/geometry execution certification.',
                               'Rejected G4 response remains unavailable; zero-response overlays are diagnostic only.',
                               'No finite-library/model uncertainty, V-R4/V-R5, MC-data or C4 admission.'])
    with (outdir / 'replacement_report.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()
    run(args.config, args.outdir)
