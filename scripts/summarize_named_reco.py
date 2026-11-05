"""Summarize explicit saved named pulse/finder controls, not native-chain physics certification."""
import argparse
import json
from pathlib import Path

from crvfastsim import tier1b_assets


def summarize(coincidences, pulses):
    if not coincidences or not pulses:
        raise ValueError('need explicit nonempty coincidence and pulse control lists')
    for row in coincidences:
        for key in ('events', 'clusters_stock', 'clusters_fast', 'matched', 'fallback_events'):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError('invalid coincidence count')
    for row in pulses:
        if type(row.get('pulses_ref')) is not int or row['pulses_ref'] <= 0:
            raise ValueError('unsupported pulse control')
    totals = {key: sum(row[key] for row in coincidences) for key in
              ('events', 'clusters_stock', 'clusters_fast', 'matched', 'fallback_events')}
    totals['all_identical_cluster_lists'] = all(row.get('identical_cluster_lists') is True for row in coincidences)
    counts = {key: sum(row[key] for row in pulses) for key in
              ('pulses_ref', 'pulses_fast', 'matched', 'good_fits_matched', 'clean_matched')}
    counts.update(all_criteria_pass=all(row.get('pass') is True for row in pulses),
                  all_finding_identical=all(row.get('finding_identical') is True for row in pulses),
                  all_clean_finding_identical=all(row.get('finding_identical_clean') is True for row in pulses),
                  min_failed_flag_agree=min(row['failed_flag_agree'] for row in pulses),
                  min_good_within_pe_time=min(row['good_within_pe_time'] for row in pulses))
    return dict(coincidence_controls=len(coincidences), pulse_controls=len(pulses), coincidences=totals,
                pulses=counts, physics_admitted=False,
                limitations=['Each pulse arm is conditional on saved stock digis; event counts in pulse reports omit zero-digi events.',
                             'Coincidence counts cover explicit controls; duplicate event/file admission is external evidence.',
                             'All-sector bias, event-relative timing, native transport, GPU and full V-R gates are not certified.'])


def run(config, out):
    config, out = Path(config).resolve(), Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    spec = json.loads(config.read_text())
    paths = [config, Path(__file__).resolve()] + [Path(path).resolve() for key in ('coincidences', 'pulses') for path in spec[key]]
    if len(paths) != len(set(paths)):
        raise ValueError('duplicate or aliased control report')
    before = {str(path): tier1b_assets.sha256(path) for path in paths}
    result = summarize(*[[json.loads(Path(path).read_text()) for path in spec[key]] for key in ('coincidences', 'pulses')])
    result['sources_sha256'] = before
    if any(tier1b_assets.sha256(path) != digest for path, digest in before.items()):
        raise ValueError('saved named reco report drift')
    with out.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args.config, args.out)
