"""Saved control summaries retain raw finding failures alongside clean-rule outcomes."""
import json

import pytest

from scripts.summarize_named_reco import run, summarize


def controls():
    c = dict(events=10, clusters_stock=9, clusters_fast=9, matched=9, fallback_events=0, identical_cluster_lists=True)
    p = dict(pulses_ref=20, pulses_fast=21, matched=20, good_fits_matched=19, clean_matched=18,
             finding_identical=False, finding_identical_clean=True, failed_flag_agree=1., good_within_pe_time=.999, **{'pass': True})
    return c, p


def test_raw_finding_failure_is_not_overruled_by_clean_criteria():
    c, p = controls()
    result = summarize([c, c], [p, p])
    assert result['coincidences']['matched'] == 18 and result['coincidences']['events'] == 20
    assert result['coincidences']['all_identical_cluster_lists']
    assert result['pulses']['all_criteria_pass'] and result['pulses']['all_clean_finding_identical']
    assert not result['pulses']['all_finding_identical']
    assert result['pulses']['pulses_fast'] - result['pulses']['pulses_ref'] == 2
    assert not result['physics_admitted']


def test_unsupported_or_negative_counts_refused():
    c, p = controls()
    with pytest.raises(ValueError, match='nonempty'):
        summarize([], [p])
    with pytest.raises(ValueError, match='invalid coincidence'):
        summarize([dict(c, matched=-1)], [p])
    with pytest.raises(ValueError, match='unsupported pulse'):
        summarize([c], [dict(p, pulses_ref=0)])


def test_hashed_exclusive_publication_and_alias_rejection(tmp_path):
    c, p = controls()
    a, b = tmp_path / 'coinc.json', tmp_path / 'pulses.json'
    a.write_text(json.dumps(c))
    b.write_text(json.dumps(p))
    config, out = tmp_path / 'config.json', tmp_path / 'out.json'
    config.write_text(json.dumps(dict(coincidences=[str(a)], pulses=[str(b)])))
    result = run(config, out)
    assert len(result['sources_sha256']) == 4 and json.loads(out.read_text()) == result
    with pytest.raises(FileExistsError):
        run(config, out)
    config.write_text(json.dumps(dict(coincidences=[str(a), str(a)], pulses=[str(b)])))
    with pytest.raises(ValueError, match='duplicate'):
        run(config, tmp_path / 'new.json')
