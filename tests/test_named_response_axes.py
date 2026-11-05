"""Saved response/finder axes retain zero events, immutable populations and declared producer settings."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from crvfastsim import runconfig
from scripts.check_named_response_axes import cluster_frame, contrast, normalized, require_metadata, summarize


def events():
    return pd.DataFrame(dict(run=[1400, 1400], subrun=[102, 102], event=[1, 2]))


def clusters():
    return events().iloc[:1].assign(pe=20., start=5., end=8., x=0., y=1., z=2.)


def test_zero_events_and_exact_paired_effects():
    a = summarize(events(), clusters())
    b = summarize(events(), clusters().assign(pe=30.))
    assert a['event_count'].tolist() == [1, 0]
    assert a['event_pe'].tolist() == [20., 0.]
    result = contrast(a, b)
    assert result['event_pe']['mean_difference'] == 5
    assert result['event_pe']['paired_event_se'] == 5
    assert result['event_pe']['changed_events'] == 1
    assert result['cluster_presence'] == dict(both=1, reference_only=0, candidate_only=0, neither=1)
    assert sum(a['histograms']['event_clusters']) == 2


def test_outside_census_and_duplicate_universe_fail():
    with pytest.raises(ValueError, match='outside saved'):
        summarize(events(), clusters().assign(event=3))
    with pytest.raises(ValueError, match='unique integer'):
        summarize(pd.concat([events(), events()]), clusters())


def test_empty_clusters_keep_zero_histogram():
    result = summarize(events(), clusters().iloc[:0])
    assert result['clusters'] == result['cluster_events'] == 0
    assert result['event_count'].tolist() == [0, 0]
    assert sum(result['histograms']['event_clusters']) == 2


def test_cluster_frame_full_identity_and_positions():
    output = dict(cl_event=np.array([(102 << 20) | 1]), cl_pe=[20.], cl_start=[5.], cl_end=[8.], cl_pos=[[1., 2., 3.]])
    result = cluster_frame(events(), output)
    assert result[['run', 'subrun', 'event']].values.tolist() == [[1400, 102, 1]]
    assert result[['x', 'y', 'z']].values.tolist() == [[1., 2., 3.]]
    output['cl_event'] = np.array([(102 << 20) | 3])
    with pytest.raises(ValueError, match='outside saved'):
        cluster_frame(events(), output)


def test_cluster_frame_rejects_capacity_and_nan():
    output = dict(cl_event=np.array([(102 << 20) | 1]), cl_pe=[20.], cl_start=[5.], cl_end=[8.], cl_pos=[[1., 2., np.nan]])
    with pytest.raises(ValueError, match='nonfinite'):
        cluster_frame(events(), output)
    with pytest.raises(ValueError, match='one run'):
        cluster_frame(events().assign(run=[1400, 1430]), output)


def metadata():
    rc = runconfig.get('run1a_v01')
    cfg = dataclasses.replace(rc.chain_config(), window=runconfig.OFFSPILL_WINDOW)
    evidence = {'test_conditions': True}
    result = dict(schema='crv-fasttables-named-v1', run_config=rc.name, stage='chain', seed=1, jax=False, noise=True,
                  conditions=evidence, chain_config=normalized(dataclasses.asdict(cfg)),
                  finder=normalized(dataclasses.asdict(cfg.finder)), configured_window_ns=list(rc.window),
                  effective_window_ns=list(runconfig.OFFSPILL_WINDOW), window_selection='offspill', window_applied=True,
                  timing_policy='fixed_absolute_crop_no_pbtime_rebase', onspill_equivalence=False)
    return rc, evidence, result


def test_metadata_binds_actual_response_and_attachment():
    rc, evidence, meta = metadata()
    require_metadata(meta, rc, evidence, rc.response(), True)
    with pytest.raises(ValueError, match='producer metadata'):
        require_metadata(meta, rc, evidence, rc.response(), False)
    changed = dataclasses.replace(rc.response(), photon_yield_scale=.91)
    with pytest.raises(ValueError, match='producer metadata'):
        require_metadata(meta, rc, evidence, changed, True)


@pytest.mark.parametrize('field,value', [('seed', True), ('seed', 2), ('noise', False), ('stage', 'reco'),
                                        ('onspill_equivalence', True), ('conditions', {'wrong': True})])
def test_metadata_rejects_different_draw_or_timing(field, value):
    rc, evidence, meta = metadata()
    meta[field] = value
    with pytest.raises(ValueError, match='producer metadata'):
        require_metadata(meta, rc, evidence, rc.response(), True)


@pytest.mark.parametrize('column,value', [('sipm', 4), ('sipm', -1), ('bar', 1280), ('bar', -1),
                                         ('time', np.nan), ('time', np.inf), ('pe', np.nan)])
def test_saved_pulse_input_rejected_before_dispatch(column, value):
    from scripts.check_named_response_axes import validate_pulses
    pulses = pd.DataFrame(dict(bar=[0], sipm=[0], pe=[20.], time=[100.]))
    pulses[column] = value
    with pytest.raises(ValueError, match='saved pulse'):
        validate_pulses(pulses, 1280)


def test_saved_pulse_input_keeps_valid_unmodified():
    from scripts.check_named_response_axes import validate_pulses
    pulses = pd.DataFrame(dict(bar=[0, 1279], sipm=[0, 3], pe=[20., 30.], time=[100., 200.]))
    before = pulses.copy(deep=True)
    validate_pulses(pulses, 1280)
    pd.testing.assert_frame_equal(before, pulses)


def test_float32_cluster_pe_is_accumulated_in_float64():
    rows = pd.concat([clusters()] * 3, ignore_index=True).assign(pe=np.array([1e8, 1., 1.], dtype=np.float32))
    result = summarize(events(), rows)
    assert result['total_cluster_pe'] == 100000002.
    assert result['event_pe'].tolist() == [100000002., 0.]
