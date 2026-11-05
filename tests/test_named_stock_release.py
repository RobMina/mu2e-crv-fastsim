"""Release controls require exact event-relative timing and calibration, not just matching labels."""
import numpy as np
import pandas as pd
import pytest

from scripts.check_named_stock_release import exact_table


def rows():
    return pd.DataFrame(dict(run=np.array([1400, 1400], np.uint32), event=np.array([1, 2], np.uint32),
                             pbtime=np.array([0., 2.], np.float64)))


def test_exact_table_is_order_independent():
    result = exact_table(rows(), rows().iloc[::-1], ['run', 'event'], ['run', 'event', 'pbtime'])
    assert result['rows'] == 2


@pytest.mark.parametrize('change', ['dtype', 'signedzero', 'time', 'duplicate', 'missing', 'nan'])
def test_exact_table_rejects_nonidentical_evidence(change):
    a, b = rows(), rows()
    if change == 'dtype':
        b['pbtime'] = b.pbtime.astype(np.float32)
    elif change == 'signedzero':
        b.loc[0, 'pbtime'] = -0.
    elif change == 'time':
        b.loc[0, 'pbtime'] = .01
    elif change == 'duplicate':
        b = pd.concat([b, b.iloc[:1]])
    elif change == 'missing':
        b = b.iloc[:1]
    else:
        b.loc[0, 'pbtime'] = np.nan
    with pytest.raises(ValueError, match='stock evidence'):
        exact_table(a, b, ['run', 'event'], ['run', 'event', 'pbtime'])


def calibration():
    return pd.DataFrame(dict(run=np.full(5120, 1400), channel=np.arange(5120), status=np.zeros(5120, dtype=int),
                             roc=np.full(5120, -1), feb=np.full(5120, -1), febch=np.full(5120, -1)))


def test_calibration_is_complete_and_event_run_bound():
    from scripts.check_named_stock_release import validate_calibration
    validate_calibration(calibration(), rows())
    with pytest.raises(ValueError, match='event run'):
        validate_calibration(calibration().assign(run=999), rows())
    with pytest.raises(ValueError, match='5120'):
        validate_calibration(calibration().iloc[:5000], rows())


@pytest.mark.parametrize('field,value', [('channel', 0.), ('run', 1400.), ('status', .5), ('status', -1),
                                        ('roc', 0), ('febch', 64)])
def test_calibration_rejects_fractional_or_invalid_hardware(field, value):
    from scripts.check_named_stock_release import validate_calibration
    with pytest.raises(ValueError, match='calibration'):
        validate_calibration(calibration().assign(**{field: value}), rows())


def test_population_rejects_equally_truncated_arms(monkeypatch):
    from scripts import check_named_stock_release as driver
    monkeypatch.setattr(driver.response_receipt, 'event_identity', lambda path: dict(count=1, sha256='same'))
    with pytest.raises(ValueError, match='pinned census'):
        driver.require_population('truncated.root', 7193, 'same')
    with pytest.raises(ValueError, match='pinned census'):
        driver.require_population('other.root', 1, 'different')
    assert driver.require_population('ok.root', 1, 'same')['count'] == 1
