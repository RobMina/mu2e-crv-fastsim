"""Clock phases cannot be shared across ROC or event triples merely because FEB numbers coincide."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.check_named_waveforms import phases


def test_full_event_roc_feb_phase_identity():
    events = pd.DataFrame(dict(run=[1400, 1400, 1401], subrun=[102] * 3, event=[1] * 3))
    cond = SimpleNamespace(roc=np.array([1, 2]), feb=np.array([1, 1]))
    result = phases(events, np.array([0, 1, 0]), np.array([2., 3., 4.]), events, np.array([0, 1, 0]), cond)
    assert result.tolist() == [2., 3., 4.]
    with pytest.raises(ValueError, match='phases disagree'):
        phases(pd.concat([events.iloc[:1]] * 2), np.array([0, 0]), np.array([2., 3.]), events, np.array([0, 1, 0]), cond)
