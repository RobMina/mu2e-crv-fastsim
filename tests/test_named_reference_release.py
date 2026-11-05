"""Stock replacements must be complete, unique and leave original paired inputs unchanged."""
import copy

import pytest

from scripts.compare_named_reference_release import replacement_pairs


def legacy():
    return dict(pairs=[dict(stock=['/old/a.root', '/old/b.root'], candidates={'seed1': '/native/one.npz'}),
                       dict(stock=['/old/c.root'], candidates={'seed1': '/native/two.npz'})],
                legacy_producer_script={'path': '/producer.sh', 'sha256': 'frozen'})


def test_replacements_preserve_candidate_identity_and_legacy_configuration():
    original = legacy()
    before = copy.deepcopy(original)
    result = replacement_pairs(original, {f'/old/{name}.root': f'/new/{name}.root' for name in 'abc'})
    assert original == before
    assert result['pairs'][0]['stock'] == ['/new/a.root', '/new/b.root']
    assert result['pairs'][0]['candidates'] == original['pairs'][0]['candidates']
    assert result['legacy_producer_script'] == original['legacy_producer_script']


@pytest.mark.parametrize('mapping', [dict(), {'/old/a.root': '/new/a.root'},
    {'/old/a.root': '/new/a.root', '/old/b.root': '/new/a.root', '/old/c.root': '/new/c.root'},
    {'/old/a.root': '/old/b.root', '/old/b.root': '/new/b.root', '/old/c.root': '/new/c.root'}])
def test_incomplete_or_aliased_replacements_rejected(mapping):
    with pytest.raises(ValueError, match='replacement'):
        replacement_pairs(legacy(), mapping)
