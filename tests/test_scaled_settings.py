import pytest
from hypothesis import settings

from tests.scaled_settings import scaled


@pytest.mark.parametrize('scale', [0.5, 1.0, 2.5])
def test_scaled(scale: float) -> None:
    """Assert `max_examples` is the scale times the session profile's baseline.

    The expected value is computed from the same ambient profile the implementation
    reads, so this checks the arithmetic but not *when* the baseline is looked up.
    """
    baseline = settings().max_examples
    assert scaled(scale).max_examples == round(scale * baseline)
