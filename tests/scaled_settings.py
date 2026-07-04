"""A `settings` factory for `@given` tests that deviate from the baseline.

The main property tests carry no explicit `max_examples`; the profiles
registered in `tests/conftest.py` supply it (200 by default, 10,000 nightly).
`scaled()` keeps a deviating test's budget proportional to the baseline under
every profile: a scale above 1 needs a demonstrated rarity argument, and a
scale below 1 should not make the default-profile budget too small.
"""

from hypothesis import settings


def scaled(scale: float, /) -> settings:
    """`settings` with `max_examples` scaled from the active profile's baseline."""
    baseline = settings().max_examples  # `settings()` inherits the active profile
    return settings(max_examples=round(scale * baseline))
