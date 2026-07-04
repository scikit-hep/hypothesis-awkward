import os

from hypothesis import settings

try:
    import icecream

    icecream.install()  # pragma: no cover
except ImportError:  # pragma: no cover
    pass

# Values not given here fall back to the profile active at registration time.
# On GitHub Actions, hypothesis auto-loads its built-in 'ci' profile
# (derandomize=True, deadline=None, database=None, print_blob=True).
settings.register_profile('default', max_examples=200)
settings.register_profile(
    'nightly',
    max_examples=10_000,
    deadline=None,
    print_blob=True,
    # The value inherited from 'ci' (True) would repeat the same examples
    # every night.
    derandomize=False,
)
settings.load_profile(os.environ.get('HYPOTHESIS_PROFILE', 'default'))
