"""Shared `settings` instances for `find()` reachability tests.

`max_examples` is pinned explicitly so `find()` budgets stay independent of the
active Hypothesis profile. `database=None` because replaying a stored example
would let a reachability test pass without exercising generation. A bare
`find()` call must be avoided: without a `settings` argument, Hypothesis falls
back to an internal default with a 2000-example budget and the database on.

Start with `FIND`; use `FIND_RARE` when the target is too rare for the default
budget; use a `_NO_SHRINK` variant only when shrinking is slow. Derive any
one-off tweak from these instances (e.g. `settings(FIND, ...)`).
"""

from hypothesis import Phase, settings

FIND = settings(max_examples=2_000, database=None)
FIND_NO_SHRINK = settings(FIND, phases=[Phase.generate])

FIND_RARE = settings(FIND, max_examples=10_000)
FIND_RARE_NO_SHRINK = settings(FIND_RARE, phases=[Phase.generate])
