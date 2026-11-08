---
paths:
  - "tests/**"
---

# Strategy Testing Patterns

Reference implementations:

- `tests/strategies/constructors/test_unmasked_array.py` — smallest example of
  drawing a strategy-valued option in advance (`KwargsDrawn`, no
  `RelatedKwargs`)
- `tests/strategies/constructors/test_indexed_array.py` — drawing in advance
  with `RelatedKwargs`; `optional_independent` merged with `related`
- `tests/strategies/constructors/test_regular_array.py` — drawing in advance
  when the drawn option shares `optional_independent` with other options
  (`_st_drawn()` and `_st_optional_independent(drawn)`)
- `tests/strategies/constructors/test_numpy_array.py` — drawing in advance when
  related options must fit the drawn value (`_st_related_kwargs(drawn)`)
- `tests/strategies/form/test_numpy_forms.py` — mode selection, with each mode
  drawing its strategy-valued options in advance (no `RelatedKwargs`)
- `tests/strategies/constructors/test_content.py` — full template with a
  recorder; `exclude` for internal recursion params
- `tests/strategies/constructors/test_array.py` — kwargs inheritance; `DEFAULTS`
  derived from the parent module's; the kwargs strategy draws the parent's and
  extends them
- `tests/strategies/numpy/test_numpy_arrays.py` — min/max pairs with `ranges()`;
  the dtype pool depends on other drawn options
- `tests/strategies/constructors/test_content_lists.py` — callable-valued kwargs
  with `RecordCallDraws`
- `tests/strategies/misc/test_ranges.py` — plain-dict kwargs without recorders
  (closest to the original template in awkward's
  `tests/properties/operations/test_flatten.py`)

Every kwargs test module under `tests/strategies/` follows this template; the
modules above are the examples to start from.

## 1. Option declarations

Each kwargs test module declares a strategy's options in up to four objects, and
a meta-test keeps them in sync with the strategy's signature:

- A `TypedDict` (`total=False`) with one key per signature parameter;
  annotations match the signature exactly (a convention the meta-test does not
  check — verify by reading; the meta-test compares only keys and defaults).
- `DEFAULTS`: the TypedDict instantiated with the signature's default values.
- `RelatedKwargs`: a second TypedDict naming the options drawn together (see
  section 2). Omitted when no options constrain each other
  (`test_unmasked_array.py`) or when the coupling takes another form, such as
  mode selection.
- `<Name>Drawn`: a TypedDict of the values drawn in advance for strategy-valued
  options (see section 2), e.g. `content: Content` in
  `RegularArrayContentsDrawn`. Only in modules that draw in advance.

The meta-test, named `test_kwargs_match_signature`, calls
`assert_kwargs_match_signature()` from `tests/funcs.py`:

```python
def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `contents()`'s parameters."""
    assert_kwargs_match_signature(
        func=st_ak.contents,
        exclude={'allow_union_root', 'allow_option_root', 'allow_indexed_root'},
        kwargs_cls=ContentsKwargs,
        defaults=DEFAULTS,
        related_cls=RelatedKwargs,
    )
```

It asserts that the TypedDict keys equal the signature's parameters minus
`exclude`, that `DEFAULTS` equals the signature's defaults, and that
`RelatedKwargs` keys are a subset. `exclude` records the parameters the kwargs
strategy deliberately does not draw: internal recursion controls (`allow_*_root`
in `contents()`; `all_option_or_none` and `st_option` in `content_lists()`), a
positional factory (`st_` in `ranges()`), or a required positional argument
(`content_len` in `_st_starts_stops()`). The test draws an excluded positional
argument first and passes it to the kwargs strategy, which draws the other
options within it (`test_ranges.py`, `test_list_array_starts_stops.py`). Adding
a parameter to a strategy turns this test red first; update the declarations
with it.

A strategy that extends another subclasses the TypedDict and derives its
`DEFAULTS` by unpacking the parent module's — see `ArraysKwargs` in
`test_array.py`.

## 2. Strategy for kwargs

`<strategy>_kwargs()` is a `@st.composite`. Its return type depends on the
strategy-valued options:

- **No strategy-valued options:** a plain kwargs dict (`test_ranges.py`,
  `test_bytestring.py`).
- **Strategy-valued options the strategy under test draws exactly once, without
  arguments** (e.g. `content` in the content wrappers, `dtypes` in
  `numpy_array_contents()`): `KwargsDrawn[MyKwargs, MyDrawn]` from
  `tests/funcs.py`, a `NamedTuple` of `kwargs` and `drawn`. The kwargs strategy
  draws the value itself and passes it either as the value or as
  `st.just(value)`; `drawn` holds the value for the assertions. No recorder is
  needed: the test does not check that the strategy draws once, which is not
  part of its contract.
- **Strategy-valued options drawn many times or called with arguments** (e.g.
  `dtypes` in `contents()`, `alphabet` in `string_contents()`, `st_content` in
  `content_lists()`): a plain kwargs dict whose option holds a recorder (see
  "Recorders" below).

A kwargs strategy for a strategy that extends another draws the parent's kwargs
strategy and adds its own options (`arrays_kwargs()` draws `contents_kwargs()`
and adds `allow_virtual`).

The body makes two draws and merges them:

- A nested `_st_related_kwargs()` composite returns `RelatedKwargs` — the
  options whose values constrain each other: min/max pairs from
  `st_ak.misc.ranges()`, feasibility-coupled flags (`test_content.py`), or pools
  that depend on other drawn values (`test_numpy_arrays.py`). A value that can
  be `None` is added to the dict only when it is not `None`, so the omitted-key
  path exercises the strategy's default.
- The independent options (`optional_independent`), where whether each key
  appears is drawn separately from its value, either with
  `st.fixed_dictionaries({}, optional={...})` or with `if draw(st.booleans()):`.

When one group is empty, its draw is skipped (`test_bytestring.py`,
`test_unmasked_array.py`). Merge with
`MyKwargs(**related, **optional_independent)`; both must be TypedDicts for mypy,
so `cast` a `fixed_dictionaries` result to `MyKwargs` first.

The kwargs strategy generates only options the strategy accepts. Options that
decide whether a call is feasible, such as `allow_*` flags of which at least one
must be `True`, are drawn together in `_st_related_kwargs()` so that every
combination is valid (`test_content.py`, `test_option.py`, `test_leaf.py`);
`test_properties` does not expect errors. When the strategy under test itself
raises on infeasible options, a dedicated `test_raise_*` test checks the error
(`test_option.py`, `test_leaf.py`); otherwise the `raise` line goes uncovered. A
strategy that only passes on a callee's error needs none: `contents()` gets its
`ValueError` from `leaf_contents()`, which `test_leaf.py` covers.

### Drawing in advance

Draw the value only when the option is passed, so the omitted path draws nothing
extra:

```python
    optional_independent = IndexedArrayContentsKwargs()
    drawn = IndexedArrayContentsDrawn(content=None)
    if draw(st.booleans()):
        content = draw(st_ak.contents(...))
        optional_independent['content'] = draw(
            st.sampled_from([content, st.just(content)])
        )
        drawn['content'] = content

    kwargs = IndexedArrayContentsKwargs(**related, **optional_independent)
    return KwargsDrawn(kwargs, drawn)
```

Here `drawn` always has the key, with `content: Content | None`, and the test
decides from `kwargs` whether the option was passed. This inline form fits when
the drawn option is the only independent option.

When the drawn option shares `optional_independent` with other options, split
the draw: `_st_drawn()` returns the drawn values, and
`_st_optional_independent(drawn)` builds the independent options from them, so
each composite returns one thing (`test_regular_array.py`,
`test_record_array.py`). In this form `<Name>Drawn` is `total=False`, and its
keys decide which options are passed:

```python
    related = draw(_st_related_kwargs())
    drawn = draw(_st_drawn())
    optional_independent = draw(_st_optional_independent(drawn))

    kwargs = RegularArrayContentsKwargs(**related, **optional_independent)
    return KwargsDrawn(kwargs, drawn)
```

When other options must fit the drawn value, the drawn option is a related
option. Draw `_st_drawn()` first and pass it to `_st_related_kwargs(drawn)`,
which constrains the other options to fit the value and builds the drawn option
itself; the value's pool stays unfiltered (`test_numpy_array.py`,
`test_leaf.py`: under `unique`, a drawn `bool` dtype caps `min_size` at 2):

```python
    drawn = draw(_st_drawn())
    related = draw(_st_related_kwargs(drawn))
```

An option that takes only a strategy or `None` (`dtypes`) is passed as
`st.just(value)` when drawn, and otherwise as `None` or omitted
(`test_numpy_array.py`). When the value is a list, build the `st.sampled_from()`
choices as an explicitly typed list, or mypy infers `object`
(`test_record_array.py`). For mutually exclusive parameter groups, define mode
composites selected with `st.one_of`, each returning its own `KwargsDrawn`
(`test_numpy_forms.py`).

A change to these draws can shift how often each form of an option is generated.
Before committing, compare a sample of the old and new kwargs strategies, with
the test subject drawn after the options as in `test_properties`; drawing the
options alone can exhaust short paths, such as an omitted option.

### Recorders

Strategy-valued kwargs that cannot be drawn in advance are wrapped in a recorder
so the test can later assert what was drawn. The recorders are test helpers in
`tests/record.py`, not part of the package: `RecordDraws(strategy)` records
every value drawn from the strategy, and `RecordCallDraws(factory)` wraps a
callable-valued kwarg, aggregating draws across all calls
(`test_content_lists.py`). Pass the recorder via `st.just(recorder)`, e.g.
`st.just(RecordDraws(st_ak.numpy.supported_dtypes()))` in `test_content.py`.

Create recorders inside the composite body, so each example gets new ones and no
reset is needed. A recorder created once, outside the per-example code (at
module level, or in a non-composite helper drawn by `@given`), is shared by
every example and accumulates draws across them. The option's pool typically
mixes concrete values with the recorder, so the concrete-value path is exercised
too.

## 3. Main property-based test

The main property test in each file is named `test_properties` — the file name
already carries the strategy name. It draws the options, calls the test subject,
and asserts every option was effective:

```python
@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `regular_array_contents()`."""
    kwargs, drawn = data.draw(regular_array_contents_kwargs(), label='opts')

    result = data.draw(st_ak.regular_array_contents(**kwargs), label='result')

    max_size = kwargs.get('max_size', DEFAULTS['max_size'])
    ...
    if 'content' in kwargs:
        assert result.content is drawn['content']
```

Defaults are read as `kwargs.get(k, DEFAULTS[k])`, never as literals. A value
drawn in advance is checked against `drawn`, whether it was passed as the value
or as `st.just(value)`, so one check covers both forms.

In a module with recorders, the kwargs strategy returns plain kwargs, so the
test draws `kwargs` alone. A kwarg that can hold either a concrete value or a
recorder is distinguished with `match` / `case`: the recorder arms
(`RecordDraws()` / `RecordCallDraws()`) assert against the recorded draws, and a
concrete-value arm asserts the result uses the passed objects.

Do not set `max_examples` on `@given` tests. The profiles registered in
`tests/conftest.py` supply it — 200 by default (PRs and local runs), 10,000
under the `nightly` profile selected with the `HYPOTHESIS_PROFILE` environment
variable (or pytest's `--hypothesis-profile` flag). A test may deviate from the
baseline with `@scaled(x)` from `tests/scaled_settings.py`, which keeps its
budget proportional to the baseline under every profile:

- above 1 only with a demonstrated rarity argument (a case the test must cover
  that the baseline misses; historical values inherited without one were
  removed);
- below 1 is possible if necessary; use with caution so the PR budget does not
  become too small.

## 4. Edge case reachability tests using `find()`

The main property test asserts that invariants hold for every draw — it tests
universal properties. It cannot assert that something is ever produced. `find()`
tests the opposite: that there exists a draw satisfying a predicate.

Always pass one of the shared `settings` instances from
`tests/find_settings.py`:

```python
from tests.find_settings import FIND


def test_draw_empty() -> None:
    """Assert that empty arrays can be drawn by default."""
    find(st_ak.numpy.numpy_arrays(), lambda a: math.prod(a.shape) == 0, settings=FIND)
```

- Never call `find()` without `settings`: Hypothesis then falls back to an
  internal default (`max_examples=2000` with the example database on), and a
  reachability test that replays a stored example passes without exercising
  generation.
- The shared instances pin `max_examples` explicitly, so `find()` budgets stay
  independent of the active Hypothesis profile, and set `database=None`.
- Start with `FIND` (2000 examples, shrinking on). Escalate to `FIND_RARE`
  (10,000) when the target is too rare, and switch to a `_NO_SHRINK` variant
  (`phases=[Phase.generate]`) only when shrinking is slow.
- Derive any one-off tweak from the shared instances (e.g.
  `settings(FIND, ...)`) instead of building `settings` inline.
- Use specific dtypes to target relevant types (e.g., `st_np.floating_dtypes()`
  for NaN tests, where `st_np` is `hypothesis.extra.numpy`)

## 5. Optional bounds with `safe_compare`

When an option like `max_size` or `min_size` may be `None`, use
`safe_compare as sc` to write concise range assertions:

```python
from hypothesis_awkward._util import safe_compare as sc

assert sc(min_size) <= len(result) <= sc(max_size)
```

`sc(None)` returns an object that is true for all inequality comparisons, so
`None` bounds are effectively ignored.

## 6. Tracking upstream Awkward Array bugs in tests

Two patterns, depending on whether the bug can be triggered directly or only
shows up as an incidental failure inside a broader property test.

### Direct repro with `xfail`

When a bug can be triggered by a small, hand-built case, write a dedicated test
that constructs it directly and mark it `xfail` with the reason naming the
broken version:

```python
"""Reproduce <library> bug with <short description>.

Fixed in <library> vX.Y.Z (likely <PR/issue>).
"""


@pytest.mark.xfail(reason='fails with <library> vX.Y')
def test_<bug_name>() -> None:
    """<what triggers the bug and what it raises>."""
    ...  # minimal case that triggers the bug
```

Full historical example:
[`test_from_buffers.py` at v0.19.0](https://github.com/scikit-hep/hypothesis-awkward/blob/v0.19.0/tests/strategies/constructors/test_from_buffers.py)
(removed once the minimum supported Awkward version moved past the fix).

- The module or test docstring names the exact broken version and, if known, the
  upstream fix.
- Once the minimum supported version moves past the fix, the test XPASSes. This
  project does not set `xfail_strict`, so an XPASS does not fail the run — watch
  for it in the test summary as the signal to delete the test, and update any
  doc that mentions it, rather than leave a dead marker in the suite.

### Exclude the broken case from a property test

When the bug only shows up inside a broader generated space and is not worth a
dedicated repro, define a local predicate that reports whether a case is
expected to work, and branch the assertion around it instead of adding an
`xfail`:

```python
def _is_<condition>(a: ak.Array) -> bool:
    """True if <operation> is expected to work without error.

    <what fails and why>
    <link to the upstream issue>
    """
    ...  # determine whether this case is affected


if _is_<condition>(a):
    ...  # normal assertion
else:
    with pytest.raises(<ExpectedError>):
        ...  # the broken case raises instead
```

Full example (pinned to v0.19.0 so the line numbers stay accurate):
[`test_numpy_arrays.py`, lines 163–185](https://github.com/scikit-hep/hypothesis-awkward/blob/v0.19.0/tests/strategies/numpy/test_numpy_arrays.py#L163-L185).

- The predicate's docstring links the upstream issue, same as the `xfail` reason
  above.
- Unlike the `xfail` pattern, there is no automatic signal when the bug is fixed
  — revisit the predicate (and its branch) when the linked issue closes.
