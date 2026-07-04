---
paths:
  - "tests/**"
---

# Strategy Testing Patterns

Reference implementations:

- `tests/strategies/contents/test_content.py` (base `OptsChain` pattern, all
  optional kwargs)
- `tests/strategies/numpy/test_numpy_arrays.py` (min/max pairs with `ranges()`)
- `tests/strategies/forms/test_numpy_forms.py` (mode selection for
  strategy-valued kwargs)
- `tests/strategies/misc/test_ranges.py` (dependent kwargs without `OptsChain`)
- `tests/strategies/contents/test_content_lists.py` (callable-valued kwargs with
  `register_callable()`)

## 1. TypedDict for Strategy kwargs

Define a `TypedDict` that mirrors the strategy's parameters:

```python
class ContentsKwargs(TypedDict, total=False):
    '''Options for `contents()` strategy.'''

    dtypes: st.SearchStrategy[np.dtype] | None
    max_size: int
    allow_nan: bool
    allow_numpy: bool
    allow_empty: bool
    allow_string: bool
    allow_bytestring: bool
    allow_regular: bool
    allow_list_offset: bool
    allow_list: bool
    allow_record: bool
    allow_union: bool
    allow_indexed: bool
    allow_indexed_option: bool
    allow_byte_masked: bool
    allow_bit_masked: bool
    allow_unmasked: bool
    max_leaf_size: int | None
    max_depth: int | None
    min_length: int
    max_length: int | None
```

## 2. Strategy for kwargs

### Base pattern with `OptsChain`

All kwargs strategies use `@st.composite` returning `st_ak.OptsChain[MyKwargs]`.
The `chain` parameter enables kwargs delegation between composable strategies.

See `tests/strategies/contents/test_content.py` for a full example.

Key techniques:

- `@st.composite` with `chain` parameter for composability
- `chain.register(strategy)` creates a `RecordDraws` wrapper that tracks drawn
  values; pass it via `st.just(recorder)` as a strategy-valued kwarg
- `chain.register_callable(factory)` creates a `RecordCallDraws` wrapper for
  callable-valued kwargs (see "Callable-valued kwargs" section below)
- `chain.extend(kwargs)` returns a new `OptsChain` with merged kwargs
- Use `st.fixed_dictionaries` with `optional` for independently drawn kwargs

### Min/max pairs with `st_ak.ranges()`

Use `st_ak.ranges()` to generate `(min, max)` pairs where `min <= max` and
either may be `None`:

```python
min_size, max_size = draw(
    st_ak.ranges(min_start=0, max_end=100, max_start=DEFAULT_MAX_SIZE)
)
```

Include non-`None` values as required keys in `st.fixed_dictionaries`:

```python
drawn = (
    ('min_size', min_size),
    ('max_size', max_size),
)

kwargs = draw(
    st.fixed_dictionaries(
        {k: st.just(v) for k, v in drawn if v is not None},
        optional={...},
    )
)

return chain.extend(cast(MyKwargs, kwargs))
```

See `tests/strategies/numpy/test_numpy_arrays.py` and `tests/util/test_draw.py`
for full examples.

### Dependent kwargs (without `OptsChain`)

See `tests/strategies/misc/test_ranges.py` for a full example.

When kwargs depend on each other, use `@st.composite` and `flatmap` to draw
values with dependencies (e.g., an upper bound that depends on a drawn lower
bound).

Key techniques:

- `@st.composite` allows multiple `draw()` calls with dependencies
- `flatmap` chains dependent strategies (e.g., max depends on min)
- Mix required and optional in `st.fixed_dictionaries`

### Mode selection for strategy-valued kwargs

See `tests/strategies/forms/test_numpy_forms.py` for a full example.

When a strategy has mutually exclusive parameter groups (e.g., `type_` mode vs
`dtypes` mode), define mode functions selected via `st.one_of`. Within a mode, a
kwarg can be drawn as either a concrete value or `st.just(recorder)` — a tracked
strategy from `chain.register()` — so the test can later distinguish the two via
`match`.

In the test, call `reset()` before drawing and use `match` for assertions:

```python
opts = data.draw(numpy_forms_kwargs(), label='opts')
opts.reset()
result = data.draw(st_ak.numpy_forms(**opts.kwargs), label='result')

match type_:
    case ak.types.NumpyType():
        assert result.primitive == type_.primitive
    case st_ak.RecordDraws():
        drawn_primitives = {t.primitive for t in type_.drawn}
        assert result.primitive in drawn_primitives
```

Key techniques:

- Register multiple recorders for different strategy-valued parameters
- `st.one_of(type_mode(), dtypes_mode())` selects between mutually exclusive
  parameter groups
- `reset()` clears all registered recorders before each draw (avoids stale
  state)
- `match` / `case` distinguishes concrete values from `st_ak.RecordDraws` in
  assertions

### Callable-valued kwargs with `register_callable()`

See `tests/strategies/contents/test_content_lists.py` for a full example.

When a parameter is a callable that returns a strategy (not a strategy itself),
use `chain.register_callable()` to create a `RecordCallDraws` wrapper. Each call
to the wrapper produces a `RecordDraws` instance, and `drawn` aggregates all
values across all calls.

In the test, use `match` to check for `RecordCallDraws` and verify drawn values:

```python
match opts.kwargs.get('st_content'):
    case RecordCallDraws() as st_content:
        assert len(st_content.drawn) == len(result)
        assert all(d is r for d, r in zip(st_content.drawn, result))
```

Key techniques:

- `chain.register_callable(factory)` creates a `RecordCallDraws` wrapper that
  tracks all calls and their drawn values
- When the kwarg is optional (callable has a default), use `optional` in
  `st.fixed_dictionaries` so it can be omitted
- `RecordCallDraws.drawn` returns a flat list of all drawn values across all
  calls, in order
- `reset()` on `OptsChain` also resets all callable recorders

## 3. Main property-based test

Test that the strategy respects all its options. By convention, the main
property test in each file is named `test_properties` — the file name already
carries the strategy name, so the function name does not need to repeat it.

```python
@settings(max_examples=200)
@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `numpy_arrays()`."""
    # Draw options
    opts = data.draw(numpy_arrays_kwargs(), label='opts')
    opts.reset()

    # Call the test subject
    n = data.draw(st_ak.numpy_arrays(**opts.kwargs), label='n')

    # Assert the options were effective
    ...
```

## 4. Edge case reachability tests using `find()`

The main property test asserts that invariants hold for every draw — it tests
universal properties. It cannot assert that something is ever produced. `find()`
tests the opposite: that there exists a draw satisfying a predicate.

```python
def test_draw_empty() -> None:
    '''Assert that empty arrays can be drawn by default.'''
    find(
        st_ak.numpy_arrays(),
        lambda a: math.prod(a.shape) == 0,
        settings=settings(phases=[Phase.generate]),
    )
```

- Use `phases=[Phase.generate]` to skip shrinking (faster)
- Use `max_examples=2000` for rare conditions
- Use specific dtypes to target relevant types (e.g., `st_np.floating_dtypes()`
  for NaN tests)

## 5. Optional bounds with `safe_compare`

When an option like `max_size` or `min_size` may be `None`, use
`safe_compare as sc` to write concise range assertions:

```python
from hypothesis_awkward.util import safe_compare as sc

assert sc(min_size) <= len(result) <= sc(max_size)
```

`sc(None)` returns an object that is true for all inequality comparisons, so
`None` bounds are effectively ignored.

## 6. Global constants

Extract shared values like default parameters:

```python
DEFAULT_MAX_SIZE = 10
```

## 7. Tracking upstream Awkward Array bugs in tests

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
