---
paths:
  - "tests/**"
---

# Strategy Testing Patterns

Reference implementations:

- `tests/strategies/contents/test_content.py` — full template with `OptsChain`;
  `exclude` for internal recursion params
- `tests/strategies/constructors/test_array.py` — kwargs inheritance; `DEFAULTS`
  derived from the parent module's
- `tests/strategies/contents/test_record_array.py` — smallest full-template
  example
- `tests/strategies/numpy/test_numpy_arrays.py` — min/max pairs with `ranges()`;
  the dtype pool depends on other drawn options
- `tests/strategies/forms/test_numpy_forms.py` — mode selection for
  strategy-valued kwargs (no `RelatedKwargs`)
- `tests/strategies/contents/test_content_lists.py` — callable-valued kwargs
  with `register_callable()`
- `tests/strategies/misc/test_ranges.py` — plain-dict kwargs without `OptsChain`
  (closest to the original template in awkward's
  `tests/properties/operations/test_flatten.py`)

Test modules not listed above may predate this template; follow the references,
not neighboring files. The remaining modules will be brought onto the template
eventually.

## 1. Option declarations

Each kwargs test module declares a strategy's options in up to three objects,
and a meta-test keeps them in sync with the strategy's signature:

- A `TypedDict` (`total=False`) with one key per signature parameter;
  annotations match the signature exactly (a convention the meta-test does not
  check — verify by reading; the meta-test compares only keys and defaults).
- `DEFAULTS`: the TypedDict instantiated with the signature's default values.
- `RelatedKwargs`: a second TypedDict naming the options drawn together (see
  section 2). Omitted when the coupling takes another form, such as mode
  selection.

The meta-test, named `test_kwargs_match_signature`, calls
`assert_kwargs_match_signature()` from `tests/funcs.py`:

```python
def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `contents()`'s parameters."""
    assert_kwargs_match_signature(
        func=st_ak.contents.contents,
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
in `contents()`; `all_option_or_none` and `st_option` in `content_lists()`) or a
positional factory (`st_` in `ranges()`). Adding a parameter to a strategy turns
this test red first; update the declarations with it.

A strategy that extends another subclasses the TypedDict and derives its
`DEFAULTS` by unpacking the parent module's — see `ArraysKwargs` in
`test_array.py`.

## 2. Strategy for kwargs

`<strategy>_kwargs()` is a `@st.composite` returning
`st_ak.OptsChain[MyKwargs]`, with a `chain` parameter so composable strategies
can delegate kwargs (`arrays_kwargs` chains `contents_kwargs`). When nothing
needs recorders or chaining, it returns a plain kwargs dict instead
(`test_ranges.py`).

The body makes two draws and merges them:

- A nested `_st_related_kwargs()` composite returns `RelatedKwargs` — the
  options whose values constrain each other: min/max pairs from
  `st_ak.ranges()`, feasibility-coupled flags (`test_content.py`), or pools that
  depend on other drawn values (`test_numpy_arrays.py`). A value that can be
  `None` is added to the dict only when it is not `None`, so the omitted-key
  path exercises the strategy's default.
- An `st.fixed_dictionaries({}, optional={...})` draw for the independent
  options; whether each key appears is drawn separately from its value.

Strategy-valued kwargs are registered on the chain so the test can later assert
what was drawn: `chain.register(strategy)` returns a tracked `RecordDraws`
wrapper (passed via `st.just(recorder)`), and `chain.register_callable(factory)`
returns a `RecordCallDraws` wrapper for callable-valued kwargs, aggregating
draws across all calls (`test_content_lists.py`). The option's pool typically
mixes concrete values with the recorder — e.g.
`st.one_of(_contents_list(), st.just(recorder))` in `test_record_array.py` — so
the concrete-value path is exercised too. For mutually exclusive parameter
groups, define mode functions selected with `st.one_of` (`test_numpy_forms.py`).

## 3. Main property-based test

The main property test in each file is named `test_properties` — the file name
already carries the strategy name. It draws the options, calls the test subject,
and asserts every option was effective:

```python
@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `numpy_arrays()`."""
    opts = data.draw(numpy_arrays_kwargs(), label='opts')
    opts.reset()  # clear recorders before each draw

    n = data.draw(st_ak.numpy_arrays(**opts.kwargs), label='n')

    max_size = opts.kwargs.get('max_size', DEFAULTS['max_size'])
    ...
```

Defaults are read as `opts.kwargs.get(k, DEFAULTS[k])`, never as literals. A
kwarg that can hold either a concrete value or a recorder is distinguished with
`match` / `case`: the recorder arms (`st_ak.RecordDraws()` /
`RecordCallDraws()`) assert against the recorded draws, and a concrete-value arm
(e.g. `case list():` in `test_record_array.py`) asserts the result uses the
passed objects.

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
    find(st_ak.numpy_arrays(), lambda a: math.prod(a.shape) == 0, settings=FIND)
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
from hypothesis_awkward.util import safe_compare as sc

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
