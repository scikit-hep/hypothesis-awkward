# Getting Started

By the end of this page you will have drawn generated Awkward Arrays, learned to
read their reprs, run your first property-based test, and shaped what is
generated. The page assumes the package is [installed](installation.md).

## Draw your first arrays

The main strategy, `arrays()`, generates Awkward Arrays — a strategy is the
object Hypothesis draws test data from. To see what it produces, save this
script as `explore_arrays.py` and run it:

```python
import hypothesis_awkward.strategies as st_ak

strategy = st_ak.constructors.arrays()
for _ in range(5):
    print(repr(strategy.example()))
```

```bash
python explore_arrays.py
```

The script prints five generated arrays. The `example()` method draws one sample
from a strategy; it is meant for exploring a strategy interactively, not for use
inside tests.

## What `arrays()` generates

Generation is random: each run prints different arrays, and yours will differ
from any shown here. The lines below, each the repr of one generated array, were
collected across many runs — a single run of the script above prints five:

```text
<Array [772, 53, -4462, -5260] type='4 * int32'>
<Array ['', '', 'K', ..., '\x00\U000d1d457Ë_', 'm'] type='6 * string'>
<Array [1535-09-05, 0965-02-21, -784-08-11, NaT] type='4 * datetime64[W]'>
<Array [-inf+-0j, 0+-6.1e+16j, ..., 0+-6.81e-42j] type='5 * complex64'>
<Array [-2.23e-309+0j, ..., 1.06e-222+nanj] type='8 * complex128'>
<Array [] type='0 * unknown'>
<Array [[b'\xa1LC2'], ...] type='2 * 1 * bytes'>
<Array [??, ??, ??, ??, ??, ..., ??, ??, ??, ??, ??] type='24 * var * ?float64'>
<Array [None, None, None, None, ..., None, None, None] type='50 * ?unknown'>
<Array [('\xa0\xad'), ('©')] type='2 * (string)'>
<Array [b'\xd0\x86\xa3', ..., '\U000ad087'] type='4 * union[string, bytes]'>
<Array [None] type='1 * option[12 * var * unknown]'>
```

The generated arrays cover numbers (including `nan` and infinities), dates and
durations, strings, bytestrings, records, unions, missing values, and nested
lists of regular and variable length, in a variety of Awkward layouts. The
[API reference](../reference/strategies/constructors.md) lists exactly which
layouts are generated.

## Reading the output

- `None` is a missing value. `NaT` ("not a time") is the missing-value
  placeholder inside date and duration types.
- A `?` prefix or `option[...]` in the type string marks an option type — a type
  whose values can be missing. The two spellings mean the same; `?` is the short
  form.
- `??` marks a virtual value: the array is backed by lazy buffers that have not
  been computed yet, so the repr shows placeholders instead of values. This
  appears in the values, unlike the type-level `?`.
- Parentheses mark a record without field names, as in `2 * (string)`.
- A type string reads left to right: `4 * int32` is four `int32` values, and
  `24 * var * ?float64` is 24 variable-length lists of `float64` values that can
  be missing. `union[...]` is a mixed type, and `unknown` is the type of an
  array with no actual values (an empty array, or one whose values are all
  missing).

## Write your first property test

The strategies are for use with [Hypothesis](https://hypothesis.works/) in
tests. Save this as `test_arrays.py`:

```python
from hypothesis import given

import awkward as ak
import hypothesis_awkward.strategies as st_ak


@given(array=st_ak.constructors.arrays())
def test_arrays(array: ak.Array) -> None:
    assert isinstance(array, ak.Array)
```

The test runs with [pytest](https://docs.pytest.org/), which is not installed
with the package; run `pip install pytest` first if you do not have it. pytest
collects functions whose names start with `test_` in files named `test_*.py`.
Run the test:

```bash
pytest test_arrays.py
```

The test passes (pytest reports `1 passed`). The `@given` decorator ran the test
function many times, each time with a newly generated array, and the assertion
held for every one of them. That is a property-based test: instead of picking
input arrays by hand, you state a property and Hypothesis generates the inputs —
the [introduction](../index.md) explains the idea. Here the property is only
that every generated value is an `ak.Array`; a real test asserts a property of
your own code. When an assertion fails, Hypothesis reports a minimal failing
example; [Generating and Shrinking Samples](generating-and-shrinking-samples.md)
explains how.

## Shape what is generated

Options of `arrays()` narrow the output. For example, `allow_virtual=False`
turns off virtual arrays, so no `??` placeholders appear. Change the strategy in
`explore_arrays.py` and run it again:

```python
import hypothesis_awkward.strategies as st_ak

strategy = st_ak.constructors.arrays(allow_virtual=False)
for _ in range(5):
    print(repr(strategy.example()))
```

```bash
python explore_arrays.py
```

A run might print, for example:

```text
<Array [] type='0 * unknown'>
<Array [[[b''], [b''], ..., [b''], [b'']], ...] type='3 * 13 * 1 * bytes'>
<Array [] type='0 * var * 1 * int32'>
<Array [b'#h\x950#\x8cMv', b'', ..., b'\xbf\xcc\x1c\xfe\xe0'] type='9 * bytes'>
<Array ['í衣䕉', ..., 'D'] type='4 * string'>
```

The [API reference](../reference/strategies/constructors.md) documents all
options.
