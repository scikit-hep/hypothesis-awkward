# Hypothesis-awkward

_Hypothesis strategies for Awkward Array._

[![pypi-python-badge]][pypi] [![pypi-badge]][pypi]
[![conda-forge-badge]][conda-forge]

[![test-badge]][test] [![codecov-badge]][codecov]

[pypi-badge]: https://img.shields.io/pypi/v/hypothesis-awkward.svg
[pypi-python-badge]:
  https://img.shields.io/pypi/pyversions/hypothesis-awkward.svg
[pypi]: https://pypi.org/project/hypothesis-awkward
[conda-forge-badge]:
  https://img.shields.io/conda/vn/conda-forge/hypothesis-awkward.svg
[conda-forge]: https://anaconda.org/conda-forge/hypothesis-awkward
[test-badge]:
  https://github.com/scikit-hep/hypothesis-awkward/actions/workflows/unit-test.yml/badge.svg
[test]:
  https://github.com/scikit-hep/hypothesis-awkward/actions/workflows/unit-test.yml
[codecov-badge]:
  https://codecov.io/gh/scikit-hep/hypothesis-awkward/graph/badge.svg?token=cffic9D2b3
[codecov]: https://codecov.io/gh/scikit-hep/hypothesis-awkward

[Awkward Array][awkward-array] represents deeply nested, variable-length, and
mixed-type data — the kind of irregular structure common in scientific datasets.
Its valid arrays therefore span a vast combinatorial space, and test data
written by hand covers only a small corner of it. The edge cases that break code
tend to hide in the parts no one thought to write down.

_Property-based testing_ addresses this. Instead of asserting specific outputs
for hand-picked inputs, you assert properties that should hold for any valid
input and let the framework generate the inputs. [Hypothesis] is a
property-based testing library for Python: its [_strategies_][hyp-st] are
composable objects that describe how to build test data, and when a test fails
Hypothesis _shrinks_ it, searching for a minimal sample that still triggers the
failure.

This package, [hypothesis-awkward], brings property-based testing to Awkward
Array with a collection of strategies for generating Awkward Arrays. Its main
strategy, [`arrays()`][api-ref-arrays], generates nearly fully general Awkward
Arrays: called with no arguments, it produces nested, variable-length, record,
and union layouts; leaf values of any NumPy dtype Awkward Array supports, as
well as strings and bytestrings; optional, masked, and missing values; and
virtual arrays — with options to constrain any of these. The goal is full
generality, so these strategies can surface edge cases in tools that use Awkward
Array, and in Awkward Array itself.

[hypothesis]: https://github.com/HypothesisWorks/hypothesis
[hyp-st]: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html
[hypothesis-awkward]: https://github.com/scikit-hep/hypothesis-awkward
[awkward-array]: https://awkward-array.org/

## Installation

You can install the package from PyPI using pip:

```bash
pip install hypothesis-awkward
```

This also installs Hypothesis and Awkward Array as dependencies unless they are
already installed.

## The strategy `arrays()`

The function `arrays()` is the main strategy. It generates Awkward Arrays with
many options to control the output arrays.

### Sample outputs of `arrays()`

You can see sample outputs of the current version of `arrays()` in the test
case:

```python
from hypothesis import given

import awkward as ak
import hypothesis_awkward.strategies as st_ak


@given(array=st_ak.constructors.arrays())
def test_array(array: ak.Array) -> None:
    print(f'{array=!r}')
```

For example, this might print:

```python
array=<Array ['', '\U000c2f9f', ..., '@ú\x94j\U000c4364e'] type='4 * string'>
array=<Array [[], [], None, [], ..., [], [], None] type='42 * option[var * ?bytes]'>
array=<Array [??, ??, ??, ??, ??, ??] type='6 * var * unknown'>
array=<Array [[], [], [], [], [], [], [], []] type='8 * var * string'>
array=<Array [??, ??, ??, ??, ??, ??, ??, ??] type='8 * var * string'>
array=<Array [b'O\x01\x14\xecE\xdb_'] type='1 * bytes'>
array=<Array [??, ??] type='2 * var * bytes'>
array=<Array [None] type='1 * ?bytes'>
array=<Array [??, ??, ??, ??] type='4 * string'>
array=<Array [NaT, NaT, ..., -9223372036854773681] type='26 * datetime64[Y]'>
array=<Array [[??, ??], [??, ??], ..., [??, ??]] type='8 * 2 * var * timedelta64[fs]'>
array=<Array [[[[], [], [], [], []]]] type='1 * 1 * var * var * timedelta64[fs]'>
array=<Array [[[[[], [], [], [], []]]]] type='1 * 1 * 1 * var * var * var * bool'>
array=<Array [[16996], [10841], ..., [10841], None] type='7 * option[1 * uint16]'>
array=<Array [[0]] type='1 * option[1 * uint16]'>
array=<Array [??] type='1 * option[1 * uint16]'>
array=<Array [[None]] type='1 * 1 * option[1 * option[var * int16]]'>
array=<Array [[]] type='1 * option[var * 0 * union[timedelta64[D], 0 * unknown]]'>
array=<Array [??, ??] type='2 * datetime64[D]'>
array=<Array [??, ??, ??, ??] type='4 * ?timedelta64[us]'>
array=<Array [??, ??, ??, ??, ??, ??, ..., ??, ??, ??, ??, ??, ??] type='14 * bytes'>
array=<Array [[], [], [], [], ..., [], [], [], []] type='55 * option[var * var * ...'>
array=<Array [0.0, inf, 0.0, nan, 0.0] type='5 * float16'>
array=<Array [None, -768614336404561008-11, ..., None] type='6 * ?datetime64[M]'>
array=<Array [??, ??] type='2 * option[var * 1 * string]'>
```

In the type strings above, a `?` marks an option type (e.g., `?int64`), whose
missing values print as `None`. Virtual arrays print as `??`.

### The options of `arrays()`

The strategy `arrays()` has many options to control the output arrays. You can
find all options in the API reference:

- [**API reference: `arrays()`**][api-ref-arrays]

[api-ref-arrays]:
  https://scikit-hep.github.io/hypothesis-awkward/dev/reference/strategies/constructors/

## Other strategies

In addition to `arrays()`, this package includes other strategies that generate
Awkward Arrays and related data types, which can be found in the API reference:

- [**API reference**][api-ref]

[api-ref]: https://scikit-hep.github.io/hypothesis-awkward/dev/reference/
