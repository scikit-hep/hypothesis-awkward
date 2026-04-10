# hypothesis-awkward

_Hypothesis strategies for Awkward Arrays._

[![pypi-badge]][pypi] [![pypi-python-badge]][pypi]
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

[Hypothesis] is a _property-based testing_ library. Its [_strategies_][hyp-st]
are Python functions that strategically generate test data that can fail test
cases in _pytest_ and other testing frameworks. Once a test fails, Hypothesis
searches for the simplest sample that causes the same error. Hypothesis
automatically explores edge cases; developers do not need to craft test data
manually.

Property-based testing is useful for finding edge cases in _array_ libraries and
in code that uses them. In fact, Hypothesis strategies for [NumPy][hyp-st-numpy]
and [pandas][hyp-st-pandas] data types are included in Hypothesis itself.
[Xarray] provides [strategies for its data structure][xarray-st]. The _Apache
Arrow_ codebase has [strategies for PyArrow][pyarrow-st], which are not
officially documented in its API reference.

This package, [hypothesis-awkward], is a collection of Hypothesis strategies for
[Awkward Array][awkward-array], which can represent a wide variety of layouts of
nested, variable-length, and mixed-type data. The current version of this
package includes strategies that generate samples with certain types of layouts.
The goal is to develop strategies that can generate fully general Awkward Arrays
with multiple options to control the layout, data types, missing values, masks,
and other array attributes. These strategies can help close in on edge cases in
tools that use Awkward Array, and Awkward Array itself.

[hypothesis]: https://github.com/HypothesisWorks/hypothesis
[hyp-st]: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html
[hyp-st-numpy]:
  https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#numpy
[hyp-st-pandas]:
  https://hypothesis.readthedocs.io/en/latest/reference/strategies.html#pandas
[xarray]: https://xarray.dev/
[xarray-st]:
  https://docs.xarray.dev/en/stable/user-guide/testing.html#hypothesis-testing
[pyarrow-st]:
  https://github.com/apache/arrow/blob/apache-arrow-22.0.0/python/pyarrow/tests/strategies.py
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

The current version generates arrays with `NumpyArray`, `EmptyArray`, string,
and bytestring as leaf contents that can be nested multiple levels deep in
`RegularArray`, `ListOffsetArray`, `ListArray`, `RecordArray`, and `UnionArray`.
Option types (`IndexedOptionArray`, `ByteMaskedArray`, `BitMaskedArray`,
`UnmaskedArray`) add nullable values shown as `None` in the output. The `?` in
type strings (e.g., `?int64`) indicates option types. Arrays might be virtual,
shown as `??` in the output.

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
