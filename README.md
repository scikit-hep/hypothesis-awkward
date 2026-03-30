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
array=<Array [] type='0 * bool'>
array=<Array [61038, 65535, 2127] type='3 * uint16'>
array=<Array [1, nan, 1.19e-07, -0, 0] type='5 * float32'>
array=<Array [-1e-05+-infj, ..., 5.76e+16+-1.19e-07j] type='36 * complex64'>
array=<Array ['pÜx\x1d½1', '', '', '', 'z'] type='5 * string'>
array=<Array [b'\xb7\xb7\xc1b\x1d=\x93', ..., b'M'] type='4 * bytes'>
array=<Array [[], []] type='2 * var * 3 * unknown'>
array=<Array [['È\x11헎spµ\U00096bad', '']] type='1 * 2 * string'>
array=<Array [[], [[]], [[], []], []] type='4 * var * 0 * unknown'>
array=<Array [[-35858-02-18T15:30:39.815212, ...]] type='1 * var * datetime64[us]'>
array=<Array [NaT, ..., -290308-12-21T19:59:05.224253] type='7 * datetime64[us]'>
array=<Array [NaT, NaT, ..., NaT, 100 hours] type='9 * timedelta64[h]'>
array=<Array [1.56e+16+-5.15e+16j, ..., inf+0j] type='5 * complex64'>
array=<Array [b'', ..., b'R\xf7\xb6l\x1d\xdd-tmXK'] type='3 * union[bytes, bytes,...'>
array=<Array [[[[], [inf], [0.0], []]], [...]] type='2 * union[var * float32, var...'>
array=<Array [] type='0 * {"": string, Z: 1 * union[string, bytes]}'>
array=<Array [179, 179, 179, 179, 179, ..., 179, 179, 179, 179] type='34 * uint32'>
array=<Array [[...]] type='1 * var * 1 * bytes'>
array=<Array [??, ??, ??, ??] type='4 * bytes'>
array=<Array [[??], [??], [??], [??], [??], [??], [??]] type='7 * 1 * complex64'>
array=<Array [??, ??, ??, ??, ??, ..., ??, ??, ??, ??] type='29 * timedelta64[us]'>
array=<Array [3389378472, 7, 234, ..., 249, 7352234684956532367] type='17 * uint64'>
array=<Array [[], []] type='2 * var * var * var * bytes'>
array=<Array [1969-12-31T23:59:50.776627963145224442, ...] type='30 * datetime64[as]'>
array=<Array [0.807+-2.93e+106j, ..., -2.23e-311+2.42e+34j] type='9 * complex128'>
```

The current version generates arrays with `NumpyArray`, `EmptyArray`, string,
and bytestring as leaf contents that can be nested multiple levels deep in
`RegularArray`, `ListOffsetArray`, `ListArray`, `RecordArray`, and `UnionArray`.
Arrays might be virtual, shown as `??` in the output.

### The options of `arrays()`

The strategy `arrays()` has many options to control the output arrays. You can
find all options in the API reference:

- [**API reference: `arrays()`**][api-ref-arrays]

[api-ref-arrays]:
  https://scikit-hep.github.io/hypothesis-awkward/dev/strategies/constructors/

## Other strategies

In addition to `arrays()`, this package includes other strategies that generate
Awkward Arrays and related data types, which can be found in the API reference:

- [**API reference**][api-ref]

[api-ref]: https://scikit-hep.github.io/hypothesis-awkward/dev/
