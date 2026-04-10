# Getting Started

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
find all options in the [API reference](../reference/strategies/constructors.md).
