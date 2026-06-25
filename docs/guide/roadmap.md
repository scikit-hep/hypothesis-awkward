# Roadmap

This page describes what `hypothesis-awkward` generates today and where it is
heading, and it is updated as the package develops. It gives no dates; the order
of the items reflects priority, not a schedule.

## What you can generate today

Today the main strategy, [`arrays()`](../reference/strategies/constructors.md),
generates arrays **in terms of their layout** — the low-level structure of an
Awkward Array, a tree of `Content` nodes — by composing those nodes directly. It
resides in the `constructors` subpackage because its development has followed
the
[direct constructors](https://awkward-array.org/doc/stable/user-guide/how-to-create-constructors.html)
section of the Awkward Array documentation.

The library includes strategies for each of `EmptyArray`, `NumpyArray`,
`RegularArray`, `ListOffsetArray`, `ListArray`, `RecordArray`, `UnionArray`,
`IndexedArray`, `IndexedOptionArray`, `ByteMaskedArray`, `BitMaskedArray`,
`UnmaskedArray`, as well as for strings and bytestrings. The main strategy
`arrays()` generates nested combinations of these, and can also generate arrays
backed by virtual (lazy) buffers. [Getting Started](getting-started.md) shows
sample outputs, and the [API reference](../reference/strategies/constructors.md)
lists the available options. The strategy for categorical data has not yet been
implemented.

The current `arrays()` is not exported from the top-level package, so it must be
called as `st_ak.constructors.arrays()` (with
`import hypothesis_awkward.strategies as st_ak`). The bare name
`st_ak.arrays()`, without `constructors`, is **reserved** for the type- and
form-directed strategy described below; it does not exist yet. So today you
shape generation by constraints — the `allow_*` flags, `dtypes`, length bounds
(`min_length`, `max_length`), and a total-size bound (`max_size`) — rather than
by an exact type. Asking for an array of a _specific type_ is the next
direction.

The project is at version 0.x, before 1.0, so the application programming
interface (API) can still change between releases (see the
[release notes](https://github.com/scikit-hep/hypothesis-awkward/releases)).

## Directions we are exploring

Awkward describes an array at three levels, from the most abstract to the most
concrete: its **type** (the datashape, such as `var * float64`), its **form**
(the structural blueprint of a layout without the data buffers; one type can
have many forms, but each form has exactly one type), and its **layout** (the
concrete tree of `Content` nodes that holds the data buffers). Every generated
array has all three. What changes across the directions below is the level you
describe it in terms of: today its layout, next its type, and later its form.

### Near-term

**Complete the layout coverage.** Add categorical-data strategies.

**Generate arrays from a type.** Add strategies that generate Awkward _types_
(building on the existing `st_ak.numpy_types()`), then strategies that generate
arrays matching a given type. This is the planned home of the reserved
`st_ak.arrays()`. It would let a test generate arrays of the specific type that
the code under test expects:

```python
# Not yet available — illustrative of the planned type-directed API.
# st_ak.arrays() does not exist yet; today, call st_ak.constructors.arrays()
# without type=.
from hypothesis import given

import hypothesis_awkward.strategies as st_ak


@given(array=st_ak.arrays(type='var * float64'))
def test_my_analysis(array):
    result = my_analysis(array)
    # A property that should hold for every generated array:
    assert len(result) == len(array)
```

Here `@given` is the Hypothesis decorator that runs the test function
repeatedly, each time on one generated array; the same test runs today if you
call `st_ak.constructors.arrays()` without `type=` — narrowing its output with
`dtypes` or the `allow_*` flags to the inputs your code accepts. See the
[introduction](../index.md) for property-based testing and
[Getting Started](getting-started.md) for using the current `arrays()`.

_Related work:_ static type stubs for Awkward data types are being explored in
parallel in the personal proof-of-concept repository
[`awkward-stubs-pilot-01`](https://github.com/TaiSakuma/awkward-stubs-pilot-01).
It would pair naturally with type-directed generation.

### Later

**Generate arrays from a form.** Because one type can have many forms — a list
type, for example, can be laid out as a `ListOffsetArray` or a `ListArray` — a
later step is to generate forms for a given type (building on the existing
`st_ak.numpy_forms()`), and then arrays matching a given form, also under the
reserved `st_ak.arrays()`. This helps when a test depends on the concrete
layout, not only the type.

## How to influence the roadmap

These priorities are open. If you build on Awkward and need a particular type,
layout, or option that the library does not yet generate, please
[open an issue](https://github.com/scikit-hep/hypothesis-awkward/issues) with
the use case. A concrete use case is the most useful way to shape what comes
next.
