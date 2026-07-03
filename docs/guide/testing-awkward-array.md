# Testing Awkward Array

[Awkward Array](https://awkward-array.org/) represents nested, variable-length,
and mixed-type data, so its valid arrays span a large combinatorial space of
layouts. Test data written by hand covers only a small part of that space, and
failures often occur on input shapes that are absent from hand-written test
cases. This makes Awkward Array hard to test thoroughly.

This page is for Awkward Array contributors and for readers assessing the
project. If you want to use the strategies in your own tests, start with
[Getting Started](getting-started.md).

## How hypothesis-awkward tests Awkward Array

This package addresses that problem with [property-based testing](../index.md):
running one test against many automatically generated inputs instead of a fixed
list. Its main strategy, `st_ak.constructors.arrays()`, generates nearly fully
general Awkward Arrays, including virtual arrays (arrays whose buffers are not
yet materialized); categorical data is the remaining gap (see the
[Roadmap](roadmap.md)). See [Getting Started](getting-started.md) for what it
produces.

These strategies are integrated into Awkward Array's continuous integration
(CI). The first property-based tests were added in
[#3887](https://github.com/scikit-hep/awkward/pull/3887) and now run on every
change (in
[`tests/properties`](https://github.com/scikit-hep/awkward/tree/main/tests/properties)).
Two kinds of properties are checked so far:

**Round-trip.** Converting an array to buffers with `ak.to_buffers()` and back
with `ak.from_buffers()` reconstructs an equal array
([#3887](https://github.com/scikit-hep/awkward/pull/3887)).

```python
from hypothesis import given

import awkward as ak
import hypothesis_awkward.strategies as st_ak


@given(a=st_ak.constructors.arrays())
def test_roundtrip(a: ak.Array) -> None:
    sent = ak.to_buffers(a)
    returned = ak.from_buffers(*sent)
    assert ak.array_equal(a, returned, equal_nan=True)
```

**Equality.** `ak.array_equal()` is reflexive (an array equals itself) and
symmetric (if `a` equals `b`, then `b` equals `a`)
([#3891](https://github.com/scikit-hep/awkward/pull/3891)).

```python
@given(a=st_ak.constructors.arrays())
def test_reflexivity(a: ak.Array) -> None:
    assert ak.array_equal(a, a, equal_nan=True)


@given(a1=st_ak.constructors.arrays(), a2=st_ak.constructors.arrays())
def test_symmetry(a1: ak.Array, a2: ak.Array) -> None:
    forward = ak.array_equal(a1, a2, equal_nan=True)
    backward = ak.array_equal(a2, a1, equal_nan=True)
    assert forward == backward
```

`equal_nan=True` treats two `NaN` values as equal — and likewise two `NaT`
values in datetimes and timedeltas — which the properties need because the
generated arrays can contain both. When a property fails, Hypothesis shrinks the
input toward a minimal failing array — a best-effort search that
[Generating and Shrinking Samples](generating-and-shrinking-samples.md) explains
— which is why the reports below reduce to small, reproducible cases.

Both kinds of properties are oracle-free: checking them needs no reference
implementation to compare against. The goal is to cover all testable properties
of Awkward Array, including operations, slicing, reducers, and kernels.

## Bugs found

These tests have found bugs in both Awkward Array and Hypothesis. This log is
reviewed at each release.

### Awkward Array

- [#3888](https://github.com/scikit-hep/awkward/issues/3888) (fixed) —
  `ak.array_equal()` raised an error on virtual arrays and returned the wrong
  result for empty unions.
- [#3921](https://github.com/scikit-hep/awkward/pull/3921) (fixed) —
  `ak.array_equal()` returned the wrong result for datetimes and timedeltas
  containing `NaT` (not-a-time).
- [#3962](https://github.com/scikit-hep/awkward/pull/3962) (fixed) —
  `ak.almost_equal()`, which backs `ak.array_equal()`, compared record-array
  fields incorrectly.
- [#4126](https://github.com/scikit-hep/awkward/issues/4126) (fixed) —
  `IndexedOptionArray.to_ByteMaskedArray` raised a `TypeError` when its content
  was an `EmptyArray`.

The test suite also reproduces or accounts for two known Awkward Array issues. A
test in
[`test_from_buffers.py`](https://github.com/scikit-hep/hypothesis-awkward/blob/main/tests/strategies/constructors/test_from_buffers.py)
reproduces an `ak.from_buffers()` bug with virtual buffers and a
`RegularArray(size=0)` inside a `BitMaskedArray`; it raised an `AssertionError`
on Awkward Array v2.9.0, is marked `xfail` (expected failure), and passes with
v2.9.1 (likely fixed by
[#3889](https://github.com/scikit-hep/awkward/pull/3889)). A NumPy property test
in
[`test_numpy_arrays.py`](https://github.com/scikit-hep/hypothesis-awkward/blob/main/tests/strategies/numpy/test_numpy_arrays.py)
accounts for [#3690](https://github.com/scikit-hep/awkward/issues/3690) (open):
`ak.to_numpy()` does not support structured arrays whose fields are not
one-dimensional.

### Hypothesis

- [#4708](https://github.com/HypothesisWorks/hypothesis/issues/4708) (fixed) —
  an `AssertionError` in `Shrinker.explain()` for unstable span labels. Fixed in
  [#4717](https://github.com/HypothesisWorks/hypothesis/pull/4717) and released
  in Hypothesis 6.152.4.

## Outlook

Automatically generated test inputs raise confidence that a change is correct
across a broad range of valid arrays, not only the cases a developer wrote by
hand.

<!-- Living log: review the "Bugs found" section each release. -->
