# Testing Awkward Array

This page is for Awkward Array contributors and for readers assessing the
project. If you want to use the strategies in your own tests, start with
[Getting Started](getting-started.md).

## How hypothesis-awkward tests Awkward Array

<!-- diataxis: explanation -->

[Awkward Array](https://awkward-array.org/) builds arrays by composing a small
set of layout nodes (the classes in `ak.contents`): numeric leaves, the
unknown-type `EmptyArray`, variable-length and regular lists, indexed
(indirection) nodes, missing values in four node forms, records, and unions. The
nodes compose recursively, subject to a few validity rules, so the valid arrays
span a large combinatorial space of layouts. Test data written by hand covers
only a small part of that space, and failures often occur on input shapes that
are absent from hand-written test cases. This makes Awkward Array hard to test
thoroughly.

[Property-based testing](../index.md) inverts the approach: instead of choosing
inputs, a test states a property that must hold for every valid array, and the
inputs are generated. This package supplies the generation. Its main strategy is
[`st_ak.constructors.arrays()`](../reference/strategies/constructors.md) (with
`import hypothesis_awkward.strategies as st_ak`), which generates nearly fully
general Awkward Arrays, including virtual arrays — arrays whose buffers are not
yet materialized. A generated array has either all buffers virtual or none; it
lives on the CPU backend and carries no node parameters other than the
`__array__` markers that make strings and bytestrings, so categorical data (the
main remaining gap; see the [Roadmap](roadmap.md)), named record types, and
custom behaviors are not yet generated. See
[Getting Started](getting-started.md) for what it produces.

Four kinds of properties are checked:

**Round-trip.** Converting an array to another representation and back
reconstructs an equal array. The first property test covered `ak.to_buffers()`
and `ak.from_buffers()` (the snippet is adapted from
[`test_to_from_buffers.py`](https://github.com/scikit-hep/awkward/blob/main/tests/properties/operations/test_to_from_buffers.py)):

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

The same property now also covers the conversions to and from NumPy, Arrow,
Parquet, Feather, and JSON. Only the buffers round trip draws from the
unrestricted strategy; the other five narrow it — with the strategy's `allow_*`
flags, restricted dtype strategies, and format-compatibility filters — to what
the target format can represent, so each conversion is checked over the subset
it is expected to support. Where a bug is still open, a restriction can also
stand in for a known-issue predicate, with the issue cited in the test module.
The NumPy round trip, for example, draws no variable-length lists, records,
unions, strings, or bytestrings — regular (fixed-size) lists stay, since they
become NumPy dimensions — and a separate test carries option types through NumPy
masked arrays. Where a conversion cannot preserve everything, the property
becomes a fixed point instead: converting the reconstructed array a second time
reproduces the same result.

**Equality.** `ak.array_equal()` is reflexive (an array equals itself) and
symmetric (the argument order does not change the result), with tests adapted
from
[`test_array_equal.py`](https://github.com/scikit-hep/awkward/blob/main/tests/properties/operations/test_array_equal.py)
(the imports are the same as above):

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
(not-a-time) values in datetimes and timedeltas — which the properties need
because the generated arrays can contain both.

**No-raise.** When the operation is expected to accept a generated array — the
test judges this from the array's form and the call's arguments — it must
complete without raising an error. `ak.flatten()`, `ak.all()`, and `ak.ravel()`
are tested this way.

**Kernel.** Awkward Array specifies each low-level kernel with a pure-Python
reference definition in
[`kernel-specification.yml`](https://github.com/scikit-hep/awkward/blob/main/kernel-specification.yml).
One kernel is covered so far, `awkward_BitMaskedArray_to_ByteMaskedArray`: its
test runs the compiled CPU kernel — and the CUDA kernel when a GPU is present —
and compares the results with the reference definition. A kernel works below the
layout level, so this test draws the kernel's own arguments — a bit mask and its
flags — with plain Hypothesis strategies rather than with this package's.

The round-trip, equality, and no-raise properties need no reference
implementation to compare against; the kernel property is the exception, and its
oracle is the specification's reference definition. When a property fails,
Hypothesis shrinks the input toward a minimal failing array — a best-effort
search that
[Generating and Shrinking Samples](generating-and-shrinking-samples.md) explains
— which is why the bug reports below are small enough to reproduce by hand.

## How the tests run in Awkward Array's CI

<!-- diataxis: explanation -->

The strategies are integrated into Awkward Array's continuous integration (CI).
The first property-based tests were added in
[#3887](https://github.com/scikit-hep/awkward/pull/3887) and live in
[`tests/properties`](https://github.com/scikit-hep/awkward/tree/main/tests/properties).
On every change they run under the `default` profile in
[`tests/properties/conftest.py`](https://github.com/scikit-hep/awkward/blob/main/tests/properties/conftest.py),
which inherits the built-in `ci` profile that Hypothesis loads automatically in
CI: generation is derandomized — seeded from a hash of each test function — so a
run repeats the same examples until the test function, the strategies, or the
Hypothesis or Python version changes. A nightly run
([`property-tests.yml`](https://github.com/scikit-hep/awkward/blob/main/.github/workflows/property-tests.yml))
selects the `nightly` profile from the same file, which raises
[`max_examples`](https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.settings.max_examples)
— the number of valid examples each test tries — and turns randomization back
on, so it reaches examples the per-change runs never generate. The nightly run
also reports Hypothesis's run statistics — how many examples each test generated
and how many were invalid — which keeps the discard rate visible as the
predicates below accumulate. A second nightly workflow
([`property-tests-gpu.yml`](https://github.com/scikit-hep/awkward/blob/main/.github/workflows/property-tests-gpu.yml))
runs the kernel test on a GPU under the same `nightly` profile.

Awkward Array pins hypothesis-awkward to an exact version, and Dependabot
proposes an update for each release: a new strategy release changes which inputs
are generated, and the pin keeps that change from surfacing as an unrelated
failure in someone else's pull request.
[Upstream test pull requests](#upstream-test-pull-requests) lists the pull
requests that built this arrangement.

A bug that stays open does not block the suite. Where the tests would otherwise
reach it, its trigger is encoded as a predicate on the array, named after its
issue number (`has_issue_4262`, for example) — shared predicates live in
[`known_issues.py`](https://github.com/scikit-hep/awkward/blob/main/tests/properties/operations/known_issues.py),
and a module that is alone affected keeps private ones next to its tests — and
the affected draws are discarded, with Hypothesis's
[`assume()`](https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.assume)
inside the test or by filtering the strategy. A predicate is deleted in the pull
request that fixes its issue, so the discarded inputs are tested again. The
predicates are reviewed and merged in Awkward Array itself
([#4275](https://github.com/scikit-hep/awkward/pull/4275),
[#4281](https://github.com/scikit-hep/awkward/pull/4281)), so the open bugs
these tests reach are acknowledged in the library's own test suite. The suite
keeps passing while every open bug remains visible in the issue tracker and in
the [log below](#bugs-found).

## What the bugs show

<!-- diataxis: explanation -->

The bugs found so far (listed in [Bugs found](#bugs-found) below) are not spread
evenly over Awkward Array; they cluster where generated inputs have an advantage
over hand-written ones.

**Conversion boundaries.** The largest group of the bugs is in the conversions —
`ak.to_numpy()`, `ak.to_arrow()`, `ak.from_json()`, `ak.from_parquet()`, and
their inverses. A conversion must map every valid layout onto another system's
representation, so each rarely-exercised layout feature is a chance for a
mismatch. Round-trip properties probe exactly that mapping, in both directions.

**Rare but valid shapes.** The triggers are almost never the common cases; they
are empty unions, zero-length lists, missing values in unusual places, offsets
that do not start at zero, and uncommon dtypes such as `datetime64[D]` or
`float16`. A developer writing test data by hand rarely constructs these; a
generator that samples the whole layout space constructs them routinely.

**Beyond the library under test.** The tests also exercise the software around
Awkward Array. The Feather round trip — `ak.to_feather()` writes Arrow's
inter-process communication (IPC) file format — produced invalid files whose
cause is in PyArrow's IPC writer; it was reported upstream after reducing to a
pure-PyArrow case. And one bug was in Hypothesis itself: after shrinking a
failing array, its explain phase ended in an internal assertion error.

## Upstream test pull requests

<!-- diataxis: reference -->

The pull requests in Awkward Array that added or changed the property-based
tests, grouped by purpose and ascending by number within each group; the
one-line descriptions follow the pull-request titles. All are merged.

**New property tests:**

- [#3887](https://github.com/scikit-hep/awkward/pull/3887) — the first
  property-based test: the `to_buffers`/`from_buffers` round trip.
- [#3891](https://github.com/scikit-hep/awkward/pull/3891) — the equality
  properties of `ak.array_equal`.
- [#4142](https://github.com/scikit-hep/awkward/pull/4142) — a property-based
  kernel test with CPU and GPU backends.
- [#4215](https://github.com/scikit-hep/awkward/pull/4215) — the
  `to_numpy`/`from_numpy` round-trip test.
- [#4218](https://github.com/scikit-hep/awkward/pull/4218) — the
  `to_parquet`/`from_parquet` round-trip test.
- [#4225](https://github.com/scikit-hep/awkward/pull/4225) — a stability
  round-trip test for `to_numpy`/`from_numpy`.
- [#4231](https://github.com/scikit-hep/awkward/pull/4231) — the
  `to_arrow`/`from_arrow` round-trip tests.
- [#4239](https://github.com/scikit-hep/awkward/pull/4239) — the
  `to_feather`/`from_feather` round-trip tests.
- [#4245](https://github.com/scikit-hep/awkward/pull/4245) — the
  `to_json`/`from_json` round-trip tests.
- [#4265](https://github.com/scikit-hep/awkward/pull/4265) — no-raise property
  tests for `ak.flatten` and `ak.all`.
- [#4284](https://github.com/scikit-hep/awkward/pull/4284) — the no-raise
  property-test module for `ak.ravel`.

**Test infrastructure:**

- [#3942](https://github.com/scikit-hep/awkward/pull/3942) — pinned
  hypothesis-awkward to an exact version.
- [#3944](https://github.com/scikit-hep/awkward/pull/3944) — added a Dependabot
  configuration for hypothesis-awkward.
- [#4128](https://github.com/scikit-hep/awkward/pull/4128) — added a nightly run
  with a large `max_examples`.
- [#4146](https://github.com/scikit-hep/awkward/pull/4146) — reported Hypothesis
  statistics in the nightly runs.
- [#4147](https://github.com/scikit-hep/awkward/pull/4147) — ran the GPU kernel
  property tests nightly under the nightly profile.

**Maintenance and documentation:**

- [#4273](https://github.com/scikit-hep/awkward/pull/4273) — documented the
  `tests/properties` naming convention in `CONTRIBUTING.md`.
- [#4275](https://github.com/scikit-hep/awkward/pull/4275) — added known-issue
  predicates for [#4255](https://github.com/scikit-hep/awkward/issues/4255) and
  [#4274](https://github.com/scikit-hep/awkward/issues/4274).
- [#4281](https://github.com/scikit-hep/awkward/pull/4281) — added known-issue
  predicates for [#4278](https://github.com/scikit-hep/awkward/issues/4278) and
  [#4280](https://github.com/scikit-hep/awkward/issues/4280).

## Bugs found

<!-- diataxis: reference -->

<!-- Declared reference: its completeness obligation is to the bugs found so
     far, not to every bug that exists. -->

Tests written with these strategies — in Awkward Array's suite and in
hypothesis-awkward's own — have found bugs in Awkward Array, in PyArrow, and in
Hypothesis. This log is reviewed at each
[release](https://github.com/scikit-hep/hypothesis-awkward/releases); status is
as of the last review. The Awkward Array list is grouped by the operation
involved, ascending by number within each group; "fixed" points to the merged
fixing pull request, and an entry says so where the fix is not yet in a released
Awkward version.

### Awkward Array

**Equality:**

- [#3888](https://github.com/scikit-hep/awkward/issues/3888) (fixed in
  [#3909](https://github.com/scikit-hep/awkward/pull/3909)) — `ak.array_equal()`
  raised an error on virtual arrays and returned the wrong result for empty
  unions.
- [#3921](https://github.com/scikit-hep/awkward/pull/3921) (reported and fixed
  in this pull request) — `ak.array_equal()` returned the wrong result for
  datetimes and timedeltas containing `NaT`.
- [#3962](https://github.com/scikit-hep/awkward/pull/3962) (reported and fixed
  in this pull request) — `ak.almost_equal()`, which backs `ak.array_equal()`,
  compared record-array fields incorrectly.

**NumPy conversion:**

- [#4217](https://github.com/scikit-hep/awkward/issues/4217) (open) —
  `ak.to_numpy()` fails for option-type `timedelta64` arrays that contain
  missing values.
- [#4226](https://github.com/scikit-hep/awkward/issues/4226) (open) —
  `ak.from_numpy()` fails for multidimensional masked string arrays with no
  masked elements.
- [#4227](https://github.com/scikit-hep/awkward/issues/4227) (open) —
  `ak.from_numpy()` raises a `ValueError` for masked arrays of three or more
  dimensions with a zero-length dimension.

**Arrow conversion:**

- [#4219](https://github.com/scikit-hep/awkward/issues/4219) (open) —
  `ak.to_arrow()` silently corrupts `datetime64[D]` values.
- [#4221](https://github.com/scikit-hep/awkward/issues/4221) (open) —
  `ak.to_arrow()` raises an `IndexError` for a non-empty option array with
  zero-length list content.
- [#4222](https://github.com/scikit-hep/awkward/issues/4222) (open) —
  `ak.to_arrow()` silently shifts list values when a nullable list's offsets do
  not start at zero.
- [#4228](https://github.com/scikit-hep/awkward/issues/4228) (open) —
  `ak.to_arrow()` raises an `IndexError` for valid unions with option-type
  children.
- [#4229](https://github.com/scikit-hep/awkward/issues/4229) (open) —
  `ak.from_arrow()` loses the length of a size-0 fixed-size list.
- [#4230](https://github.com/scikit-hep/awkward/issues/4230) (open) —
  `ak.from_arrow()` raises an `AttributeError` for a table with a null-type
  column.
- [#4255](https://github.com/scikit-hep/awkward/issues/4255) (open) —
  `ak.from_arrow()` fails on an option record with an
  `IndexedArray`-of-`EmptyArray` field.
- [#4274](https://github.com/scikit-hep/awkward/issues/4274) (open) —
  `ak.to_arrow_table()` fails for an `UnmaskedArray` of records with a
  uint32-indexed field.

**Parquet conversion:**

- [#4220](https://github.com/scikit-hep/awkward/issues/4220) (open) —
  `ak.from_parquet()` fails on `NaT` or out-of-range `datetime64[ms]`/`[us]`
  row-group statistics.

**Feather conversion:**

- [#4238](https://github.com/scikit-hep/awkward/issues/4238) (open) —
  `ak.to_feather()` writes union-type arrays as invalid files, and
  `ak.from_feather()` crashes reading them; the cause is
  [#50623](https://github.com/apache/arrow/issues/50623) in PyArrow's IPC
  writer.

**JSON conversion:**

- [#4241](https://github.com/scikit-hep/awkward/issues/4241) (open) —
  `ak.from_json()` parses floats at reduced precision, so values drift on each
  reparse.
- [#4242](https://github.com/scikit-hep/awkward/issues/4242) (open) —
  `ak.from_json()` raises a `TypeError` when JSON mixes null with heterogeneous
  values.
- [#4243](https://github.com/scikit-hep/awkward/issues/4243) (open) —
  `ak.from_json()` silently wraps JSON integers in `[2**63, 2**64)` to negative
  int64 values.
- [#4244](https://github.com/scikit-hep/awkward/issues/4244) (open) —
  `ak.to_json()` writes lone-surrogate escapes that `ak.from_json()` rejects.

**Flattening and broadcasting:**

- [#4214](https://github.com/scikit-hep/awkward/issues/4214) (fixed in
  [#4246](https://github.com/scikit-hep/awkward/pull/4246)) —
  `ak.flatten(axis=None)` dropped data or raised an `IndexError` on nested
  unions, a regression in Awkward Array 2.11.0; the fix is not yet in a released
  version as of the last review.
- [#4247](https://github.com/scikit-hep/awkward/issues/4247) (open) —
  `ak.broadcast_arrays()` silently skips broadcasting on some union-containing
  arrays.
- [#4260](https://github.com/scikit-hep/awkward/issues/4260) (open) —
  `ak.flatten()` with a negative `axis` raises an `AssertionError` when record
  field depths differ.
- [#4261](https://github.com/scikit-hep/awkward/issues/4261) (open) —
  `ak.ravel()` raises internal errors when the collected leaves cannot merge.
- [#4262](https://github.com/scikit-hep/awkward/issues/4262) (open) —
  `ak.broadcast_arrays()` fails for a flat array and a regular list of
  variable-length lists.
- [#4263](https://github.com/scikit-hep/awkward/issues/4263) (open) —
  `ak.ravel()` raises an `AssertionError` with no message on a union that no
  values reach.
- [#4278](https://github.com/scikit-hep/awkward/issues/4278) (open) —
  `ak.ravel()` raises an `OverflowError` when temporal leaves need an
  overflowing unit conversion.
- [#4280](https://github.com/scikit-hep/awkward/issues/4280) (open) —
  `ak.ravel()` raises an `OverflowError` when a temporal value does not fit the
  merged unit.
- [#4282](https://github.com/scikit-hep/awkward/issues/4282) (open) —
  `ak.ravel()` fails or silently reorders values when a union branch is an
  all-regular list.
- [#4283](https://github.com/scikit-hep/awkward/issues/4283) (open) —
  `ak.ravel()` raises an `AssertionError` on a record under an option beside a
  non-record leaf.

**Reducers and sorting:**

- [#4259](https://github.com/scikit-hep/awkward/issues/4259) (open) — nearly all
  reducers and sorting raise a `KeyError` on `float16`, `float128`, and
  `complex256` leaves.
- [#4264](https://github.com/scikit-hep/awkward/issues/4264) (open) — reducers
  fail on an option type inside a `RegularArray`'s untrimmed content.

**Layout internals and virtual arrays:**

- [#4126](https://github.com/scikit-hep/awkward/issues/4126) (fixed in
  [#4127](https://github.com/scikit-hep/awkward/pull/4127)) —
  `ak.contents.IndexedOptionArray.to_ByteMaskedArray` raised an unhelpful
  `TypeError` when a non-empty array's content was an `EmptyArray`; the fix
  rejects the conversion with a clear error.
- [#4288](https://github.com/scikit-hep/awkward/issues/4288) (open) — virtual
  arrays lose the `length` of a nested zero-field `RecordArray`.

A NumPy property test in
[`test_numpy_arrays.py`](https://github.com/scikit-hep/hypothesis-awkward/blob/main/tests/strategies/numpy/test_numpy_arrays.py)
accounts for [#3690](https://github.com/scikit-hep/awkward/issues/3690) (open):
`ak.to_numpy()` does not support structured arrays whose fields are not
one-dimensional. This is a pre-existing limitation the tests work around, not a
bug they found.

### PyArrow

- [#50623](https://github.com/apache/arrow/issues/50623) (open) — Arrow's IPC
  serialization drops a union's buffers when the union is wrapped in an
  extension type, producing invalid data that segfaults on read. Reported with a
  pure-PyArrow reproduction after the Feather round trip surfaced it as
  [#4238](https://github.com/scikit-hep/awkward/issues/4238).

### Hypothesis

- [#4708](https://github.com/HypothesisWorks/hypothesis/issues/4708) (fixed) —
  an `AssertionError` in `Shrinker.explain()` for unstable span labels. Fixed in
  [#4717](https://github.com/HypothesisWorks/hypothesis/pull/4717) and released
  in Hypothesis 6.152.4.

## Outlook

<!-- diataxis: explanation -->

The goal is to cover all testable properties of Awkward Array — more operations,
slicing, reducers, and the remaining kernels — following the pattern the suite
has established: add a property test, file the bugs it finds, and discard their
triggers until the fixes are merged. Automatically generated test inputs raise
confidence that a change is correct across a broad range of valid arrays, not
only the cases a developer wrote by hand.
