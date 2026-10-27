# Hypothesis-awkward

<!-- diataxis: explanation -->

_Hypothesis strategies for Awkward Array._

Hypothesis-awkward is a collection of [Hypothesis] strategies that generate
[Awkward Array][awkward-array] data for tests, both of Awkward Array itself and
of code that uses it. Tests written with these strategies [run in Awkward
Array's continuous integration (CI)][ci]. Together with the package's own tests,
they have found bugs in Awkward Array, PyArrow, and Hypothesis; the [bugs-found
log][bugs-found] lists them. To cite Hypothesis-awkward, see
[Citation][citation].

[Awkward Array][awkward-array] represents nested lists, lists of variable
length, records, missing values, and values of mixed types, nested to any depth.
Two arrays can hold the same values and have the same type, such as
`var * ?float64`, yet be stored differently. Each array is stored as a _layout_,
a tree of nodes that hold the data: the variable-length lists in this example
can be stored in either of two node classes, the missing values in any of four,
a list can skip data at the start of its content, the layout can contain data
that no element reaches, and the buffers can be virtual, loaded only when an
operation needs them. Ordinary operations, such as slicing and masking, produce
many of these layouts, and the choices multiply with each level of nesting. Code
that accepts Awkward Arrays is expected to work on all of them.

Many tests check predefined samples: inputs chosen by a developer, or generated
by AI, and written into the test. For Awkward Array data, these samples cover a
small part of the possible arrays, and the tests miss the edge cases outside it.
Predefined samples also let an implementation be fitted to them: an AI coding
agent, for example, can produce code that returns the expected outputs for those
inputs without implementing the behavior for others. Either way, the tests pass
while the code is wrong. That matters because software is developed iteratively,
in cycles of refactoring and feature changes, each made safe by tests that pass
only when the code behaves as expected — and AI coding agents now make the
cycles shorter.

[Hypothesis] addresses both problems with property-based testing. A test states
properties that must hold for every valid input — for example, that writing an
array to a file and reading it back returns an equal array. A Hypothesis
[_strategy_][hyp-st] describes the inputs, and Hypothesis generates many
different samples from it at run time, so no sample is written into the test for
an implementation to be fitted to. When a test fails, Hypothesis _shrinks_ the
failing sample to the simplest one it can find that still fails in the same way.

The main strategy of Hypothesis-awkward, [`st_ak.constructors.arrays()`][arrays]
(where `st_ak` is `hypothesis_awkward.strategies`), can produce almost any
layout, nested to any depth. It is layout-based: its options, listed in the [API
reference][api-ref], allow or exclude kinds of layout nodes and limit sizes, so
a test can restrict it to the arrays the code under test accepts. [Generating
and Shrinking Samples][gen-shrink] explains how it builds and shrinks samples.
Strategies that generate arrays of a given type are planned; the
[Roadmap][roadmap] describes them and names what the current strategy cannot
generate yet, such as categorical data.

[awkward-array]: https://awkward-array.org/
[hypothesis]: https://hypothesis.readthedocs.io/
[hyp-st]: https://hypothesis.readthedocs.io/en/latest/reference/strategies.html
[arrays]: reference/strategies/constructors.md
[api-ref]: reference/index.md
[gen-shrink]: guide/generating-and-shrinking-samples.md
[roadmap]: guide/roadmap.md
[bugs-found]: guide/testing-awkward-array.md#bugs-found
[citation]: guide/citation.md
[ci]: guide/testing-awkward-array.md#how-the-tests-run-in-awkward-arrays-ci

Next steps:

- [Guide](guide/index.md)
- [API Reference](reference/index.md)
