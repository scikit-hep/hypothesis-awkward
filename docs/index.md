# hypothesis-awkward

_Hypothesis strategies for Awkward Arrays._

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

Next steps:

- [Guide](guide/index.md)
- [API Reference](reference/index.md)
