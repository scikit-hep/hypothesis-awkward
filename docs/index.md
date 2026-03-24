# API Reference for hypothesis-awkward

[hypothesis-awkward] is a collection of Hypothesis strategies for [Awkward
Array][awkward-array]. This doc is its API reference.

[hypothesis-awkward]: https://github.com/scikit-hep/hypothesis-awkward
[awkward-array]: https://awkward-array.org/

**Modules:**

| Name           | Description                                                             |
| -------------- | ----------------------------------------------------------------------- |
| [constructors] | The main strategy.                                                      |
| [contents]     | Strategies for contents (layouts).                                      |
| [numpy]        | Strategies related to NumPy in the context of Awkward Array.            |
| [builtins]     | Strategies for built-in Python objects in the context of Awkward Array. |

[constructors]: strategies/constructors.md
[contents]: strategies/contents.md
[numpy]: strategies/numpy.md
[builtins]: strategies/builtins.md
