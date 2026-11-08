# Directory Structure: `constructors` Module

- **Date:** 2026-11-08
- **Status:** Implemented
- **Related:**
  [public-namespaces-API](../api/2026-11-08-01-Public-namespaces-API.md)

## Overview

When the content strategies moved to the `st_ak.constructors` namespace, their
modules moved from `strategies/contents/` (renamed `content/` on the branch) to
`strategies/constructors/`, so that the files match the import path, as they do
for `numpy/` and `misc/`.

## Layout

Flat: `constructors/` holds `array_.py`, the 19 content modules, and
`_types.py`. Tests mirror it in `tests/strategies/constructors/`.

```text
strategies/
├── __init__.py          # exports only the namespaces constructors, numpy, misc
├── constructors/
│   ├── __init__.py      # __all__: arrays() first, then the 18 content strategies
│   ├── array_.py
│   ├── content.py       # contents(), content_lists()
│   ├── leaf.py, option.py, masked.py
│   ├── <class>_array.py # one module per Content subclass
│   ├── string.py, bytestring.py, categorical.py
│   └── _types.py        # StContent, StOption protocols
├── numpy/
├── misc/
├── builtin/             # private
├── form/                # private
└── type_/               # private
```

A nested `constructors/content/` subpackage was rejected: importing it binds
`st_ak.constructors.content`, a public-looking name that is easy to confuse with
`contents()`, and would be the one subpackage in a namespace of strategies.

## Imports

Modules in `constructors/` import from the same package with relative imports,
not through `st_ak` (the rule is in `.claude/CLAUDE.md`, Code Style).

`content.py` imports the wrapper modules at the top, and the wrappers call
`contents()` for a default child. A top-level `from .content import contents` in
a wrapper would be circular, so the wrappers import it inside the function that
calls it, with the comment "Imported here because content.py imports this
module." `record_array.py` and `union_array.py` already imported
`content_lists()` this way.

- Top-level relative imports: `array_.py`, `leaf.py`, `categorical.py`, and
  `content.py` itself.
- Function-local imports: `bit_masked_array`, `byte_masked_array`,
  `indexed_array`, `indexed_option_array`, `list_array`, `list_offset_array`,
  `regular_array`, `unmasked_array`, `option`, `record_array`, and
  `union_array`.
- In `union_array_contents()`, whose parameter is named `contents`, the function
  is imported as `st_contents`.

The test of `arrays()` patches `contents` in `array_.py` directly.
