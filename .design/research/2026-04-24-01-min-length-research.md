# `min_length` Research

Date: 2026-04-24

## Overview

This document explores adding a lower-bound counterpart to the existing
`max_length` parameter on `contents()` and the surrounding content strategies.
With both bounds available, callers can constrain the immediate `len(result)` to
a specific range — and, by setting both bounds equal, pin the result to an exact
length.

## Motivation

### Exact-length use case

`max_length` already caps `len(result)`. There is no symmetric way to require
that `len(result)` reaches a floor. The most concrete request that surfaces this
gap is "give me content of length exactly N." Without `min_length`, the caller
has to filter the strategy and pay shrinker pressure to converge on a particular
length.

### Internal callers already want it

The `_from_contents` bridges already draw an exact target length and forward it
to the standalone `*_contents` strategies as `max_length`, but cannot pin the
floor. Two TODOs in the current code anticipate this addition:

- `src/hypothesis_awkward/strategies/contents/list_offset_array.py:209` —
  `# TODO: Add min_length=length when min_length is implemented.`
- `src/hypothesis_awkward/strategies/contents/list_array.py:313` — same TODO.

In both cases, the bridge wants `min_length == max_length == length`. The result
today only respects the upper bound, so the realised length can be shorter than
what the bridge intended.

## Semantic frame

`min_length` is the lower-bound counterpart to `max_length`. Each strategy that
exposes one should expose the other, with the same semantics:

- It constrains the immediate `len()` of the strategy's output, not the depth of
  any nested content and not the total scalar count.
- At the entry points (`contents()` and `arrays()`), it applies at the outermost
  level only; nested nodes inside the result are unconstrained by it. This
  matches how `max_length` behaves today.
- It is a per-strategy contract: each strategy guarantees
  `min_length <= len(result)` and is free to choose how to deliver that (drawing
  children long enough, looping until the floor is met, drawing the length first
  and constraining children, etc.). Callers do not need to know which mechanism
  is used.

## Survey of strategies

Which strategies have an immediate-length parameter today, and what name it goes
by:

| Strategy                          | Existing length-control parameter | Lower bound today           |
| --------------------------------- | --------------------------------- | --------------------------- |
| `contents()`                      | `max_length`                      | none                        |
| `arrays()`                        | `max_length`                      | none                        |
| `leaf_contents()`                 | `min_size` / `max_size`           | already symmetric           |
| `numpy_array_contents()`          | `min_size` / `max_size`           | already symmetric           |
| `string_contents()`               | `min_size` / `max_size`           | already symmetric           |
| `bytestring_contents()`           | `min_size` / `max_size`           | already symmetric           |
| `empty_array_contents()`          | none (always 0)                   | n/a                         |
| `regular_array_contents()`        | `max_length` (groups)             | none                        |
| `list_offset_array_contents()`    | `max_length`                      | none                        |
| `list_array_contents()`           | `max_length`                      | none                        |
| `record_array_contents()`         | `max_length`                      | none                        |
| `union_array_contents()`          | `max_length`                      | none                        |
| `indexed_option_array_contents()` | `max_size` (= index length)       | none                        |
| `byte_masked_array_contents()`    | none — length tracks `content`    | n/a (forwarded via wrapper) |
| `bit_masked_array_contents()`     | none — length tracks `content`    | n/a (forwarded via wrapper) |
| `unmasked_array_contents()`       | none — length tracks `content`    | n/a (forwarded via wrapper) |
| `*_from_contents` bridges         | `max_length`                      | none                        |

Two patterns of name follow from existing code:

- Strategies that already expose `max_length` get a `min_length` counterpart.
- Strategies that use `max_size` to control the immediate length (currently only
  `indexed_option_array_contents()`, where `max_size` sizes the index buffer and
  therefore equals `len(result)`) get a `min_size` counterpart — matching how
  `numpy_array_contents()` and the string/bytestring strategies already expose
  `min_size` alongside `max_size`.

## Where the addition lands

### Entry points

`contents()` and `arrays()` need a new top-level parameter. The plumbing in
`contents()` is symmetric to the existing `leaf_max_size`: a `leaf_min_size`
local is computed and forwarded to `leaf_contents(...)` as `min_size`. The leaf
strategies do not change.

### Wrapper standalone strategies

Each of `regular_array_contents`, `list_offset_array_contents`,
`list_array_contents`, `record_array_contents`, and `union_array_contents` needs
to accept a lower bound and guarantee `len(result) >= min_length` for the values
it returns.

### `_from_contents` bridges

Each bridge needs to forward the lower bound through to its standalone strategy.
The two TODOs in `list_offset_array.py` and `list_array.py` are satisfied by
this work — the bridge can pass `min_length == max_length == length` to pin the
exact length it drew.

### Option types

`indexed_option_array_contents` exposes `max_size` to control the index length
and gets a matching `min_size`. The masked/unmasked option strategies do not
control their own length, so the bound is delivered by the wrapper that drew the
inner content; the `_from_contents` bridges for these types forward `min_length`
to the recursive `content(...)` call so the inner content meets the floor.

### `content_lists()`

Independent. `content_lists` already exposes `min_len`/`max_len`, which control
the number of children in the list rather than the immediate length of any
single content. No change needed.

## EmptyArray at the outermost level

`leaf_contents()` already excludes `empty_array_contents` when `min_size > 0`:

```python
if allow_empty and min_size <= 0 <= max_size:
    options.append(st_ak.contents.empty_array_contents())
```

When `contents()` plumbs `min_length` through to `leaf_min_size`, this guard
fires automatically: a non-zero floor at the outermost level rules out
`EmptyArray` as the outermost choice. Deeper-level `EmptyArray` leaves are
unaffected because the floor is only applied at the entry point.

## Infeasibility handling

The decision is: no eager validation. Strategies raise only when a draw step
becomes structurally impossible — typically when an empty `st.one_of(...)` or
`st.sampled_from(...)` is reached. This matches the existing approach for
`max_length` and `max_size` interactions: rather than enumerate all incompatible
combinations up front, the strategy proceeds and lets the first unworkable draw
site raise.

Examples of combinations that may surface as runtime errors rather than eager
`ValueError`s:

- `min_length > max_length`.
- `min_length > 0` with all length-bearing leaf options disabled.
- `min_length > 0` for a wrapper whose budget (`max_size`, `max_leaf_size`) is
  too tight to reach the floor.

A future iteration may add explicit validation, but the first cut keeps the
implementation small.

## Per-strategy mechanics

Sketches only — concrete signatures and defaults are out of scope for this note.
The point is to confirm each strategy has a workable path to enforce the floor;
the API note will turn these into the actual contract.

### `regular_array_contents`

`len(result) == zeros_length` when `size == 0`, otherwise
`len(content) // size`. The floor is delivered by:

- when `size > 0`: bounding the size selector so that `size` does not exceed
  `len(content) // min_length` (already symmetric to the existing `max_length`
  ceiling on `min_group_size`).
- when `size == 0`: floor `zeros_length` at `min_length`.

### `list_offset_array_contents` and `list_array_contents`

`len(result)` is the number of sublists. The existing offsets/starts–stops
helpers already accept `max_length`; they need a `min_length` counterpart so the
drawn list of split points has at least `min_length + 1` (offsets) or
`min_length` (starts/stops) entries.

### `record_array_contents`

`len(result)` is the shared field length. Today it is set to
`min(min(len(c) for c in contents), max_length)`. For `min_length`, each child
must have `len(c) >= min_length` for the result to satisfy the floor; the
simplest first cut requires the children passed in (or drawn via
`content_lists`) to already meet the floor and falls back on `assume()`
otherwise.

### `union_array_contents`

`len(result)` is the total of child lengths (capped by `max_length` via
truncation). For `min_length`, the children's combined length must reach the
floor; the simplest first cut forwards `min_length` to the child draw so the
total naturally meets it, with `assume()` as a backstop.

### `indexed_option_array_contents`

`len(result)` is the index length, which the existing code derives from
`max_size`. Adding a `min_size` counterpart turns the
`st.lists(..., max_size=upper)` draw into
`st.lists(..., min_size=lower, max_size=upper)`.

### `byte_masked_array_contents`, `bit_masked_array_contents`, `unmasked_array_contents`

`len(result) == len(content)`. The standalone variants do not need a new
parameter; the floor is satisfied by the wrapper drawing inner content of
adequate length.

## Implementation order

Bottom-up, mirroring the order used for `max_length`:

1. `regular_array_contents` — `size`/`zeros_length` floor.
2. `list_offset_array_contents` — offsets minimum length.
3. `list_array_contents` — starts/stops minimum length.
4. `record_array_contents` — shared field-length floor.
5. `union_array_contents` — total-length floor via children.
6. `indexed_option_array_contents` — `min_size` on the index.
7. The four `*_from_contents` bridges — forward the floor; pin the exact length
   in `list_offset_array_from_contents` and `list_array_from_contents`, retiring
   the two TODOs.
8. `contents()` — new parameter; compute `leaf_min_size` symmetrically to
   `leaf_max_size`; forward to wrappers and leaf draws.
9. `arrays()` — new parameter; forward to `contents()`.

## Testing

The existing test scaffolding has the right shape. Tests for each strategy
should:

- Add the lower bound to the strategy's `*Kwargs` `TypedDict`.
- Draw `(min_length, max_length)` pairs as a single coordinated kwarg using the
  existing `st_ak.ranges()` helper, which already produces ordered
  `(start, end)` pairs with optional `None` for either side.
- Assert `min_length <= len(result) <= max_length` (treating `None` on either
  side as unbounded, e.g. via `safe_compare`).

## Future work

Out of scope for the first cut, anticipated for later:

1. **Draw length first, then content.** For `regular_array_contents` and the
   list strategies, drawing the length up front and asking the child for content
   of a matching length would give more uniform coverage and remove any reliance
   on `assume()`. This is the same direction noted in the `max_length` research
   for improving `regular_array_contents` size coverage; `min_length` makes it
   implementable for the list strategies too.
2. **Eager validation.** Once all strategies are wired, the entry points can
   detect impossible combinations (`min_length > max_length`, etc.) and raise
   `ValueError` early.

## Open questions

1. Should `record_array_contents` enforce the floor by truncating children that
   overshoot, similar to how `union_array_contents` truncates tags/index when
   the total exceeds `max_length`? Likely no — record fields share length and
   truncating on one field implies truncating on all, which the constructor
   already handles via the explicit `length` argument. But worth confirming when
   writing the API note.
2. Should `min_length` default value mirror `max_length` (`None`, meaning
   unbounded) or `min_size` (`0`, explicit floor)? Either is consistent with
   some part of the existing surface. The API note will pick one.
