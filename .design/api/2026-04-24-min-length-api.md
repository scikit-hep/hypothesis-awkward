# API Design: `min_length` Parameter

- **Date:** 2026-04-24
- **Status:** Implemented — `min_length` is exposed on `arrays()`, `contents()`,
  and the wrapper/option strategies; `min_size` is on
  `indexed_option_array_contents()`; the `*_from_contents` bridges forward the
  floor and the two list-strategy TODOs are retired.
- **Author:** Claude (with developer collaboration)

## Overview

This document specifies the API for a lower-bound counterpart to the existing
`max_length` parameter on `contents/` strategies. With both bounds available,
callers can constrain the immediate `len(result)` to a range — and, by setting
both bounds equal, pin the result to an exact length.

See [min-length-research](../research/2026-04-24-min-length-research.md) for
motivation and the per-strategy survey. See
[max-length-api](./2026-02-23-max-length-api.md) for the symmetric upper-bound
contract that this design mirrors.

## Per-strategy contract

Every strategy that gains the new bound makes the same guarantee:
`min_length <= len(result)`. How the strategy delivers that guarantee is an
implementation concern, not part of the contract. Callers do not need to know
whether the strategy draws a longer child, loops until the floor is met, or
relies on `assume()` to filter undershooting draws.

### Wrapper strategies

These take a positional `content` (or `contents`) plus existing keyword-only
configuration. Each gains a `min_length` keyword argument paired with the
existing `max_length`.

#### `regular_array_contents()`

`min_length` constrains `len(result)`, where:

- when `size > 0`: `len(result) == len(content) // size`
- when `size == 0`: `len(result) == zeros_length`

#### `list_offset_array_contents()`

`min_length` constrains the number of sublists, i.e.,
`len(offsets) - 1 >= min_length`.

#### `list_array_contents()`

`min_length` constrains the number of sublists, i.e.,
`len(starts) >= min_length`.

#### `record_array_contents()`

`min_length` constrains the shared field length. Since
`len(result) <= min(len(c) for c in contents)`, every child content must itself
satisfy `len(c) >= min_length`.

#### `union_array_contents()`

`min_length` constrains the union length, i.e.,
`sum(len(c) for c in contents) >= min_length` (before any `max_length`-driven
truncation). `max_length` continues to truncate overshoot; `min_length` is never
satisfied by truncation.

### Index-controlled option strategy

#### `indexed_option_array_contents()`

The index length equals `len(result)`. Because this strategy expresses the upper
bound as `max_size` (sizing the index buffer), the lower bound is spelled
`min_size` for symmetry with `numpy_array_contents()`, `string_contents()`, and
`bytestring_contents()` — not `min_length`.

### Mask-controlled option strategies

`byte_masked_array_contents()`, `bit_masked_array_contents()`, and
`unmasked_array_contents()` do not control their own length —
`len(result) == len(content)`. The standalone strategies gain no new parameter.
The corresponding `_from_contents` bridges forward `min_length` to the inner
`content(...)` recursive call so the inner content meets the floor.

### Leaf strategies

`leaf_contents()`, `numpy_array_contents()`, `string_contents()`, and
`bytestring_contents()` already expose a symmetric `min_size`/`max_size` pair.
They gain no new parameter.

### `_from_contents` bridges

Each bridge accepts `min_length` and forwards it to its standalone counterpart
(or, for the mask-controlled option types, to the inner `content(...)` call).
The `_StFromContents` protocol gains the new keyword so all bridges share the
same shape.

`list_offset_array_from_contents` and `list_array_from_contents` use
`min_length == max_length == length` to pin the exact length they drew, retiring
the two TODOs in those modules.

### Entry points

#### `contents()`

`contents()` gains a `min_length` keyword. Plumbing mirrors the existing
`leaf_max_size`:

- a local `leaf_min_size` is computed from `min_length`,
- it is forwarded to `leaf_contents(...)` as `min_size` when `contents()` draws
  a flat leaf,
- and it is forwarded to wrapper bridges via the `_StFromContents` protocol.

The bound applies only at the outermost level (the value seen by the caller). It
is not threaded into recursive `contents()` calls.

#### `arrays()`

`arrays()` gains a `min_length` keyword that forwards to `contents()`.

#### `content_lists()`

No change. Its existing `min_len`/`max_len` parameters control the number of
children, not the immediate length of any single content.

## Parameter design

### Name

- `min_length` for strategies that already use `max_length`.
- `min_size` for `indexed_option_array_contents()`, paired with its existing
  `max_size`.

### Type

`int`. Not `int | None`: the natural "no floor" sentinel is `0`, which is
already an integer, so the optional wrapper buys nothing.

### Default

`0`. Matches the convention used by `min_size` on every leaf strategy.

### Position

Keyword-only. Placed immediately before its `max_length` (or `max_size`)
counterpart. This matches the existing `numpy_array_contents()` ordering of
`min_size, max_size, max_length` and keeps related bounds adjacent.

## Interaction with existing parameters

### `min_length` and `max_length`

Both hold simultaneously: `min_length <= len(result) <= max_length`. When
`min_length > max_length`, no draw is possible; the strategy reaches an empty
`st.integers(...)` or `st.lists(..., min_size > max_size)` site and raises. No
eager validation.

### `min_length` and `min_size` at leaves

`contents()` translates its `min_length` into `leaf_min_size` for the leaf draw.
Leaf strategies receive it as `min_size`. The leaf-level upper bound remains the
existing `min(max_size, max_length)`; the lower bound becomes
`max(min_size_from_caller, leaf_min_size)`. In practice `leaf_contents()` is
invoked from `contents()` only with the computed `leaf_min_size`, so the two
never collide outside direct user calls.

### `min_length` and `max_size` / `max_leaf_size`

Orthogonal in dimension but interactive in feasibility. A high floor with a
tight scalar budget may be unsatisfiable; this surfaces at draw time as
`Unsatisfied` rather than eagerly.

### `min_length > 0` and `EmptyArray`

`leaf_contents()` already excludes `empty_array_contents` when `min_size > 0`
(via the existing guard `if allow_empty and min_size <= 0 <= max_size`). When
`contents()` plumbs a non-zero `min_length` into `leaf_min_size`, that guard
fires automatically: `EmptyArray` is excluded as the outermost choice.
Deeper-level `EmptyArray` leaves remain reachable because the floor only applies
at the outermost level.

## Infeasibility handling

No eager validation. Strategies raise only when a draw site has no choices —
typically an empty `st.one_of(...)`, an empty `st.sampled_from(...)`, or an
`st.integers(min_value=a, max_value=b)` with `a > b`. This matches how
`max_length` interacts with `max_size` today and keeps the first cut small.

A future iteration may add explicit validation at the entry points; that work is
out of scope for this design.

## Design decisions

### 1. Default `0`, not `None`

Lower bounds default to a concrete `0`, not `None`. This matches `min_size`
across the leaf strategies and avoids an awkward "unbounded below" notion when
the natural lower bound is zero anyway.

### 2. Per-strategy local enforcement

Each strategy is responsible for guaranteeing `len(result) >= min_length` for
the values it returns. The contract does not specify the mechanism — some
strategies push the floor to children and rely on natural construction, others
fall back to `assume()` for caller-supplied contents. Symmetric to how
`max_length` is enforced today.

### 3. No eager validation

Surfacing impossible combinations only at the first unworkable draw site keeps
the first cut small and matches existing convention. Eager validation can be
added later as a separate change.

### 4. Naming follows the existing length-control parameter

`min_length` for strategies that have `max_length`; `min_size` for
`indexed_option_array_contents()` (which uses `max_size` to size the index).
This avoids inventing parallel names and keeps each strategy's two bounds named
symmetrically.

### 5. Position before `max_*`

Placed immediately before the matching upper bound. Matches the
`min_size, max_size` ordering used today by `numpy_array_contents()`,
`string_contents()`, `bytestring_contents()`, and `leaf_contents()`.

### 6. `record_array_contents` does not truncate to satisfy the floor

`UnionArray` truncates its tags/index when the total length exceeds
`max_length`; the symmetric move for `min_length` would be padding, which
neither the constructor nor the data model supports. The strategy instead
forwards `min_length` to children (in the `None` branch) and uses `assume()` (in
the concrete/strategy branches) when caller-supplied contents are too short. No
silent padding.

## Implementation order

Bottom-up, matching the order used for `max_length`:

1. `regular_array_contents()` — `size`/`zeros_length` floor.
2. `list_offset_array_contents()` — offsets minimum count.
3. `list_array_contents()` — starts/stops minimum count.
4. `record_array_contents()` — shared field-length floor (children carry it).
5. `union_array_contents()` — total-length floor via children.
6. `indexed_option_array_contents()` — `min_size` on the index.
7. The four `_from_contents` bridges — forward the floor; pin the exact length
   in `list_offset_array_from_contents` and `list_array_from_contents`, retiring
   the two TODOs.
8. `contents()` — new keyword; compute `leaf_min_size`; forward to wrappers and
   leaves.
9. `arrays()` — new keyword; forward to `contents()`.

## Testing

Per-strategy tests follow the existing `*Kwargs` `TypedDict` and `*_kwargs()`
strategy pattern documented in
[`testing-patterns.md`](../../.claude/rules/testing-patterns.md). Specifics for
`min_length`:

- Add the new bound to each strategy's `*Kwargs` `TypedDict`.
- Draw `(min_length, max_length)` (or `(min_size, max_size)`) as a coordinated
  pair via the existing `st_ak.ranges()` helper, which already produces ordered
  `(start, end)` tuples with optional `None` for either side.
- In the main property test, assert `min_length <= len(result) <= max_length`.
  Use `safe_compare` so `None` on either side is effectively unbounded;
  `min_length=0` requires no special handling.
- Add edge-case reachability tests via `find()`:
  - can produce `len(result) == min_length` exactly,
  - can produce `len(result) == max_length` exactly,
  - can produce `min_length == max_length == N` exactly for at least one
    nontrivial `N`.

## Open questions

1. **Eager validation later?** Once all strategies expose `min_length`, the
   entry points could detect `min_length > max_length` and other obvious
   contradictions and raise `ValueError` early. Out of scope for this design;
   revisit after the first cut lands.
2. **Drawing length first, then content.** For `regular_array_contents` and the
   list strategies, drawing the length up front and asking the child for content
   of a matching length would give more uniform coverage and eliminate
   `assume()` reliance. Same direction noted in the `max_length` research;
   `min_length` makes it newly implementable for the list strategies. Out of
   scope here, captured for future work.
