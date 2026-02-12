'''Tests for `OptsChain` with `register()` and `extend()`.'''

from __future__ import annotations

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from hypothesis_awkward.strategies.misc.record import OptsChain, RecordDraws


@st.composite
def draw_opts_chain(
    draw: st.DrawFn,
) -> tuple[OptsChain[Any], list[OptsChain[Any]]]:
    '''Build a chain of OptsChain produced by repeated extend() calls.

    Returns
    -------
    final : OptsChain
        The OptsChain after all extend() calls.
    chain : list[OptsChain]
        All OptsChain in the chain, from base (index 0) to final (index -1).
    '''
    depth = draw(st.integers(min_value=0, max_value=4), label='depth')

    chain: list[OptsChain[Any]] = []
    opts: OptsChain[Any] = OptsChain({})
    chain.append(opts)

    for level in range(depth):
        level_kwargs = draw(
            _level_kwargs(opts=opts),
            label=f'level_{level}_kwargs',
        )

        opts = opts.extend(level_kwargs)
        chain.append(opts)

    return opts, chain


_VALUE_STRATEGIES: list[st.SearchStrategy[Any]] = [
    st.integers(),
    st.floats(allow_nan=False),
    st.booleans(),
    st.none(),
    st.text(max_size=10),
    st.tuples(st.integers()),
]


@st.composite
def _draw_value(
    draw: st.DrawFn,
    *,
    opts: OptsChain[Any],
    max_depth: int = 2,
) -> Any:
    '''Draw a value that may be plain, a registered recorder, or a nested structure.'''
    choices = ['plain', 'strategy']
    if max_depth > 0:
        choices += ['list', 'dict']

    kind = draw(st.sampled_from(choices), label='kind')

    match kind:
        case 'plain':
            return draw(st.one_of(*_VALUE_STRATEGIES))
        case 'strategy':
            strategy = draw(st.sampled_from(_VALUE_STRATEGIES))
            return opts.register(strategy)
        case 'list':
            n = draw(st.integers(min_value=0, max_value=3))
            return [
                draw(_draw_value(opts=opts, max_depth=max_depth - 1)) for _ in range(n)
            ]
        case 'dict':
            keys = draw(
                st.lists(
                    st.text(min_size=1, max_size=5),
                    min_size=0,
                    max_size=3,
                    unique=True,
                )
            )
            return {
                k: draw(_draw_value(opts=opts, max_depth=max_depth - 1)) for k in keys
            }
        case _:  # pragma: no cover
            raise AssertionError(kind)


@st.composite
def _level_kwargs(
    draw: st.DrawFn,
    *,
    opts: OptsChain[Any],
) -> dict[str, Any]:
    '''Draw kwargs for a single level in the opts chain.

    Values can be plain, registered recorders, or nested lists/dicts
    containing recorders. Strategies are registered on *opts* via
    ``opts.register()``.
    '''
    fresh_key = st.text(min_size=1, max_size=10).filter(lambda k: k not in opts.kwargs)
    keys = draw(
        st.lists(fresh_key, min_size=0, max_size=10, unique=True),
        label='keys',
    )

    return {key: draw(_draw_value(opts=opts), label=key) for key in keys}


@settings(max_examples=200)
@given(data=st.data())
def test_opts_chain(data: st.DataObject) -> None:
    '''Test that register(), extend(), and reset() work together.'''
    final, chain = data.draw(draw_opts_chain(), label='opts_chain')
    all_recorders = final.recorders

    # 1. kwargs merging: final.kwargs equals the merge of all levels
    expected_kwargs: dict[str, Any] = {}
    for opts in chain:
        expected_kwargs.update(opts.kwargs)
    assert dict(final.kwargs) == expected_kwargs

    # 2. register returns RecordDraws
    for r in all_recorders:
        assert isinstance(r, RecordDraws)

    # 3. draw populates drawn lists
    drawn_values: list[Any] = []
    for r in all_recorders:
        v = data.draw(r, label='draw_from_recorder')
        drawn_values.append(v)
    for r, v in zip(all_recorders, drawn_values):
        assert len(r.drawn) == 1
        assert r.drawn[0] == v

    # 4. reset clears all recorders
    final.reset()
    for r in all_recorders:
        assert r.drawn == [], f'Expected empty drawn after reset, got {r.drawn}'

    # 5. reset isolates: draw again after reset
    new_values: list[Any] = []
    for r in all_recorders:
        v = data.draw(r, label='draw_after_reset')
        new_values.append(v)
    for r, v in zip(all_recorders, new_values):
        assert len(r.drawn) == 1
        assert r.drawn[0] == v


@settings(max_examples=200)
@given(data=st.data())
def test_extend_does_not_affect_parent(data: st.DataObject) -> None:
    '''Test that resetting a child does not lose parent-only recorders,
    and that resetting a parent leaves child-only recorders untouched.
    '''
    final, chain = data.draw(draw_opts_chain(), label='opts_chain')

    # Need at least depth >= 1 for this test
    if len(chain) < 2:
        return

    parent = chain[0]
    all_recorders = final.recorders
    parent_recorders = parent.recorders
    parent_recorder_ids = {id(r) for r in parent_recorders}
    child_only_recorders = [
        r for r in all_recorders if id(r) not in parent_recorder_ids
    ]

    # 1. Draw from all recorders to populate drawn
    for r in all_recorders:
        data.draw(r, label='populate')

    # 2. Reset the final (child) opts — all recorders should clear
    final.reset()
    for r in all_recorders:
        assert r.drawn == []

    # 3. Draw again from a parent-level recorder
    for r in parent_recorders:
        data.draw(r, label='draw_parent_after_child_reset')

    # 4. Draw from child-only recorders
    for r in child_only_recorders:
        data.draw(r, label='draw_child_only')

    # 5. Reset the parent only
    parent.reset()

    # 6. Parent recorders should be cleared
    for r in parent_recorders:
        assert r.drawn == []

    # 7. Child-only recorders should still hold their values
    for r in child_only_recorders:
        assert len(r.drawn) == 1


def test_empty() -> None:
    '''OptsChain({}) — reset() does not raise, kwargs == {}.'''
    opts: OptsChain[Any] = OptsChain({})
    opts.reset()
    assert opts.kwargs == {}


def test_register_on_empty() -> None:
    '''OptsChain({}).register(st.integers()) works and returns a RecordDraws.'''
    opts: OptsChain[Any] = OptsChain({})
    recorder = opts.register(st.integers())
    assert isinstance(recorder, RecordDraws)

    # Draw from it and verify it records
    recorder.example()
    assert len(recorder.drawn) > 0

    # Reset clears it
    opts.reset()
    assert recorder.drawn == []


def test_extend_empty() -> None:
    '''OptsChain({'a': 1}).extend({}) — kwargs unchanged.'''
    opts: OptsChain[Any] = OptsChain({'a': 1})
    extended = opts.extend({})
    assert extended.kwargs == {'a': 1}
