from typing import TypedDict, cast

import numpy as np
from hypothesis import find, given
from hypothesis import strategies as st

import awkward as ak
from hypothesis_awkward import strategies as st_ak
from hypothesis_awkward.util import SUPPORTED_DTYPE_NAMES
from tests.find_settings import FIND_NO_SHRINK
from tests.funcs import KwargsDrawn, assert_kwargs_match_signature


class NumpyFormsKwargs(TypedDict, total=False):
    """Options for `numpy_forms()` strategy."""

    type_: ak.types.NumpyType | st.SearchStrategy[ak.types.NumpyType] | None
    dtypes: st.SearchStrategy[np.dtype] | None
    allow_datetime: bool
    inner_shape: tuple[int, ...] | st.SearchStrategy[tuple[int, ...]] | None
    allow_inner_shape: bool


DEFAULTS = NumpyFormsKwargs(
    type_=None,
    dtypes=None,
    allow_datetime=True,
    inner_shape=None,
    allow_inner_shape=True,
)


class NumpyFormsDrawn(TypedDict, total=False):
    """Values drawn in advance for the options of `numpy_forms()`."""

    type_: ak.types.NumpyType
    dtypes: np.dtype
    inner_shape: tuple[int, ...]


def test_kwargs_match_signature() -> None:
    """Assert the option declarations agree with `numpy_forms()`'s parameters."""
    assert_kwargs_match_signature(
        func=st_ak.numpy_forms,
        kwargs_cls=NumpyFormsKwargs,
        defaults=DEFAULTS,
    )


def _inner_shape_strategies() -> st.SearchStrategy[tuple[int, ...]]:
    """Strategy for generating small inner_shape tuples."""
    return st.lists(
        st.integers(min_value=1, max_value=5),
        min_size=1,
        max_size=3,
    ).map(tuple)


@st.composite
def numpy_forms_kwargs(
    draw: st.DrawFn,
) -> KwargsDrawn[NumpyFormsKwargs, NumpyFormsDrawn]:
    """Strategy for options for `numpy_forms()` strategy.

    Two modes:
    - type_ mode: type_ is set, other params omitted.
    - dtypes mode: type_ omitted, dtypes/allow_datetime/inner_shape/allow_inner_shape drawn.
    """

    @st.composite
    def _st_type_mode(
        draw: st.DrawFn,
    ) -> KwargsDrawn[NumpyFormsKwargs, NumpyFormsDrawn]:
        type_ = draw(st_ak.numpy_types())
        choices = list[ak.types.NumpyType | st.SearchStrategy[ak.types.NumpyType]](
            [type_, st.just(type_)]
        )
        kwargs = NumpyFormsKwargs(type_=draw(st.sampled_from(choices)))
        return KwargsDrawn(kwargs, NumpyFormsDrawn(type_=type_))

    @st.composite
    def _st_dtypes_mode(
        draw: st.DrawFn,
    ) -> KwargsDrawn[NumpyFormsKwargs, NumpyFormsDrawn]:
        kwargs = cast(
            NumpyFormsKwargs,
            draw(
                st.fixed_dictionaries(
                    {},
                    optional={
                        'allow_datetime': st.booleans(),
                        'allow_inner_shape': st.booleans(),
                    },
                )
            ),
        )
        drawn = NumpyFormsDrawn()
        if draw(st.booleans()):
            match draw(st.sampled_from(['none', 'strategy'])):
                case 'none':
                    kwargs['dtypes'] = None
                case 'strategy':
                    dtype = draw(st_ak.supported_dtypes())
                    kwargs['dtypes'] = st.just(dtype)
                    drawn['dtypes'] = dtype
        if draw(st.booleans()):
            match draw(st.sampled_from(['none', 'value', 'strategy'])):
                case 'none':
                    kwargs['inner_shape'] = None
                case 'value':
                    shape = draw(
                        st.lists(
                            st.integers(min_value=1, max_value=5),
                            min_size=0,
                            max_size=3,
                        ).map(tuple)
                    )
                    kwargs['inner_shape'] = shape
                    drawn['inner_shape'] = shape
                case 'strategy':
                    shape = draw(_inner_shape_strategies())
                    kwargs['inner_shape'] = st.just(shape)
                    drawn['inner_shape'] = shape
        return KwargsDrawn(kwargs, drawn)

    return draw(st.one_of(_st_type_mode(), _st_dtypes_mode()))


DATETIME_PRIMITIVES = frozenset(
    n for n in SUPPORTED_DTYPE_NAMES if n.startswith(('datetime64', 'timedelta64'))
)


@given(data=st.data())
def test_properties(data: st.DataObject) -> None:
    """Assert the results of `numpy_forms()`."""
    # Draw options
    kwargs, drawn = data.draw(numpy_forms_kwargs(), label='opts')

    # Call the test subject
    result = data.draw(st_ak.numpy_forms(**kwargs), label='result')

    # Assert the result is always a NumpyForm
    assert isinstance(result, ak.forms.NumpyForm)

    # Assert parameters and form_key are always None
    assert result._parameters is None
    assert result._form_key is None

    # Assert the options were effective
    allow_datetime = kwargs.get('allow_datetime', DEFAULTS['allow_datetime'])
    allow_inner_shape = kwargs.get('allow_inner_shape', DEFAULTS['allow_inner_shape'])

    if 'type_' in drawn:
        # type_ mode: primitive matches type, inner_shape is ()
        assert result.primitive == drawn['type_'].primitive
        assert result.inner_shape == ()
    else:
        # dtypes mode
        if 'dtypes' in drawn:
            assert result.primitive == drawn['dtypes'].name
        else:
            assert result.primitive in SUPPORTED_DTYPE_NAMES
            if not allow_datetime:
                assert result.primitive not in DATETIME_PRIMITIVES

        if 'inner_shape' in drawn:
            assert result.inner_shape == drawn['inner_shape']
        elif not allow_inner_shape:
            assert result.inner_shape == ()


def test_draw_empty_inner_shape() -> None:
    """Assert that forms with empty inner_shape can be drawn."""
    find(st_ak.numpy_forms(), lambda f: f.inner_shape == (), settings=FIND_NO_SHRINK)


def test_draw_nonempty_inner_shape() -> None:
    """Assert that forms with non-empty inner_shape can be drawn."""
    find(st_ak.numpy_forms(), lambda f: len(f.inner_shape) > 0, settings=FIND_NO_SHRINK)


def test_draw_datetime_primitive() -> None:
    """Assert that datetime64 primitive can be drawn."""
    find(
        st_ak.numpy_forms(),
        lambda f: f.primitive.startswith('datetime64'),
        settings=FIND_NO_SHRINK,
    )


def test_draw_integer_primitive() -> None:
    """Assert that integer primitives can be drawn."""
    find(
        st_ak.numpy_forms(),
        lambda f: f.primitive in ('int8', 'int16', 'int32', 'int64'),
        settings=FIND_NO_SHRINK,
    )


def test_draw_from_type() -> None:
    """Assert that a form from a NumpyType matches the type."""
    t = ak.types.NumpyType('float64')
    f = find(st_ak.numpy_forms(type_=t), lambda f: True, settings=FIND_NO_SHRINK)
    assert f.primitive == 'float64'
    assert f.inner_shape == ()
