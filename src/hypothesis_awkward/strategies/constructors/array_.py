import functools
from collections.abc import Callable

import numpy as np
from hypothesis import strategies as st

import awkward as ak
import hypothesis_awkward.strategies as st_ak
from hypothesis_awkward.util.draw import CountdownDrawer

_ContentsFn = Callable[
    [st.SearchStrategy[ak.contents.Content]],
    st.SearchStrategy[ak.contents.Content],
]


@st.composite
def arrays(
    draw: st.DrawFn,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    max_size: int = 10,
    allow_nan: bool = False,
    allow_regular: bool = True,
    allow_list_offset: bool = True,
    allow_list: bool = True,
    max_depth: int = 5,
) -> ak.Array:
    '''Strategy for Awkward Arrays.

    The current implementation generates arrays with NumpyArray as leaf contents that can
    be nested multiple levels deep in RegularArray, ListOffsetArray, and ListArray lists.

    Parameters
    ----------
    dtypes
        A strategy for NumPy scalar dtypes used in ``NumpyArray``. If ``None``, the
        default strategy that generates any scalar dtype supported by Awkward Array is
        used.
    max_size
        Maximum total number of scalar values in the generated array.
    allow_nan
        No ``NaN``/``NaT`` values are generated if ``False``.
    allow_regular
        No ``RegularArray`` is generated if ``False``.
    allow_list_offset
        No ``ListOffsetArray`` is generated if ``False``.
    allow_list
        No ``ListArray`` is generated if ``False``.
    max_depth
        Maximum depth of nested arrays.

    Examples
    --------
    >>> arrays().example()
    <Array ... type='...'>

    '''
    layout = draw(
        contents(
            dtypes=dtypes,
            max_size=max_size,
            allow_nan=allow_nan,
            allow_regular=allow_regular,
            allow_list_offset=allow_list_offset,
            allow_list=allow_list,
            max_depth=max_depth,
        )
    )
    return ak.Array(layout)


@st.composite
def contents(
    draw: st.DrawFn,
    dtypes: st.SearchStrategy[np.dtype] | None = None,
    max_size: int = 10,
    allow_nan: bool = False,
    allow_regular: bool = True,
    allow_list_offset: bool = True,
    allow_list: bool = True,
    max_depth: int = 5,
) -> ak.contents.Content:
    '''Strategy for Awkward Array content layouts.

    The current implementation generates layouts with NumpyArray as leaf contents that can
    be nested multiple levels deep in RegularArray, ListOffsetArray, and ListArray lists.

    Parameters
    ----------
    dtypes
        A strategy for NumPy scalar dtypes used in ``NumpyArray``. If ``None``, the
        default strategy that generates any scalar dtype supported by Awkward Array is
        used.
    max_size
        Maximum total number of scalar values in the generated array.
    allow_nan
        No ``NaN``/``NaT`` values are generated if ``False``.
    allow_regular
        No ``RegularArray`` is generated if ``False``.
    allow_list_offset
        No ``ListOffsetArray`` is generated if ``False``.
    allow_list
        No ``ListArray`` is generated if ``False``.
    max_depth
        Maximum depth of nested arrays.

    Examples
    --------
    >>> contents().example()
    <NumpyArray ...>

    '''
    content_fns: list[_ContentsFn] = []
    if allow_regular:
        content_fns.append(st_ak.contents.regular_array_contents)
    if allow_list_offset:
        content_fns.append(st_ak.contents.list_offset_array_contents)
    if allow_list:
        content_fns.append(st_ak.contents.list_array_contents)

    st_ = functools.partial(st_ak.contents.numpy_array_contents, dtypes, allow_nan)

    if not content_fns or max_size == 0:
        return draw(st_(min_size=0, max_size=max_size))

    draw_content = CountdownDrawer(draw, st_, max_size_total=max_size)

    # Draw nesting depth, then choose a content function for each level.
    depth = draw(st.integers(min_value=0, max_value=max_depth))
    chosen: list[_ContentsFn] = [
        draw(st.sampled_from(content_fns)) for _ in range(depth)
    ]

    content = draw_content()
    if content is None:
        return draw(st_(min_size=0, max_size=0))

    for fn in reversed(chosen):
        content = draw(fn(st.just(content)))

    return content
