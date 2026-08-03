"""Check the Diátaxis markers in `docs/` against the rules.

The rules are in `.claude/rules/diataxis-review.md`. Each unit of content declares its
quadrant with an HTML comment directly below the unit's heading: one marker below the H1
when the whole page is one unit, or one below every H2 when the H1 is a container.
"""

import re
from pathlib import Path
from typing import NamedTuple

import pytest

DOCS = Path(__file__).parents[2] / 'docs'

QUADRANTS = frozenset({'tutorial', 'how-to', 'reference', 'explanation'})

MARKER = re.compile(r'^<!-- diataxis: (?P<value>.+) -->$')
HEADING = re.compile(r'^(?P<hashes>#{1,6}) (?P<text>.+)$')
FENCE = re.compile(r'^\s*(```|~~~)')

PAGES = sorted(DOCS.rglob('*.md'))


class Heading(NamedTuple):
    """A Markdown heading and the marker declaring it, if any."""

    lineno: int
    level: int
    text: str
    marker: str | None


def test_pages_found() -> None:
    """Assert the page set is non-empty, so an empty glob cannot pass silently."""
    assert PAGES


@pytest.mark.parametrize('page', PAGES, ids=lambda p: str(p.relative_to(DOCS)))
def test_markers(page: Path) -> None:
    """Assert the page declares its units as the marker rules require."""
    headings = _parse(page)

    h1s = [h for h in headings if h.level == 1]
    assert len(h1s) == 1, f'{page}: expected exactly one H1, found {len(h1s)}'
    h2s = [h for h in headings if h.level == 2]

    if h1s[0].marker is not None:
        marked = [h.text for h in h2s if h.marker is not None]
        assert not marked, (
            f'{page}: the H1 is marked, so it is the only marked unit; '
            f'remove the marker below {marked}'
        )
        return

    assert h2s, f'{page}: no marker below the H1 and no H2 to carry one'
    unmarked = [h.text for h in h2s if h.marker is None]
    assert not unmarked, (
        f'{page}: the H1 is a container, so every H2 needs a marker; '
        f'missing below {unmarked}'
    )


def _parse(page: Path) -> list[Heading]:
    """Return the headings outside fenced code, each with its marker."""
    headings = list[Heading]()
    in_fence = False

    for i, line in enumerate(page.read_text().splitlines()):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        if marker := MARKER.match(line):
            value = marker['value']
            assert value in QUADRANTS or value.startswith('none'), (
                f'{page}:{i + 1}: unknown quadrant {value!r}'
            )
            assert headings and headings[-1].lineno == i - 2, (
                f'{page}:{i + 1}: a marker goes directly below its heading, '
                'separated by one blank line'
            )
            headings[-1] = headings[-1]._replace(marker=value)
            continue

        if heading := HEADING.match(line):
            headings.append(
                Heading(
                    lineno=i,
                    level=len(heading['hashes']),
                    text=heading['text'],
                    marker=None,
                )
            )

    return headings
