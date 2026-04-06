"""Generate index.html listing all doc versions on gh-pages."""

import re
import sys
from pathlib import Path
from string import Template

from packaging.version import InvalidVersion, Version


def find_versions() -> list[str]:
    """Find version directories (e.g., 0.2.1, 1.0.0) sorted descending."""
    versions: list[tuple[Version, str]] = []
    for d in Path('.').iterdir():
        if not d.is_dir() or not re.match(r'\d', d.name):
            continue
        try:
            versions.append((Version(d.name), d.name))
        except InvalidVersion:
            continue
    versions.sort(key=lambda item: item[0], reverse=True)
    return [name for _, name in versions]


def find_latest_version(versions: list[str]) -> str | None:
    """Return the version that /latest/ points to, if any."""
    if not versions or not Path('latest').is_dir():
        return None
    return versions[0]


def build_version_list(versions: list[str], latest_version: str | None) -> str:
    """Build HTML list items for all versions."""
    lines: list[str] = []
    if latest_version:
        lines.append(
            f'    <li><a href="latest/">latest</a> <span class="meta">({latest_version})</span></li>'
        )
    if Path('dev').is_dir():
        lines.append('    <li><a href="dev/">dev</a></li>')
    for v in versions:
        lines.append(f'    <li><a href="{v}/">{v}</a></li>')
    return '\n'.join(lines)


def main() -> None:
    repo_name = sys.argv[1]
    repo_url = sys.argv[2]

    template_path = Path(__file__).parent / 'index.html'
    template = Template(template_path.read_text())

    versions = find_versions()
    latest_version = find_latest_version(versions)
    version_list = build_version_list(versions, latest_version)

    html = template.substitute(
        repo_name=repo_name,
        repo_url=repo_url,
        version_list=version_list,
    )

    Path('index.html').write_text(html)


main()
