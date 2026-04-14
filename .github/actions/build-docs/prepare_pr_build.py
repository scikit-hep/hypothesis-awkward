"""Prepare ``zensical.toml`` for a PR-preview build.

Runs two edits on the config in-place:

1. Append ``pr/<N>/`` (or whichever subdir is passed) to ``site_url`` so
   canonical URLs in the preview point to the preview location.
2. Remove ``project.extra.version`` so the preview does not render a
   version selector (mike's ``versions.json`` lives at the gh-pages root,
   not under ``pr/<N>/``).
"""

import sys
from pathlib import Path

import tomlkit

CONFIG_PATH = Path('zensical.toml')


def main():
    subdir = sys.argv[1]
    config = tomlkit.loads(CONFIG_PATH.read_text())

    project = config.get('project') or {}
    url = project.get('site_url')
    if url:
        project['site_url'] = url.rstrip('/') + '/' + subdir.strip('/') + '/'

    extra = project.get('extra') or {}
    if 'version' in extra:
        del extra['version']

    CONFIG_PATH.write_text(tomlkit.dumps(config))


main()
