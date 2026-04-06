"""Override site_url in zensical.toml for subdirectory deployment."""

import sys
from pathlib import Path

import tomlkit

CONFIG_PATH = Path('zensical.toml')


def main():
    config = tomlkit.loads(CONFIG_PATH.read_text())
    url = config.get('project', {}).get('site_url')
    if url:
        subdir = sys.argv[1]
        new_url = url.rstrip('/') + '/' + subdir.strip('/') + '/'
        config['project']['site_url'] = new_url
    CONFIG_PATH.write_text(tomlkit.dumps(config))


main()
