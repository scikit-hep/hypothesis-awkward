import subprocess
import sys
from types import ModuleType

import pytest

import hypothesis_awkward.strategies
import hypothesis_awkward.strategies.constructors
import hypothesis_awkward.strategies.content
import hypothesis_awkward.strategies.misc
import hypothesis_awkward.strategies.numpy


@pytest.mark.parametrize(
    'subpackage',
    [
        hypothesis_awkward.strategies,
        hypothesis_awkward.strategies.constructors,
        hypothesis_awkward.strategies.content,
        hypothesis_awkward.strategies.misc,
        hypothesis_awkward.strategies.numpy,
    ],
    ids=lambda m: m.__name__,
)
def test_subpackage_attribute(subpackage: ModuleType) -> None:
    """Test if `subpackage` is an attribute after importing only its parent.

    Run in a fresh interpreter, where no other import has loaded `subpackage`.
    """
    parent, _, _ = subpackage.__name__.rpartition('.')
    code = f'import {parent}; {subpackage.__name__}'
    subprocess.run([sys.executable, '-c', code], check=True)
