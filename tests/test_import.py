import subprocess
import sys


def test_strategies_attribute() -> None:
    """Test if `strategies` is an attribute after importing only the package.

    Run in a fresh interpreter, where no other import has loaded `strategies`.
    """
    import hypothesis_awkward
    import hypothesis_awkward.strategies

    package = hypothesis_awkward.__name__
    module = hypothesis_awkward.strategies.__name__
    code = f'import {package}; {module}'
    subprocess.run([sys.executable, '-c', code], check=True)
