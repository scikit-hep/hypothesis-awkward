import hypothesis_awkward


def test_version() -> None:
    """Test if the version string is attached to the module."""
    hypothesis_awkward.__version__
