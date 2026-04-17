from hypothesis import Phase, settings

try:
    import icecream

    icecream.install()  # pragma: no cover
except ImportError:  # pragma: no cover
    pass

# Exclude `explain` from the Hypothesis phases for the issue:
# https://github.com/HypothesisWorks/hypothesis/issues/4708
# NOTE: Remove these lines when the issue is resolved.
phases = set(Phase) - {Phase.explain}
settings.register_profile('default', phases=phases)
settings.load_profile('default')
