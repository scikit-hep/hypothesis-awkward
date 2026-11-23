# Development cycle

Develop test-first, one behavior at a time: make the change easy, then make the
easy change. How to write the tests themselves is in `testing-patterns.md`.

Before a cycle's red step, settle the design of the behavior that cycle adds:
its signature and options so far, and how it shrinks. The design can grow across
cycles: `datetime_units()` had no options in its first cycle, and the second
cycle added `allow_multiples`. If the design changes during a cycle, go back to
that cycle's red.

## Steps

0. **Prepare** (optional). Refactor while the suite passes, so that the next
   change is easy. Only the next test drives it. A non-trivial prepare step gets
   its own commit.
1. **Red.** Write the tests of one behavior, and confirm that they fail for the
   expected reason: on behavior, not on a missing name or argument.
   - A cycle's tests take the forms in `testing-patterns.md`: assertions in
     `test_properties`, reachability tests with `find()`, and shrink tests when
     the behavior affects shrinking.
   - A new strategy gets a stub with its signature whose body is
     `raise NotImplementedError`, so the tests cannot pass by accident.
   - A parameter is added in the red of the cycle whose behavior uses it,
     accepted and ignored. Without it, the test fails with `TypeError`, which
     only confirms that the parameter is missing; with it, the test fails on its
     assertions, which confirms that the test checks the behavior. No parameter
     is added before a cycle uses it.
   - If the red fails at collection because `@given(x=strategy())` calls a stub,
     draw inside the test with `data.draw(strategy())` instead, so that the
     other tests run.
   - One behavior per red step. A parametrized table of examples of one rule is
     fine; cases of different behaviors in one table are not.
2. **Green.** Write the minimum code that makes the tests pass, and run the
   whole suite, including the doctests in docstrings and Markdown files.
3. **Refactor**, then run `mypy src tests` and `ruff check src tests`.
4. **Mutation check.** Run the `mutation-check` skill, whose script must exit 0:
   the change must make some test fail, and the restored code must pass. A
   mutation that survives gets a stronger test in this step, not in a refactor.
   A module-level `assert` is part of the code, not a test. A mutation it
   catches is invalid; choose one the tests must catch.
5. **Commit**, only while the suite passes, one commit per behavior. Starting a
   cycle authorizes its commits on a development branch; on a production branch
   (`main`), the git rules apply: show the message and wait.

Run the cycle without stopping, through step 5 and into the next cycle, unless
the user says otherwise. Stop early when a step's result differs from what was
expected.

## Rules

- A change to a test's expected output is a behavior change, never a refactor.
- A refactor done after the behavior's commit is a late step 3, not step 0.
- A test that protects existing behavior has no red step. Check it with a
  mutation instead.
- Refactor one side only: change the tests or the code, never both in one step.
  The side that stays fixed is the only check on the other. After a refactor of
  the tests, rerun a mutation they caught before.
- An interface change takes three one-sided steps. Choose the final name in the
  first step; renaming back is another interface change.
  1. Code only: add the new function, and turn the old one into a wrapper over
     it.
  2. Tests only: move the tests to the new function.
  3. Code only: delete the wrapper.
- Read the output of every command before the next step. A restored run of
  `mutate.py` that does not pass stops the process.
- Report every deviation from these steps, with the reason for it.
