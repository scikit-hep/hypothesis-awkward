---
name: mutation-check
description:
  Check that the tests catch a deliberate change to the code. Use at step 4 of
  the development cycle, after a behavior's test passes, and to check a test
  that protects existing behavior.
---

# Mutation check

Change the code on purpose, and confirm that a test fails. A test that still
passes on wrong code does not test that code.

## Run

From the repository root, in the project's virtual environment:

```bash
python -I .claude/skills/mutation-check/scripts/mutate.py FILE OLD NEW [PYTEST_ARG ...]
```

The script replaces the single occurrence of `OLD` in `FILE` with `NEW`, runs
pytest, restores `FILE`, and runs pytest again. Pass the test module of the
behavior as `PYTEST_ARG`: the whole suite takes minutes, and the script runs it
twice.

## Choose the mutation

- Break the behavior that the test is meant to check, and only that: flip a
  condition, drop an option, change a bound, reverse an order.
- Make `OLD` long enough to occur once in `FILE`, e.g. include the rest of the
  line.
- Do not mutate a file while another mutation runs on it.

## Read the result

| Exit | Output                | Meaning                                          |
| ---- | --------------------- | ------------------------------------------------ |
| 0    | (no verdict)          | A test failed on the mutant; the restored passes |
| 1    | `MUTANT SURVIVED`     | No test failed: strengthen the test              |
| 1    | `INVALID MUTANT`      | pytest could not run, e.g. a syntax error        |
| 1    | `RESTORED CODE FAILS` | The original code fails: stop and report         |
| 2    | usage or `OLD` error  | Fix the arguments                                |

Check that the failing test under `--- mutant:` is the one meant to catch the
change. Report it when a different test catches it.

A surviving mutant gets a stronger test in the same step, not in a later
refactor. Hypothesis draws at random, so a property test can also miss a mutant
by chance.
