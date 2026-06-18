# Design Documents

This directory contains UX research findings and API design documents for
hypothesis-awkward.

## Directory Structure

```text
.design/
├── README.md          # This file
├── research/          # UX research interviews and findings
├── api/               # API design proposals and decisions
├── impl/              # Implementation decisions (directory structure, internals)
└── notes/             # Progress tracking and working notes
```

## Purpose

- **research/**: Store technical research and UX findings. Topics include
  Awkward Array internals (type system, form system, direct constructors), user
  interviews, and feasibility studies for new strategies.

- **api/**: Store API design proposals, alternatives considered, and final
  decisions with rationale.

- **impl/**: Store implementation decisions that don't affect the user-facing
  API, such as directory structure, module organization, and internal
  architecture.

- **notes/**: Store progress tracking and working notes for ongoing work.

## Naming Convention

Every file is prefixed with its date and a two-digit intra-day sequence:
`YYYY-MM-DD-NN-<topic>.md`. The `NN` orders files created on the same day within
a directory (`01`, `02`, …); use `01` when a file is the only one that day. This
keeps a directory listing in chronological order. Topic words use sentence case
(capitalize the first word); acronyms and proper nouns keep their canonical
casing (README, API, UX, NumPy, PBT). Per-type topic conventions:

- Research files: `YYYY-MM-DD-NN-<topic>-research.md` (e.g.,
  `2026-02-02-01-NumPy-form-research.md`). UX interviews use
  `YYYY-MM-DD-NN-UX-interview-MM.md`, where `MM` is the interview number.
- API design files: `YYYY-MM-DD-NN-<feature>-API.md`
- Implementation files: `YYYY-MM-DD-NN-<topic>.md`
- Notes: `YYYY-MM-DD-NN-<topic>.md`
