---
paths:
  - "README.md"
  - "docs/**"
---

# Documentation Voice

The audience is international scientific-Python users. These rules apply to
user-facing documentation prose (the README and `docs/`), not to code, to
`.design/` notes, or to commit and pull-request text.

- Avoid slang.
- Avoid idioms and figures of speech that mainly native English speakers use (no
  sports, military, or cooking metaphors).
- Prefer literal expressions over figurative ones ("found the bug", not "hunted
  down the bug").
- Established technical terms are fine even when metaphorical ("shrink", "tree",
  "leaf"); the idiom and figurative-language rules target decorative figures of
  speech, not standard vocabulary.
- Prefer precise words: specific numbers, named technologies, exact counts — not
  "many" or "several". But do not invent precision: a vague quantifier is honest
  when the exact number is unknown, and avoid literal values that go stale when
  a stable reference exists.
- State things directly — no hedging like "I think" or "I believe" unless they
  genuinely describe a thought or belief. Expressing calibrated uncertainty is
  information, not hedging; do not upgrade uncertain claims to fact.
- Spell out each acronym on first use in a document, then use the short form.
  Skip expansions the audience certainly knows better than the long form (URL,
  API).
- Explain ideas in your own words rather than lightly rewording a source.
- Put reused external wording in quotation marks, attribute it, and link the
  source; never present quoted phrasing as your own.
- Keep quotes short and load-bearing; do not quote whole paragraphs of prose.
- Summarize even first-party material (docstrings, `.design/` notes) rather than
  paste it.
- Verbatim reproduction is correct when exact wording is the point (error
  messages, command output, falsifying examples); quote or fence it rather than
  paraphrase or retype.
