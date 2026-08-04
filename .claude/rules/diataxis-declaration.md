# Diátaxis declaration

How units of content declare their Diátaxis quadrants in the pages under
`docs/`. The review rules that use these declarations are in the
`persona-review` skill's `references/diataxis-review.md` (the shared core); this
file supplies the marker syntax, the page shapes, and the declaration record the
core defers to.

## Markers

A unit of content declares its quadrant with an HTML comment placed directly
below the unit's heading, separated by one blank line:

```markdown
## Bugs found

<!-- diataxis: reference -->
```

The value is one of `tutorial`, `how-to`, `reference`, `explanation`, or
`none (<reason>)` for a page that is not a Diátaxis unit, such as a navigation
index. Markers are invisible to site readers: they do not change heading
anchors, the table of contents, or the search index.

In the core's declaration pass, a well-formed marker is directly below its
heading with a value from this legend. A unit marked `none (<reason>)` has no
reader question: check only that the reason still holds and that the unit has
not started doing a quadrant's work.

## Page shapes

A page takes one of two shapes:

- **One unit** — a single marker below the H1, and no other marker on the page.
- **Container** — no marker below the H1, and one below every H2. The material
  between the H1 and the first H2 is an **orientation preamble**: framing,
  audience, scope, and links out. It carries no marker and is not a unit; once
  it starts doing a quadrant's work, it has to become a marked section.

Content that belongs to no unit is out of quadrant by construction; the
orientation preamble is the one sanctioned exception.

Markers sit at one level per branch: a heading whose subsections carry their own
markers is a container and carries none. Every page under `docs/` carries at
least one marker. Shape breaches for the core's declaration pass: markers at
more than one level of a branch, a container with an unmarked H2, or an H1
marker alongside section markers.

Nothing enforces this mechanically: a unit with no marker gives its reviewers no
question to answer, so it surfaces in the declaration pass of the self-check.
Note what that covers — the page under review, when a run happens. A page edited
outside a run keeps whatever markers it has until its next review.

A one-unit page that grows a section of another quadrant changes shape: the H1
marker is removed and every H2 gets one. A run may do this while relocating
content into a new section on the same page, because the change preserves every
existing unit's quadrant — only the markers' placement moves, and no value
changes. Changing a value is reclassification, and stays a scoping decision.

Converting a page this way demotes the material under the H1, which was part of
the declared unit and becomes an unmarked preamble. Do not leave it there:
whatever does the unit's work moves below the first section's heading, or
becomes the first marked section. Only framing, audience, scope, and links out
stay in the preamble.

## Record

The markers in the shipped page are the record of its quadrants; there is no
separate table to keep in sync. Declarations travel with the text: each draft
carries its own markers, so the declaration under review is always the one in
the draft. The review brief repeats each unit's quadrant with the matching
reader question, and gives the page's primary audience from the chapters table
(`.design/docs/Chapters.md`).
