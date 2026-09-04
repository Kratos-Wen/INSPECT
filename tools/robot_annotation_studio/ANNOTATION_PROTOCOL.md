# Fixed-Lattice Inspection Annotation Protocol

This protocol defines the production labeling process for the INSPECT robot
evaluation. The unit of work is one physical setup with six synchronized views.
Annotators judge only visible evidence in the selected image. Physical setup
truth is recorded once in the separate setup workflow and must not be inferred
while labeling a view in blind mode.

## Roles and workflow

- **Primary annotator:** reviews setup metadata and labels all assigned views.
- **Independent annotator:** labels the prespecified overlap setups in a separate
  database without seeing the primary annotations.
- **Reviewer:** resolves disagreements and records the adjudicated label as a new
  revision. Existing rows are never overwritten outside the application.

Keep all six views of a setup with the same primary annotator. Use a separate
database for independent overlap annotation so that annotations cannot leak
between annotators.

## Stage 1: setup truth

Open **Setup truth**, apply the imported prefill, and verify it against the setup
record. Confirm the assembly family, physical parts, active claim, physical claim
outcome, and error type. Complete this stage once per setup. Setup truth describes
the world; it does not state what any individual camera can prove.

## Stage 2: per-view evidence

Keep **Blind view mode** enabled and label one view at a time. Use this decision
tree:

1. Does this image alone expose enough evidence to decide the active claim?
   - Yes: assign utility `2` and choose `supported` or `contradicted`.
   - No: continue to Question 2.
2. Does the image expose claim-relevant evidence, even though it is incomplete?
   - Yes: assign utility `1` and choose `insufficient`.
   - No: assign utility `0` and choose `insufficient`.

Do not use another lattice view, the setup filename, the known physical outcome,
or a policy-selected destination when making a per-view decision.

## Evidence roles

Mark a role **visible** only when the current image provides usable evidence for
that role. Mark it **missing** when the role is required by the active claim but
cannot be judged from the current image.

| Role | Positive criterion |
|---|---|
| Identity disambiguation | The target can be separated from its hard-pair alternative. |
| Insertion verification | Insertion depth or completion is visible. |
| Containment verification | Inside/outside containment is visually decidable. |
| Slot relation | Target-to-slot placement or orientation is visible. |
| Gap visibility | A relevant assembly gap, or its absence, is visible. |
| Boundary alignment | Mating boundaries can be compared for alignment. |
| Contact verification | Required contact or separation can be judged. |
| Claim disambiguation | The evidence separates the active claim from a named counterfactual. |
| Occlusion recovery | The view removes an occlusion that blocked a required role. |

Unrated is not equivalent to missing. Use **Mark remaining required as missing**
only after reviewing the visible roles.

## Objects, relations, and geometry

Object visibility and relations are view-level observations. Record a relation
only when both participants and the relevant boundary are visible. Use
`uncertain` rather than guessing.

Bounding boxes and keypoints are selective diagnostics, not mandatory work on
every image:

- draw a target box for identity and object-visibility diagnostics;
- draw target/slot centers and orientation markers for insertion or alignment;
- draw cover-boundary and gap endpoints for seating and gap diagnostics;
- skip redundant geometry when it does not support a reported metric.

All geometry is stored in normalized image coordinates so exports remain valid
after display resizing.

## Completion and quality control

The application blocks completion when:

- utility `0` or `1` is paired with a committed support/contradiction decision;
- utility `2` is paired with `insufficient`;
- utility `2` has no visible evidence role;
- geometry lies outside the image;
- the view decision conflicts with setup truth without an explanatory note.

Imported setup records and the existing 132 utility labels begin as **drafts**.
They must be reviewed before completion. Use **Quality review** to resolve flagged
records, then export the ZIP containing flattened CSVs, full JSON, the exact
manifest, and the append-only audit log.

## Efficiency policy

1. Review the 22 active-inspection setups first (`trial_025`--`trial_040` and
   `trial_055`--`trial_060`); their 132 utility labels are already prefilled.
2. Label required evidence roles with the quick decisions and role shortcuts.
3. Complete object and identity diagnostics on the remaining step-1 views.
4. Add boxes or keypoints only to the metric-specific geometry subset.
5. Double-label a stratified overlap containing each claim, physical outcome,
   error type, and utility level before final adjudication.

This ordering reuses reliable existing labels, separates independent judgments,
and reserves the most expensive geometry annotation for experiments that consume
it directly.
