# Robot View Decidability Studio

This local tool collects only the human judgements needed for the fixed-lattice
robot-view protocol:

1. whether one RGB view alone makes the active claim decidable; and
2. when it does not, whether the claim-critical region is explicitly occluded.

The project preparation step imports existing human utility labels. A legacy
`2 clear` label becomes a locked decidable label. Legacy `0 not useful` and
`1 helpful` labels become locked not-decidable labels, leaving only the binary
occlusion check. The browser manifest intentionally excludes physical setup
truth, outcome labels, error types, and oracle utility.

## Prepare

```powershell
python tools\robot_decidability_studio\prepare_project.py
```

Preparation refuses to overwrite an existing annotation database unless
`--overwrite-db` is supplied. The generated project is stored under
`outputs\robot_decidability_studio`.

## Run

```powershell
powershell -ExecutionPolicy Bypass -File outputs\robot_decidability_studio\launch.ps1
```

Open `http://127.0.0.1:8787`. Annotations are committed to SQLite immediately.
Use **Export** to download a ZIP containing CSV, JSON, the audit log, and the
public manifest.

## Keyboard

- `1`: decidable
- `0`: cannot decide
- `O`: explicitly occluded
- `V`: not explicitly occluded
- `J`: next pending view
- `K`: previous pending view
