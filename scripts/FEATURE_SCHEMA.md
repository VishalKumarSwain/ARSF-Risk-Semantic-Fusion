# KSERESNET-RS Shared Feature Schema (frozen)

This is the **identical structured feature vector** given to both B4 (GBDT baseline) and the
LLM semantic scorer (Qwen3-8B and other Stage-1 candidates). Identity of input is what makes
B4 a valid test of "is the LLM doing more than see more features" — any feature present to one
must be present to the other, and vice versa. Do not add a feature to one consumer without
adding it to both and updating this document.

## Source

Derived purely from road geometry (the `.xodr` OpenDRIVE `road_points`), NOT from the
`sim_data.json` sensor telemetry — because telemetry only exists for 33,373/36,006 scenarios
(7.3% missing), while geometry is available for all 36,006. Using geometry-only features keeps
the full pool usable for every component (KSERESNET, GBDT, LLM) without a coverage mismatch.

## Explicitly EXCLUDED: `duration`

`test_duration` is **not a feature** for any risk-scoring component (KSERESNET, GBDT, or LLM).
Reason: in this dataset, a FAIL test's simulation halts immediately when the car goes off-road,
so `duration` is *caused by* the outcome, not a pre-execution property of the scenario. Using it
as a predictive input would leak the label through the back door. `duration` is used **only** as
the cost term in APFDc computation, never as model input.

## Feature vector (10 fields)

| Field | Type | Description | Computation |
|---|---|---|---|
| `generator` | categorical | Road generator: `ambiegen` / `frenetic` / `freneticv` | Parsed from `test_id` |
| `num_points` | int | Raw waypoint count before any resampling | `len(road_points)` |
| `road_length` | float | Total path length (sum of segment lengths) | Σ Euclidean distance between consecutive points |
| `straight_line_distance` | float | Euclidean distance from first to last point | `dist(p0, p_last)` |
| `tortuosity` | float | `road_length / straight_line_distance` — how much the road winds relative to a straight line | ratio (1.0 = perfectly straight) |
| `mean_curvature` | float | Mean absolute heading-angle change between consecutive segments (radians) | mean(\|Δheading_i\|) |
| `max_curvature` | float | Sharpest single turn in the road (radians) | max(\|Δheading_i\|) |
| `std_curvature` | float | Variability of turning along the road | std(\|Δheading_i\|) |
| `sharp_turn_count` | int | Number of segments with \|Δheading\| > 0.3 rad (~17°) | count |
| `bbox_aspect_ratio` | float | width/height of the road's bounding box | `bbox_w / max(bbox_h, 1e-6)` |

## Representations

- **GBDT (B4)**: numeric vector, `generator` one-hot encoded → 12-dim input (3 one-hot + 9 numeric).
- **LLM (Qwen3-8B, Stage 1 candidates)**: same 10 fields rendered as a structured text block in
  the prompt (see `llm_prompt_template.md`), plus explicit natural-language framing of what each
  field means for driving risk. No ground-truth label is ever included in the LLM prompt.
- **KSERESNET**: unaffected — continues to use its own original raw kinematic (dx,dy,ax,ay)
  representation, frozen from the ICST submission, not this schema. This schema is only for the
  new B4/LLM components being added in the JSS extension.

## Feature-availability audit (required by the frozen design, Section 9 of prior spec)

KSERESNET's own input (raw resampled dx/dy/ax/ay from up to 200 points) is a *different, richer*
per-point representation than this 10-field summary. This is a genuine asymmetry: KSERESNET sees
full per-point kinematics; B4/LLM see only summary statistics of the same underlying geometry. This
must be disclosed explicitly in the paper's feature-availability table (Part 9 of the frozen
methodology) — it is not a leak (both ultimately derive from the same road_points, nothing external
is added), but it is an information-representation difference worth stating plainly rather than
letting a reviewer discover it.

## Versioning

Schema version: v1, frozen 2026-08-29. Any change requires updating this file, `compute_features.py`,
the GBDT training script, and the LLM prompt template together, and re-running all three from scratch.
