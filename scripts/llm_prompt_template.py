"""
Builds the structured LLM risk-scoring prompt from the frozen shared
feature schema (FEATURE_SCHEMA.md) -- the SAME 10 fields given to B4.
No ground-truth outcome is ever included.
"""

SYSTEM_PROMPT = """You are a safety analyst for autonomous-driving simulation testing. \
You will be given a structured summary of a road scenario's geometry (from a self-driving \
car test track). Based ONLY on the geometric properties given, assess how likely this \
scenario is to cause the autonomous vehicle to fail (drive off the road boundary).

You do not have access to sensor data, weather, or traffic information -- only road shape. \
Reason about which geometric patterns (sharp turns, high curvature variability, tortuous \
paths, etc.) are associated with vehicle control difficulty and off-road risk.

Respond with ONLY a JSON object in this exact schema, no other text:
{
  "risk_score": <float between 0.0 and 1.0>,
  "risk_factors": [<list of short strings naming the geometric factors driving your score>],
  "rationale": "<1-2 sentence explanation>"
}"""

USER_PROMPT_TEMPLATE = """Road scenario (generator: {generator}):

- Number of waypoints: {num_points}
- Total road length: {road_length:.2f} units
- Straight-line distance (start to end): {straight_line_distance:.2f} units
- Tortuosity (road_length / straight_line_distance): {tortuosity:.2f}
- Mean turn angle between segments: {mean_curvature:.4f} radians
- Maximum single turn angle: {max_curvature:.4f} radians
- Turn angle variability (std): {std_curvature:.4f} radians
- Number of sharp turns (>0.3 rad / ~17deg): {sharp_turn_count}
- Bounding-box aspect ratio (width/height): {bbox_aspect_ratio:.3f}

Assess the failure risk of this scenario."""


def build_prompt(feature_record):
    """feature_record: one record from compute_features.py output (must NOT include outcome)."""
    user_prompt = USER_PROMPT_TEMPLATE.format(
        generator=feature_record["generator"],
        num_points=feature_record["num_points"],
        road_length=feature_record["road_length"],
        straight_line_distance=feature_record["straight_line_distance"],
        tortuosity=feature_record["tortuosity"],
        mean_curvature=feature_record["mean_curvature"],
        max_curvature=feature_record["max_curvature"],
        std_curvature=feature_record["std_curvature"],
        sharp_turn_count=feature_record["sharp_turn_count"],
        bbox_aspect_ratio=feature_record["bbox_aspect_ratio"],
    )
    return SYSTEM_PROMPT, user_prompt
