"""
Validate the LLM (served via vLLM's OpenAI-compatible API) inference
pipeline on a small sample (~100 scenarios) before committing to the
full 1,000-scenario Stage 1 screening run. Checks: does the model
respond, is output valid JSON matching the schema, malformed-output
rate, per-scenario latency.
"""
import argparse
import json
import re
import sys
import time

import requests

sys.path.insert(0, ".")
from llm_prompt_template import build_prompt


def call_vllm(host, model, system_prompt, user_prompt, timeout=60, max_tokens=300):
    resp = requests.post(
        f"{host}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "seed": 42,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def try_parse(raw_text):
    m = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if "risk_score" not in obj:
        return None
    try:
        obj["risk_score"] = float(obj["risk_score"])
    except (TypeError, ValueError):
        return None
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--sample-ids", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=300)
    args = ap.parse_args()

    with open(args.features, "r", encoding="utf-8") as f:
        feat_by_id = {r["test_id"]: r for r in json.load(f)}
    with open(args.sample_ids, "r", encoding="utf-8") as f:
        candidate_ids = json.load(f)

    import random
    rng = random.Random(args.seed)
    sample_ids = rng.sample(candidate_ids, min(args.n, len(candidate_ids)))

    print(f"Testing model={args.model} host={args.host} on {len(sample_ids)} scenarios ...", file=sys.stderr)

    results = []
    n_ok, n_malformed, n_error = 0, 0, 0
    latencies = []

    for i, tid in enumerate(sample_ids):
        rec = feat_by_id.get(tid)
        if rec is None:
            continue
        clean_rec = {k: v for k, v in rec.items() if k not in ("outcome", "duration")}
        system_prompt, user_prompt = build_prompt(clean_rec)

        t0 = time.time()
        try:
            raw = call_vllm(args.host, args.model, system_prompt, user_prompt, max_tokens=args.max_tokens)
            elapsed = time.time() - t0
            latencies.append(elapsed)
            parsed = try_parse(raw)
            if parsed is None:
                n_malformed += 1
                results.append({"test_id": tid, "status": "malformed", "raw": raw[:500], "latency": elapsed})
            else:
                n_ok += 1
                results.append({
                    "test_id": tid, "status": "ok", "latency": elapsed,
                    "risk_score": parsed["risk_score"],
                    "risk_factors": parsed.get("risk_factors"),
                    "rationale": parsed.get("rationale"),
                    "true_outcome": rec["outcome"],
                })
        except Exception as e:
            n_error += 1
            results.append({"test_id": tid, "status": "error", "error": str(e)})

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(sample_ids)}: ok={n_ok} malformed={n_malformed} error={n_error}", file=sys.stderr)

    print(f"\n=== VALIDATION SUMMARY ===", file=sys.stderr)
    print(f"Total: {len(sample_ids)}", file=sys.stderr)
    print(f"OK: {n_ok} ({100*n_ok/len(sample_ids):.1f}%)", file=sys.stderr)
    print(f"Malformed: {n_malformed} ({100*n_malformed/len(sample_ids):.1f}%)", file=sys.stderr)
    print(f"Errors: {n_error} ({100*n_error/len(sample_ids):.1f}%)", file=sys.stderr)
    if latencies:
        print(f"Latency: mean={sum(latencies)/len(latencies):.2f}s min={min(latencies):.2f}s max={max(latencies):.2f}s", file=sys.stderr)
        est_1000 = sum(latencies) / len(latencies) * 1000
        print(f"Estimated time for 1,000-scenario Stage-1 run: {est_1000/60:.1f} minutes", file=sys.stderr)

    ok_results = [r for r in results if r["status"] == "ok"]
    if ok_results:
        fail_scores = [r["risk_score"] for r in ok_results if r["true_outcome"] == "FAIL"]
        pass_scores = [r["risk_score"] for r in ok_results if r["true_outcome"] == "PASS"]
        if fail_scores and pass_scores:
            print(f"\nSanity check (informal):", file=sys.stderr)
            print(f"  Mean risk_score for true FAIL: {sum(fail_scores)/len(fail_scores):.3f} (n={len(fail_scores)})", file=sys.stderr)
            print(f"  Mean risk_score for true PASS: {sum(pass_scores)/len(pass_scores):.3f} (n={len(pass_scores)})", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote full results to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
