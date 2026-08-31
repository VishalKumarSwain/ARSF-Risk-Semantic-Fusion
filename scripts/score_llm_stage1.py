"""
Stage-1 LLM scoring: runs a model over the fixed 1,000-scenario Stage-1 set,
producing BOTH (a) the standard resource-profile CSV/summary/config (same
instrumentation as the 100-scenario validation runs) and (b) a rankings CSV
(rank,scenario_id,score,true_outcome) in the exact format compute_ranking_metrics.py
expects. Same prompt/schema/settings as validated in the 100-scenario pass --
nothing is changed after seeing those results.
"""
import argparse
import csv
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

import requests

sys.path.insert(0, ".")
from llm_prompt_template import build_prompt


def call_vllm(host, model, system_prompt, user_prompt, timeout, max_tokens, temperature, seed):
    resp = requests.post(
        f"{host}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    content = choice["message"]["content"]
    usage = data.get("usage", {})
    return content, usage, choice.get("finish_reason")


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


def get_vllm_version():
    try:
        import vllm
        return vllm.__version__
    except Exception:
        return "unknown"


def get_nvidia_driver_cuda():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip().split("\n")[0]
        cuda_out = subprocess.check_output(["nvidia-smi"], text=True)
        m = re.search(r"CUDA Version:\s*([\d.]+)", cuda_out)
        return out, (m.group(1) if m else "unknown")
    except Exception:
        return "unknown", "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--stage1-ids", required=True)
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-path", default="")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--out-per-scenario-csv", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-config-json", required=True)
    ap.add_argument("--out-ranking-csv", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--model-load-time-sec", type=float, default=None)
    ap.add_argument("--config-notes", default="")
    args = ap.parse_args()

    driver, cuda = get_nvidia_driver_cuda()
    config = {
        "model_served_name": args.model, "model_path": args.model_path,
        "dtype": args.dtype, "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "temperature": args.temperature, "seed": args.seed,
        "max_output_tokens": args.max_tokens, "request_timeout_sec": args.timeout,
        "vllm_version": get_vllm_version(), "cuda_version": cuda,
        "nvidia_driver_version": driver, "python_version": platform.python_version(),
        "os": platform.platform(), "model_load_time_sec": args.model_load_time_sec,
        "retry_policy": "no retries", "enable_thinking": False,
        "stage": "Stage-1 1000-scenario screening",
        "notes": args.config_notes,
    }
    with open(args.out_config_json, "w") as f:
        json.dump(config, f, indent=2)

    with open(args.features) as f:
        feat_by_id = {r["test_id"]: r for r in json.load(f)}
    with open(args.stage1_ids) as f:
        stage1_ids = json.load(f)

    print(f"Scoring {args.model} on {len(stage1_ids)} Stage-1 scenarios ...", file=sys.stderr)

    per_scenario_rows = []
    ranking_rows = []
    n_ok = n_malformed = n_error = 0
    run_start = time.time()

    for i, tid in enumerate(stage1_ids):
        rec = feat_by_id[tid]
        clean_rec = {k: v for k, v in rec.items() if k not in ("outcome", "duration")}
        system_prompt, user_prompt = build_prompt(clean_rec)

        start_ts = datetime.now(timezone.utc).isoformat()
        t0 = time.time()
        status, risk_score, finish_reason, error_msg = "error", None, "", ""
        input_tokens = output_tokens = total_tokens = 0

        try:
            content, usage, finish_reason = call_vllm(
                args.host, args.model, system_prompt, user_prompt,
                timeout=args.timeout, max_tokens=args.max_tokens,
                temperature=args.temperature, seed=args.seed,
            )
            elapsed = time.time() - t0
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
            parsed = try_parse(content)
            if parsed is None:
                status = "malformed"
                n_malformed += 1
                risk_score = 0.0  # neutral fallback score for ranking purposes; excluded from reliability stats
            else:
                status = "ok"
                n_ok += 1
                risk_score = parsed["risk_score"]
        except Exception as e:
            elapsed = time.time() - t0
            error_msg = str(e)
            n_error += 1
            risk_score = 0.0

        end_ts = datetime.now(timezone.utc).isoformat()
        per_scenario_rows.append({
            "scenario_id": tid, "start_time": start_ts, "end_time": end_ts,
            "latency_sec": round(elapsed, 4), "status": status, "retry_count": 0,
            "finish_reason": finish_reason, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "total_tokens": total_tokens,
            "risk_score": risk_score, "error": error_msg,
        })
        ranking_rows.append({"scenario_id": tid, "score": risk_score, "true_outcome": rec["outcome"]})

        if (i + 1) % 50 == 0:
            elapsed_total = time.time() - run_start
            print(f"  {i+1}/{len(stage1_ids)}: ok={n_ok} malformed={n_malformed} error={n_error} "
                  f"elapsed={elapsed_total:.1f}s", file=sys.stderr)

    total_runtime = time.time() - run_start

    ranking_rows.sort(key=lambda r: (-r["score"], r["scenario_id"]))
    for i, r in enumerate(ranking_rows):
        r["rank"] = i + 1
    with open(args.out_ranking_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "scenario_id", "score", "true_outcome"])
        writer.writeheader()
        writer.writerows(ranking_rows)

    with open(args.out_per_scenario_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_scenario_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_scenario_rows)

    import statistics
    latencies = sorted(r["latency_sec"] for r in per_scenario_rows)
    n_total = len(stage1_ids)
    summary = {
        "model": args.model, "total_requests": n_total, "successful_requests": n_ok,
        "malformed_json": n_malformed, "inference_errors": n_error,
        "success_rate_pct": round(100 * n_ok / n_total, 2),
        "mean_latency_sec": round(statistics.mean(latencies), 4),
        "median_latency_sec": round(statistics.median(latencies), 4),
        "std_latency_sec": round(statistics.pstdev(latencies), 4) if len(latencies) > 1 else None,
        "min_latency_sec": round(min(latencies), 4), "max_latency_sec": round(max(latencies), 4),
        "total_runtime_sec": round(total_runtime, 2),
        "scenarios_per_minute": round(n_total / total_runtime * 60, 2) if total_runtime else None,
    }
    with open(args.out_summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== STAGE-1 SUMMARY: {args.model} ===", file=sys.stderr)
    for k, v in summary.items():
        print(f"{k}: {v}", file=sys.stderr)
    print(f"Wrote ranking to {args.out_ranking_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
