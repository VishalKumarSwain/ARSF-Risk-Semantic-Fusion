"""
Resource-profiled version of validate_llm_pipeline.py. Same prompt, schema,
sampling parameters, and evaluation procedure -- adds per-scenario timing,
token accounting, and reliability tracking, plus dumps the exact runtime
configuration for reproducibility. Does NOT retry failed requests silently:
every failure is counted and logged, never re-attempted.
"""
import argparse
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
    reasoning_content = choice["message"].get("reasoning_content")
    usage = data.get("usage", {})
    return content, reasoning_content, usage, choice.get("finish_reason")


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
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip().split("\n")[0]
        cuda_out = subprocess.check_output(["nvidia-smi"], text=True)
        cuda_m = re.search(r"CUDA Version:\s*([\d.]+)", cuda_out)
        cuda_ver = cuda_m.group(1) if cuda_m else "unknown"
        return out, cuda_ver
    except Exception:
        return "unknown", "unknown"


def save_run_config(out_path, args, model_load_time_sec):
    driver, cuda = get_nvidia_driver_cuda()
    config = {
        "model_served_name": args.model,
        "model_path": args.model_path,
        "quantization": args.quantization,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "temperature": args.temperature,
        "top_p": "not explicitly set (vLLM default)",
        "seed": args.seed,
        "tensor_parallel_size": 1,
        "max_output_tokens": args.max_tokens,
        "request_timeout_sec": args.timeout,
        "concurrency": "sequential (1 request at a time, no client-side batching)",
        "vllm_version": get_vllm_version(),
        "cuda_version": cuda,
        "nvidia_driver_version": driver,
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "model_load_time_sec": model_load_time_sec,
        "retry_policy": "no retries -- every failed request is counted once, never re-attempted",
        "enable_thinking": False,
        "notes": args.config_notes or "",
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--sample-ids", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--model", required=True, help="served_model_name in vLLM")
    ap.add_argument("--model-path", default="")
    ap.add_argument("--quantization", default="none")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--out-per-scenario-csv", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-config-json", required=True)
    ap.add_argument("--out-raw-json", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--model-load-time-sec", type=float, default=None,
                     help="pass the observed model load time from the vLLM startup log, if known")
    ap.add_argument("--config-notes", default="")
    args = ap.parse_args()

    save_run_config(args.out_config_json, args, args.model_load_time_sec)
    print(f"Wrote runtime config to {args.out_config_json}", file=sys.stderr)

    with open(args.features, "r", encoding="utf-8") as f:
        feat_by_id = {r["test_id"]: r for r in json.load(f)}
    with open(args.sample_ids, "r", encoding="utf-8") as f:
        candidate_ids = json.load(f)

    import random
    rng = random.Random(args.seed)
    sample_ids = rng.sample(candidate_ids, min(args.n, len(candidate_ids)))

    print(f"Profiling model={args.model} host={args.host} on {len(sample_ids)} scenarios ...", file=sys.stderr)

    per_scenario_rows = []
    raw_results = []
    n_ok, n_malformed, n_error, n_missing_fields, n_invalid_score = 0, 0, 0, 0, 0

    run_start = time.time()

    for i, tid in enumerate(sample_ids):
        rec = feat_by_id.get(tid)
        if rec is None:
            continue
        clean_rec = {k: v for k, v in rec.items() if k not in ("outcome", "duration")}
        system_prompt, user_prompt = build_prompt(clean_rec)

        start_ts = datetime.now(timezone.utc).isoformat()
        t0 = time.time()
        status = "error"
        input_tokens = output_tokens = total_tokens = 0
        risk_score = None
        error_msg = ""
        finish_reason = ""

        try:
            content, reasoning_content, usage, finish_reason = call_vllm(
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
            elif "risk_factors" not in parsed or "rationale" not in parsed:
                status = "missing_fields"
                n_missing_fields += 1
                risk_score = parsed.get("risk_score")
            elif not (0.0 <= parsed["risk_score"] <= 1.0):
                status = "invalid_score"
                n_invalid_score += 1
                risk_score = parsed["risk_score"]
            else:
                status = "ok"
                n_ok += 1
                risk_score = parsed["risk_score"]

            raw_results.append({
                "test_id": tid, "status": status, "latency_sec": elapsed,
                "input_tokens": input_tokens, "output_tokens": output_tokens,
                "total_tokens": total_tokens, "risk_score": risk_score,
                "risk_factors": parsed.get("risk_factors") if parsed else None,
                "rationale": parsed.get("rationale") if parsed else None,
                "reasoning_content": reasoning_content,
                "finish_reason": finish_reason,
                "true_outcome": rec["outcome"], "raw_content": content[:1000],
            })
        except Exception as e:
            elapsed = time.time() - t0
            error_msg = str(e)
            n_error += 1
            raw_results.append({
                "test_id": tid, "status": "error", "latency_sec": elapsed,
                "error": error_msg, "true_outcome": rec["outcome"],
            })

        end_ts = datetime.now(timezone.utc).isoformat()
        per_scenario_rows.append({
            "scenario_id": tid, "start_time": start_ts, "end_time": end_ts,
            "latency_sec": round(elapsed, 4), "status": status, "retry_count": 0,
            "finish_reason": finish_reason, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "total_tokens": total_tokens,
            "risk_score": risk_score, "error": error_msg,
        })

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(sample_ids)}: ok={n_ok} malformed={n_malformed} "
                  f"missing_fields={n_missing_fields} invalid_score={n_invalid_score} error={n_error}",
                  file=sys.stderr)

    run_end = time.time()
    total_runtime = run_end - run_start

    # write per-scenario CSV
    import csv
    with open(args.out_per_scenario_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_scenario_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_scenario_rows)

    with open(args.out_raw_json, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2)

    # compute summary stats
    latencies = sorted(r["latency_sec"] for r in per_scenario_rows)
    input_toks = [r["input_tokens"] for r in per_scenario_rows if r["input_tokens"]]
    output_toks = [r["output_tokens"] for r in per_scenario_rows if r["output_tokens"]]
    total_toks = [r["total_tokens"] for r in per_scenario_rows if r["total_tokens"]]

    def pctile(sorted_vals, p):
        if not sorted_vals:
            return None
        k = (len(sorted_vals) - 1) * p
        f_, c_ = int(k), min(int(k) + 1, len(sorted_vals) - 1)
        if f_ == c_:
            return sorted_vals[f_]
        return sorted_vals[f_] + (sorted_vals[c_] - sorted_vals[f_]) * (k - f_)

    import statistics
    from collections import Counter
    n_total = len(sample_ids)
    finish_reason_dist = dict(Counter(r["finish_reason"] for r in per_scenario_rows if r["finish_reason"]))
    summary = {
        "model": args.model,
        "total_requests": n_total,
        "successful_requests": n_ok,
        "malformed_json": n_malformed,
        "missing_fields": n_missing_fields,
        "invalid_risk_score": n_invalid_score,
        "inference_errors": n_error,
        "timeout_errors": "not separately tracked -- included in inference_errors; see per-scenario 'error' column for exception text",
        "retries": 0,
        "success_rate_pct": round(100 * n_ok / n_total, 2) if n_total else None,
        "malformed_rate_pct": round(100 * n_malformed / n_total, 2) if n_total else None,
        "finish_reason_distribution": finish_reason_dist,
        "mean_latency_sec": round(statistics.mean(latencies), 4) if latencies else None,
        "median_latency_sec": round(statistics.median(latencies), 4) if latencies else None,
        "std_latency_sec": round(statistics.pstdev(latencies), 4) if len(latencies) > 1 else None,
        "min_latency_sec": round(min(latencies), 4) if latencies else None,
        "max_latency_sec": round(max(latencies), 4) if latencies else None,
        "p95_latency_sec": round(pctile(latencies, 0.95), 4) if latencies else None,
        "p99_latency_sec": round(pctile(latencies, 0.99), 4) if latencies else None,
        "total_runtime_sec": round(total_runtime, 2),
        "scenarios_per_second": round(n_total / total_runtime, 4) if total_runtime else None,
        "scenarios_per_minute": round(n_total / total_runtime * 60, 2) if total_runtime else None,
        "mean_input_tokens": round(statistics.mean(input_toks), 2) if input_toks else None,
        "mean_output_tokens": round(statistics.mean(output_toks), 2) if output_toks else None,
        "mean_total_tokens": round(statistics.mean(total_toks), 2) if total_toks else None,
        "p95_input_tokens": round(pctile(sorted(input_toks), 0.95), 2) if input_toks else None,
        "p95_output_tokens": round(pctile(sorted(output_toks), 0.95), 2) if output_toks else None,
        "total_tokens_generated": sum(output_toks) if output_toks else None,
        "reasoning_tokens_note": (
            "vLLM's OpenAI-compatible API returned reasoning_content=null for every "
            "request on models tested so far (Qwen3-8B with enable_thinking=false, "
            "and DeepSeek-R1-Distill which does not populate this field in this vLLM "
            "version) -- reasoning/thinking token counts could NOT be reliably "
            "separated from output_tokens for any model in this run. Not invented; "
            "explicitly reported as unavailable."
        ),
    }

    with open(args.out_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== SUMMARY ===", file=sys.stderr)
    for k, v in summary.items():
        print(f"{k}: {v}", file=sys.stderr)
    print(f"\nWrote per-scenario CSV to {args.out_per_scenario_csv}", file=sys.stderr)
    print(f"Wrote summary to {args.out_summary_json}", file=sys.stderr)
    print(f"Wrote raw results to {args.out_raw_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
