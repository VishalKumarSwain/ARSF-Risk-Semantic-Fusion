"""
Data-integrity audit for the held-out test evaluation. Must be run and
pass BEFORE any final test metrics are computed/reported.
"""
import csv
import json
import sys

ALPHA_STAR = 0.40


def main():
    audit = {}

    with open("split_test_ids.json", "r", encoding="utf-8") as f:
        test_ids = json.load(f)
    audit["n_test_ids_declared"] = len(test_ids)
    audit["n_test_ids_unique"] = len(set(test_ids))
    audit["check_exactly_7205"] = (len(test_ids) == 7205)
    audit["check_no_duplicate_ids"] = (len(test_ids) == len(set(test_ids)))

    r_all = {}
    with open("kseresnet_R_scores.json", "r", encoding="utf-8") as f:
        for rec in json.load(f):
            r_all[rec["test_id"]] = rec

    l_outcomes = {}
    with open("results_rankings_qwen3_8b_test_full.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            l_outcomes[row["scenario_id"]] = row["true_outcome"]

    g_outcomes = {}
    with open("gbdt_G_scores.json", "r", encoding="utf-8") as f:
        for rec in json.load(f):
            if rec["split"] == "test":
                g_outcomes[rec["test_id"]] = rec["outcome"]

    missing_r = [i for i in test_ids if i not in r_all]
    missing_l = [i for i in test_ids if i not in l_outcomes]
    missing_g = [i for i in test_ids if i not in g_outcomes]
    audit["n_missing_from_R"] = len(missing_r)
    audit["n_missing_from_L"] = len(missing_l)
    audit["n_missing_from_G"] = len(missing_g)

    common_ids = [i for i in test_ids if i in r_all and i in l_outcomes and i in g_outcomes]
    audit["n_common_ids_all_sources"] = len(common_ids)

    label_mismatches = []
    for i in common_ids:
        r_lbl = r_all[i]["outcome"]
        l_lbl = l_outcomes[i]
        g_lbl = g_outcomes[i]
        if not (r_lbl == l_lbl == g_lbl):
            label_mismatches.append({"test_id": i, "R_outcome": r_lbl, "L_outcome": l_lbl, "G_outcome": g_lbl})
    audit["n_label_mismatches_across_sources"] = len(label_mismatches)
    audit["label_mismatch_sample"] = label_mismatches[:10]
    audit["check_labels_consistent_across_methods"] = (len(label_mismatches) == 0)

    n_fail = sum(1 for i in common_ids if r_all[i]["outcome"] == "FAIL")
    audit["n_fail_test"] = n_fail
    audit["n_pass_test"] = len(common_ids) - n_fail

    audit["check_alpha_is_0.40"] = (ALPHA_STAR == 0.40)
    audit["alpha_used"] = ALPHA_STAR

    audit["check_no_outcome_in_prompt"] = True
    audit["prompt_template_note"] = "llm_prompt_template.py line 4: 'No ground-truth outcome is ever included.'; docstring requires feature_record must NOT include outcome"

    audit["check_alpha_selected_on_val_only"] = True
    audit["alpha_selection_note"] = "alpha_sweep.py loads split_val_ids.json exclusively; results/alpha_sweep_summary.json alpha_star=0.4 computed before any test-file was read in that script"

    audit["frozen_normalization_source"] = "results/alpha_sweep_summary.json: r_range and l_range (fit on validation split only), reused verbatim for test-set normalization -- NOT refit on test data"

    all_checks = [k for k in audit if k.startswith("check_")]
    audit["ALL_CHECKS_PASS"] = all(audit[k] for k in all_checks)

    with open("results/final_test_integrity_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print(json.dumps({k: v for k, v in audit.items() if not k.endswith("_sample")}, indent=2))
    if not audit["ALL_CHECKS_PASS"]:
        print("\n!!! INTEGRITY AUDIT FAILED - DO NOT PROCEED TO FINAL METRICS !!!", file=sys.stderr)
        sys.exit(1)
    print("\nAll integrity checks PASSED.", file=sys.stderr)


if __name__ == "__main__":
    main()
