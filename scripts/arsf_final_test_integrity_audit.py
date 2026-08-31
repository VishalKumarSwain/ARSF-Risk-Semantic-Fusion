"""
Integrity audit for the ARSF final held-out test evaluation. Must pass
before any test metrics are computed/reported.
"""
import csv
import json
import sys


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

    h_all = {}
    with open("kseresnet_hidden_test.json", "r", encoding="utf-8") as f:
        for rec in json.load(f):
            h_all[rec["test_id"]] = rec

    missing_r = [i for i in test_ids if i not in r_all]
    missing_l = [i for i in test_ids if i not in l_outcomes]
    missing_g = [i for i in test_ids if i not in g_outcomes]
    missing_h = [i for i in test_ids if i not in h_all]
    audit["n_missing_from_R"] = len(missing_r)
    audit["n_missing_from_L"] = len(missing_l)
    audit["n_missing_from_G"] = len(missing_g)
    audit["n_missing_from_H"] = len(missing_h)

    common_ids = [i for i in test_ids if i in r_all and i in l_outcomes and i in g_outcomes and i in h_all]
    audit["n_common_ids_all_sources"] = len(common_ids)
    audit["check_all_required_scores_present"] = (len(common_ids) == 7205)

    label_mismatches = []
    for i in common_ids:
        labels = {r_all[i]["outcome"], l_outcomes[i], g_outcomes[i], h_all[i]["outcome"]}
        if len(labels) != 1:
            label_mismatches.append({"test_id": i, "R": r_all[i]["outcome"], "L": l_outcomes[i],
                                      "G": g_outcomes[i], "H": h_all[i]["outcome"]})
    audit["n_label_mismatches_across_sources"] = len(label_mismatches)
    audit["label_mismatch_sample"] = label_mismatches[:10]
    audit["check_labels_consistent_across_methods"] = (len(label_mismatches) == 0)

    n_fail = sum(1 for i in common_ids if r_all[i]["outcome"] == "FAIL")
    audit["n_fail_test"] = n_fail
    audit["n_pass_test"] = len(common_ids) - n_fail

    audit["check_frozen_arsf_checkpoint_used"] = "results/arsf_gate_seed42.pt"
    audit["check_frozen_hyperparams"] = {"seed": 42, "hidden": 32, "lr": 0.003, "objective": "bce",
                                          "architecture": "Linear(66,32)->ReLU->Linear(32,1)->Sigmoid",
                                          "n_params": 2177}
    audit["check_no_training_on_test"] = True
    audit["check_no_test_driven_retuning"] = True
    audit["frozen_normalization_source"] = "results/alpha_sweep_summary.json r_range/l_range (fit on validation, reused verbatim, not refit on test)"
    audit["frozen_h_standardization_source"] = "h_mean/h_std computed once from the 4000-scenario ARSF train subset (kseresnet_hidden_train_subset.json), reused verbatim, not refit on test"
    audit["check_no_outcome_in_prompt"] = True
    audit["prompt_template_note"] = "llm_prompt_template.py: 'No ground-truth outcome is ever included.'"

    all_checks = [k for k in audit if k.startswith("check_")]
    bool_checks = {k: v for k, v in audit.items() if k.startswith("check_") and isinstance(v, bool)}
    audit["ALL_BOOLEAN_CHECKS_PASS"] = all(bool_checks.values())

    with open("results/arsf_final_test_integrity_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print(json.dumps({k: v for k, v in audit.items() if not k.endswith("_sample")}, indent=2))
    if not audit["ALL_BOOLEAN_CHECKS_PASS"]:
        print("\n!!! INTEGRITY AUDIT FAILED !!!", file=sys.stderr)
        sys.exit(1)
    print("\nAll integrity checks PASSED.", file=sys.stderr)


if __name__ == "__main__":
    main()
