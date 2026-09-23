"""
evaluate_scale_pipeline_accuracy_v3.py

Measures the accuracy of the EXACT scoring pipeline used by
find_clones_at_scale_v3.py (or _v4.py) -- TF-IDF+SVD content vectors, PLUS (new) the
same structure-similarity blend the script now supports -- against your
GitHub-confirmed ground truth in labeled_pairs.csv.

WHY THIS VERSION VALIDATES TWO SEPARATE THRESHOLDS
------------------------------------------------------
find_clones_at_scale_v3.py can gate its results on either CONTENT similarity
or OVERALL similarity (content + structure, averaged, matching Pandu's
S_overall). These are genuinely different score distributions -- a tau
validated for one does NOT automatically transfer to the other (the same
mistake as reusing one method's tau for a different method). This script
runs the full naive-fit + cross-validation procedure TWICE: once against
content_score, once against overall_score, so you get two independently
validated tau values -- one for each --threshold-on mode in
find_clones_at_scale_v3.py.

METHODOLOGY (per score type)
------------------------------
1. Embed the ENTIRE corpus exactly as the selected engine does
   (shared TF-IDF vectorizer + SVD across every role in --datasets-dir).
2. Compute each role's structure path-set exactly as find_clones_at_scale_v3.py
   does (get_role_paths / structure_similarity), unless --no-structure.
3. For every labeled pair: content_score = cosine similarity of the
   corpus-wide vectors; structure_score = Jaccard of relative file paths;
   overall_score = (content_score + structure_score) / 2.
4. Report BOTH naive fit (reference only -- same data picks and evaluates
   tau) and cross-validated (the number to trust) for content_score AND
   overall_score.

USAGE
-----
    python3 evaluate_scale_pipeline_accuracy_v3.py --datasets-dir datasets/mysql --pairs labeled_pairs_all3.csv --tech mysql --folds 10

    # content-only, skip structure entirely (faster, matches old behavior):
    python3 evaluate_scale_pipeline_accuracy_v3.py --datasets-dir datasets/mysql --pairs labeled_pairs_all3.csv --tech mysql --folds 10 --no-structure

    python3 evaluate_scale_pipeline_accuracy_v3.py --datasets-dir datasets/mysql --pairs labeled_pairs.csv --tech mysql --folds 10
    python3 evaluate_scale_pipeline_accuracy_v3.py --datasets-dir datasets/jenkins --pairs labeled_pairs.csv --tech jenkins --folds 10
    python3 evaluate_scale_pipeline_accuracy_v3.py --datasets-dir datasets/elastic_search --pairs labeled_pairs.csv --tech elastic_search --folds 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
)

# Which scoring engine to mirror. v4 includes every file (EXCLUDED_FILENAMES
# is empty); v3 excludes legal boilerplate (LICENSE/COPYING/NOTICE) from the
# TF-IDF text. The tau this script validates is only valid for the engine
# that produced it, so this MUST match the find_clones_at_scale_vN.py you
# deploy. Read from sys.argv directly because the import has to happen
# before argparse runs.
if "--exclude-license" in sys.argv:
    from find_clones_at_scale_v3 import load_roles, embed_roles_tfidf, get_role_paths, structure_similarity, filter_boilerplate, BOILERPLATE_MODES
    ENGINE = "v3 (LICENSE/COPYING/NOTICE excluded from content)"
else:
    from find_clones_at_scale_v4 import load_roles, embed_roles_tfidf, get_role_paths, structure_similarity, filter_boilerplate, BOILERPLATE_MODES
    ENGINE = "v4 (all files included)"


def read_labeled_pairs(csv_path, tech=None):
    import csv
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if tech and row["technology"] != tech:
                continue
            rows.append({
                "role_a": row["role_a"].strip(),
                "role_b": row["role_b"].strip(),
                "is_fork": int(row["is_fork"]),
            })
    return rows


def score_labeled_pairs(pairs, role_vectors, role_paths=None):
    """
    Scores each labeled pair using PRECOMPUTED corpus-wide vectors (the
    same ones the FAISS index was built from) -- not a fresh, separate
    computation per pair. If role_paths is provided, also computes
    structure and overall scores, identically to find_clones_at_scale_v3.py.
    """
    scored = []
    for p in pairs:
        va = role_vectors.get(p["role_a"])
        vb = role_vectors.get(p["role_b"])
        if va is None or vb is None:
            missing = p["role_a"] if va is None else p["role_b"]
            print(f"  Skipping pair ({p['role_a']}, {p['role_b']}): "
                  f"'{missing}' not found in --datasets-dir corpus")
            continue
        # Vectors are already L2-normalized (see embed_roles_tfidf), so
        # inner product = cosine similarity directly.
        content_score = float(np.dot(va, vb)) * 100

        row = {"role_a": p["role_a"], "role_b": p["role_b"],
               "is_fork": p["is_fork"], "content_score": content_score}

        if role_paths is not None:
            pa = role_paths.get(p["role_a"])
            pb = role_paths.get(p["role_b"])
            if pa is not None and pb is not None:
                structure_score = structure_similarity(pa, pb)
                row["structure_score"] = structure_score
                row["overall_score"] = (content_score + structure_score) / 2
            else:
                row["structure_score"] = None
                row["overall_score"] = content_score

        scored.append(row)
    return scored


def descriptive_stats(df, score_col, label):
    """
    Mean/median similarity of KNOWN FORK PAIRS only, using the ACTUAL
    deployed pipeline's scoring (corpus-wide TF-IDF+SVD, and structure if
    enabled) -- not the simplified pairwise scoring
    evaluate_accuracy_pandu_style.py uses. This is the direct check for
    whether Pandu's approach (eyeball the mean/median, pick a round
    number below it) would actually work HERE, on the real pipeline you
    intend to deploy -- before spending time on full cross-validation.
    """
    positives = df[df["is_fork"] == 1][score_col]
    negatives = df[df["is_fork"] == 0][score_col]
    mean = round(positives.mean(), 2) if len(positives) else float("nan")
    median = round(positives.median(), 2) if len(positives) else float("nan")
    neg_mean = round(negatives.mean(), 2) if len(negatives) else float("nan")
    print(f"\n  {label} -- known fork pairs (n={len(positives)}): mean={mean}%  median={median}%")
    print(f"  {label} -- known non-fork pairs (n={len(negatives)}): mean={neg_mean}%")
    gap = mean - neg_mean if (mean == mean and neg_mean == neg_mean) else float("nan")
    if gap == gap:
        verdict = "GOOD separation" if gap > 15 else ("THIN separation -- a simple round number is risky" if gap > 0 else "OVERLAPPING -- no round number will work well")
        print(f"  {label} -- gap between fork mean and non-fork mean: {gap:.2f} points  ({verdict})")
    return {"score_type": label, "fork_mean": mean, "fork_median": median, "nonfork_mean": neg_mean}


def sweep_best_threshold(scores, labels):
    best_tau, best_f1 = 0.0, -1.0
    for tau in sorted(set(scores)):
        preds = [1 if s >= tau else 0 for s in scores]
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_tau = f1, tau
    return best_tau


def naive_fit_evaluation(df, score_col, label):
    scores, labels = df[score_col].tolist(), df["is_fork"].tolist()
    tau = sweep_best_threshold(scores, labels)
    preds = [1 if s >= tau else 0 for s in scores]
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan")
    result = {
        "score_type": label, "tau": round(tau, 2), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": round(accuracy_score(labels, preds), 3),
        "precision": round(precision_score(labels, preds, zero_division=0), 3),
        "recall": round(recall_score(labels, preds, zero_division=0), 3),
        "f1": round(f1_score(labels, preds, zero_division=0), 3),
        "auc": round(auc, 3) if auc == auc else auc,
    }
    print(f"\n  NAIVE FIT on {label} (tau chosen AND evaluated on same {len(df)} pairs -- for reference only)")
    print(f"    tau={result['tau']}  Accuracy={result['accuracy']}  Precision={result['precision']}  "
          f"Recall={result['recall']}  F1={result['f1']}  AUC={result['auc']}")
    return result


def cross_validated_evaluation(df, score_col, label, n_folds, seed=42):
    scores = np.array(df[score_col].tolist(), dtype=float)
    labels = np.array(df["is_fork"].tolist(), dtype=int)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_rows = []
    for fold_i, (train_idx, test_idx) in enumerate(skf.split(scores, labels), 1):
        tau = sweep_best_threshold(scores[train_idx], labels[train_idx])
        preds = (scores[test_idx] >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels[test_idx], preds, labels=[0, 1]).ravel()
        fold_rows.append({
            "fold": fold_i, "tau": tau,
            "accuracy": accuracy_score(labels[test_idx], preds),
            "precision": precision_score(labels[test_idx], preds, zero_division=0),
            "recall": recall_score(labels[test_idx], preds, zero_division=0),
            "f1": f1_score(labels[test_idx], preds, zero_division=0),
        })

    fold_df = pd.DataFrame(fold_rows)
    auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan")

    result = {
        "score_type": label,
        "mean_tau": round(fold_df["tau"].mean(), 2),
        "accuracy_mean": round(fold_df["accuracy"].mean(), 3), "accuracy_std": round(fold_df["accuracy"].std(), 3),
        "precision_mean": round(fold_df["precision"].mean(), 3), "precision_std": round(fold_df["precision"].std(), 3),
        "recall_mean": round(fold_df["recall"].mean(), 3), "recall_std": round(fold_df["recall"].std(), 3),
        "f1_mean": round(fold_df["f1"].mean(), 3), "f1_std": round(fold_df["f1"].std(), 3),
        "auc": round(auc, 3) if auc == auc else auc,
    }
    print(f"\n  CROSS-VALIDATED on {label} ({n_folds}-fold, tau chosen on training folds only -- TRUST THIS ONE)")
    print(f"    mean_tau={result['mean_tau']}")
    print(f"    Accuracy  = {result['accuracy_mean']} +/- {result['accuracy_std']}")
    print(f"    Precision = {result['precision_mean']} +/- {result['precision_std']}")
    print(f"    Recall    = {result['recall_mean']} +/- {result['recall_std']}")
    print(f"    F1        = {result['f1_mean']} +/- {result['f1_std']}")
    print(f"    AUC (pooled, threshold-free) = {result['auc']}")
    return result, fold_df


def main():
    ap = argparse.ArgumentParser(description="Validates BOTH a content-only tau and an overall (content+structure) tau against ground truth.")
    ap.add_argument("--exclude-license", action="store_true",
                     help="Mirror find_clones_at_scale_v3.py (LICENSE/COPYING/NOTICE excluded from "
                          "the TF-IDF text) instead of v4. MUST match the detector you deploy.")
    ap.add_argument("--datasets-dir", required=True, help="Same corpus folder used with find_clones_at_scale_v3.py")
    ap.add_argument("--pairs", required=True, help="labeled_pairs.csv")
    ap.add_argument("--tech", default=None, help="Filter labeled pairs to this technology")
    ap.add_argument("--svd-components", type=int, default=300,
                     help="Must match what you used with find_clones_at_scale_v3.py for a true apples-to-apples result")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--filter-content-boilerplate", action="store_true",
                    help="Remove `ansible-galaxy init` TEMPLATE TEXT from the TF-IDF content. "
                         "MUST match find_clones_at_scale_v3.py, exactly like "
                         "--svd-components and --boilerplate-mode: the tau this script "
                         "produces is only valid for the engine it mirrors.")
    ap.add_argument("--filter-boilerplate", action="store_true",
                     help="Exclude boilerplate paths from structure similarity. MUST match "
                          "the settings used on find_clones_at_scale_v3.py, or the tau this "
                          "produces will not apply to the detector's scores.")
    ap.add_argument("--boilerplate-mode", choices=list(BOILERPLATE_MODES), default="frequency",
                     help="How boilerplate is defined; must match the detector.")
    ap.add_argument("--boiler-frac", type=float, default=0.5,
                     help="Only used by --boilerplate-mode frequency (default 0.5)")
    ap.add_argument("--no-structure", action="store_true", help="Content-only, skip structure/overall entirely")
    ap.add_argument("--out-dir", default="scale_pipeline_accuracy_output")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading corpus from {args.datasets_dir}...")
    root_dir = Path(args.datasets_dir)
    roles = load_roles(root_dir, strip_content=args.filter_content_boilerplate)
    print(f"Loaded {len(roles)} roles")

    print(f"\nFitting TF-IDF + SVD across the whole corpus "
          f"(target {args.svd_components} dense dims) -- same as the selected engine...")
    role_names, matrix = embed_roles_tfidf(roles, args.svd_components)
    role_vectors = {name: matrix[i] for i, name in enumerate(role_names)}

    role_paths = None
    if not args.no_structure:
        print("Computing directory structure (relative file paths) -- same as the selected engine...")
        role_paths = get_role_paths(root_dir, role_names)
        if args.filter_boilerplate:
            role_paths, _ = filter_boilerplate(role_paths, args.boilerplate_mode,
                                               args.boiler_frac)

    pairs = read_labeled_pairs(args.pairs, args.tech)
    print(f"\nLoaded {len(pairs)} labeled pairs"
          + (f" for tech='{args.tech}'" if args.tech else ""))

    scored = score_labeled_pairs(pairs, role_vectors, role_paths)
    df = pd.DataFrame(scored)
    if df.empty:
        print("No labeled pairs could be scored -- check --datasets-dir matches the corpus used to build labeled_pairs.csv.")
        return
    n_pos = df["is_fork"].sum()
    print(f"Scored {len(df)}/{len(pairs)} pairs ({n_pos} forks, {len(df) - n_pos} non-forks)")

    score_cols = [("content_score", "CONTENT")]
    if role_paths is not None:
        score_cols.append(("overall_score", "OVERALL (content+structure)"))

    print(f"\n{'=' * 70}")
    print("STEP 1 -- DESCRIPTIVE STATS ON KNOWN FORK PAIRS (Pandu-style check, real pipeline scoring)")
    print(f"{'=' * 70}")
    all_stats = [descriptive_stats(df, col, label) for col, label in score_cols]
    pd.DataFrame(all_stats).to_csv(out_dir / "descriptive_stats.csv", index=False)

    all_naive, all_cv = [], []
    for col, label in score_cols:
        print(f"\n{'=' * 70}")
        print(f"STEP 2 -- VALIDATING TAU FOR: {label}")
        print(f"{'=' * 70}")
        all_naive.append(naive_fit_evaluation(df, col, label))
        cv_result, fold_df = cross_validated_evaluation(df, col, label, args.folds)
        all_cv.append(cv_result)
        fold_df.to_csv(out_dir / f"cv_folds_{col}.csv", index=False)

    df.to_csv(out_dir / "scored_pairs.csv", index=False)
    pd.DataFrame(all_naive).to_csv(out_dir / "naive_fit_summary.csv", index=False)
    pd.DataFrame(all_cv).to_csv(out_dir / "cv_summary.csv", index=False)

    print(f"\n{'=' * 70}")
    print("SUMMARY -- use these mean_tau values with find_clones_at_scale_v3.py's matching --threshold-on")
    print(f"{'=' * 70}")
    for r in all_cv:
        print(f"  {r['score_type']:30s} mean_tau={r['mean_tau']}  "
              f"F1={r['f1_mean']}+/-{r['f1_std']}  AUC={r['auc']}")

    print(f"\nSaved results to {out_dir}/")


if __name__ == "__main__":
    main()
