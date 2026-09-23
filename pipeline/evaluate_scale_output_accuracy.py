"""
evaluate_scale_output_accuracy.py

Computes TP/FP/FN/TN, accuracy, precision, recall, F1 (and AUC where
possible) for the ACTUAL clone_pairs.csv file produced by
find_clones_at_scale_v3.py -- the real, deployed output, exactly as it was
generated (including whatever top-k retrieval and threshold you actually
ran it with) -- checked directly against your GitHub-confirmed ground
truth in labeled_pairs.csv. This is the same shape as Pandu's Table 2
validation, applied to your real pipeline's real output file.

WHY THIS IS DIFFERENT FROM THE OTHER TWO ACCURACY SCRIPTS
------------------------------------------------------------
evaluate_accuracy_cv.py (superseded; in previous_work/) -- PAIRWISE-only TF-IDF
                                  (fit on just 2 docs at a time)
evaluate_scale_pipeline_accuracy_v3.py
                               -- re-embeds the corpus and computes a
                                  FRESH similarity score for every labeled
                                  pair directly, bypassing top-k retrieval
                                  entirely (an "oracle" version -- assumes
                                  every pair got compared)
evaluate_scale_output_accuracy.py (this script)
                               -- checks the REAL clone_pairs.csv file
                                  you already generated. If a labeled
                                  pair isn't in that file, it counts as a
                                  negative prediction -- whether that's
                                  because its score fell below your
                                  threshold, OR because top-k retrieval
                                  never even checked it. This is the
                                  truest measure of "how accurate is what
                                  I actually got," including every real
                                  limitation of the deployed pipeline.

WHAT THIS CANNOT DO
--------------------
AUC needs a continuous score for every pair, including ones NOT found.
clone_pairs.csv only records scores for pairs that were actually found --
for a labeled pair that's missing from it, we don't know whether its true
score was just below threshold or nowhere near it, since top-k retrieval
may never have compared it at all. So this script reports AUC only over
the SUBSET of labeled pairs that happen to have a recorded score (found
pairs), clearly labeled as such -- for a full, unbiased AUC across every
labeled pair, use evaluate_scale_pipeline_accuracy_v3.py instead.

INPUT
-----
labeled_pairs.csv : technology,role_a,role_b,is_fork
clone_pairs.csv   : role_a,role_b,similarity_pct  (from find_clones_at_scale_v3.py)

USAGE
-----
    python3 evaluate_scale_output_accuracy.py --labeled labeled_pairs.csv --clone-pairs clone_scale_output/clone_pairs.csv --tech mysql
"""

import argparse
import csv
from pathlib import Path

from sklearn.metrics import confusion_matrix, roc_auc_score


def load_labeled_pairs(csv_path, tech=None):
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


# find_clones_at_scale_v3/v4 emit content_/structure_/overall_similarity_pct
# rather than a single similarity_pct column. Prefer whichever the detector
# actually gated on, so the score reported here is the one that made the
# include/exclude decision.
SCORE_COLUMN_PREFERENCE = [
    "similarity_pct", "content_similarity_pct", "overall_similarity_pct",
]


def pick_score_column(fieldnames, requested=None):
    if requested:
        if requested not in fieldnames:
            raise SystemExit(f"--score-column '{requested}' not in {csv_path_hint(fieldnames)}")
        return requested
    for c in SCORE_COLUMN_PREFERENCE:
        if c in fieldnames:
            return c
    raise SystemExit(f"No recognised score column found; got {list(fieldnames)}. "
                     f"Pass --score-column explicitly.")


def csv_path_hint(fieldnames):
    return f"available columns: {list(fieldnames)}"


def load_found_pairs(csv_path, score_column=None):
    """Returns {frozenset({role_a, role_b}): score}"""
    found = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        col = pick_score_column(reader.fieldnames or [], score_column)
        print(f"  Using score column: {col}")
        for row in reader:
            key = frozenset({row["role_a"], row["role_b"]})
            found[key] = float(row[col])
    return found


def main():
    ap = argparse.ArgumentParser(
        description="Confusion matrix + accuracy metrics for the actual find_clones_at_scale_v3.py output.")
    ap.add_argument("--labeled", required=True, help="labeled_pairs.csv (ground truth)")
    ap.add_argument("--clone-pairs", required=True, help="clone_pairs.csv from find_clones_at_scale_v3.py")
    ap.add_argument("--tech", default=None, help="Filter labeled pairs to this technology")
    ap.add_argument("--score-column", default=None,
                     help="Which column holds the score the detector gated on "
                          "(default: auto-detect similarity_pct > content_ > overall_).")
    ap.add_argument("--out", default="scale_output_accuracy.csv")
    args = ap.parse_args()

    labeled = load_labeled_pairs(args.labeled, args.tech)
    found = load_found_pairs(args.clone_pairs, args.score_column)

    n_pos = sum(p["is_fork"] for p in labeled)
    n_neg = len(labeled) - n_pos
    print(f"Loaded {len(labeled)} labeled pairs"
          + (f" for tech='{args.tech}'" if args.tech else "")
          + f"  ({n_pos} confirmed forks, {n_neg} confirmed non-forks)")
    print(f"Loaded {len(found)} pairs actually found by the scale run\n")

    y_true, y_pred = [], []
    scores_for_auc, labels_for_auc = [], []
    rows_out = []

    for p in labeled:
        key = frozenset({p["role_a"], p["role_b"]})
        predicted = 1 if key in found else 0
        y_true.append(p["is_fork"])
        y_pred.append(predicted)

        score = found.get(key)
        if score is not None:
            scores_for_auc.append(score)
            labels_for_auc.append(p["is_fork"])

        rows_out.append({
            "role_a": p["role_a"], "role_b": p["role_b"],
            "true_is_fork": p["is_fork"], "predicted": predicted,
            "similarity_pct": score if score is not None else "",
        })

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / len(y_true) if y_true else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"{'=' * 60}")
    print("ACTUAL DEPLOYED OUTPUT vs. GROUND TRUTH")
    print(f"{'=' * 60}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Accuracy  = {accuracy:.3f}")
    print(f"  Precision = {precision:.3f}   (of pairs flagged as clones, how many really are)")
    print(f"  Recall    = {recall:.3f}   (of real clones, how many did we actually catch)")
    print(f"  F1        = {f1:.3f}")

    if len(set(labels_for_auc)) > 1:
        auc = roc_auc_score(labels_for_auc, scores_for_auc)
        print(f"\n  AUC (over the {len(scores_for_auc)} labeled pairs that HAVE a recorded")
        print(f"  score in clone_pairs.csv -- excludes pairs top-k never compared): {auc:.3f}")
        print("  For a full, unbiased AUC across every labeled pair, use")
        print("  evaluate_scale_pipeline_accuracy_v3.py instead.")
    else:
        print("\n  AUC: not computable (need both classes represented among found-pair scores)")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["role_a", "role_b", "true_is_fork", "predicted", "similarity_pct"])
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nSaved per-pair detail to {args.out}")

    if fp > 0:
        print(f"\n{fp} FALSE POSITIVE(S) -- confirmed non-fork pairs that got flagged as clones:")
        for r in rows_out:
            if r["true_is_fork"] == 0 and r["predicted"] == 1:
                print(f"  {r['role_a']}  <->  {r['role_b']}  (score={r['similarity_pct']})")

    if fn > 0:
        print(f"\n{fn} FALSE NEGATIVE(S) -- confirmed forks that were missed:")
        for r in rows_out:
            if r["true_is_fork"] == 1 and r["predicted"] == 0:
                print(f"  {r['role_a']}  <->  {r['role_b']}")


if __name__ == "__main__":
    main()
