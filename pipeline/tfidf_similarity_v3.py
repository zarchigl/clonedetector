"""
tfidf_similarity_v3.py (Execlude LICENSE files from TF-IDF vectorization)

TF-IDF based clone / similarity detection for Ansible roles.

WHY THIS SCRIPT EXISTS
-----------------------
git_diff_v5.py and git_diff_heatmap_v2.py detect clones by running
`git diff` file-by-file and counting changed lines. That works, but it
inherits two limitations we identified in the existing pipeline:

  1. It compares files, not whole roles as single units.
  2. Every pair requires an actual diff computation, which is why the
     all-pairs version (git_diff_heatmap_v2.py) scales as O(n^2).

This script takes a different approach, suggested by Pandu: treat each
ROLE as one document (concatenating every relevant file inside it),
turn every role into a TF-IDF vector, and compare roles using cosine
similarity between vectors instead of diffing text. This also sidesteps
the "codesim only compares one file to another" problem, since we are
no longer comparing file-to-file at all -- we compare role-to-role.

TWO MODES, matching the structure of the existing pipeline:
  --mode baseline  : compare every role against ONE reference role per
                      technology (same shape as git_diff_v5.py)
  --mode allpairs  : compare every role against every other role
                      (same shape as git_diff_heatmap_v2.py)

USAGE
-----
    python tfidf_similarity_v3.py --tech nginx --mode baseline
    python tfidf_similarity_v3.py --tech nginx --mode allpairs
    python tfidf_similarity_v3.py --all --mode baseline

Requires: pandas, scikit-learn  (pip install pandas scikit-learn)
"""

import argparse
import csv
from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Extension allowlist removed on request -- every text-based file in a role
# now counts as content, not just YAML/Jinja2/shell/config. Two remaining
# guards:
#   1. A binary-content sniff (is_probably_binary), since reading a
#      genuinely binary file (image, compiled .pyc) as text would inject
#      garbage bytes rather than any meaningful signal.
#   2. A small legal-boilerplate blocklist (EXCLUDED_FILENAMES). License
#      files (LICENSE, COPYING, etc.) are legally required, near-identical
#      across thousands of unrelated projects that happen to share the
#      same license type, and can be hundreds of lines long -- long enough
#      to dominate a TF-IDF vector and produce false-positive similarity
#      between roles that share nothing but a license choice. This was
#      found empirically: two roles with completely different task logic
#      and completely different meta/main.yml (different author,
#      description, tags) scored 99.71% similar, traced directly to both
#      declaring the same GPLv3 license.
EXCLUDED_FILENAMES = {
    "license", "license.md", "license.txt", "license.rst",
    "copying", "copying.md", "copying.txt",
    "notice", "notice.md", "notice.txt",
}


def is_probably_binary(file_path: Path, sample_size: int = 1024) -> bool:
    """Peek at a file's first bytes for a null character -- a standard,
    cheap heuristic for 'this is binary, not text'. Text files essentially
    never contain a null byte; binary files (images, compiled code,
    archives) almost always do somewhere in the first KB."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(sample_size)
        return b"\x00" in chunk
    except Exception:
        return True  # unreadable at all -> treat as unusable, skip


def read_role_as_text(role_path: Path) -> str:
    """
    Concatenate every text-based file inside a role folder into one text
    blob -- ANY file type (no extension filter), except files that sniff
    as binary or are known legal boilerplate (see EXCLUDED_FILENAMES).
    Files are read in sorted path order so the result is deterministic --
    running this twice on the same role always produces the same text.
    """
    chunks = []
    for file_path in sorted(role_path.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.name.lower() in EXCLUDED_FILENAMES:
            continue
        if is_probably_binary(file_path):
            continue
        try:
            chunks.append(file_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception as e:
            print(f"  Warning: could not read {file_path}: {e}")
    return "\n".join(chunks)


def load_roles(tech_dir: Path) -> dict:
    """
    Returns {role_name: concatenated_text} for every role folder under tech_dir.
    Roles with no readable content are skipped with a warning, rather than
    silently producing an empty/meaningless vector.
    """
    roles = {}
    for entry in sorted(tech_dir.iterdir()):
        if entry.is_dir():
            text = read_role_as_text(entry)
            if text.strip():
                roles[entry.name] = text
            else:
                print(f"  Skipping {entry.name} -- no readable content found")
    return roles


def read_baseline_repo(csv_path: Path) -> dict:
    """Returns {technology: reference_role_name} from baseline_repo.csv"""
    baseline = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            baseline[row["technology"]] = row["top1repo"]
    return baseline


def run_baseline_mode(tech: str, roles: dict, reference_role: str, out_dir: Path):
    """
    Compare every role against the single reference role for this technology.
    Mirrors git_diff_v5.py's baseline-vs-all structure, but scores similarity
    via TF-IDF cosine similarity instead of git diff line counts.
    """
    if reference_role not in roles:
        print(f"  ERROR: reference role '{reference_role}' not found in "
              f"datasets/{tech}/ -- skipping.")
        return

    role_names = list(roles.keys())
    documents = [roles[name] for name in role_names]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    ref_idx = role_names.index(reference_role)
    similarities = cosine_similarity(tfidf_matrix[ref_idx], tfidf_matrix).flatten()

    results = []
    for name, score in zip(role_names, similarities):
        if name == reference_role:
            continue
        results.append({
            "technology": tech,
            "reference_role": reference_role,
            "compared_role": name,
            "tfidf_cosine_similarity": round(float(score), 4),
        })

    results.sort(key=lambda r: r["tfidf_cosine_similarity"], reverse=True)

    out_path = out_dir / f"{tech}_tfidf_baseline_similarities.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"  Saved {len(results)} results to {out_path}")
    if results:
        top = results[0]
        print(f"  Most similar to {reference_role}: "
              f"{top['compared_role']} ({top['tfidf_cosine_similarity']})")


def run_allpairs_mode(tech: str, roles: dict, out_dir: Path):
    """
    Compare every role against every other role.
    Mirrors git_diff_heatmap_v2.py's all-pairs structure -- this is still
    O(n^2) pairs by definition, but each individual comparison is now a
    cheap vector-distance lookup rather than an actual git diff subprocess
    call, which is much faster once the vectors are built.
    """
    role_names = list(roles.keys())
    documents = [roles[name] for name in role_names]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    sim_matrix = cosine_similarity(tfidf_matrix)

    results = []
    for i, j in combinations(range(len(role_names)), 2):
        results.append({
            "technology": tech,
            "role_a": role_names[i],
            "role_b": role_names[j],
            "tfidf_cosine_similarity": round(float(sim_matrix[i, j]), 4),
        })

    results.sort(key=lambda r: r["tfidf_cosine_similarity"], reverse=True)

    out_path = out_dir / f"{tech}_tfidf_allpairs_similarities.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"  Saved {len(results)} pairs to {out_path}")

    # Also save the full similarity matrix -- useful later for the same
    # kind of heatmap / plagiarism-band analysis the existing pipeline does.
    matrix_df = pd.DataFrame(sim_matrix, index=role_names, columns=role_names)
    matrix_path = out_dir / f"{tech}_tfidf_similarity_matrix.csv"
    matrix_df.to_csv(matrix_path)
    print(f"  Saved similarity matrix to {matrix_path}")


def main():
    parser = argparse.ArgumentParser(
        description="TF-IDF based similarity detection for Ansible roles.")
    parser.add_argument("--tech",
                         help="Technology folder under datasets/ to process (e.g. nginx)")
    parser.add_argument("--all", action="store_true",
                         help="Process every technology folder found under datasets/")
    parser.add_argument("--mode", choices=["baseline", "allpairs"], default="baseline",
                         help="baseline = compare all roles to the reference role; "
                              "allpairs = compare every role to every other role")
    parser.add_argument("--datasets-dir", default="datasets",
                         help="Root folder containing datasets/<technology>/ subfolders")
    parser.add_argument("--baseline-csv", default="baseline_repo.csv",
                         help="Path to baseline_repo.csv (required for --mode baseline)")
    parser.add_argument("--out-dir", default="tfidf_output",
                         help="Where to write result CSVs")
    args = parser.parse_args()

    if not args.tech and not args.all:
        parser.error("Specify --tech <technology> or --all")

    datasets_dir = Path(args.datasets_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_map = {}
    if args.mode == "baseline":
        baseline_csv_path = Path(args.baseline_csv)
        if not baseline_csv_path.exists():
            parser.error(f"{args.baseline_csv} not found -- required for --mode baseline")
        baseline_map = read_baseline_repo(baseline_csv_path)

    if args.all:
        technologies = [d.name for d in datasets_dir.iterdir() if d.is_dir()]
    else:
        technologies = [args.tech]

    for tech in technologies:
        tech_dir = datasets_dir / tech
        if not tech_dir.exists():
            print(f"Skipping {tech}: {tech_dir} does not exist")
            continue

        print(f"\n{'=' * 60}\nProcessing technology: {tech}\n{'=' * 60}")
        roles = load_roles(tech_dir)
        print(f"  Loaded {len(roles)} roles with readable content")

        if len(roles) < 2:
            print(f"  Not enough roles to compare -- skipping.")
            continue

        if args.mode == "baseline":
            reference_role = baseline_map.get(tech)
            if not reference_role:
                print(f"  ERROR: no reference role found for '{tech}' in "
                      f"{args.baseline_csv} -- skipping.")
                continue
            run_baseline_mode(tech, roles, reference_role, out_dir)
        else:
            run_allpairs_mode(tech, roles, out_dir)


if __name__ == "__main__":
    main()
