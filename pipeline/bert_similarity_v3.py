"""
bert_similarity.py

SentenceTransformer (BERT-style) based clone / similarity detection for
Ansible roles -- a third method alongside git_diff_v5.py and tfidf_similarity.py.

WHY THIS SCRIPT EXISTS
-----------------------
TF-IDF and git-diff both compare roles on surface form: exact tokens (TF-IDF)
or exact lines (git-diff). Neither can tell that

    mysql_port: 3306
and
    db_port: "3306"

are doing the same thing with different names -- to a bag-of-words model
those are just different tokens. A sentence-embedding model instead encodes
MEANING: it reads the role's content and produces a dense vector positioned
by semantic similarity, so renamed variables, reworded comments, or
reordered-but-equivalent tasks can still land close together in vector
space, which neither TF-IDF nor line-diffing can detect.

This follows Pandu's suggestion directly: concatenate each role into a
single document (same as tfidf_similarity.py's read_role_as_text()) and
embed the whole thing as one vector, rather than forcing a file-to-file
comparison the way traditional code-clone tools (e.g. CodeSim) do.

A NOTE ON ROLE LENGTH
-----------------------
Sentence-transformer models have a token limit (the default model below
truncates at 256 tokens). A full Ansible role's concatenated text is often
much longer than that, so this script chunks each role's text into
~200-word windows, embeds every chunk, and averages the chunk vectors into
one role-level vector (mean pooling). This is the "pool per role" option
Pandu mentioned as an alternative to embedding one document per role
outright -- it lets long roles still get a single representative vector
without silently truncating away most of their content.

TWO MODES, matching tfidf_similarity.py and git_diff_v5.py:
  --mode baseline  : compare every role against ONE reference role per
                      technology
  --mode allpairs  : compare every role against every other role

USAGE
-----
    python bert_similarity.py --tech nginx --mode baseline
    python bert_similarity.py --tech nginx --mode allpairs
    python bert_similarity.py --all --mode baseline

Requires: pandas, sentence-transformers, scikit-learn, numpy
    pip install pandas sentence-transformers scikit-learn numpy

First run will download the model (~80MB, all-MiniLM-L6-v2) -- needs
internet access once; cached locally afterwards.
"""

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Extension allowlist removed on request -- every text-based file counts
# as content now, except binary files and legal boilerplate (see
# EXCLUDED_FILENAMES) -- matches tfidf_similarity.py's version exactly,
# so both methods still see identical input content.
EXCLUDED_FILENAMES = {
    "license", "license.md", "license.txt", "license.rst",
    "copying", "copying.md", "copying.txt",
    "notice", "notice.md", "notice.txt",
}


def is_probably_binary(file_path: Path, sample_size: int = 1024) -> bool:
    """Peek at a file's first bytes for a null character -- a cheap,
    standard heuristic for 'this is binary, not text'."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(sample_size)
        return b"\x00" in chunk
    except Exception:
        return True

# Small, fast, well-established sentence-embedding model. Good default for
# a first pass; swap for a larger model later if you want to trade speed
# for accuracy once you've validated the approach on a subset.
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Chunk size in words. Roughly maps to under the model's 256-token limit
# for typical YAML/code text (which tokenizes denser than prose).
CHUNK_WORDS = 200


def read_role_as_text(role_path: Path) -> str:
    """
    Concatenate every text-based file inside a role folder into one text
    blob -- ANY file type (no extension filter), except files that sniff
    as binary or are known legal boilerplate. Identical logic to
    tfidf_similarity.py's version, so both methods score the exact same
    input content.
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
    Identical logic to tfidf_similarity.py's load_roles().
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


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS) -> list:
    """Split text into ~chunk_words-word windows so nothing gets silently
    truncated by the model's token limit."""
    words = text.split()
    if not words:
        return [""]
    return [
        " ".join(words[i:i + chunk_words])
        for i in range(0, len(words), chunk_words)
    ]


def embed_roles(roles: dict, model: SentenceTransformer) -> dict:
    """
    Returns {role_name: embedding_vector}. Each role's text is chunked,
    every chunk is embedded, and the chunk vectors are mean-pooled into
    one vector per role (see module docstring for why).
    """
    embeddings = {}
    role_names = list(roles.keys())

    for i, name in enumerate(role_names, 1):
        chunks = chunk_text(roles[name])
        chunk_vecs = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
        embeddings[name] = np.mean(chunk_vecs, axis=0)
        if i % 25 == 0 or i == len(role_names):
            print(f"    embedded {i}/{len(role_names)} roles")

    return embeddings


def run_baseline_mode(tech: str, embeddings: dict, reference_role: str, out_dir: Path):
    """
    Compare every role against the single reference role for this technology.
    Mirrors tfidf_similarity.py's run_baseline_mode() exactly, but scores
    cosine similarity between pooled sentence-embedding vectors instead of
    TF-IDF vectors.
    """
    if reference_role not in embeddings:
        print(f"  ERROR: reference role '{reference_role}' not found in "
              f"datasets/{tech}/ -- skipping.")
        return

    role_names = list(embeddings.keys())
    matrix = np.stack([embeddings[name] for name in role_names])
    ref_idx = role_names.index(reference_role)

    similarities = cosine_similarity(matrix[ref_idx:ref_idx + 1], matrix).flatten()

    results = []
    for name, score in zip(role_names, similarities):
        if name == reference_role:
            continue
        results.append({
            "technology": tech,
            "reference_role": reference_role,
            "compared_role": name,
            "bert_cosine_similarity": round(float(score), 4),
        })

    results.sort(key=lambda r: r["bert_cosine_similarity"], reverse=True)

    out_path = out_dir / f"{tech}_bert_baseline_similarities.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"  Saved {len(results)} results to {out_path}")
    if results:
        top = results[0]
        print(f"  Most similar to {reference_role}: "
              f"{top['compared_role']} ({top['bert_cosine_similarity']})")


def run_allpairs_mode(tech: str, embeddings: dict, out_dir: Path):
    """
    Compare every role against every other role.
    Mirrors tfidf_similarity.py's run_allpairs_mode() exactly. Still O(n^2)
    pairs by definition (same caveat as the TF-IDF all-pairs mode) -- the
    per-pair cost is a cheap vector dot product, but the pair COUNT is
    unchanged, so this is not a fix for the corpus-wide scaling problem.
    For that, see the MinHash/LSH or vector-database prefiltering approach
    discussed separately.
    """
    role_names = list(embeddings.keys())
    matrix = np.stack([embeddings[name] for name in role_names])
    sim_matrix = cosine_similarity(matrix)

    results = []
    for i, j in combinations(range(len(role_names)), 2):
        results.append({
            "technology": tech,
            "role_a": role_names[i],
            "role_b": role_names[j],
            "bert_cosine_similarity": round(float(sim_matrix[i, j]), 4),
        })

    results.sort(key=lambda r: r["bert_cosine_similarity"], reverse=True)

    out_path = out_dir / f"{tech}_bert_allpairs_similarities.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"  Saved {len(results)} pairs to {out_path}")

    matrix_df = pd.DataFrame(sim_matrix, index=role_names, columns=role_names)
    matrix_path = out_dir / f"{tech}_bert_similarity_matrix.csv"
    matrix_df.to_csv(matrix_path)
    print(f"  Saved similarity matrix to {matrix_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SentenceTransformer (BERT-style) based similarity detection for Ansible roles.")
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
    parser.add_argument("--out-dir", default="bert_output",
                         help="Where to write result CSVs")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                         help=f"SentenceTransformer model name (default: {DEFAULT_MODEL})")
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

    print(f"Loading model '{args.model}'...")
    model = SentenceTransformer(args.model)

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

        print(f"  Embedding {len(roles)} roles (chunked, mean-pooled)...")
        embeddings = embed_roles(roles, model)

        if args.mode == "baseline":
            reference_role = baseline_map.get(tech)
            if not reference_role:
                print(f"  ERROR: no reference role found for '{tech}' in "
                      f"{args.baseline_csv} -- skipping.")
                continue
            run_baseline_mode(tech, embeddings, reference_role, out_dir)
        else:
            run_allpairs_mode(tech, embeddings, out_dir)


if __name__ == "__main__":
    main()
