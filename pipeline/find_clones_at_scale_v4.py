"""
find_clones_at_scale_v4.py (No EXCLUDE_FILENAMES, no LICENSE blocklist)

Finds similarity/clone relationships across an ENTIRE corpus of roles
without ever running a role-to-role comparison for every possible pair.

THE CORE IDEA
-------------
git-diff's cost is stuck at O(n^2) PAIRS no matter how it's engineered,
because each comparison requires an actual diff subprocess between two
specific texts -- there's no way to "look up" a result without doing the
work for that exact pair.

TF-IDF and BERT change what's being compared: instead of two texts, you
get two FIXED-LENGTH VECTORS. Comparing vectors is a dot product -- and
computing EVERY pairwise dot product across n vectors is literally one
matrix multiplication (n x d) x (d x n), which BLAS-optimized libraries
(what NumPy/FAISS use under the hood) execute as vectorized bulk arithmetic
rather than n^2 separate subprocess calls. The pair COUNT is unchanged;
the cost PER PAIR drops by several orders of magnitude (a git-diff
subprocess call vs. a fraction of a microsecond of vectorized math).

On top of that, FAISS lets you skip most of the matrix entirely: instead
of "compare this role to all 32,868 others," you ask "give me this role's
20 most similar roles" via an index built for exactly that query, which is
what actually removes the O(n^2) requirement at real scale (hundreds of
thousands to millions of roles). At tens of thousands of roles (our
current scale), FAISS's exact flat index still computes the full
comparison, just fast enough that it doesn't matter.

OUTPUT
------
clone_pairs.csv     -- role_a, role_b, content_similarity_pct,
                        structure_similarity_pct, overall_similarity_pct,
                        for every pair at or above --threshold (checked
                        against CONTENT similarity -- see find_candidate_pairs()
                        docstring for why). structure_similarity_pct is the
                        Jaccard coefficient of relative file paths (same
                        formula as Pandu's original S_structure).
                        overall_similarity_pct = (content + structure) / 2,
                        matching Pandu's S_overall. Use --no-structure to
                        skip this and revert to content-only (faster).
clone_families.csv  -- role, family_id, family_size -- pairs are grouped
                        into clusters via union-find, so A~B and B~C
                        become one family of 3, not two disconnected pairs
                        (same idea as the commit-SHA/blob-SHA clustering
                        approach discussed earlier)

USAGE
-----
    python3 find_clones_at_scale_v4.py --datasets-dir datasets/mysql --threshold 80
    python3 find_clones_at_scale_v4.py --datasets-dir datasets_all --top-k 20
    python3 find_clones_at_scale_v4.py --datasets-dir datasets/mysql --method tfidf --threshold 52.28 --top-k 488
    python3 find_clones_at_scale_v4.py --datasets-dir datasets/mysql --method bert --threshold 88 --top-k 488
    python3 find_clones_at_scale_v4.py --datasets-dir datasets/mysql --method tfidf --threshold-on overall --threshold 52.28 --top-k 488
    python3 find_clones_at_scale_v4.py --datasets-dir datasets/jenkins --method tfidf --threshold-on overall --threshold 41.73 --top-k 250
    python3 find_clones_at_scale_v4.py --datasets-dir datasets/elastic_search --method tfidf --threshold-on overall --threshold 44.95 --top-k 250
"""

import os
# Must be set before torch/faiss are imported anywhere in the process.
# On macOS, PyTorch (used internally by sentence-transformers) and FAISS
# each bundle their own OpenMP runtime; loading both in one process can
# segfault without this. See: https://github.com/facebookresearch/faiss/issues/1006
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from tfidf_similarity_v4 import read_role_as_text
from bert_similarity_v4 import chunk_text, DEFAULT_MODEL


def load_roles(root_dir: Path) -> dict:
    """Returns {role_name: concatenated_text} for every immediate
    subfolder under root_dir with readable content."""
    roles = {}
    for entry in sorted(root_dir.iterdir()):
        if entry.is_dir():
            text = read_role_as_text(entry)
            if text.strip():
                roles[entry.name] = text
            else:
                print(f"  Skipping {entry.name} -- no readable content found")
    return roles


def embed_roles_bert(roles: dict, model) -> tuple:
    """
    Returns (role_names, embedding_matrix) where embedding_matrix is
    L2-normalized so inner product = cosine similarity.
    """
    role_names = list(roles.keys())
    vectors = []
    for i, name in enumerate(role_names, 1):
        chunks = chunk_text(roles[name])
        chunk_vecs = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
        pooled = np.mean(chunk_vecs, axis=0)
        vectors.append(pooled)
        if i % 100 == 0 or i == len(role_names):
            print(f"    embedded {i}/{len(role_names)} roles")

    matrix = np.array(vectors, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid divide-by-zero for any empty-content edge case
    matrix = matrix / norms
    return role_names, matrix


def embed_roles_tfidf(roles: dict, n_components: int) -> tuple:
    """
    TF-IDF produces a SPARSE vector whose length equals the corpus
    vocabulary size, and that size changes every time you add a role --
    neither property works with FAISS, which needs dense, fixed-length
    vectors. Truncated SVD (the same technique behind Latent Semantic
    Analysis) compresses the sparse TF-IDF matrix down to a small, fixed
    number of dense dimensions, preserving most of the similarity
    structure while making it FAISS-compatible.

    This is an approximation of "full" TF-IDF cosine similarity (some
    variance is discarded in the compression), not a mathematically
    identical result to tfidf_similarity_v4.py's uncompressed vectors -- but
    it's what makes TF-IDF usable at scale here, the same way it makes
    BERT usable at scale.
    """
    role_names = list(roles.keys())
    documents = [roles[name] for name in role_names]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    vocab_size = tfidf_matrix.shape[1]

    # SVD requires n_components < min(n_samples, n_features) -- clamp
    # down automatically for small corpora instead of erroring out.
    safe_components = max(1, min(n_components, tfidf_matrix.shape[0] - 1, vocab_size - 1))
    if safe_components < n_components:
        print(f"    (corpus too small for {n_components} SVD components; using {safe_components} instead)")

    svd = TruncatedSVD(n_components=safe_components, random_state=42)
    reduced = svd.fit_transform(tfidf_matrix)
    print(f"    reduced TF-IDF from {vocab_size} vocabulary terms to {safe_components} dense dimensions "
          f"(explained variance: {svd.explained_variance_ratio_.sum():.1%})")

    matrix = reduced.astype("float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    return role_names, matrix


def get_role_paths(root_dir: Path, role_names: list) -> dict:
    """
    Returns {role_name: set of relative file paths} -- ALL files, not just
    the content-readable ones used for embedding. This matches Pandu's
    original structure-similarity input exactly (GetFiles() over the
    whole role, including README, meta, tests, etc.), since directory
    layout is what's being compared here, not file content.
    """
    paths = {}
    for name in role_names:
        role_dir = root_dir / name
        paths[name] = {str(p.relative_to(role_dir)) for p in role_dir.rglob("*") if p.is_file()}
    return paths


# ---------------------------------------------------------------------------
# Boilerplate definitions.
#
# `ansible-galaxy init` writes the same skeleton for every role regardless of
# what the role does, so two unrelated roles share it and structure similarity
# scores them highly. The same is true of repository furniture (licences, CI
# config, linter config) and of files written by tooling rather than authors.
#
# Validated for coverage against 27,583 Ansible roles spanning all
# technologies (previous_work/datasets, with this project's three technologies
# excluded): this list covers 10 of the 11 paths present in >=40% of roles, and
# no technology-specific convention appears above 7.9%. Boilerplate in Ansible
# is technology-independent, which is why a fixed list generalises where a
# per-corpus frequency threshold does not.
# ---------------------------------------------------------------------------

# exactly what `ansible-galaxy init` creates (files only -- files/ and
# templates/ are empty directories)
GALAXY_INIT_PATHS = {
    "readme.md", "defaults/main.yml", "handlers/main.yml", "meta/main.yml",
    "tasks/main.yml", "tests/inventory", "tests/test.yml", "vars/main.yml",
}

# written by tooling or the OS, never by an author
TOOLING_ARTIFACTS = {
    "meta/.galaxy_install_info", ".ds_store", ".gitkeep",
}

# repository furniture: present because of how the repo is maintained, not
# because of what the role does. Licences are already excluded from CONTENT
# (EXCLUDED_FILENAMES in tfidf_similarity_vN.py); this applies the same
# reasoning to the path set.
REPO_FURNITURE = {
    "license", "license.md", "license.txt", "license.rst", "mit-license",
    "copying", "copying.md", "copying.txt", "notice", "notice.md", "notice.txt",
    "changelog.md", "changelog", "changelog.rst", "contributing.md",
    "code_of_conduct.md", "security.md", "authors", "authors.md",
    "requirements.txt", "requirements.yml", "vagrantfile", "makefile",
    ".travis.yml", ".gitignore", ".gitattributes", ".yamllint", ".yamllint.yml",
    ".ansible-lint", ".editorconfig", ".pre-commit-config.yaml", "tox.ini",
    "setup.cfg", ".flake8",
}

# whole directory trees that are CI / test harness scaffolding
SCAFFOLD_PREFIXES = (".github/", "molecule/", ".circleci/", ".gitlab/")

BOILERPLATE_MODES = ("frequency", "galaxy", "galaxy_ci", "none")


def boilerplate_paths(paths_by_role: dict, min_frac: float) -> set:
    """Frequency rule: paths present in at least min_frac of the corpus's roles.

    Corpus-derived, so it adapts to a technology -- but it needs a corpus,
    produces a different set per technology, and min_frac is a hyperparameter
    that has to be justified. Prefer a fixed mode for new technologies.
    """
    n = len(paths_by_role)
    if not n:
        return set()
    counts = {}
    for ps in paths_by_role.values():
        for p in ps:
            counts[p] = counts.get(p, 0) + 1
    return {p for p, c in counts.items() if c / n >= min_frac}


def _fixed_match(path: str, exact: set, prefixes: tuple = ()) -> bool:
    lower = path.lower()
    return lower in exact or any(lower.startswith(x) for x in prefixes)


def filter_boilerplate(role_paths: dict, mode: str = "frequency",
                       min_frac: float = 0.5, verbose: bool = True):
    """Removes boilerplate paths from every role's path set.

    mode:
      frequency  paths in >= min_frac of THIS corpus's roles (needs a corpus)
      galaxy     the fixed `ansible-galaxy init` skeleton + tooling artifacts
      galaxy_ci  the above plus repository/CI furniture  [recommended]
      none       no filtering

    Returns (filtered_paths, excluded_description).
    """
    if mode not in BOILERPLATE_MODES:
        raise ValueError(f"boilerplate mode must be one of {BOILERPLATE_MODES}, got {mode!r}")

    if mode == "none":
        return dict(role_paths), "nothing"

    if mode == "frequency":
        boiler = boilerplate_paths(role_paths, min_frac)
        filtered = {n: (ps - boiler) for n, ps in role_paths.items()}
        desc = f"{len(boiler)} path(s) present in >={min_frac:.0%} of this corpus"
        if verbose:
            print(f"  Boilerplate mode 'frequency': excluding {desc}:")
            for p in sorted(boiler):
                print(f"    - {p}")
    else:
        exact = GALAXY_INIT_PATHS | TOOLING_ARTIFACTS
        prefixes = ()
        if mode == "galaxy_ci":
            exact = exact | REPO_FURNITURE
            prefixes = SCAFFOLD_PREFIXES
        filtered = {n: {p for p in ps if not _fixed_match(p, exact, prefixes)}
                    for n, ps in role_paths.items()}
        removed = sum(len(role_paths[n]) - len(filtered[n]) for n in role_paths)
        desc = (f"fixed '{mode}' list ({len(exact)} filenames"
                + (f" + {len(prefixes)} directory prefixes" if prefixes else "") + ")")
        if verbose:
            print(f"  Boilerplate mode '{mode}': excluding {desc}")
            print(f"    removed {removed:,} path instances across {len(role_paths)} roles")

    if verbose:
        empties = sum(1 for ps in filtered.values() if not ps)
        if empties:
            print(f"  NOTE: {empties} role(s) have NO non-boilerplate files left; "
                  f"their structure similarity will be 0 against everything.")
    return filtered, desc


def structure_similarity(paths_a: set, paths_b: set) -> float:
    """
    Jaccard coefficient of relative file paths -- identical formula to
    Pandu's S_structure: 100 x |P_i intersect P_j| / |P_i union P_j|.
    Two roles that share no files at all score 0; two roles with
    identical directory layouts score 100.
    """
    union = paths_a | paths_b
    if not union:
        return 0.0
    return 100.0 * len(paths_a & paths_b) / len(union)


def build_index(matrix: np.ndarray):
    """Exact cosine-similarity index via inner product on normalized vectors.
    Flat = exact (not approximate) -- fine up to roughly hundreds of
    thousands of roles; swap for faiss.IndexIVFFlat or IndexHNSWFlat if
    the corpus grows past what a flat index can hold comfortably in RAM."""
    import faiss
    faiss.omp_set_num_threads(1)  # extra safety alongside KMP_DUPLICATE_LIB_OK above
    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    return index


def find_candidate_pairs(role_names, matrix, index, top_k, threshold, role_paths=None, threshold_on="content"):
    """
    For every role, query its top_k nearest neighbors (instead of
    comparing against all n-1 others) and keep only those at or above
    --threshold, checked against whichever score --threshold-on selects.
    Returns deduplicated (role_a, role_b, content_pct, structure_pct,
    overall_pct) tuples with role_a < role_b alphabetically, so A-B and
    B-A collapse into one row instead of two.

    IMPORTANT: threshold_on="overall" requires role_paths (structure
    can't be computed without it). Also -- and this matters -- a
    threshold validated via cross-validation against CONTENT-only scores
    (e.g. evaluate_scale_pipeline_accuracy_v3.py's mean_tau) is NOT automatically valid
    for OVERALL scores, since overall = (content + structure) / 2 is a
    different distribution entirely. Re-validate against ground truth
    before trusting a specific --threshold value under --threshold-on
    overall -- don't just reuse the content-validated number.
    """
    if threshold_on == "overall" and role_paths is None:
        raise ValueError("--threshold-on overall requires structure similarity -- remove --no-structure")

    n = len(role_names)
    k = min(top_k + 1, n)  # +1 because a role's own vector is always its own top match
    similarities, indices = index.search(matrix, k)

    seen_pairs = set()
    pairs = []
    for i in range(n):
        for sim, j in zip(similarities[i], indices[i]):
            if j == i or j == -1:
                continue
            content_pct = float(sim) * 100
            key = tuple(sorted((role_names[i], role_names[j])))

            if role_paths is not None:
                structure_pct = structure_similarity(role_paths[key[0]], role_paths[key[1]])
                overall_pct = (content_pct + structure_pct) / 2
            else:
                structure_pct = None
                overall_pct = content_pct

            gate_score = overall_pct if threshold_on == "overall" else content_pct
            if gate_score < threshold:
                continue
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            pairs.append((key[0], key[1], round(content_pct, 2),
                          round(structure_pct, 2) if structure_pct is not None else None,
                          round(overall_pct, 2)))

    return pairs


class UnionFind:
    """Standard union-find for clustering pairwise relationships into
    families -- if A~B and B~C, all three land in one family rather than
    being reported as two separate, disconnected pairs."""

    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def cluster_families(role_names, pairs):
    uf = UnionFind(role_names)
    for pair in pairs:
        a, b = pair[0], pair[1]
        uf.union(a, b)

    families = {}
    for role in role_names:
        root = uf.find(role)
        families.setdefault(root, []).append(role)

    rows = []
    family_id = 0
    for root, members in families.items():
        if len(members) < 2:
            continue  # not part of any clone relationship -- skip solo roles
        family_id += 1
        for m in members:
            rows.append({"role": m, "family_id": family_id, "family_size": len(members)})
    return rows


def main():
    ap = argparse.ArgumentParser(description="Find clone/similarity clusters across a full corpus via vector search.")
    ap.add_argument("--datasets-dir", required=True, help="Folder containing one subfolder per role")
    ap.add_argument("--method", choices=["bert", "tfidf"], default="bert",
                     help="Which method produces the vectors FAISS indexes (default: bert)")
    ap.add_argument("--threshold", type=float, default=80.0, help="Similarity %% cutoff -- use a method-specific "
                     "validated tau (from evaluate_scale_pipeline_accuracy_v3.py), not the same number across methods")
    ap.add_argument("--threshold-on", choices=["content", "overall"], default="content",
                     help="Which score --threshold is checked against. 'overall' requires structure similarity "
                          "(i.e. --no-structure NOT set), and needs its OWN validated tau -- a tau validated for "
                          "content does not automatically transfer to overall.")
    ap.add_argument("--top-k", type=int, default=20, help="Neighbors to check per role, instead of all n-1 others")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model (only used with --method bert)")
    ap.add_argument("--svd-components", type=int, default=300,
                     help="Dense dimensions to reduce TF-IDF vectors to (only used with --method tfidf)")
    ap.add_argument("--out-dir", default="output/clone_scale_output")
    ap.add_argument("--no-structure", action="store_true",
                     help="Skip structure similarity (content-only, faster -- matches the original behavior)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root_dir = Path(args.datasets_dir)
    roles = load_roles(root_dir)
    print(f"Loaded {len(roles)} roles with readable content")
    if len(roles) < 2:
        print("Not enough roles to compare.")
        return

    if args.method == "bert":
        print(f"\nLoading model '{args.model}'...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.model)

        print(f"\nEmbedding {len(roles)} roles (chunked + mean-pooled)...")
        role_names, matrix = embed_roles_bert(roles, model)
    else:
        print(f"\nVectorizing {len(roles)} roles with TF-IDF + SVD "
              f"(target {args.svd_components} dense dimensions)...")
        role_names, matrix = embed_roles_tfidf(roles, args.svd_components)

    print(f"\nBuilding FAISS index over {matrix.shape[0]} vectors (dim={matrix.shape[1]})...")
    index = build_index(matrix)

    print(f"\nQuerying top-{args.top_k} neighbors per role "
          f"({len(roles) * args.top_k:,} lookups instead of "
          f"{len(roles) * (len(roles) - 1) // 2:,} brute-force pairs)...")

    role_paths = None
    if not args.no_structure:
        print("\nComputing directory structure (relative file paths) for structure similarity...")
        role_paths = get_role_paths(root_dir, role_names)

    if args.threshold_on == "overall" and role_paths is None:
        print("\nERROR: --threshold-on overall requires structure similarity. Remove --no-structure.")
        return

    pairs = find_candidate_pairs(role_names, matrix, index, args.top_k, args.threshold, role_paths, args.threshold_on)
    print(f"Found {len(pairs)} pairs at or above {args.threshold}% {args.threshold_on.upper()} similarity")

    if role_paths is not None:
        pairs_df = pd.DataFrame(pairs, columns=["role_a", "role_b", "content_similarity_pct",
                                                  "structure_similarity_pct", "overall_similarity_pct"])
        pairs_df = pairs_df.sort_values("overall_similarity_pct", ascending=False)
    else:
        pairs_df = pd.DataFrame(pairs, columns=["role_a", "role_b", "content_similarity_pct",
                                                  "structure_similarity_pct", "overall_similarity_pct"])
        pairs_df = pairs_df.sort_values("content_similarity_pct", ascending=False)

    pairs_path = out_dir / "clone_pairs.csv"
    pairs_df.to_csv(pairs_path, index=False)
    print(f"Saved pairs to {pairs_path}")

    family_rows = cluster_families(role_names, pairs)
    families_df = pd.DataFrame(family_rows, columns=["role", "family_id", "family_size"])
    if not families_df.empty:
        families_df = families_df.sort_values(["family_size", "family_id"], ascending=[False, True])
    families_path = out_dir / "clone_families.csv"
    families_df.to_csv(families_path, index=False)

    n_families = families_df["family_id"].nunique() if not families_df.empty else 0
    n_in_families = len(families_df)
    print(f"Saved families to {families_path}")
    print(f"\n{n_families} clone families found, covering {n_in_families}/{len(roles)} roles "
          f"({n_in_families / len(roles) * 100:.1f}% of the corpus)")

    if not families_df.empty:
        print("\nLargest families:")
        for fid, group in families_df.groupby("family_id"):
            if group["family_size"].iloc[0] >= families_df["family_size"].max() - 1:
                print(f"  family {fid} ({group['family_size'].iloc[0]} roles): "
                      f"{', '.join(group['role'].tolist()[:6])}"
                      f"{' ...' if len(group) > 6 else ''}")


if __name__ == "__main__":
    main()
