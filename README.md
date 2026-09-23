# Ansible Role Clone Detection

Detects cloned/copied Ansible roles across a corpus of GitHub roles for three
technologies (MySQL, Jenkins, Elasticsearch), and measures how accurate that
detection is against a manually verified ground truth.

The pipeline scores every role pair by TF-IDF + SVD content similarity, gates on
a cross-validated threshold (tau), and clusters the surviving pairs into clone
families.

## Repository layout

```
mywork/
  run_pipeline.sh              the entry point -- runs Steps 1-3 for all three technologies
  README.md                    this file
  requirements.txt
  labeled_pairs.csv            256 labelled pairs -- the ground truth (section 9)
  role_github_map.csv          role -> github_user/github_repo for those 890 roles

  pipeline/                    the 9 modules the pipeline needs (its own README)
    evaluate_scale_pipeline_accuracy_v3.py   Step 1  derive & cross-validate tau
    find_clones_at_scale_v3.py               Step 2  detection
    evaluate_scale_output_accuracy.py        Step 3  deployed accuracy
    tfidf_similarity_v3.py                   read_role_as_text()
    bert_similarity_v3.py                    chunk_text, DEFAULT_MODEL
    content_template.py                      ansible-galaxy template lines (section 5)
    find_clones_at_scale_v4.py  tfidf_similarity_v4.py  bert_similarity_v4.py
                                             the no-licence-exclusion lineage (section 6)

  output/                      EVERY result
    step1_threshold_<mode>/    per technology, one folder per MODE
    step2_clones_<mode>/
    step3_accuracy_<mode>/
    *_k500/                    the exhaustive top-k comparison (section 7)
    variants_output/  adaptive_output/  clone_scale_output/   experiment results
    before_after.csv  comparison_all.csv                      comparison tables
    structure_driven_pairs.csv  structure_driven_diagnosis.csv

  ground_truth/                the scripts that BUILT labeled_pairs.csv, and their
                               evidence: negatives_review*.csv, role_accessibility.csv,
                               ground_truth_decay_affected.csv (its own README)

  docs/                        deliverables -- .pptx decks, methodology .docx, commands.txt
  archive/                     superseded v1/v2 scripts; nothing imports them
  backups/                     .bak / .pre_* snapshots + backup_20260908_210650/
  corpus_scale/                the 28,474-role corpus-wide work, 10 numbered stages

  compare_configurations.py           experiments cited in sections 4-5
  diagnose_structure_driven.py
  evaluate_adaptive_weighting.py
  evaluate_boilerplate_variants.py

  datasets/                    mysql/ 489, jenkins/ 212, elastic_search/ 189
  fulldatasets/                28,474 roles -- used ONLY by corpus_scale/
```

Only two data files stay in the root, and both are there because a great many
scripts reference them by bare relative path: `labeled_pairs.csv` (17 scripts)
and `role_github_map.csv` (6).

### Where results go

Everything writes under `output/`. `run_pipeline.sh` takes an `OUTDIR`
environment variable, defaulting to `output`:

```bash
OUTDIR=/tmp/scratch MODE=galaxyci ./run_pipeline.sh
```

The experiment scripts default there too --
`evaluate_adaptive_weighting.py` to `output/adaptive_output`,
`evaluate_boilerplate_variants.py` to `output/variants_output`,
`find_clones_at_scale_v3.py` to `output/clone_scale_output`.
`compare_configurations.py` reads the `_k500` directories from `output/` as well.

### Imports across folders

The pipeline modules import each other by bare module name, which resolves
because Python puts the running script's own directory first on `sys.path`, and
`run_pipeline.sh` invokes them as `pipeline/<script>.py`.

Two places reach in from outside and were wired up when `pipeline/` was created:

* `corpus_scale/_paths.py` adds `PROJECT/pipeline` to `sys.path`, so the
  corpus-scale stages can still `from find_clones_at_scale_v3 import ...`.
* The four experiment scripts in the root each carry a two-line shim appending
  `pipeline/` to `sys.path`.

Move `pipeline/` again and those are the two things to update.

---

## 1. Requirements

**Python 3.9.6** with the virtualenv used for this project:

```bash
/Users/maydagon/Documents/Dissertation/previous_work/paper4_clone_detection_code/.venv/bin/python
```

The system `python3` will **not** work — it has no scikit-learn. Either use the
full path above, or activate the venv first:

```bash
source /Users/maydagon/Documents/Dissertation/previous_work/paper4_clone_detection_code/.venv/bin/activate
```

Packages (see `requirements.txt`):

| Package | Version | Used for |
|---|---|---|
| scikit-learn | 1.6.1 | TF-IDF, SVD, StratifiedKFold, metrics |
| pandas | 2.3.3 | CSV I/O |
| numpy | 2.0.2 | vectors |
| scipy | 1.13.1 | sklearn dependency |
| faiss-cpu | 1.13.0 | exact nearest-neighbour index (`IndexFlatIP`) |
| sentence-transformers | 5.1.2 | required at import time (see below) |
| torch | 2.8.0 | sentence-transformers dependency |
| requests | 2.32.5 | GitHub API calls in `build_labeled_pairs.py` |

> **sentence-transformers and torch are required even when running
> `--method tfidf`.** `find_clones_at_scale_v3.py` imports `bert_similarity_v3`
> at module level, which imports `sentence_transformers` at module level. You
> cannot run the TF-IDF path without them installed.

### Inputs that must exist

| Path | What it is |
|---|---|
| `datasets/<tech>/<role_name>/` | one folder per role, containing the role's files |
| `labeled_pairs.csv` | ground truth: `technology,role_a,role_b,is_fork` |
| `role_github_map.csv` | role_name -> GitHub user/repo |

Technology names are **`mysql`, `jenkins`, `elastic_search`** — with an
underscore. These must match between the `datasets/` folder name and the
`technology` column of `labeled_pairs.csv`.

---

## 2. Running the pipeline

Three configurations, differing in how structure similarity is treated:

| Mode | Gate | Structure | Boilerplate defined by |
|---|---|---|---|
| `content` (default) | `content >= tau` | never computed | — |
| `overall` | `(content + structure)/2 >= tau` | **all** file paths | — |
| `filtered` | `(content + structure)/2 >= tau` | non-boilerplate paths | this corpus, `>= BOILER_FRAC` |
| **`galaxyci`** | `(content + structure)/2 >= tau` | non-boilerplate paths | **fixed list** (recommended) |
| **`galaxyci_content`** | `(content + structure)/2 >= tau` | non-boilerplate paths | fixed list, **and** generated template TEXT removed from content |

```bash
MODE=content ./run_pipeline.sh
```

```bash
MODE=overall ./run_pipeline.sh
```

```bash
MODE=filtered ./run_pipeline.sh
```

```bash
MODE=galaxyci ./run_pipeline.sh
```

```bash
TOPK=500 MODE=galaxyci_content ./run_pipeline.sh
```

`filtered` and `galaxyci` both exist because unrelated roles score 80–100%
structure simply for both having been created by `ansible-galaxy init`. They
exclude that skeleton from the path set before computing Jaccard; they differ
in **how boilerplate is defined** — see section 5.

**Use `galaxyci`.** It is a fixed list, so it needs no corpus, has no
hyperparameter, and is identical across technologies. `filtered` derives the
list from whichever corpus it is given, which does not generalise. `BOILER_FRAC`
(default 0.5) affects `filtered` only:

```bash
BOILER_FRAC=0.4 MODE=filtered ./run_pipeline.sh
```

Each mode picks its own tau, passes the matching `--threshold-on`, tells Step 3
which column to score, and writes to `output/step1_threshold_<mode>/`,
`output/step2_clones_<mode>/`, `output/step3_accuracy_<mode>/` — so the runs never
overwrite each other.

`TOPK` defaults to 250. **Use `TOPK=500` for these corpora** — it exceeds the
largest (489 roles), making the search exhaustive so no pair is excluded by
retrieval. Results below were produced this way.

Overridable via environment variables:

```bash
TOPK=500 MODE=overall ./run_pipeline.sh         # exhaustive search (recommended)
TAG=_k500 TOPK=500 ./run_pipeline.sh            # write to a separate output set
PAIRS=labeled_pairs_all3.csv ./run_pipeline.sh  # different ground truth
PY=/path/to/python ./run_pipeline.sh            # different interpreter
```

Edit the `TECHS` array
(`tech:content:overall:filtered:galaxyci:galaxyci_content`) to change a threshold
or run a subset.

### Or step by step, per technology

Substitute `<tech>` = `mysql` | `jenkins` | `elastic_search`, and `<tau>` from
the table in section 3. Set `$PY` first:

```bash
PY=/Users/maydagon/Documents/Dissertation/previous_work/paper4_clone_detection_code/.venv/bin/python
```

**Step 1 — derive and cross-validate the threshold**

```bash
$PY pipeline/evaluate_scale_pipeline_accuracy_v3.py \
    --datasets-dir datasets/<tech> --pairs labeled_pairs.csv \
    --tech <tech> --folds 10 --exclude-license --no-structure \
    --out-dir output/step1_threshold_content/<tech>
```

Drop `--no-structure` to validate an `overall` tau as well — with structure on,
Step 1 reports both a CONTENT and an OVERALL tau in one run.

Sweeps every observed score for the one maximising F1, then reports it two ways:
*naive fit* (tau chosen and evaluated on the same pairs — biased, reference
only) and *10-fold cross-validated* (tau chosen on training folds only — the
number to trust). Writes `cv_summary.csv`, `naive_fit_summary.csv`,
`cv_folds_*.csv`, `descriptive_stats.csv`, `scored_pairs.csv`.

**Step 2 — corpus-wide detection**

```bash
$PY pipeline/find_clones_at_scale_v3.py \
    --datasets-dir datasets/<tech> --method tfidf \
    --threshold <tau> --top-k 250 \
    --threshold-on content --no-structure \
    --out-dir output/step2_clones_content/<tech>
```

For the other configuration use `--threshold-on overall` with the overall tau,
and drop `--no-structure` (structure must be computed for the gate to exist).

Embeds the whole corpus, retrieves each role's top-k nearest neighbours via
FAISS, keeps pairs whose gated score >= tau, then clusters them into families
with union-find. Writes `clone_pairs.csv` and `clone_families.csv`.

**Step 3 — accuracy of the deployed output**

```bash
$PY pipeline/evaluate_scale_output_accuracy.py \
    --clone-pairs output/step2_clones_content/<tech>/clone_pairs.csv \
    --labeled labeled_pairs.csv --tech <tech> \
    --score-column content_similarity_pct \
    --out output/step3_accuracy_content/<tech>.csv
```

Use `--score-column overall_similarity_pct` for the overall run. Omitting the
flag auto-detects, preferring `content_similarity_pct` — which is wrong for an
overall-gated run, so pass it explicitly.

---

## 3. Validated thresholds

Licence files excluded. These are **cross-validated `mean_tau`** values:
the threshold is chosen on 9 folds, applied to the held-out 10th, and the
deployed value is the mean of the 10.

| Technology | content tau | overall tau | filtered tau | **galaxyci tau** | **galaxyci_content tau** |
|---|---:|---:|---:|---:|---:|
| mysql | 46.36 | 52.49 | 36.74 | **36.56** | **31.84** |
| jenkins | 51.01 | 40.78 | 34.29 | **26.43** | **26.07** |
| elastic_search | 50.51 | 44.79 | 35.28 | **34.46** | **34.23** |

### Why cross-validated, not naive

Earlier versions of this table used the **naive fit** — the threshold maximising
F1 over *all* of a technology's labelled pairs. That value is chosen on the same
pairs it is then scored against. Measured across 18 technology x score
combinations, naive F1 exceeded cross-validated F1 in **18 of 18 cases**, by a
mean of **+0.029** (max +0.058). A systematic bias in one direction, not noise.

Switching costs almost nothing in headline numbers — two configurations improve,
two are identical, two lose ground, and every difference sits inside the fold
standard deviation (+/-0.05 to +/-0.13 at these sample sizes):

| | naive | CV | naive | CV | naive | CV |
|---|---:|---:|---:|---:|---:|---:|
| | **mysql** | | **jenkins** | | **elastic_search** | |
| content | 44.83 | 46.36 | 53.91 | 51.01 | 49.97 | 50.51 |
| overall | 52.28 | 52.49 | 41.73 | 40.78 | 44.95 | 44.79 |
| filtered | 40.05 | 36.74 | 35.29 | 34.29 | 35.21 | 35.28 |
| galaxyci | 39.77 | 36.56 | 24.84 | 26.43 | 34.73 | 34.46 |

It also matches `corpus_scale/`, which derives tau with `cv_mean_tau()`. Mixing
bases between the two halves of the project would make them incomparable — and
the comparison between them (one global tau reaching within 1.5 F1 points of
technology-specific ones) is a generalisation claim that only holds if the
threshold never saw the data it is evaluated on.

The naive table is preserved in `run_pipeline.sh.naive_basis`.

A tau is only valid for the exact score it was fitted on. Passing a content tau
to `--threshold-on overall` (or the reverse) produces a silently wrong run — no
error, just bad numbers. The same applies across the five modes: excluding
boilerplate shifts the distribution **down**, and by a different amount for each
definition — jenkins runs at 40.78 unfiltered but **26.43** under `galaxyci`,
and mysql at 52.49 unfiltered, 36.56 under `galaxyci`, **31.84** under
`galaxyci_content`. `run_pipeline.sh` selects the right tau per mode; if you
invoke the scripts directly, `--filter-boilerplate`, `--boilerplate-mode` and
`--filter-content-boilerplate` MUST all match between Step 1 and Step 2.

---

## 4. Content vs content + structure

Both runs use the same corpus, engine, licence handling and top-k; only the
gating column changes.

**Cross-validated (10-fold) — the honest comparison:**

| Technology | Mode | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---:|---:|---:|---:|---:|
| mysql | content | 0.951 | 0.940 | 0.983 | 0.957 | **0.987** |
| mysql | overall | 0.928 | 0.930 | 0.952 | 0.934 | 0.957 |
| jenkins | content | 0.902 | 0.950 | 0.867 | 0.891 | 0.981 |
| jenkins | overall | 0.900 | 0.920 | 0.933 | 0.910 | 0.979 |
| elastic_search | content | 0.869 | 0.902 | 0.850 | 0.865 | 0.925 |
| elastic_search | overall | 0.883 | 0.960 | 0.817 | 0.872 | **0.957** |

**Deployed (Step 2 + Step 3):**

| Technology | Mode | Pairs | Families | Largest | TP | FP | FN | TN | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mysql | content | 11,307 | 27 | 256 | 64 | 5 | 0 | 59 | 0.961 | 0.928 | 1.000 | 0.962 |
| mysql | overall | 10,327 | 32 | 228 | 61 | 6 | 3 | 58 | 0.930 | 0.910 | 0.953 | 0.931 |
| jenkins | content | 961 | 16 | 69 | 28 | 0 | 3 | 31 | 0.952 | 1.000 | 0.903 | 0.949 |
| jenkins | overall | 1,274 | 18 | 105 | 29 | 2 | 2 | 29 | 0.935 | 0.935 | 0.935 | 0.935 |
| elastic_search | content | 970 | 8 | 112 | 28 | 4 | 5 | 29 | 0.864 | 0.875 | 0.848 | 0.862 |
| elastic_search | overall | 1,168 | 9 | 112 | 27 | 0 | 6 | 33 | 0.909 | 1.000 | 0.818 | 0.900 |

### On F1 the two are statistically indistinguishable

Every F1 gap is smaller than the fold-to-fold spread:

| Technology | F1 gap (overall − content) | pooled SD |
|---|---:|---:|
| mysql | −2.3 pts | 6.7 pts |
| jenkins | +1.9 pts | 10.7 pts |
| elastic_search | +0.7 pts | 12.7 pts |

Jenkins and elastic_search have only 62 and 66 labelled pairs, so each fold
tests 6–7 items. Do not claim a winner from a 1–2 point F1 difference at this
sample size.

### Structure does not trade precision for recall consistently

Direction of change when structure is added (cross-validated, points):

| Technology | Precision | Recall | |
|---|---:|---:|---|
| mysql | −1.0 | −3.1 | both fall |
| jenkins | −3.0 | +6.6 | recall up |
| elastic_search | +5.8 | −3.3 | precision up |

Three technologies, three directions. Each tau is re-fitted independently, so
each configuration lands at a different point on its own precision–recall curve.
There is no general rule to state here.

### AUC is where they genuinely separate

AUC is threshold-free, so it survives both the tau rounding and the re-fitting:
content wins mysql (0.987 vs 0.957), structure wins elastic_search (0.925 vs
0.957), jenkins is a tie (0.981 vs 0.979).

**Conclusion: content-only is the better default.** It is simpler and cheaper,
never clearly worse except on elastic_search, and measurably better on mysql.
Adding structure helps only where forks reorganise files less than they rewrite
them.


---

## 5. Boilerplate filtering (`filtered` / `galaxyci` / `galaxyci_content`)

### The problem

Most roles are generated by `ansible-galaxy init`, so unrelated roles share a
skeleton. Structure similarity over the full path set therefore scores unrelated
roles highly. Two real examples from the corpus, both verified non-clones:

```
content=86.03  structure=100.00   sujankumar4593.mysql <-> muhametkaqandolli.ansible_role_mysql
content= 4.52  structure= 85.71   Northys.elasticsearch-hunspell <-> eyeem.elasticsearch-restart
```

The first pair has ten files each, all ten shared, nothing unique — but that
list *is* the generator's output. Raw structure AUC is only 0.852 on mysql.

Scale of the damage, measured on the `overall`-gated output: **1,284 pairs**
were flagged with content below their technology's content threshold *and*
structure >= 80% (mysql 1,183, jenkins 75, elastic_search 26). None would have
been detected by content alone. A hand-verified sample of 13, drawn from the
lowest-content end, were **all confirmed false positives** — zero shared
authored code, no fork relationship, demonstrably different purposes. The full
list is in `structure_driven_pairs.csv`.

### Two ways to define boilerplate

| | `MODE=filtered` | `MODE=galaxyci` |
|---|---|---|
| Definition | paths in >= 50% of **this corpus** | a **fixed list** |
| Needs a corpus | yes | no |
| Same across technologies | no — a different set each time | yes |
| Hyperparameter | `BOILER_FRAC` | none |

`galaxyci` excludes the `ansible-galaxy init` skeleton, tooling artifacts
(`meta/.galaxy_install_info`, `.DS_Store`), and repository furniture (licences,
`CHANGELOG.md`, `requirements.*`, `.travis.yml`, `.gitignore`, linter configs)
plus the `.github/`, `molecule/`, `.circleci/` and `.gitlab/` trees — 45
filenames matched case-insensitively, and 4 directory prefixes. The list lives
in `find_clones_at_scale_v3.py`.

**Why the frequency rule does not generalise.** At 0.5 it caught all 8 skeleton
files for mysql but only **5 of 8** for jenkins and elastic_search —
`tests/inventory`, `tests/test.yml` and `vars/main.yml` all sit just under the
line there. It also has no natural cut point outside mysql: mysql has an 18.8
point gap in its frequency distribution around 50%, so the value does not
matter; jenkins and elastic_search have no such gap, and their results move
with it.

**Validated for coverage at scale.** Against **27,583 roles spanning all
technologies** (`previous_work/datasets`, with this project's three excluded),
the `galaxy_ci` list covers 10 of the 11 paths present in >=40% of roles, 13 of
14 above 20%, and 16 of 17 above 10%. No technology-specific convention appears
above 7.9%. Ansible boilerplate is technology-independent, which is why a fixed
list works and a per-corpus threshold is unnecessary.

Removing only the download artifacts is not enough — it fixes just 59 of the
1,284 pairs (4.6%). Jaccard is intersection over union, so removing a path both
roles have shrinks both sides equally: 10-of-10 becomes 8-of-8, still 100%.
Only removing the **whole** skeleton empties the intersection.

### Result

Deployed (Step 3) at cross-validated tau, all five modes:

| Technology | MODE | tau | TP | FP | FN | TN | Accuracy | Precision | Recall | F1 | Clone pairs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mysql | content | 46.36 | 63 | 5 | 1 | 59 | 95.3% | 92.6% | 98.4% | 95.5% | 10,546 |
| mysql | overall | 52.49 | 61 | 6 | 3 | 58 | 93.0% | 91.0% | 95.3% | 93.1% | 10,268 |
| mysql | filtered | 36.74 | 62 | 2 | 2 | 62 | 96.9% | 96.9% | 96.9% | 96.9% | 5,194 |
| mysql | galaxyci | 36.56 | 62 | 2 | 2 | 62 | 96.9% | 96.9% | 96.9% | 96.9% | 5,251 |
| mysql | **galaxyci_content** | 31.84 | **63** | 2 | **1** | 62 | **97.7%** | 96.9% | **98.4%** | **97.7%** | **4,341** |
| jenkins | content | 51.01 | 28 | 0 | 3 | 31 | 95.2% | 100.0% | 90.3% | 94.9% | 1,106 |
| jenkins | overall | 40.78 | 30 | 2 | 1 | 29 | 95.2% | 93.8% | 96.8% | 95.2% | 1,394 |
| jenkins | filtered | 34.29 | 30 | 2 | 1 | 29 | 95.2% | 93.8% | 96.8% | 95.2% | 1,007 |
| jenkins | **galaxyci** | 26.43 | 30 | 1 | 1 | 30 | **96.8%** | 96.8% | 96.8% | **96.8%** | 1,298 |
| jenkins | galaxyci_content | 26.07 | 30 | 1 | 1 | 30 | 96.8% | 96.8% | 96.8% | 96.8% | 1,241 |
| elastic_search | content | 50.51 | 28 | 4 | 5 | 29 | 86.4% | 87.5% | 84.8% | 86.2% | 911 |
| elastic_search | overall | 44.79 | 28 | 1 | 5 | 32 | 90.9% | 96.6% | 84.8% | 90.3% | 1,188 |
| elastic_search | filtered | 35.28 | 27 | 0 | 6 | 33 | 90.9% | 100.0% | 81.8% | 90.0% | 781 |
| elastic_search | **galaxyci** | 34.46 | 29 | 0 | 4 | 33 | **93.9%** | 100.0% | **87.9%** | **93.5%** | **698** |
| elastic_search | galaxyci_content | 34.23 | 29 | 0 | 4 | 33 | 93.9% | 100.0% | 87.9% | 93.5% | 692 |

Pooled over all 256 labelled pairs:

| MODE | TP | FP | FN | TN | Accuracy | Precision | Recall | F1 | Clone pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| content | 119 | 9 | 9 | 119 | 93.0% | 93.0% | 93.0% | 93.0% | 12,563 |
| overall | 119 | 9 | 9 | 119 | 93.0% | 93.0% | 93.0% | 93.0% | 12,850 |
| filtered | 119 | 4 | 9 | 124 | 94.9% | 96.7% | 93.0% | 94.8% | 6,982 |
| galaxyci | 121 | 3 | 7 | 125 | 96.1% | 97.6% | 94.5% | 96.0% | 7,247 |
| **galaxyci_content** | **122** | **3** | **6** | **125** | **96.5%** | **97.6%** | **95.3%** | **96.4%** | **6,274** |

Monotonic improvement down the table, and every step also shrinks the output.

**`galaxyci` or `galaxyci_content` is best or tied-best on every technology.**

- **mysql** is where the content filter earns its place: `galaxyci_content`
  recovers one more true fork with no new false positive (F1 96.9% ->
  **97.7%**) while emitting **910 fewer clone pairs** (5,251 -> 4,341).
- **jenkins** gains from the structural filter and nothing more:
  `galaxyci` lifts F1 95.2% -> **96.8%**, and the content filter changes no cell.
- **elastic_search** likewise peaks at `galaxyci` — recall 81.8% -> **87.9%**,
  F1 90.0% -> **93.5%**, with the smallest output of any configuration.

### `galaxyci_content`: excluding generated template TEXT

`galaxyci` removes generated **paths** from structure similarity. It does
nothing to content: the TF-IDF vectoriser still reads the template prose inside
those files. `ansible-galaxy init` writes 2,927 bytes, of which README.md
(1,328) and meta/main.yml (1,209) are 86% — and both are mostly English.

Measured over a 4,000-role sample of the 28,474-role corpus:

| line | % of roles | | line | % of roles |
|---|---:|---|---|---:|
| `galaxy_info:` | 100.0% | | `License` | 46.5% |
| `platforms:` | 91.5% | | `Role Variables` | 45.1% |
| `versions:` | 89.0% | | `Example Playbook` | 43.7% |
| `dependencies: []` | 76.2% | | `Requirements` | 40.1% |
| `galaxy_tags:` | 67.2% | | `Author Information` | 38.1% |

About 30% of roles never edited the scaffold comments (`# tasks file for ...`),
and ~9% still carry the entire unedited README.

The line list comes from running `ansible-galaxy init` live, unioned with a
hard-coded list of what **older** generators emitted — necessary because the
`#SPDX-License-Identifier: MIT-0` header today's version writes appears in
**0.0%** of this December 2023 corpus.

**It helps mysql and nothing else**, which is the same asymmetry the structural
filter showed. The reason is that TF-IDF's inverse-document-frequency term
already discounts text this common: `galaxy_info:` is in 100% of roles, so its
weight is already near zero. Jaccard has no such mechanism, which is why the
equivalent filter matters far more for structure than for content.

Tested corpus-wide on 14,582 labelled pairs the same way, the effect is
+0.03 F1 — MySQL's gain diluted across 56 technologies. See
`corpus_scale/10_content_boilerplate/`.

### Alternatives that were tested and rejected

`evaluate_adaptive_weighting.py` and `evaluate_boilerplate_variants.py` compare
five schemes, including down-weighting structure by the boilerplate fraction of
the shared paths (`w = 0.5*(1-f)`) rather than filtering the paths. Under nested
5x5 cross-validation, tuning the weight **degraded** held-out performance in 5
of 6 tuned-vs-untuned comparisons, and the selected weights were unstable across
folds (jenkins picked w in {0.1, 0.3, 1.0}). Filtering the paths and keeping a
fixed 50/50 was best or tied-best everywhere. Filtering keeps more information
than discounting: a pair sharing five real task files and ten skeleton files
still scores well, whereas the adaptive weight sees a high boilerplate fraction
and discards the whole signal.

### Caveats

- **`MODE=filtered` has a tuned hyperparameter; `MODE=galaxyci` does not.**
  Under `filtered`, mysql AUC is 0.998 at `BOILER_FRAC` 0.4–0.5 and falls to
  0.963 at 0.8, and that value was chosen on the same data it was measured on.
  `galaxyci` removes the parameter entirely, which is the main reason to prefer
  it in the write-up.
- **Coverage is validated at scale; accuracy is not.** The 27,583-role test
  establishes that the fixed list *captures boilerplate* across all
  technologies. It does **not** establish that filtering *improves detection
  accuracy* there — that needs labelled fork pairs, which exist for three
  technologies only. State the two claims separately.
- **Filtering empties some roles entirely** — 125 mysql (26%), 26 jenkins and
  11 elastic_search under `galaxy_ci` (112/4/3 under `frequency`). After
  Their path sets are empty, so structure similarity is 0 against everything,
  including genuine forks — they are judged on content alone. The run prints
  this as a NOTE. It points at a separate question: whether roles consisting
  only of generated scaffolding belong in a clone-detection corpus at all.
- **1,271 of the 1,284 structure-driven pairs remain unverified.** 13 were
  hand-checked and all were false positives, but 13 of 1,284 is a 1% sample
  drawn from the most extreme end. A random verified sample would be needed to
  put a confidence interval on corpus-wide precision.

---

## 6. Script versions: v3 vs v4

The difference is `EXCLUDED_FILENAMES`:

| | v3 | v4 |
|---|---|---|
| `tfidf_similarity_vN.py` | excludes `LICENSE`/`COPYING`/`NOTICE` from the TF-IDF text | empty set — includes every file |
| `find_clones_at_scale_vN.py` | supports `--threshold-on content\|overall`; writes `clone_pairs.csv` | supports `--threshold-on`; writes `clone_pairs_<tech>.csv` |

`--threshold-on` was added to v3 so the content-vs-structure comparison could be
run with licences excluded. Previously it existed only in v4, which cannot
exclude licences — so comparing the two gates meant changing the licence
handling at the same time, which is not a valid ablation. v3 defaults to
`content`, preserving its original behaviour.

**v3 is the production configuration.** Including licence files added 948 pairs
across the three corpora without improving accuracy on ground truth, and made
Elasticsearch worse (precision 1.000 -> 0.933). Licence text is byte-identical
across unrelated projects by design, so it inflates content similarity without
evidence of copying. Keep v4 only to reproduce that ablation.

`evaluate_scale_pipeline_accuracy_v3.py` mirrors whichever engine you ask for:
`--exclude-license` selects v3, omitting it selects v4. **The tau it produces is
only valid for the engine that produced it** — always pass `--exclude-license`
to match the v3 detector.

---

## 7. Known gotchas

- **`elastic_search`, not `elasticsearch`.** Passing the wrong spelling to
  `--tech` matches zero labelled pairs and silently produces an empty
  evaluation rather than an error.
- **`--top-k` caps recall independently of tau**, so set it above the corpus
  size. `k = min(top_k + 1, n)`, so at `--top-k 500` all three corpora (489 /
  212 / 189 roles) are searched exhaustively and no pair is excluded by
  retrieval. Measured effect of moving 250 -> 500: only mysql was truncated at
  250, and only its overall-gated run changed, gaining 9 pairs out of 10,327
  (0.09%). Every confusion matrix, metric and family count was identical.
- **Score column names.** v3/v4 emit `content_similarity_pct` /
  `structure_similarity_pct` / `overall_similarity_pct`, not `similarity_pct`.
  Step 3 auto-detects and prints which column it used; override with
  `--score-column`.
- **Balanced ground truth vs real prevalence.** Tau is tuned on a 50/50 labelled
  set, but real prevalence is under 1% of the pair space (MySQL: 128 labelled
  pairs vs 119,316 possible). Precision measured on the labelled set does not
  transfer to the corpus. Report Step 1 as *threshold validation* and Step 3 as
  *deployed accuracy* — they are different quantities.
- **Family clustering over-merges.** Union-find takes transitive closure, so
  `A~B` and `B~C` merge even when A and C are unrelated. The largest MySQL
  family covers 256 of 489 roles. Report the family-size distribution, not just
  the count.
- **Empty roles.** Some roles are bare `ansible-galaxy init` skeletons with no
  authored tasks. They score high against each other purely on generated
  scaffolding and should be excluded before scoring. All five MySQL false
  positives are of this kind.
- **Retrieval is always by content.** FAISS cannot search on structure, since
  structure is not part of the embedded vector. `--threshold-on overall`
  therefore *filters* a content-retrieved candidate set — it can reject a pair,
  never introduce one that content retrieval missed. This is fully neutralised
  by an exhaustive `--top-k`; with `TOPK=500` it no longer applies. Verified
  against a full exhaustive search of all 119,316 mysql pairs: the content gate
  missed 0 qualifying pairs at k=250, the overall gate missed 9 (0.09%), and no
  labelled pair was affected (all 16 unretrieved labelled pairs were true
  negatives scoring far below both thresholds).
- **A tau belongs to one score, not one column.** `content`, `overall` and
  `filtered` are three different distributions. Excluding boilerplate shifts
  `overall` down 6–12 points, so `overall_tau` and `filtered_tau` are not
  interchangeable — mysql is 52.28 vs 40.05. Passing the wrong one produces no
  error, just bad numbers. `run_pipeline.sh` picks the right one per mode; if
  you invoke the scripts directly, the `--filter-boilerplate` flag MUST match
  between Step 1 and Step 2.
- **Filtering empties some roles entirely.** 112 mysql roles (23%), 4 jenkins
  and 3 elastic_search consist only of generated scaffolding. Under
  `MODE=filtered` their path sets are empty, so structure similarity is 0
  against everything including real forks. The run prints this as a NOTE — read
  it, because it is also a signal about corpus quality.
- **A tau belongs to one column.** Content and overall are different score
  distributions. Passing a content tau to `--threshold-on overall`, or the
  reverse, silently produces a wrong run — there is no error, just bad numbers.
- **Carry full precision on tau.** Step 1 reports tau rounded to 2 dp. For
  elastic_search the sweep chose 49.96762573719025, reported as 49.97; deployed
  at the rounded value the pair that *defined* the threshold falls 0.003 below
  it and is missed. Passing the unrounded value lifts recall 84.8% -> 87.9%.
- **Differences smaller than the fold spread are not results.** With 62–66
  labelled pairs each fold tests 6–7 items, so cross-validated F1 carries a
  ±7–13 point spread. Every content-vs-structure F1 gap sits inside it.

---

## 8. Script inventory

### `pipeline/` — everything the live pipeline needs

| Script | Role |
|---|---|
| `evaluate_scale_pipeline_accuracy_v3.py` | Step 1 — derive & cross-validate tau |
| `find_clones_at_scale_v3.py` | Step 2 — detection over a technology corpus |
| `evaluate_scale_output_accuracy.py` | Step 3 — deployed accuracy vs ground truth |
| `tfidf_similarity_v3.py` | `read_role_as_text()`, imported by Step 2 |
| `bert_similarity_v3.py` | `chunk_text`, `DEFAULT_MODEL` — a module-level import in Step 2, so it loads even in TF-IDF mode and pulls in `sentence_transformers` |
| `content_template.py` | `ansible-galaxy init` template lines + `strip_template()`; used only with `--filter-content-boilerplate` (§5) |
| `find_clones_at_scale_v4.py` `tfidf_similarity_v4.py` `bert_similarity_v4.py` | the no-licence-exclusion lineage. Step 1 imports v4 when `--exclude-license` is omitted; `diagnose_structure_driven.py` uses it directly (§6) |

`run_pipeline.sh` stays in the root and invokes these as `pipeline/<script>.py`.

### Root — experiments cited in this README

| Script | Role |
|---|---|
| `evaluate_adaptive_weighting.py` | boilerplate-penalised structure weight (§5) |
| `evaluate_boilerplate_variants.py` | sweep of 5 scores x 4 boilerplate definitions (§5) |
| `diagnose_structure_driven.py` | pairs flagged on structure alone (§5) |
| `compare_configurations.py` | consolidates every evaluated configuration into comparison tables |

Not needed to run the pipeline, but §4–§5 cite their results. Each carries a
two-line `sys.path` shim so its bare imports still find `pipeline/`.

### `ground_truth/` — one-off, already applied

| Script | Role |
|---|---|
| `map_dataset_to_github.py` | role folder name -> github_user/github_repo |
| `build_labeled_pairs.py` | fork ancestry -> positives; sampling with sibling exclusion -> negatives |
| `extract_negatives_for_review.py` `extract_negatives_for_review_v2.py` | pulled negatives out for manual GitHub checking |
| `refresh_dead_negatives.py` | replaced negatives whose repo went 404 |
| `check_ground_truth_decay.py` | measured link rot across the labelled roles |
| `compare_fork_pairs.py` `fork_pair_variants.py` | fork-relationship cross-checks |

**Re-running these invalidates every threshold.** Tau is derived from
`labeled_pairs.csv`; change it and you must re-run Step 1 for all five modes and
update the table in `run_pipeline.sh` — silently, with no error, otherwise.

### `archive/` — superseded, nothing imports them

All `_v1` and `_v2` variants plus `evaluate_scale_pipeline_accuracy_v2.py`.
They form a closed cluster (the v1/v2 detectors import `tfidf_similarity_v2` and
`bert_similarity_v2`, both in there), so they still run from that folder.

### `corpus_scale/` — the corpus-wide extension

Ten numbered stages taking detection from these three technologies to all
28,474 roles: indexing, technology grouping, Galaxy and GitHub metadata, a gate
test for corpus-wide scoring, threshold derivation, a sparse detector, and
evaluation. It has its own README; its thresholds are derived with
`cv_mean_tau()`, the same basis as §3.

---

## 9. Ground truth

`labeled_pairs.csv` is balanced: 128 positive (`is_fork=1`) and 128 negative
pairs, split 64/33/31 per technology. Positives are GitHub-confirmed fork
relationships; negatives are randomly sampled non-fork pairs.

All 128 negatives were manually verified. Three were found to be mislabelled —
pairs sharing an upstream ancestor through manual copying rather than a GitHub
fork edge, which the sampler cannot detect because no fork edge exists. They
were replaced with verified-independent pairs; the removed pairs and the
evidence are kept in `negatives_review_removed_mislabeled.csv`.

Link rot: of 890 roles collected, 845 were still reachable as of 2026-08-30
(5.1% decay). See `role_accessibility.csv`.
