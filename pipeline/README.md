# pipeline/

The nine modules the clone-detection pipeline needs. Run everything through
`../run_pipeline.sh`; these are not meant to be invoked from here directly,
though they work if you do.

## The three steps

| module | step |
|---|---|
| `evaluate_scale_pipeline_accuracy_v3.py` | 1 — derive and cross-validate tau |
| `find_clones_at_scale_v3.py` | 2 — detection over a technology corpus |
| `evaluate_scale_output_accuracy.py` | 3 — deployed accuracy vs ground truth |

## Imported by those

| module | what for |
|---|---|
| `tfidf_similarity_v3.py` | `read_role_as_text()` — concatenates a role's files into one document |
| `bert_similarity_v3.py` | `chunk_text`, `DEFAULT_MODEL` — a module-level import in Step 2, so it loads even in TF-IDF mode and pulls in `sentence_transformers` |
| `content_template.py` | the `ansible-galaxy init` template line list + `strip_template()`, used only when `--filter-content-boilerplate` is passed |

## The v4 variant

`find_clones_at_scale_v4.py`, `tfidf_similarity_v4.py`, `bert_similarity_v4.py`
are the no-licence-exclusion lineage. Step 1 imports v4 when
`--exclude-license` is NOT passed, and `../diagnose_structure_driven.py` uses it
directly. `run_pipeline.sh` always passes the flag, so the live pipeline is v3.
See README section 6 for the difference.

## Why bare imports still work

These modules import each other by plain module name (`from tfidf_similarity_v3
import ...`). That resolves because Python puts the *script's own directory*
first on `sys.path`, and `run_pipeline.sh` invokes them as
`pipeline/<script>.py`.

Two other places reach in, and both were rewired when this folder was created:

* `../corpus_scale/_paths.py` adds `PROJECT/pipeline` to `sys.path`, so the
  corpus-scale stages can still `from find_clones_at_scale_v3 import ...`.
* The experiment scripts left in the project root
  (`evaluate_adaptive_weighting.py`, `evaluate_boilerplate_variants.py`,
  `diagnose_structure_driven.py`, `compare_configurations.py`) each carry a
  two-line shim that appends this folder to `sys.path`.

If you move this folder again, update both.
