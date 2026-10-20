# Vendored IFEval scorer

The checker implementation in this directory is pinned from
`EleutherAI/lm-evaluation-harness` revision
`d6de81643928d653435c431bae19945d41d32520`, directory
`lm_eval/tasks/ifeval/`.

The implementation is based on the Google Research IFEval reference checker.
The original Google-derived files retain their Apache-2.0 headers. The
lm-evaluation-harness license and the Google Research license are included in
this directory.

Local adaptations are intentionally limited to:

1. changing package imports from `lm_eval.tasks.ifeval` to this local package;
2. disabling implicit NLTK downloads during import and replacing them with an
   explicit local-resource preflight;
3. enabling postponed annotation evaluation for Python 3.9 compatibility;
4. using the scorer through `grade_ifeval.py`, which seeds `langdetect` for
   reproducible language checks.

These changes do not alter the strict or loose instruction-checking rules.
