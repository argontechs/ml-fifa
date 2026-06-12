# Lessons

- 2026-06-12: `pytest -q 2>&1 | tail -1 && git commit` commits even when tests fail —
  the pipe makes `tail`'s exit code win. Use `set -o pipefail` or check pytest separately
  before committing.
- 2026-06-12: position-grouped standardization removes BETWEEN-group structure by design;
  synthetic clustering tests must encode styles WITHIN a group, or they test the wrong thing.
- 2026-06-12: never trust remembered FIFA/team codes or feed shapes — verify against the
  actual source (Wikipedia cells, live feed bytes) before pinning constants; three separate
  "obvious" assumptions (trigraphs, feed locations, odds-API team spellings) were all wrong.
- 2026-06-12: judge prediction outcomes on argmax(W/D/L probabilities), not on the modal
  scoreline — a draw can be the most likely single score while a win is the most likely outcome.
