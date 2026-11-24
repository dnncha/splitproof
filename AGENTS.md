# Working on cutsite

Inner loop: `.venv/bin/python -m pytest -q`
Gate, before any commit: `./scripts/gate.sh`

[NET.md](NET.md) holds the feedback net: every signal and its trigger, the risk surfaces
and their coverage, the ranked holes, and the escape log. Read it before adding a check,
and add a row to its escape log whenever a bug surfaces later than it should have.

## Rules this codebase enforces on itself

The product's whole claim is that it refuses to overclaim. That applies to its own code.

- **No pair collection is built outside `PairCollector`.** It is the only thing that
  charges the pair budget, and the pair graph is quadratic in the worst case.
  `tools/no_unbounded_pairs.py` fails the gate on a `PairSet` built any other way, and on
  any `.toarray()` or `.todense()` in `src/`.
- **A backend declares its own exactness and never borrows another's.** `exact` means
  every pair was compared. If a backend cannot promise that, it returns `heuristic` or
  `lower_bound`, and a clean result becomes `satisfied_unproven` rather than `satisfied`.
- **No invented thresholds.** cutsite fails only on constraints the user declared. A
  numeric limit may exist where the reason is physical, such as memory, and then it is a
  flag with a generous default and an error that names it. Never a limit fitted to a
  measurement.
- **k-mer Jaccard is never called identity**, in code, output or docs.
- **No code comments.** Explanation goes in names, types, small functions, tests and
  commit messages. Module and class docstrings that state an invariant are fine.
- **Numbers in the README are real measured output.** If behaviour changes the numbers,
  regenerate them in the same change.

## Layout

| Path | Holds |
|---|---|
| `src/cutsite/policy.py` | the declarative policy and every rejection path |
| `src/cutsite/units.py` | union-find over constraints into independent units |
| `src/cutsite/assign.py` | packing units into folds, and refusing when it cannot |
| `src/cutsite/evaluate.py` | measuring constraints and the leakage diagnostics |
| `src/cutsite/cert.py` | building and re-proving certificates |
| `src/cutsite/sim/` | similarity backends, each declaring exactness |
| `src/cutsite/mcp.py` | the stdio MCP server |
| `agent/SKILL.md` | how an agent should use cutsite and read its output |
