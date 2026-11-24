# Feedback net

Inner loop: `.venv/bin/python -m pytest -q` (~9 s)
Gate: `scripts/gate.sh` (~12 s)
Scheduled: none yet (see Holes)

## Signals by tier

| Tier | Signal | Command | Runs | Time | Status |
|---|---|---|---|---|---|
| edit | Unbounded-accumulation rule | `python3 tools/no_unbounded_pairs.py .` | inner, gate | 0.1 s | honest |
| edit | NET.md conformance | `python3 tools/netcheck.py NET.md` | gate | 0.1 s | honest |
| inner | Unit and integration tests | `pytest -q` | inner, gate | 9 s | honest |
| inner | Backend budget conformance | `pytest -q tests/test_net.py` | inner, gate | 1 s | honest, parametrised over the backend registry |
| gate | End-to-end split, verify, audit | `scripts/gate.sh` | gate, CI | 3 s | honest |
| gate | Python matrix (3.10, 3.12, 3.13) | GitHub Actions `test` | gate (push, PR) | 1 min | honest |

## Risk surfaces

| Surface | Paths | Risk (L×I) | Covered by | Status |
|---|---|---|---|---|
| Similarity pair graph (quadratic blowup) | `src/cutsite/sim/` | high × high | static rule, registry budget test, scale tests | covered |
| Constraint evaluation (soundness of a "proven" claim) | `src/cutsite/evaluate.py` | high × high | unit tests on violation detection, exactness downgrade test | covered |
| Certificate verification (tamper detection) | `src/cutsite/cert.py` | med × high | tests for swapped assignment, edited data, swapped policy | covered |
| Assignment feasibility | `src/cutsite/assign.py` | med × med | giant-unit, empty-fold, temporal-span tests | covered |
| Policy parsing (bad input accepted) | `src/cutsite/policy.py` | med × med | unit tests on each rejection path | late: no property test over malformed policies |
| MMseqs2 backend (identity metric) | `src/cutsite/sim/mmseqs.py` | med × high | TSV parser unit test, command construction | hole: binary not installed anywhere we run, so the live path is unexercised |
| RDKit backends (tanimoto, scaffold) | `src/cutsite/sim/chem.py` | med × med | registry budget test (skipped) | hole: rdkit not in any test environment |
| Data loading (CSV/TSV/FASTA/Parquet) | `src/cutsite/dataset.py` | med × med | duplicate-id and bad-date tests | late: no fuzz over malformed files |

## Holes, ranked

1. MMseqs2 path never executed → install `mmseqs` in CI and run one real identity split (gate, ~20 min setup). This is the metric users will actually declare, and it is the only `heuristic`-exactness backend, so its measured values feed a certificate nobody has verified end to end.
2. RDKit backends never executed → add a `chem` CI job with `pip install cutsite[chem]` and a 20-molecule scaffold split (gate, ~15 min).
3. No property test over policy parsing → generate malformed policies and assert every rejection names a `where` locator rather than raising an unhandled error (inner, ~30 min).
4. No scheduled tier at all → nothing runs mutation testing or a longer scale sweep. Add a weekly GitHub Actions schedule running mutation testing over `evaluate.py` and `assign.py`, red opening an issue (~45 min).
5. No fuzz over data loading → malformed CSV/FASTA should fail with a `DataError`, never a traceback (inner, ~30 min).

## Escape log

| Date | Symptom | Root cause | Class | Caught by | Fixed (+siblings) | Class guard | Rung | Time to red | Proved |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-24 | Dense dataset would be OOM-killed instead of failing cleanly; 5000-sequence single-family blob peaked at 2714 MB | Similarity pair graph materialised as Python tuples and grew quadratically, with no budget | unbounded all-pairs / dense accumulation | own scale benchmark, after the code was already pushed public | `sim/kmer.py` exact + minhash, `sim/duplicate.py`, `sim/chem.py` tanimoto + scaffold, `sim/mmseqs.py`; siblings: `evaluate.py` accumulated *and sorted* every crossing pair, `kmer.nearest` built a dense `.toarray()` block, `chem` copied the fingerprint tail per row | `tools/no_unbounded_pairs.py` bans `.toarray()`/`.todense()` and any `PairSet` built outside `PairCollector`, which is now the only route that charges the budget; plus a test parametrised over the backend registry asserting every backend raises `TOO_MANY_PAIRS` | 3 rule | 0.09 s | yes: red on the restored `.toarray()` at `kmer.py:156`, red on a planted backend bypassing the collector at `duplicate.py:54` (both the rule and the registry test), green on the fix |

Runs names what triggers each signal: `compiler`, `hook`, `inner`, `gate`, `release`, `live`, or `scheduled: <scheduler, cadence, where red goes>`.

Rung is the prevention ladder in dragnet's escapes.md: 1 impossible, 2 deleted path, 3 rule, 4 sweep, 5 assertion, 6 property or fuzz, 7 regression test, 8 production.

Check it with `python3 tools/netcheck.py NET.md`, which the gate runs.
