# Changelog

All notable changes to `splitproof` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Policy schema (`splitproof: 1`) and certificate schema (`schema: 1`) are versioned
and may evolve before `1.0.0`.

## [0.1.0] - 2026-11-24

Initial public release.

### Added

- Declarative YAML/JSON split policies (`splitproof: 1`) supporting similarity constraints,
  disjoint group constraints, and one-directional temporal cutoffs across `holdout`,
  `kfold`, and `leave-unit-out` schemes.
- Similarity backends with explicit exactness reporting (`exact`, `heuristic`, or
  `lower_bound`):
  - `identity` via MMseqs2 (`mmseqs` on `PATH`, `heuristic`)
  - `kmer_jaccard` via exact sparse Jaccard (`exact`) and deterministic blake2b-keyed
    MinHash LSH above `--exact-max-n` (`lower_bound`)
  - `tanimoto`, `ring_systems`, and `canonical_smiles` via RDKit (`splitproof[chem]`,
    `exact`), sharing salt stripping, largest-fragment selection, and charge neutralisation
  - `exact_duplicate` (`exact`, case-folded for sequences and case-preserved for SMILES)
- Single-linkage union-find into independent units with deterministic, row-order-invariant,
  quantile-stratified fold allocation that prioritises requested fold fractions and refuses
  infeasible `GIANT_UNIT` splits instead of hiding homology blobs.
- Full-certificate generation (`splitproof.cert.json`) and scratch recomputation
  (`splitproof verify`) covering policy, data, and assignment digests, unit structure,
  per-constraint measurements, and leakage diagnostics (nearest-neighbour similarity
  percentiles and 1-NN similarity baseline).
- CLI commands (`init`, `units`, `split`, `audit`, `verify`, `explain`, `mcp`) with
  `--json` output, structured errors, and `--require-exact` enforcement.
- Zero-dependency stdio MCP server (`splitproof mcp`) with working-directory write
  confinement and typed argument validation, plus `agent/SKILL.md`, `llms.txt`, and
  `server.json`.
- Real-data benchmark audit (`docs/real-data.md` and `scripts/realdata.py`) across six
  MoleculeNet datasets (BACE, BBBP, ClinTox, ESOL, Lipophilicity, HIV) and 20,429
  reviewed human proteins in UniProt.
- Dataset loaders for CSV, TSV, FASTA, and Parquet (`splitproof[parquet]`) with strict
  ragged-row, duplicate-header, NUL-byte, and ISO date validation, preserving literal
  `NA`, `None`, and `Null` group names.
- Memory budgets (`PairCollector` and `guard_candidate_budget`, `--max-pairs`) and
  untrusted-certificate hardening (`policy_params` allowlist and working-tree path
  containment).
