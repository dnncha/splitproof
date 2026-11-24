---
name: splitproof
description: Build, audit, and re-verify leakage-free train/test splits for biological and chemical data. Use before splitting protein/DNA/RNA, small-molecule, cell, or patient datasets; whenever you receive an existing benchmark split; or whenever a biological ML score looks surprisingly high.
---

# splitproof

Random train/test splits on biological and chemical data leak because rows are not independent. Test proteins share homology with training proteins. Test compounds share ring systems or differ only by protonation state or salt counterion. Test cells come from a training donor, plate, or site.

`splitproof` turns split hygiene into a declared YAML/JSON policy, a deterministic split, and a JSON certificate (`splitproof.cert.json`) that `splitproof verify` recomputes from scratch against the data and fold assignments.

## When to reach for it

- **Before splitting** any dataset of sequences (protein, DNA, RNA), small molecules (SMILES), cells, donors, plates, sites, or dated depositions.
- **When handed an existing split** from a benchmark or collaborator. Audit it before trusting any metric on it.
- **When a model score looks high.** Run `audit` and report the 1-NN `nearest_neighbour_baseline` next to the model score.
- **In CI**, to run `splitproof verify` and catch dataset drift, edited certificates, or accidental split changes.

Skip `splitproof` when rows are genuinely independent and exchangeable, or when you need capabilities splitproof does not implement yet (see Scope and limits below).

## Which command or MCP tool to call

| Situation | CLI command | MCP tool |
| --- | --- | --- |
| Start from a raw CSV, TSV, FASTA, or Parquet file | `splitproof init data.csv --out policy.yaml` | `splitproof_init` |
| Check effective sample size / homology blobs before splitting | `splitproof units policy.yaml --json` | `splitproof_units` |
| Build a deterministic split and certificate | `splitproof split policy.yaml --json` | `splitproof_split` |
| Grade an existing split column or two-column id,fold file | `splitproof audit policy.yaml --split-column fold --json` | `splitproof_audit` |
| Re-prove a certificate from scratch against data and split | `splitproof verify splitproof.cert.json --split split.csv --json` | `splitproof_verify` |
| Read an existing certificate in plain text or Markdown (no recomputation) | `splitproof explain splitproof.cert.json` | `splitproof_explain` |

Every CLI command accepts `--json` and returns a stable schema with a `summary` string. Exit codes:

- `0`: clean (split built, audit passed, or certificate verified)
- `1`: a declared constraint is violated, or `--require-exact` rejected a `satisfied_unproven` result
- `2`: usage, data, policy, or verification failure
- `3`: infeasible split (such as `GIANT_UNIT` or fewer independent units than folds)

## Standard workflow

```sh
splitproof init data.csv --out policy.yaml     # inspect columns; always review and edit the guess
splitproof units policy.yaml --json            # check independent unit count and largest_unit_fraction
splitproof split policy.yaml --report split.md --json
splitproof verify splitproof.cert.json --split split.csv --json
```

When auditing a published split:

```sh
splitproof audit policy.yaml --split-column fold --report leakage.md --json
```

## Reading results honestly

Inspect `status` on **every** entry in `constraints`, not only the top-level status:

- `satisfied` (rendered as `proven`): an exhaustive backend compared every pair on the declared constraint, with zero absent or unparsable rows on that key, and zero violations crossed folds. This proves only the declared constraint at the declared threshold.
- `satisfied_unproven` (rendered as `no violation found (not proven)`): no crossing pair was found, but either the backend is heuristic/approximate (MMseqs2 or MinHash LSH) or some rows had missing or unparsable values and sat unconstrained. Treat this as evidence, not proof. Pass `--require-exact` (or `require_exact: true` in MCP) when you must fail on unproven results.
- `violated`: at least one declared constraint failed. Inspect `constraints[*].examples` for the worst crossing pairs or shared groups.

`splitproof` never invents a threshold. A lax policy (such as `kmer_jaccard` at `0.95`) produces `satisfied` truthfully because no pair at `0.95` crossed folds. Always read the similarity diagnostics alongside the status word.

## The three numbers to report

From `split` or `audit` JSON output:

1. `units.largest_unit_fraction`: share of the dataset sitting in one connected component. If this is large, the effective sample size is far smaller than the row count. Report the unit count and largest unit fraction beside the row count.
2. `diagnostics.keys[*].nearest_reference_similarity`: `min`, `p50`, `p90`, `p99`, and `max` similarity from held-out items to their nearest item in the reference fold (`train`, or the rest of the folds under k-fold).
3. `diagnostics.keys[*].nearest_neighbour_baseline`: 1-NN copy score (`metric`, `value`, `compared`) alongside `with_a_neighbour` and `queries`. Report this next to any model score. Similarity reasoning is a legitimate model, so this is neither a hard floor nor a ceiling: a model that does not clearly beat it has not shown it learned anything beyond similarity, and beating it does not by itself prove the split is leakage-free.

## Backends and exactness

| Metric | Key kind | Backend | Requirement | Exactness |
| --- | --- | --- | --- | --- |
| `identity` | protein, dna, rna | `mmseqs2` | `mmseqs` on PATH | `heuristic` |
| `kmer_jaccard` | protein, dna, rna | `exact_jaccard` (n <= `--exact-max-n`, default 20,000) | core install | `exact` |
| `kmer_jaccard` | protein, dna, rna | `minhash` (n > `--exact-max-n`) | core install | `lower_bound` |
| `tanimoto` | molecule | `tanimoto` (RDKit Morgan fingerprints, desalted + neutralised) | `splitproof[chem]` | `exact` |
| `ring_systems` | molecule | `ring_systems` (canonical ring-system set over RDKit) | `splitproof[chem]` | `exact` |
| `canonical_smiles` | molecule | `canonical_smiles` (RDKit desalted + neutralised canonical SMILES) | `splitproof[chem]` | `exact` |
| `exact_duplicate` | sequence, molecule, text | `exact_duplicate` (case-folded for sequences, case-preserved for SMILES) | core install | `exact` |

Never call `kmer_jaccard` "sequence identity". On 20,429 reviewed human proteins in UniProt, MMseqs2 identity (>= 0.3, coverage 0.8) found 137,696 related pairs where k-mer Jaccard (k=5, >= 0.3) found 4,126 (a 33x gap).

## Failure codes and how to recover

- `GIANT_UNIT`: one connected component exceeds a fold's target size plus `--tolerance`. Do not silently raise the threshold to force a pass without telling the user; raising the threshold accepts more leakage. Call `splitproof_units`, report `largest_unit_fraction`, and ask whether to switch to `scheme.kind: leave-unit-out`, adjust `--tolerance`, or change the threshold explicitly.
- `SPLIT_INFEASIBLE`: fewer independent units than folds, or constraints leave a fold empty. Inspect `splitproof_units` and adjust the scheme or fold count.
- `BACKEND_UNAVAILABLE`: usually `identity` requested without `mmseqs` on PATH, or a chemistry metric requested without `splitproof[chem]`. Install the dependency or switch metrics explicitly (and never relabel `kmer_jaccard` as `identity`).
- `CONSTRAINT_VACUOUS`: zero rows have a usable value for a declared similarity, group, or date key. Fix the column name or data; splitproof refuses to certify an unbinding constraint as clean.
- `TOO_MANY_PAIRS`: the threshold relates more pairs than `--max-pairs` (default 20,000,000). Use a coarser equivalence key (`ring_systems`, group), raise the threshold deliberately, or raise `--max-pairs` if memory allows.
- `DATA_INVALID` / `POLICY_INVALID` / `CERTIFICATE_MISMATCH`: read `error.message`, `error.remedy`, and `error.details`. Note that only empty or whitespace-only cells are treated as missing; strings like `NA`, `None`, and `Null` are kept as literal group values.

## MCP setup

`splitproof mcp` runs over stdio with no extra dependencies beyond `splitproof`. Output paths written by MCP tools (`out`, `cert`, `report`) are confined to the current working directory.

Claude Code:

```sh
claude mcp add splitproof -- splitproof mcp
```

Cursor (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "splitproof": {
      "command": "splitproof",
      "args": ["mcp"]
    }
  }
}
```

Codex (`~/.codex/config.toml`):

```toml
[mcp_servers.splitproof]
command = "splitproof"
args = ["mcp"]
```

## Scope and limits

`splitproof` is evaluation infrastructure for proteins, nucleic acids, small molecules, cells, and clinical/patient cohorts. It measures and constrains splits; it hosts no optimisation loop or screening oracle.

Not yet implemented in 0.1.0 (do not promise or claim these):
- 3D structural similarity (Foldseek or binding-pocket similarity)
- 2D or bipartite interaction splits (cold-ligand / cold-target / double-cold for drug-target, PPI, or TCR-pMHC)
- In-memory pandas/Polars/AnnData DataFrame API (use CSV, TSV, FASTA, or Parquet files)
- Opt-in bridge-entity dropping (`--max-drop`) to break giant single-linkage components
