# splitproof

<!-- mcp-name: io.github.dnncha/splitproof -->

[![CI](https://github.com/dnncha/splitproof/actions/workflows/test.yml/badge.svg)](https://github.com/dnncha/splitproof/actions/workflows/test.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Build, audit, and re-verify leakage-free train/test splits for biological and chemical datasets, with a JSON certificate anyone can recompute from scratch. Built for ML researchers in computational biology and chemistry, benchmark maintainers, and coding agents.

## Quickstart

Three commands to install `splitproof`, build a certified split on the bundled example, and re-verify every measurement from scratch:

```sh
git clone https://github.com/dnncha/splitproof.git && cd splitproof && pip install .
splitproof split examples/proteins.yaml
splitproof verify splitproof.cert.json --split split.csv
```

To audit the random split column in that same file (exits `1` with crossing pairs and a 1-NN baseline):

```sh
splitproof audit examples/proteins.yaml --split-column fold_random
```

Assignment files have two columns for the entity ID and fold. For a header, use `id` or
the dataset's ID column name, followed by `fold`, `split`, or `set`.

On your own CSV, TSV, FASTA, or Parquet file, run `splitproof init data.csv` to draft a starter `policy.yaml`, review and edit the constraints, then run `splitproof split policy.yaml`. Install chemistry backends with `pip install '.[chem]'` (RDKit) and Parquet support with `pip install '.[parquet]'`.

## What splitproof found on public benchmarks

Measured on 24 November 2026 across six MoleculeNet datasets and every reviewed human protein in UniProt ([full tables and reproduction script](docs/real-data.md)):

- **Every random 80/20 split of six MoleculeNet benchmarks leaks.** On HIV (41,127 molecules), 55 duplicate compounds and 2,455 pairs at Tanimoto >= 0.8 cross train/test. On BACE (1,513 molecules), 1 duplicate compound, 491 pairs at Tanimoto >= 0.8, and 3,774 pairs sharing ring systems cross the split. Lipophilicity is the only set with zero duplicate compounds, and it still puts 411 pairs at Tanimoto >= 0.8 on both sides.
- **BACE's official split puts the same compound in both train and test.** Auditing BACE's published `Model` column (train vs valid+test) flags `BACE_72` in `train` and `BACE_669` in `test`: the same molecule, neutral in train and protonated at the guanidine in test. Raw SMILES comparison misses it; RDKit charge neutralisation catches it.
- **MMseqs2 sequence identity finds 33x as many related human proteins as k-mer Jaccard.** Across all 20,429 reviewed human proteins in UniProt (Swiss-Prot), MMseqs2 identity (>= 0.3, coverage 0.8) finds 137,696 related pairs (46,874 crossing a random 80/20 split), compared with 4,126 pairs (1,316 crossing) for k-mer Jaccard (k=5, >= 0.3).

| Benchmark | Rows | Constraint audited | Related pairs | Crossing train/test |
| --- | --- | --- | --- | --- |
| BACE (random 80/20) | 1,513 | Tanimoto >= 0.8 | 1,624 | 491 |
| BACE (random 80/20) | 1,513 | same ring systems | 11,565 | 3,774 |
| BACE (official `Model`) | 1,513 | same compound (neutralised) | 1 | 1 (`BACE_72` / `BACE_669`) |
| BBBP (random 80/20) | 2,050 | same compound / Tanimoto >= 0.8 | 78 / 251 | 25 / 64 |
| ClinTox (random 80/20) | 1,484 | same compound / Tanimoto >= 0.8 | 61 / 169 | 17 / 48 |
| ESOL (random 80/20) | 1,128 | same compound / Tanimoto >= 0.8 | 11 / 224 | 3 / 79 |
| Lipophilicity (random 80/20) | 4,200 | same compound / Tanimoto >= 0.8 | 0 / 1,217 | 0 / 411 |
| HIV (random 80/20) | 41,127 | same compound / Tanimoto >= 0.8 | 197 / 7,455 | 55 / 2,455 |
| UniProt human (random 80/20) | 20,429 | k-mer Jaccard >= 0.3 (k=5) | 4,126 | 1,316 |
| UniProt human (random 80/20) | 20,429 | MMseqs2 identity >= 0.3, cov 0.8 | 137,696 | 46,874 |

## How it works

A random split fails on biological data because rows share homology, chemical scaffolds, donors, plates, sites, or deposition dates. Instead of keeping a one-off clustering script in a paper repository, you declare your leakage rules in a policy file:

```yaml
splitproof: 1

dataset:
  path: data.csv
  id: accession
  label: activity
  similarity:
    sequence: { column: seq, kind: protein }
  groups: [donor_id, plate]
  dates:
    deposited: deposit_date

scheme:
  kind: holdout
  fractions: { train: 0.8, test: 0.2 }

constraints:
  - kind: similarity
    key: sequence
    metric: identity
    threshold: 0.30
  - kind: disjoint
    key: donor_id
  - kind: temporal
    key: deposited
    cutoff: 2025-06-01

seed: 7
```

This policy enforces three rules: no sequence pair at or above 30% identity may cross folds, no donor may appear in two folds, and nothing deposited on or after `2025-06-01` may land in `train`.

`splitproof` builds a graph of everything that must stay together (similarity edges at or above the threshold plus shared group values), takes its single-linkage connected components as atomic independent units, and packs those units into folds with a deterministic, row-order-invariant, quantile-stratified allocator under `seed`. Because components are single-linkage at the declared threshold, zero pairs at or above that threshold can land on opposite sides.

## The certificate and verification

Every `splitproof split` writes `splitproof.cert.json`: small, diffable, and safe to commit alongside your code.

```json
{
  "status": "satisfied_unproven",
  "policy_digest": "sha256:...",
  "dataset": { "digest": "sha256:...", "entities": 4817 },
  "assignment": { "digest": "sha256:..." },
  "constraints": [
    {
      "id": "sequence:identity@0.3",
      "status": "satisfied_unproven",
      "measured": {
        "cross_fold_pairs_at_or_above_threshold": 0,
        "exactness": "heuristic",
        "backend": "mmseqs2"
      }
    }
  ]
}
```

`splitproof verify` rebuilds the certificate from scratch from the data and fold assignments and compares every section field by field: policy, digests, unit structure, backends and parameters, per-constraint measurements, and leakage diagnostics. Editing a single measurement fails verification and names the exact path:

```text
FAIL constraints: every constraint re-measures to exactly the recorded values
       constraints[0].measured.max_cross_fold_similarity: certificate says 0.0 (float), recomputed None (NoneType)
```

Exit codes across all commands:

- `0`: clean (split built, audit passed, or certificate verified)
- `1`: a declared constraint is violated, or `--require-exact` rejected an unproven result
- `2`: usage, data, policy, or verification failure
- `3`: infeasible split (for example `GIANT_UNIT` or fewer units than folds)

The certificate anchors measurements to content digests, so it verifies on another machine when `--data` is supplied. It is a reproducibility and integrity anchor rather than a cryptographic signature; sign the file with GPG or Sigstore if you need authorship proof. See [SECURITY.md](SECURITY.md) for the untrusted-certificate threat model.

## What the status word does not mean

`splitproof` checks only the constraints your policy declares, so a lax policy gets a clean status. Auditing the example's random split with `kmer_jaccard` at `0.95` reports `proven` truthfully because no pair reaches `0.95`, and prints the worst held-out similarity on the very next line:

```text
240 entities in 240 independent units; folds: test=74, train=166.
Constraint status: proven.
Worst held-out similarity to the reference fold: 0.87013 on sequence:kmer_jaccard@0.95 (74 of 74 held-out items have any neighbour).
That status covers only the constraints the policy declared. Read the numbers below before treating this split as hard.
- no pair above kmer_jaccard 0.95 on 'sequence' may cross folds: proven.
Similarity baseline on sequence:kmer_jaccard@0.95: spearman=0.881475 over 74 held-out items. Report any model score next to this one.
Held-out items' nearest reference similarity on sequence:kmer_jaccard@0.95: median 0.630189, p99 0.792531, max 0.87013.
```

Quote the status word and the worst nearest-neighbour similarity together.

## Honesty rules

**No invented thresholds.** `splitproof` fails only on constraints you declared. It enforces no hidden cutoff of its own.

**Proof separated from evidence.** Every backend declares its exactness, and the certificate records it:

| exactness | meaning | status when clean |
| --- | --- | --- |
| `exact` | every pair was compared | `satisfied` (`proven`) |
| `heuristic` | sensitive but incomplete search (MMseqs2) | `satisfied_unproven` |
| `lower_bound` | approximate candidate search (MinHash LSH) | `satisfied_unproven` |

Any row with an absent or unparsable key value also downgrades a clean status to `satisfied_unproven` with a count of unconstrained rows. Pass `--require-exact` in CI to fail unless every constraint is `satisfied`.

**k-mer Jaccard is never called identity.** Without MMseqs2 installed, the built-in sequence backend computes exact k-mer Jaccard and labels it `kmer_jaccard`.

**Bounded memory on dense graphs.** Similarity pair graphs are quadratic in the worst case. Edges live in numpy arrays inside `PairCollector`, and exceeding `--max-pairs` (default 20,000,000) raises `TOO_MANY_PAIRS` with a remedy instead of an OOM kill. A 2,000-sequence single-family blob at 0.3 produces 1.9M edges in 0.8s; 5,000 produces 11.7M edges in 4.9s and 233MB of edge storage.

**Unbinding constraints fail instead of passing.** If zero rows carry a usable value for a similarity, group, or date key, `splitproof` raises `CONSTRAINT_VACUOUS`. (RDKit 2026.09's `MurckoDecompose` raises on every molecule; an early build swallowed that into "no scaffolds" and reported `satisfied`, which led to this guard.)

**Fractions first, impossible splits refused.** Fold sizes take priority and label stratification breaks exact ties, so `splitproof` reports fold drift transparently. When a single homology component is too large to fit in any fold, it refuses with `GIANT_UNIT` and points you to `splitproof units`, `leave-unit-out`, or `--tolerance`.

## Diagnostics

Both `splitproof split` and `splitproof audit` measure three diagnostics:

- **Nearest-neighbour similarity distribution** (`min`, `p50`, `p90`, `p99`, `max`) from held-out rows to the reference fold (`train`, or the rest of the folds under `kfold`).
- **1-NN similarity baseline**, which predicts each held-out row's label from its closest training neighbour. Similarity reasoning is a legitimate model, so this is neither a hard floor nor a ceiling: a model that fails to beat it has not shown it learned beyond similarity, and beating it does not by itself prove the split is sound. Always report the coverage (`with_a_neighbour` / `queries`) alongside the score.
- **Independent unit structure**: total unit count and `largest_unit_fraction`, showing your real effective sample size.

## Worked example

`examples/proteins.csv` is a synthetic 240-sequence dataset (40 homologous families across 6 donor groups) with a 70/30 random split in `fold_random`. Auditing `fold_random`:

```text
$ splitproof audit examples/proteins.yaml --split-column fold_random
240 entities in 6 independent units; folds: test=74, train=166.
Constraint status: VIOLATED.
Worst held-out similarity to the reference fold: 0.87013 on sequence:kmer_jaccard@0.4 (74 of 74 held-out items have any neighbour).
- no pair above kmer_jaccard 0.4 on 'sequence' may cross folds: VIOLATED.
- no value of 'donor_id' may appear in more than one fold: VIOLATED.
Similarity baseline on sequence:kmer_jaccard@0.4: spearman=0.881475 over 74 held-out items. Report any model score next to this one.
Held-out items' nearest reference similarity on sequence:kmer_jaccard@0.4: median 0.630189, p99 0.792531, max 0.87013.
```

On the random split, 217 of 500 related pairs cross folds, the median held-out sequence has 0.63 k-mer Jaccard to training, and copying the nearest training neighbour scores Spearman `0.881475`.

Splitting the same data with `splitproof`:

```text
$ splitproof split examples/proteins.yaml
240 entities in 6 independent units; folds: test=42, train=198.
Constraint status: proven.
Worst held-out similarity to the reference fold: 0.00232 on sequence:kmer_jaccard@0.4 (38 of 42 held-out items have any neighbour).
That status covers only the constraints the policy declared. Read the numbers below before treating this split as hard.
- no pair above kmer_jaccard 0.4 on 'sequence' may cross folds: proven.
- no value of 'donor_id' may appear in more than one fold: proven.
Similarity baseline on sequence:kmer_jaccard@0.4: spearman=-0.243314 over 38 held-out items. Report any model score next to this one.
Held-out items' nearest reference similarity on sequence:kmer_jaccard@0.4: median 0.00232, p99 0.00232, max 0.00232.
```

The 1-NN baseline drops from `0.881475` (over 74/74 items) to `-0.243314` (over the 38/42 items that share any k-mer with train), and the worst held-out similarity drops from `0.87013` to `0.00232`. Because six donor groups merge the 240 rows into six units, achievable fold sizes are 42/198 rather than 48/192, and `splitproof` reports that drift.

With MMseqs2 installed (`examples/proteins-identity.yaml`, `identity >= 0.3`), auditing `fold_random` reports true sequence identity:

```json
"cross_fold_pairs_at_or_above_threshold": 248,
"cross_fold_pairs_by_fold_pair": {"test|train": 248},
"max_cross_fold_similarity": 0.986,
"nearest_reference_similarity": {"min": 0.877, "p50": 0.954, "max": 0.986},
"nearest_neighbour_baseline": {"metric": "spearman", "value": 0.880675, "compared": 74}
```

Every held-out sequence has a training relative at 87.7% identity or higher (median 95.4%). When a split passes MMseqs2 identity, its status is `satisfied_unproven` because MMseqs2 is heuristic.

## Using splitproof with AI agents

`splitproof` exposes the same workflow over a zero-dependency stdio MCP server (`splitproof mcp`) and via `--json` CLI output with structured error codes (`code`, `message`, `remedy`, `details`), deterministic seeds, and working-directory write confinement.

MCP tools: `splitproof_init`, `splitproof_units`, `splitproof_split`, `splitproof_audit`, `splitproof_verify`, `splitproof_explain`.

### Claude Code

```sh
claude mcp add splitproof -- splitproof mcp
```

Or add `.mcp.json` in your project root:

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

### Cursor

Add `.cursor/mcp.json` (or `~/.cursor/mcp.json`):

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

### Codex

```sh
codex mcp add splitproof -- splitproof mcp
```

Or add to `~/.codex/config.toml`:

```toml
[mcp_servers.splitproof]
command = "splitproof"
args = ["mcp"]
```

Drop [`agent/SKILL.md`](agent/SKILL.md) into your agent's skill directory for full operating rules. Machine-readable discovery files: [`llms.txt`](llms.txt) and [`server.json`](server.json).

## Backends and formats

| metric | backend | needs | exactness |
| --- | --- | --- | --- |
| `identity` | MMseqs2 | `mmseqs` on `PATH` | `heuristic` |
| `kmer_jaccard` | exact sparse Jaccard (`n <= --exact-max-n`, default 20,000) | core install | `exact` |
| `kmer_jaccard` | MinHash LSH (`n > --exact-max-n`) | core install | `lower_bound` |
| `tanimoto` | RDKit Morgan fingerprints (desalted, neutralised) | `splitproof[chem]` | `exact` |
| `ring_systems` | canonical ring-system set over RDKit | `splitproof[chem]` | `exact` |
| `canonical_smiles` | RDKit canonical SMILES (desalted, neutralised) | `splitproof[chem]` | `exact` |
| `exact_duplicate` | normalised equality | core install | `exact` |

Reads CSV, TSV, FASTA, and Parquet (`splitproof[parquet]`). Only empty or whitespace-only cells are treated as missing; literal values like `NA`, `None`, and `Null` remain valid group names.

- `exact_duplicate` folds case for sequences and preserves case for SMILES (where upper-casing changes aromaticity).
- `canonical_smiles`, `ring_systems`, and `tanimoto` share one RDKit standardisation step: salt stripping, largest-fragment selection, and charge neutralisation (inorganic salts without carbon are kept whole).
- `ring_systems` groups molecules by their canonical set of ring systems (element and aromaticity, stereochemistry stripped) rather than Bemis-Murcko scaffolds, because linkers are excluded (grouping more widely, the safe direction for a split) and RDKit 2026.09's `MurckoDecompose` raises on every input.
- Numeric labels are stratified by quantile bucket; high-cardinality non-numeric columns are reported as unstratified rather than treated as thousands of classes.

## Scope and current limits

`splitproof` is evaluation infrastructure for proteins, nucleic acids, small molecules, cells, donors/patients, and temporal splits. It hosts no optimisation loop or screening oracle.

Not yet implemented in `0.1.0` (tracked in [NET.md](NET.md)):

- 3D structural similarity (Foldseek or pocket similarity)
- 2D or bipartite splits (cold-ligand, cold-target, or double-cold interaction splits)
- In-memory pandas/Polars/AnnData DataFrame API
- Opt-in bridge-entity dropping (`--max-drop`) for breaking giant single-linkage components

## Status, citation, and license

Crossing counts separate fold names with `|` and encode literal `%` and `|` in names as `%25`
and `%7C`. Regenerate certificates that recorded crossing counts under the old ambiguous keys.

v0.1.0 ([CHANGELOG.md](CHANGELOG.md)). Policy schema (`splitproof: 1`) and certificate schema (`schema: 1`) are versioned and may evolve before `1.0`.

- Citation metadata: [CITATION.cff](CITATION.cff)
- Security and threat model: [SECURITY.md](SECURITY.md)
- Website: [dnncha.github.io/splitproof](https://dnncha.github.io/splitproof/)
- Part of [Bunsen](https://bunsen.science/), scientific software from Ireland
- License: [Apache-2.0](LICENSE)
