# cutsite

**Leakage-free train/test splits for biological data — built, audited, and provable.**

A cut site is the exact, reproducible position where a sequence gets cleaved. `cutsite`
is the exact, reproducible position where you cut your dataset — and the proof that you
cut it where you said you did.

```sh
pip install cutsite

cutsite init data.csv                 # guess a starter policy from the columns
cutsite split cutsite.yaml            # build the split + write a certificate
cutsite verify cutsite.cert.json --split split.csv   # re-prove it in seconds
cutsite audit cutsite.yaml --split-column fold       # measure leakage in someone else's split
```

## Why

Biology does not lack data or models. It lacks trustworthy scoreboards.

A random train/test split on biological data is almost always wrong, because the rows
are not independent. Test proteins are homologs of training proteins. Test compounds
share a scaffold with training compounds. Test cells come from a training donor, on a
training plate, from a training site. The model learns the overlap and the benchmark
rewards it.

The usual fix is a one-off clustering script in the paper's repo that nobody can rerun
and no reviewer checks. `cutsite` turns that script into an artifact: a declared policy,
a measured result, and a certificate anyone can re-verify against the data.

## How it works

You declare what must not leak. `cutsite` makes the split that honours it, measures what
it achieved, and signs the result.

```yaml
cutsite: 1

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

That policy says three things: no pair of sequences at or above 30% identity may cross
folds, no donor may appear in two folds, and nothing deposited on or after 2025-06-01
may end up in training.

`cutsite` enforces them by building a graph of everything that must stay together —
similarity edges above the threshold, plus shared group values — taking its connected
components as the atomic units of assignment, and packing those units into folds with a
deterministic, label-stratified, seeded allocator. Because the components are
single-linkage at exactly the declared threshold, no pair at or above the threshold can
end up on opposite sides. That is a property of the construction, not a hope.

## The certificate

Every split writes a `cutsite.cert.json`: small, diffable, and safe to commit.

```json
{
  "status": "satisfied",
  "policy_digest": "sha256:…",
  "dataset": { "digest": "sha256:…", "entities": 4817 },
  "assignment": { "digest": "sha256:…" },
  "constraints": [
    {
      "id": "sequence:identity@0.3",
      "status": "satisfied",
      "measured": {
        "cross_fold_pairs_at_or_above_threshold": 0,
        "exactness": "exact",
        "backend": "exact_jaccard"
      }
    }
  ]
}
```

`cutsite verify` recomputes the data digest, the assignment digest, and every constraint
measurement, then exits non-zero if anything moved. Put it in CI and your split stops
being a claim in a methods section.

## Honesty rules

These are load-bearing, not decoration.

**It never invents a threshold.** `cutsite` fails only on constraints you declared. It
has no opinion about what identity cutoff your field should use, and it will not quietly
grade your data against one.

**It distinguishes proof from evidence.** Every backend declares its exactness, and the
certificate records it.

| exactness | meaning | status when clean |
| --- | --- | --- |
| `exact` | every pair was compared | `satisfied` |
| `heuristic` | a sensitive but incomplete search (MMseqs2) | `satisfied_unproven` |
| `lower_bound` | approximate candidate generation (MinHash LSH) | `satisfied_unproven` |

A clean result from an approximate backend is reported as *no violation found*, never as
proven. `cutsite` will not launder a heuristic into a guarantee.

**It does not call k-mer Jaccard "identity".** Without MMseqs2 installed, the sequence
backend computes exact k-mer Jaccard and says so. It is a real, reproducible,
exactly-computed similarity — and it is not sequence identity, so it does not get that
name.

**It refuses impossible splits instead of producing bad ones.** If the largest connected
unit cannot fit in any fold, you get `GIANT_UNIT` and an explanation: your data is one
homology blob at that threshold, and a balanced split does not exist. Raise the
threshold, switch to `leave-unit-out`, or widen `--tolerance` — but you will not get a
split that silently buries the problem.

## Diagnostics worth more than the split

`cutsite split` and `cutsite audit` both report:

- **Nearest-neighbour similarity distribution** for every held-out item against the
  reference fold — median, p99, max. The shape of your actual generalisation gap.
- **The nearest-neighbour baseline.** A 1-NN model that just copies its most similar
  training neighbour's label. If your transformer scores 0.86 and this scores 0.85, the
  model learned nothing beyond similarity. This number belongs in every paper and is in
  almost none.
- **Unit structure.** How many independent units exist, and what fraction of your data
  sits in the largest one. Most people have never looked, and are surprised.

## A worked example

`examples/proteins.csv` holds 240 sequences in 40 homologous families, with a 70/30
random split in the `fold_random` column — the split almost every pipeline produces by
default. Auditing it:

```
$ cutsite audit proteins.yaml --split-column fold_random
240 entities in 6 independent units; folds: test=74, train=166.
Constraint status: VIOLATED.
- no pair above kmer_jaccard 0.4 on 'sequence' may cross folds: VIOLATED.
- no value of 'donor_id' may appear in more than one fold: VIOLATED.
Nearest-neighbour baseline on sequence:kmer_jaccard@0.4: spearman=0.881475 over 74
  held-out items. Beat this or the model learned nothing.
Held-out items' nearest reference similarity: median 0.630189, p99 0.792531, max 0.87013.
```

217 of the 500 homologous pairs straddle the split, the median held-out sequence is 63%
similar to something in training, and **a model that does nothing but copy its nearest
training neighbour's label scores 0.88.** Any result near 0.9 on this split is
indistinguishable from having learned nothing.

The same data, split by `cutsite`:

```
$ cutsite split proteins.yaml
240 entities in 6 independent units; folds: test=36, train=204.
Constraint status: proven.
Nearest-neighbour baseline: spearman=-0.027364 over 36 held-out items.
Held-out items' nearest reference similarity: median 0.00232, p99 0.004651, max 0.004651.
```

The baseline collapses from 0.88 to −0.03. That gap is how much of the original score
was leakage.

Note the fold sizes: 36/204, not the requested 48/192. Six donor groups across 240 rows
means only six independent units exist, so the achievable fractions are coarse.
`cutsite` reports the drift rather than hiding it.

## Auditing somebody else's split

The fastest way to find out whether a benchmark means anything:

```sh
cutsite audit policy.yaml --split-column split --report leakage.md
```

Exit code 1 and a list of the worst crossing pairs, by similarity, with ids.

## Built for agents

`cutsite` is designed to be driven by an AI agent as much as by a human.

- Every command takes `--json` and emits a stable schema with a `summary` field written
  to be read by a model.
- Errors are structured: a machine-parseable `code`, a human `message`, an actionable
  `remedy`, and `details`. `GIANT_UNIT` tells the agent what to change.
- Exit codes are meaningful: `0` clean, `1` constraint violated, `2` usage or data
  error, `3` the split is infeasible as specified.
- Deterministic under `seed`, so an agent's run reproduces exactly.
- No network access, no telemetry, no hidden state.

## Backends

| metric | backend | needs | exactness |
| --- | --- | --- | --- |
| `identity` | MMseqs2 | `mmseqs` on PATH | heuristic |
| `kmer_jaccard` | exact sparse Jaccard | core install | exact |
| `kmer_jaccard` | MinHash LSH (large n) | core install | lower bound |
| `tanimoto` | RDKit Morgan fingerprints | `cutsite[chem]` | exact |
| `scaffold` | RDKit Bemis-Murcko | `cutsite[chem]` | exact |
| `exact_duplicate` | normalised equality | core install | exact |

Reads CSV, TSV, FASTA, and Parquet with `cutsite[parquet]`.

## Scope

`cutsite` is for splitting and auditing datasets in clinical genomics, cell biology,
ADMET and toxicology, imaging, and proteomics. It is evaluation infrastructure: it
measures and constrains, and it hosts no optimisation loop or screening oracle.

## Status

v0.1. The policy schema and certificate schema are versioned; both will change before
1.0 and the version field is how you will know.

Apache-2.0.
