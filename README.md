# splitproof

**Leakage-free train/test splits for biological data — built, audited, and provable.**

You declare what must not leak between your folds. `splitproof` builds a split that
honours it, measures what it actually achieved, and writes a certificate anyone can
re-derive from your data. Point it at someone else's split and it tells you what they
really tested on.

```sh
pip install splitproof

splitproof init data.csv                     # guess a starter policy from the columns
splitproof split policy.yaml                 # build the split, write a certificate
splitproof verify splitproof.cert.json --split split.csv    # re-derive every number
splitproof audit policy.yaml --split-column fold            # grade a split you were handed
```

## Why

Biology does not lack data or models. It lacks trustworthy scoreboards.

A random train/test split on biological data is almost always wrong, because the rows
are not independent. Test proteins are homologs of training proteins. Test compounds
share a scaffold with training compounds. Test cells come from a training donor, on a
training plate, from a training site. The model learns the overlap and the benchmark
rewards it.

The usual fix is a one-off clustering script in the paper's repo that nobody can rerun
and no reviewer checks. `splitproof` turns that script into an artifact: a declared policy,
a measured result, and a certificate anyone can re-verify against the data.

## How it works

You declare what must not leak. `splitproof` makes the split that honours it, measures what
it achieved, and anchors the result to digests of the policy, the data and the
assignment, so any change is detectable.

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

That policy says three things: no pair of sequences at or above 30% identity may cross
folds, no donor may appear in two folds, and nothing deposited on or after 2025-06-01
may end up in training.

`splitproof` enforces them by building a graph of everything that must stay together —
similarity edges above the threshold, plus shared group values — taking its connected
components as the atomic units of assignment, and packing those units into folds with a
deterministic, label-stratified, seeded allocator. Because the components are
single-linkage at exactly the declared threshold, no pair at or above the threshold can
end up on opposite sides. That is a property of the construction, not a hope.

## The certificate

Every split writes a `splitproof.cert.json`: small, diffable, and safe to commit.

```json
{
  "status": "satisfied",
  "policy_digest": "sha256:…",
  "dataset": { "digest": "sha256:…", "entities": 4817 },
  "assignment": { "digest": "sha256:…" },
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

`splitproof verify` rebuilds the whole certificate from the data and the fold column and
compares it field by field: the digests, every constraint's measured values, the unit
structure, the backends and their parameters, and the leakage diagnostics. A certificate
whose numbers were edited by hand does not verify, and the failure names the exact path:

```
FAIL constraints: every constraint re-measures to exactly the recorded values
       constraints[0].measured.max_cross_fold_similarity: certificate says 0.0, recomputed None
```

Exit codes: `0` verified, `1` the certificate verifies and records a violated constraint
(or `--require-exact` was set and it is not proven exhaustively), `2` verification itself
failed. The certificate records the data's content digest, not its path, so it verifies on
another machine. Put it in CI and your split stops being a claim in a methods
section.

The certificate is a reproducibility anchor, not a cryptographic signature. It detects
drift — edited data, a swapped assignment, a changed policy — and it lets a third party
re-derive the measurements themselves. It does not attest to who produced it. Sign the
file with your own key if you need that.

## What the status word does not mean

`splitproof` fails only on constraints you declared, which means **a lax policy gets a clean
status**. Declare k-mer Jaccard at 0.95 on a set of close homologs and you will be told
`proven`, truthfully, because no pair at 0.95 crosses folds.

This is deliberate: a tool that invented its own identity cutoff would be asserting an
opinion your field has not agreed on. But it means the status word alone is not a quality
claim, so splitproof prints the number that contradicts it on the very next line:

```
Constraint status: proven.
Worst held-out similarity to the reference fold: 0.87013 on sequence:kmer_jaccard@0.95
  (74 of 74 held-out items have any neighbour).
That status covers only the constraints the policy declared. Read the numbers below
  before treating this split as hard.
```

Quote the status and the worst similarity together, or neither.

## Honesty rules

These are load-bearing, not decoration.

**It never invents a threshold.** `splitproof` fails only on constraints you declared. It
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
proven. `splitproof` will not launder a heuristic into a guarantee.

**It does not call k-mer Jaccard "identity".** Without MMseqs2 installed, the sequence
backend computes exact k-mer Jaccard and says so. It is a real, reproducible,
exactly-computed similarity — and it is not sequence identity, so it does not get that
name.

**It refuses to exhaust your memory.** The pair graph is quadratic in the worst case: a
dataset where everything is related to everything at the declared threshold has as many
edges as it has pairs. Edges are stored in numpy arrays rather than Python tuples, and
past `--max-pairs` you get `TOO_MANY_PAIRS` with a remedy instead of an OOM kill. A
2000-sequence single-family blob at 0.3 produces 1.9M edges in 0.8s; 5000 produces 11.7M
in 4.9s and 233MB of edge storage.

**A constraint that cannot bind is an error, not a pass.** If every entity lacks a usable
value for a similarity key, the constraint could never fire and a clean result would
prove nothing, so you get `CONSTRAINT_VACUOUS`. This is not hypothetical: RDKit
2026.09's `MurckoDecompose` raises on every molecule, and an earlier build of splitproof
swallowed that into "no scaffolds" and reported the split as satisfied.

**It honours the fractions you asked for.** Fold sizes come first and label
stratification only breaks exact ties, so splitproof will not quietly hand you a 75/25
split because that balanced the labels better. Granularity is bounded by your largest
unit, and the drift is always reported.

**It refuses impossible splits instead of producing bad ones.** If the largest connected
unit cannot fit in any fold, you get `GIANT_UNIT` and an explanation: your data is one
homology blob at that threshold, and a balanced split does not exist. Raise the
threshold, switch to `leave-unit-out`, or widen `--tolerance` — but you will not get a
split that silently buries the problem.

## Diagnostics worth more than the split

`splitproof split` and `splitproof audit` both report:

- **Nearest-neighbour similarity distribution** for every held-out item against the
  reference fold — median, p99, max. The shape of your actual generalisation gap.
- **The similarity baseline.** A 1-NN model that just copies its most similar training
  neighbour's label. Similarity reasoning is a legitimate model, especially in chemistry,
  so this is neither a floor nor a ceiling: a model that does not clearly beat it has not
  shown it learned anything beyond similarity, and beating it is not on its own evidence
  that the split is sound. This number belongs next to every reported score and is in
  almost none.
- **Unit structure.** How many independent units exist, and what fraction of your data
  sits in the largest one. Most people have never looked, and are surprised.

## A worked example

`examples/proteins.csv` holds 240 sequences in 40 homologous families, with a 70/30
random split in the `fold_random` column — the split almost every pipeline produces by
default. Auditing it:

```
$ splitproof audit proteins.yaml --split-column fold_random
240 entities in 6 independent units; folds: test=74, train=166.
Constraint status: VIOLATED.
- no pair above kmer_jaccard 0.4 on 'sequence' may cross folds: VIOLATED.
- no value of 'donor_id' may appear in more than one fold: VIOLATED.
Worst held-out similarity to the reference fold: 0.87013 (74 of 74 held-out items have
  any neighbour).
Similarity baseline on sequence:kmer_jaccard@0.4: spearman=0.881475 over 74 held-out
  items. Report any model score next to this one.
Held-out items' nearest reference similarity: median 0.630189, p99 0.792531, max 0.87013.
```

217 of the 500 homologous pairs straddle the split, the median held-out sequence is 63%
similar to something in training, and **a model that does nothing but copy its nearest
training neighbour's label scores 0.88.** Any result near 0.9 on this split is
indistinguishable from having learned nothing.

The same data, split by `splitproof`:

```
$ splitproof split proteins.yaml
240 entities in 6 independent units; folds: test=42, train=198.
Constraint status: proven.
Worst held-out similarity to the reference fold: 0.00232 (38 of 42 held-out items have
  any neighbour).
Similarity baseline: spearman=-0.243314 over 38 held-out items.
```

The baseline collapses from 0.88 to -0.24, and the worst held-out similarity falls from
0.87 to 0.002. That gap is how much of the original score was leakage.

Note the baseline's denominator: 38 of the 42 held-out items, because only those share
any k-mer with the training fold at all. The other 4 have no nearest neighbour to copy,
which is the point of the split. `with_a_neighbour` in the JSON always reports that
coverage, so the baseline is never quoted without it.

The example data is synthetic and built to make the point: 40 homologous families with
near-constant labels, and six donor groups that collapse them into six units. It
demonstrates the mechanism; it is not evidence about any real dataset.

Note the fold sizes: 42/198, not the requested 48/192. Six donor groups across 240 rows
means only six independent units exist, so the achievable fractions are coarse.
`splitproof` reports the drift rather than hiding it.

## With real sequence identity

Install MMseqs2 and the same audit speaks in the units a biologist argues in:

```
$ splitproof audit proteins-identity.yaml --split-column fold_random
Constraint status: VIOLATED.
- no pair above identity 0.3 on 'sequence' may cross folds: VIOLATED.
```

```json
"cross_fold_pairs_at_or_above_threshold": 248,
"cross_fold_pairs_by_fold_pair": {"test|train": 248},
"max_cross_fold_similarity": 0.986,
"nearest_reference_similarity": {"min": 0.877, "p50": 0.954, "max": 0.986},
"nearest_neighbour_baseline": {"metric": "spearman", "value": 0.880675, "compared": 74}
```

**Every single held-out sequence has a training relative at 87.7% identity or better**,
median 95.4%. That is the sentence to put in front of anyone defending a random split.

Note the status is `satisfied_unproven` rather than `satisfied` when such a split comes
back clean: MMseqs2 is sensitive but heuristic, so absence of a crossing pair is strong
evidence and not a proof. splitproof says which it has.

## Auditing somebody else's split

The fastest way to find out whether a benchmark means anything:

```sh
splitproof audit policy.yaml --split-column split --report leakage.md
```

Exit code 1 and a list of the worst crossing pairs, by similarity, with ids.

## Built for agents

`splitproof` is designed to be driven by an AI agent as much as by a human.

- Every command takes `--json` and emits a stable schema with a `summary` field written
  to be read by a model.
- Errors are structured: a machine-parseable `code`, a human `message`, an actionable
  `remedy`, and `details`. `GIANT_UNIT` tells the agent what to change.
- Exit codes are meaningful: `0` clean, `1` constraint violated, `2` usage or data
  error, `3` the split is infeasible as specified.
- Deterministic under `seed`, so an agent's run reproduces exactly.
- No network access, no telemetry, no hidden state.
- `--require-exact` fails the build unless every constraint was proven exhaustively, so
  an agent cannot quietly accept `satisfied_unproven` as a pass.

`splitproof mcp` serves the same operations over stdio with no extra dependencies:

```json
{ "mcpServers": { "splitproof": { "command": "splitproof", "args": ["mcp"] } } }
```

Tools: `splitproof_init`, `splitproof_split`, `splitproof_audit`, `splitproof_verify`,
`splitproof_units`, `splitproof_explain`. [`agent/SKILL.md`](agent/SKILL.md) is a drop-in skill
file telling an agent when to reach for it and how to read the result honestly.

## Backends

| metric | backend | needs | exactness |
| --- | --- | --- | --- |
| `identity` | MMseqs2 | `mmseqs` on PATH | heuristic |
| `kmer_jaccard` | exact sparse Jaccard | core install | exact |
| `kmer_jaccard` | MinHash LSH (large n) | core install | lower bound |
| `tanimoto` | RDKit Morgan fingerprints | `splitproof[chem]` | exact |
| `ring_systems` | canonical ring-system set over RDKit | `splitproof[chem]` | exact |
| `canonical_smiles` | RDKit canonical SMILES, desalted | `splitproof[chem]` | exact |
| `exact_duplicate` | normalised equality | core install | exact |

Reads CSV, TSV, FASTA, and Parquet with `splitproof[parquet]`.

`exact_duplicate` folds case only for sequence keys, where the alphabet is
case-insensitive. It does not fold case for molecules, because in SMILES case is
semantic: upper-casing turns benzene into cyclohexane. For "is this the same
compound", use `canonical_smiles`, which compares RDKit canonical SMILES of the
largest covalent fragment, so a salt and its parent and two spellings of one molecule
are one compound while enantiomers are not.

The `ring_systems` backend groups molecules by their canonical set of ring systems,
computed as SMILES fragments of the parent molecule so nothing is edited and nothing
needs re-sanitising. It is deliberately not called `scaffold`: linkers between ring
systems are not part of the key, so it groups more widely than a Bemis-Murcko scaffold,
which is the safe direction for a split. RDKit's own `MurckoScaffoldSmiles` is unusable
because `MurckoDecompose` raises on every input in RDKit 2026.09, and a hand-rolled
Murcko reduction crashed on indole, caffeine, tryptophan and every nucleobase.

Numeric labels are stratified by quantile bucket rather than by raw value, and a label
with more distinct values than a sensible class count is reported as not stratified
rather than silently treated as thousands of classes.

## Scope

`splitproof` is for splitting and auditing datasets in clinical genomics, cell biology,
ADMET and toxicology, imaging, and proteomics. It is evaluation infrastructure: it
measures and constrains, and it hosts no optimisation loop or screening oracle.

## Status

v0.1. The policy schema and certificate schema are versioned; both will change before
1.0 and the version field is how you will know.

Apache-2.0.
