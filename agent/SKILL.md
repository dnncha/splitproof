---
name: splitproof
description: Build, audit and prove leakage-free train/test splits for biological data. Use whenever you are about to split a biological dataset, whenever you are handed a benchmark split you did not build, and whenever a model score on biological data looks surprisingly good.
---

# splitproof

A random train/test split on biological data is almost always wrong, because the rows
are not independent. Test proteins are homologs of training proteins. Test compounds
share scaffolds with training compounds. Test cells come from a training donor, plate or
site. A model that exploits that overlap scores well and has learned nothing.

`splitproof` makes a split a provable artifact: you declare what must not leak, it builds a
split that honours the declaration, and it writes a certificate anyone can re-verify.

## When to reach for it

- **Before** splitting any dataset of sequences, structures, molecules, cells or patients.
- **On arrival** of any benchmark split you did not build yourself. Audit it before you
  trust a single number from it.
- **When a score looks too good.** Compare it against the nearest-neighbour baseline that
  `splitproof audit` reports. That is the floor; a model below or near it has learned nothing.
- **In CI**, so a split cannot drift without the build going red.

Do not use it for splitting data whose rows genuinely are independent and exchangeable.

## The loop

```sh
splitproof init data.csv --out policy.yaml     # guesses columns; read and correct it
splitproof units policy.yaml                   # how much independent data do you actually have?
splitproof split policy.yaml --report split.md # build and certify
splitproof verify splitproof.cert.json --split split.csv
```

Every command takes `--json` and returns a stable schema with a `summary` field written
to be read by a model. Exit codes: `0` clean, `1` a declared constraint is violated, `2`
usage or data error, `3` the split is infeasible as specified.

## Reading the result honestly

Check `status` on every constraint, not just the overall result:

- `satisfied` — every pair was compared and none crosses folds. This is proof.
- `satisfied_unproven` — the backend is heuristic or approximate, so no crossing pair was
  *found*. That is evidence, not proof. Pass `--require-exact` in CI if you need proof.
- `violated` — a declared constraint is broken. `examples` lists the worst offenders.

`splitproof` never invents a threshold. It fails only on constraints you wrote down, so a
clean result means exactly what your policy said and nothing more.

## The three numbers that matter

From `split` or `audit`:

1. **`units.largest_unit_fraction`** — what share of your data sits in a single connected
   blob. If this is large, your effective sample size is far below your row count and no
   split can fix that. Say so rather than reporting the row count.
2. **`diagnostics.keys[*].nearest_reference_similarity`** — how similar held-out items
   actually are to training items. The real generalisation gap.
3. **`diagnostics.keys[*].nearest_neighbour_baseline`** — what a 1-NN copy of the training
   labels scores. **Report this next to any model score.** A model that does not clearly
   beat it has learned nothing beyond similarity.

## Failure modes and what to do

- **`GIANT_UNIT`** — the data is one homology blob at the declared threshold, so a balanced
  split does not exist. Do not lower the threshold just to make it pass; that is choosing
  leakage. Run `splitproof units`, report the blob to the user, and consider
  `scheme.kind: leave-unit-out`.
- **`BACKEND_UNAVAILABLE` for `identity`** — MMseqs2 is not installed. Either install it or
  switch to `metric: kmer_jaccard`, which splitproof computes exactly. Do not call k-mer
  Jaccard "identity" in anything you write.
- **Empty fold / `SPLIT_INFEASIBLE`** — there are fewer independent units than folds.
- **A unit spans a temporal cutoff** — decide deliberately via `on_span`; the default
  (`test`) is the conservative choice.

## MCP

`splitproof mcp` serves the same operations over stdio as `splitproof_init`, `splitproof_split`,
`splitproof_audit`, `splitproof_verify`, `splitproof_units` and `splitproof_explain`.

```json
{ "mcpServers": { "splitproof": { "command": "splitproof", "args": ["mcp"] } } }
```

## Scope

`splitproof` is evaluation infrastructure for clinical genomics, cell biology, ADMET and
toxicology, imaging and proteomics. It measures and constrains. It hosts no optimisation
loop and no screening oracle, and it should not be wired into one.
