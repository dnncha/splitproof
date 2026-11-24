# Show HN draft

Status: draft only, not posted.

---

## Title options

1. `Show HN: Splitproof: Verifiable train/test splits for biological and chemical data`
2. `Show HN: Splitproof, a CLI and MCP tool that audits and certifies bio/chem splits`

URL: `https://github.com/dnncha/splitproof`

---

## Opening comment

I built `splitproof` (Python, Apache-2.0) to make train/test splits for biological and chemical datasets reproducible and auditable instead of living in one-off clustering scripts.

You write a YAML policy naming what must not cross folds: sequence identity (MMseqs2), k-mer Jaccard, molecular Tanimoto / ring systems / desalted neutralised canonical SMILES (RDKit), exact duplicates, disjoint group columns (donor, patient, plate, site), or a temporal cutoff. `splitproof` takes single-linkage connected components as atomic units, packs them into folds deterministically under a seed, measures nearest-neighbour similarity and a 1-NN baseline, and emits a JSON certificate (`splitproof.cert.json`). Anyone with the data and split file can run `splitproof verify` to recompute every digest, unit count, constraint measurement, and diagnostic from scratch.

Running `splitproof audit` across six public MoleculeNet datasets and all 20,429 reviewed human proteins in UniProt turned up three concrete results (all numbers and the reproduction script are in `docs/real-data.md`):

1. Every random 80/20 split of the six MoleculeNet sets leaked Tanimoto >= 0.8 pairs, and five of the six leaked exact duplicate compounds after desalting and charge neutralisation. On HIV (41,127 rows), 55 duplicate compounds and 2,455 pairs at Tanimoto >= 0.8 crossed train/test. On BACE (1,513 rows), 1 duplicate compound, 491 Tanimoto >= 0.8 pairs, and 3,774 shared ring-system pairs crossed.
2. Auditing BACE's own official `Model` split (train vs valid+test) caught the same compound on both sides: `BACE_72` in train and `BACE_669` in test. They are the same molecule, neutral in train and protonated at the guanidine in test, so raw SMILES string matching misses it while charge neutralisation catches it.
3. On all 20,429 reviewed human proteins in UniProt, MMseqs2 sequence identity (>= 0.3, coverage 0.8) found 137,696 related pairs (46,874 crossing a random 80/20 split), compared with 4,126 pairs (1,316 crossing) for k-mer Jaccard (k=5, >= 0.3), a 33x gap. Because of that gap, `splitproof` refuses to call k-mer Jaccard "sequence identity".

A few design choices I tried to be strict about:

- Exactness is tracked per backend. Exact sparse Jaccard, RDKit Tanimoto, ring systems, and canonical SMILES can produce `satisfied` ("proven") on the declared constraint. MMseqs2 (heuristic) and MinHash LSH (lower bound) max out at `satisfied_unproven` ("no violation found"), and any missing or unparsable row on a constrained key also downgrades a clean result to `satisfied_unproven`. `--require-exact` fails CI unless every constraint is proven exhaustively.
- Certificates are treated as untrusted input in `verify`: backends only accept allowlisted numeric parameters from policies (no executable path can come from a certificate), and `verify` refuses a certificate's recorded data path if it escapes the working directory unless you pass `--data` explicitly.
- Single-linkage pair graphs can blow up quadratically, so pair accumulation and LSH bucket candidate generation are budgeted (`--max-pairs`, default 20M) and fail cleanly with `TOO_MANY_PAIRS`.
- It also ships a zero-dependency stdio MCP server (`splitproof mcp`) and `--json` output so coding agents can audit or build splits without ad-hoc scripts.

What is still missing in 0.1.0 (tracked openly in `NET.md`): 3D structural similarity (Foldseek / pocket metrics), 2D or bipartite splits for drug-target / PPI / TCR-pMHC data, an in-memory DataFrame API, and opt-in bridge-entity dropping to break giant single-linkage components.

Happy to hear critiques on the threat model, the chemistry standardisation choices (especially `ring_systems` vs Bemis-Murcko, since RDKit 2026.09's `MurckoDecompose` raises on every input), or benchmarks we should audit next.
