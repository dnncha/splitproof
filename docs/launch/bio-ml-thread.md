# Bio-ML thread draft (Bluesky / X / Mastodon)

Status: draft only, not posted.

---

**1/6**
We audited BACE's official train/valid/test split (`Model` column) with `splitproof`.

The same compound sits on both sides:
- `BACE_72` in train (neutral)
- `BACE_669` in test (protonated at the guanidine)

Raw SMILES comparison misses it. RDKit charge neutralisation catches it.

**2/6**
On random 80/20 splits of six MoleculeNet benchmarks (BACE, BBBP, ClinTox, ESOL, Lipophilicity, HIV), every set leaked Tanimoto >= 0.8 pairs, and five of six leaked exact duplicate compounds after salt stripping and neutralisation:
- HIV: 55 duplicate compounds, 2,455 Tanimoto >= 0.8 pairs crossing
- BACE: 1 duplicate, 491 Tanimoto >= 0.8, 3,774 shared ring-system pairs crossing
- Lipophilicity: 0 duplicates, 411 Tanimoto >= 0.8 pairs crossing

**3/6**
We also audited all 20,429 reviewed human proteins in UniProt (Swiss-Prot) on a random 80/20 split:
- k-mer Jaccard >= 0.3 (k=5): 4,126 related pairs (1,316 crossing)
- MMseqs2 identity >= 0.3, cov 0.8: 137,696 related pairs (46,874 crossing)

MMseqs2 finds 33x as many related pairs. A protein split checked only with k-mers is unchecked for remote homology.

**4/6**
So I built `splitproof`: a Python CLI and stdio MCP server that turns split rules into a YAML policy, builds deterministic connected-component splits, and emits a JSON certificate (`splitproof.cert.json`) that `splitproof verify` recomputes from scratch against the data and split.

**5/6**
Four rules baked into the code:
- Never invents thresholds (fails only on what you declared, and prints worst nearest-neighbour similarity + a 1-NN baseline right beside the status)
- Separates exhaustive `satisfied` from heuristic `satisfied_unproven` (MMseqs2, MinHash, or unparsable SMILES)
- Never calls k-mer Jaccard "identity"
- Refuses giant homology blobs (`GIANT_UNIT`) and unbinding constraints (`CONSTRAINT_VACUOUS`) instead of silently passing

**6/6**
0.1.0 is Apache-2.0. It covers sequences (MMseqs2 / k-mer Jaccard), molecules (RDKit Tanimoto, ring systems, neutralised canonical SMILES), disjoint groups (donor/plate/site), and temporal cutoffs. Still open for future releases: 3D structural similarity, 2D/bipartite splits, and a DataFrame API.

Code and full benchmark audit: https://github.com/dnncha/splitproof
Docs: https://dnncha.github.io/splitproof/
