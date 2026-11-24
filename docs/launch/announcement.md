# Short announcement draft (blog / newsletter / mailing list / Slack)

Status: draft only, not posted.

---

**splitproof 0.1.0: verifiable train/test splits for biological and chemical data**

Random splits on biological datasets routinely put homologous proteins, near-identical compounds, or cells from the same donor on both sides of train and test. Even curated splits can slip: when we audited six MoleculeNet datasets and 20,429 reviewed human proteins from UniProt with `splitproof`, every random 80/20 split leaked Tanimoto >= 0.8 pairs (including 55 exact duplicate compounds and 2,455 Tanimoto >= 0.8 pairs crossing on HIV), and BACE's official `Model` split had the same compound in both train and test (`BACE_72` neutral in train, `BACE_669` protonated at the guanidine in test). On the 20,429 human proteins, MMseqs2 sequence identity (>= 0.3, coverage 0.8) found 137,696 related pairs where k-mer Jaccard (k=5, >= 0.3) found 4,126, a 33x difference.

`splitproof` turns split hygiene into a versioned YAML policy, a deterministic split, and a JSON certificate (`splitproof.cert.json`) that anyone can recompute from scratch with `splitproof verify`.

What it does in 0.1.0:
- Declares similarity constraints (`identity` via MMseqs2, exact `kmer_jaccard` / MinHash LSH, RDKit `tanimoto`, `ring_systems`, `canonical_smiles`, `exact_duplicate`), disjoint groups (donor, patient, plate, site), and one-directional temporal cutoffs.
- Builds single-linkage connected components into independent units, packs them deterministically under a seed, and refuses impossible homology blobs (`GIANT_UNIT`) rather than burying them.
- Reports nearest-neighbour similarity percentiles, independent unit counts, and a 1-NN similarity baseline on every split and audit.
- Separates exhaustive proof (`satisfied`) from heuristic or approximate evidence (`satisfied_unproven`, with `--require-exact` for CI) and never labels k-mer Jaccard as sequence identity.
- Runs from the CLI (`--json` on every command) or over stdio MCP (`splitproof mcp`) for Claude Code, Cursor, and Codex.

What 0.1.0 does not implement yet: 3D structural similarity (Foldseek/pocket), 2D/bipartite splits for interaction data, and an in-memory DataFrame API.

Repo and full benchmark tables: https://github.com/dnncha/splitproof
Website: https://dnncha.github.io/splitproof/
License: Apache-2.0
