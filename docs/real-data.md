# splitproof on public benchmarks

Measured on 24 November 2026 with splitproof between commits f39d777 and fe73796, RDKit
2026.09, on a Colab CPU runtime. The commits in that range change speed and fold packing, not
any audit number below. Every split splitproof built from these policies re-verified against
its own certificate, except two not run to save time: HIV under Tanimoto (seven minutes per
verify) and human proteins under MMseqs2 identity (twenty). No command printed a traceback.

`scripts/realdata.py` reproduces every number here. The datasets are MoleculeNet CSVs from DeepChem's public bucket, read as published. Each row
got an id where the file had none. The "random" column is a seeded 80/20 split, the split most
pipelines start from.

## Random 80/20 splits leak on every chemistry benchmark

| Dataset | Rows | Key | Related pairs | Pairs crossing train/test | Unparsable SMILES |
|---|---|---|---|---|---|
| BACE | 1513 | same compound (canonical SMILES) | 1 | 1 | 0 |
| BACE | 1513 | Tanimoto ≥ 0.8 | 1624 | 491 | 0 |
| BACE | 1513 | same ring systems | 11565 | 3774 | 0 |
| BBBP | 2050 | same compound | 78 | 25 | 11 |
| BBBP | 2050 | Tanimoto ≥ 0.8 | 251 | 64 | 11 |
| ClinTox | 1484 | same compound | 61 | 17 | 4 |
| ClinTox | 1484 | Tanimoto ≥ 0.8 | 169 | 48 | 4 |
| ESOL | 1128 | same compound | 11 | 3 | 0 |
| ESOL | 1128 | Tanimoto ≥ 0.8 | 224 | 79 | 0 |
| Lipophilicity | 4200 | same compound | 0 | 0 | 0 |
| Lipophilicity | 4200 | Tanimoto ≥ 0.8 | 1217 | 411 | 0 |
| HIV | 41127 | same compound | 197 | 55 | 7 |
| HIV | 41127 | Tanimoto ≥ 0.8 | 7455 | 2455 | 7 |

"Same compound" is after salt stripping, fragment choice and charge neutralisation, with
stereochemistry kept. Lipophilicity is the only set with no duplicate compounds; it still puts
411 near-identical pairs across a random split.

Rows whose SMILES RDKit cannot parse are counted, not dropped, and a clean result on a key with
any unparsable row is reported as `satisfied_unproven`, never `proven`.

## Human proteins

All 20429 reviewed human proteins in UniProt (Swiss-Prot, organism 9606), downloaded on the
day, with the same seeded 80/20 random split:

| Key | Related pairs | Pairs crossing train/test | Independent units | Largest unit |
|---|---|---|---|---|
| k-mer Jaccard ≥ 0.3 (k=5) | 4126 | 1316 | 18877 | 27 |
| MMseqs2 identity ≥ 0.3, coverage 0.8 | 137696 | 46874 | 12899 | 430 |

k-mer Jaccard is a lower bound on homology, not identity, and misses most of it: sequence
identity finds 33 times as many related pairs. A protein benchmark checked only with k-mers
should be read as unchecked for remote homology.

The identity audit took 20 minutes on one core, because splitproof runs MMseqs2 with a single
thread.

## BACE's own split puts one compound on both sides

BACE ships a `Model` column assigning each row to train, valid or test. Audited with valid and
test together as the held-out side, one compound sits in both:

| Id | Fold | SMILES |
|---|---|---|
| BACE_72 | train | `Clc1cc(cc(Cl)c1NC(=O)C)CNC(NC(=O)Cn1c2cc(ccc2cc1)C#N)=N` |
| BACE_669 | test | `Clc1cc(cc(Cl)c1NC(=O)C)CNC(=[NH2+])NC(=O)Cn1c2cc(ccc2cc1)C#N` |

They are the same molecule, neutral in one row and protonated at the guanidine in the other.
A comparison of raw SMILES strings does not see it; charge neutralisation does.

## What the runs found in splitproof

Two performance defects, both fixed before these numbers were taken:

- Hashing the dataset rebuilt an n-item list for every cell, so every command was quadratic in
  rows. 470,000 rows took more than five minutes; they now take under two seconds.
- Each molecule rebuilt RDKit's salt remover, re-reading its dictionary file. On HIV that was
  about 23 of 70 seconds per command.
