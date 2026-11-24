# cutsite report

**Status: VIOLATED**

- entities: 240
- data digest: `sha256:ee6f7f5294593cba55ebed4fcd8a33de0ace7ea96e04d8001628072889fb205b`
- policy digest: `sha256:8aac50768045d463a6892285f82a76cf561b25079163ea1e638a1c685397e109`
- assignment digest: `sha256:9ad332034a61e75f7ad36a854b55582776bfacb636db191434dc2d71cc594300`
- cutsite 0.1.0, seed 7

## Folds

| fold | entities | target | drift | units |
| --- | --- | --- | --- | --- |
| test | 74 | 74 | +0.0% | 0 |
| train | 166 | 166 | +0.0% | 0 |

## Constraints

### sequence:kmer_jaccard@0.4

no pair above kmer_jaccard 0.4 on 'sequence' may cross folds — **VIOLATED**

- backend: exact_jaccard
- cross_fold_pairs_at_or_above_threshold: 217
- exactness: exact
- max_cross_fold_similarity: 0.87013
- pairs_at_or_above_threshold: 500

Worst offenders:

- `{'a': 'F25M0', 'b': 'F25M5', 'similarity': 0.87013, 'a_fold': 'train', 'b_fold': 'test'}`
- `{'a': 'F33M0', 'b': 'F33M4', 'similarity': 0.792531, 'a_fold': 'train', 'b_fold': 'test'}`
- `{'a': 'F03M0', 'b': 'F03M1', 'similarity': 0.763265, 'a_fold': 'test', 'b_fold': 'train'}`
- `{'a': 'F03M0', 'b': 'F03M3', 'similarity': 0.763265, 'a_fold': 'test', 'b_fold': 'train'}`
- `{'a': 'F19M0', 'b': 'F19M2', 'similarity': 0.741935, 'a_fold': 'train', 'b_fold': 'test'}`
- `{'a': 'F01M0', 'b': 'F01M4', 'similarity': 0.728, 'a_fold': 'train', 'b_fold': 'test'}`
- `{'a': 'F07M0', 'b': 'F07M4', 'similarity': 0.728, 'a_fold': 'train', 'b_fold': 'test'}`
- `{'a': 'F14M0', 'b': 'F14M1', 'similarity': 0.728, 'a_fold': 'train', 'b_fold': 'test'}`
- `{'a': 'F00M0', 'b': 'F00M2', 'similarity': 0.721116, 'a_fold': 'train', 'b_fold': 'test'}`
- `{'a': 'F16M0', 'b': 'F16M1', 'similarity': 0.721116, 'a_fold': 'test', 'b_fold': 'train'}`

### disjoint:donor_id

no value of 'donor_id' may appear in more than one fold — **VIOLATED**

- distinct_values: 6
- missing_values: 0
- values_in_more_than_one_fold: 6

Worst offenders:

- `{'value': 'D0', 'folds': ['test', 'train']}`
- `{'value': 'D1', 'folds': ['test', 'train']}`
- `{'value': 'D2', 'folds': ['test', 'train']}`
- `{'value': 'D3', 'folds': ['test', 'train']}`
- `{'value': 'D4', 'folds': ['test', 'train']}`
- `{'value': 'D5', 'folds': ['test', 'train']}`

## Independent units

- units: 6
- largest unit: 42 (17.5%)
- singleton units: 0
- size percentiles: {'p50': 42, 'p90': 42, 'p99': 42, 'max': 42}
- similarity exactness: exact

## Leakage diagnostics

Reference fold: `train`

### sequence:kmer_jaccard@0.4

- metric: kmer_jaccard via exact_jaccard
- held-out items compared: 74
- nearest reference similarity: {'min': 0.325153, 'p50': 0.630189, 'p90': 0.721116, 'p99': 0.792531, 'max': 0.87013, 'mean': 0.620862}
- at or above the declared threshold: 73
- nearest-neighbour baseline: spearman=0.881475 over 74 items
- a nearest-neighbour copy of the reference fold scores this much; a model that does not clearly beat it has learned nothing beyond similarity

## What this does and does not prove

cutsite proves exactly the constraints the policy declares, measured by the named backend. A `proven` status means no pair at or above the declared threshold crosses folds, and the backend searched every pair. A `no violation found` status means the backend is sensitive but heuristic, so absence of a crossing pair is evidence, not proof. cutsite sets no thresholds of its own and passes no judgement on constraints you did not declare.
