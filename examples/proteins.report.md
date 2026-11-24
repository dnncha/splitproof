# splitproof report

**Status: proven**

- entities: 240
- data digest: `sha256:ee6f7f5294593cba55ebed4fcd8a33de0ace7ea96e04d8001628072889fb205b`
- policy digest: `sha256:8aac50768045d463a6892285f82a76cf561b25079163ea1e638a1c685397e109`
- assignment digest: `sha256:bf38f87a2c2d6d40544adff5d096212c4f07bcad82efd6ebd4d6f25f0e4478d7`
- splitproof 0.1.0, seed 7

## Folds

| fold | entities | target | drift | units |
| --- | --- | --- | --- | --- |
| test | 42 | 48 | -12.5% | 1 |
| train | 198 | 192 | +3.1% | 5 |

## Constraints

### sequence:kmer_jaccard@0.4

no pair above kmer_jaccard 0.4 on 'sequence' may cross folds — **proven**

- backend: exact_jaccard
- cross_fold_pairs_at_or_above_threshold: 0
- cross_fold_pairs_by_fold_pair: {}
- exactness: exact
- max_cross_fold_similarity: None
- pairs_at_or_above_threshold: 500
- values_absent: 0
- values_unparsable: 0

### disjoint:donor_id

no value of 'donor_id' may appear in more than one fold — **proven**

- distinct_values: 6
- missing_values: 0
- values_in_more_than_one_fold: 0

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
- held-out items compared: 42
- nearest reference similarity: {'min': 0.00232, 'p50': 0.00232, 'p90': 0.00232, 'p99': 0.00232, 'max': 0.00232, 'mean': 0.00232}
- at or above the declared threshold: 0
- nearest-neighbour baseline: spearman=0.117907 over 31 items
- what a nearest-neighbour copy of the reference fold scores. Similarity reasoning is a legitimate model, so this is neither a floor nor a ceiling: a model that does not clearly beat it has not shown it learned anything beyond similarity, and beating it is not on its own evidence that the split is sound

## What this does and does not prove

splitproof proves exactly the constraints the policy declares, measured by the named backend. A `proven` status means no pair at or above the declared threshold crosses folds, and the backend searched every pair. A `no violation found` status means the backend is sensitive but heuristic, so absence of a crossing pair is evidence, not proof. splitproof sets no thresholds of its own and passes no judgement on constraints you did not declare.
