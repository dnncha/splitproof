# cutsite report

**Status: proven**

- entities: 240
- data digest: `sha256:ee6f7f5294593cba55ebed4fcd8a33de0ace7ea96e04d8001628072889fb205b`
- policy digest: `sha256:8aac50768045d463a6892285f82a76cf561b25079163ea1e638a1c685397e109`
- assignment digest: `sha256:c5b409bbc5c8a5f9e89c147199b4359e877bd82433acb5fe15e71f4f9ab8355a`
- cutsite 0.1.0, seed 7

## Folds

| fold | entities | target | drift | units |
| --- | --- | --- | --- | --- |
| test | 36 | 48 | -25.0% | 1 |
| train | 204 | 192 | +6.2% | 5 |

## Constraints

### sequence:kmer_jaccard@0.4

no pair above kmer_jaccard 0.4 on 'sequence' may cross folds — **proven**

- backend: exact_jaccard
- cross_fold_pairs_at_or_above_threshold: 0
- exactness: exact
- max_cross_fold_similarity: None
- pairs_at_or_above_threshold: 500

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
- held-out items compared: 36
- nearest reference similarity: {'min': 0.0, 'p50': 0.00232, 'p90': 0.004651, 'p99': 0.004651, 'max': 0.004651, 'mean': 0.001871}
- at or above the declared threshold: 0
- nearest-neighbour baseline: spearman=-0.027364 over 36 items
- a nearest-neighbour copy of the reference fold scores this much; a model that does not clearly beat it has learned nothing beyond similarity

## What this does and does not prove

cutsite proves exactly the constraints the policy declares, measured by the named backend. A `proven` status means no pair at or above the declared threshold crosses folds, and the backend searched every pair. A `no violation found` status means the backend is sensitive but heuristic, so absence of a crossing pair is evidence, not proof. cutsite sets no thresholds of its own and passes no judgement on constraints you did not declare.
