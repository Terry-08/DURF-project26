# Order-to-space TDR results

## Analysis set

- Combined: 40 ensembles, mean test accuracy 0.963.
- Separated: 40 ensembles, mean test accuracy 0.964.
- Balanced combined: 40 ensembles, mean test accuracy 0.969.
- Balanced separated: 39 ensembles, mean test accuracy 0.970.

Balanced-separated ensemble 9 was excluded by the preregistered 0.90 accuracy
threshold. The primary analysis uses correct trials, five-fold held-out TDR,
equal weighting of order-location-sequence strata, and value1, value2, and
sequence nuisance regressors.

## Cluster-corrected results

| Model | Unique order | Unique location | Unique space |
|---|---:|---:|---:|
| Combined | not significant | 1500-3200 ms | 1500-3200 ms |
| Separated | 1900-2640 ms | 1500-3200 ms | 1500-3200 ms |
| Balanced combined | not significant | 1500-3200 ms | 1560-3200 ms |
| Balanced separated | not significant | 1500-3200 ms | 1740-3200 ms |

Significance uses an ensemble-level one-sided cluster sign-flip test with
5,000 permutations and family-wise error controlled at 0.05. Exact cluster
masses and p-values are in `figures/tdr_significant_clusters.csv`.

## Interpretation

Location information is stable from 1500 ms onward in every model. This is
expected because the current task supplies offer location during the offer
epochs rather than introducing a new left/right mapping only at target onset.

Unique spatial-choice information emerges at 1500 ms in the original models
and later in the balanced models (1560 or 1740 ms), then grows through the
memory, target, and response periods. The balanced manipulation therefore
delays but does not remove early spatial-choice formation.

After controlling offer values and sequence, a reliable unique chosen-order
component appears only in the separated model, from 1900 to 2640 ms. The other
three architectures can contain order-correlated regression axes, but those
axes do not explain held-out activity beyond value, location, and chosen side.
Consequently, the current evidence does not support a universal serial process
in which a stable order representation is later replaced by a spatial one.
It is more consistent with direct or early spatial-choice construction, with
a temporary independent order component in the separated architecture.

The mean raw order-space axis angles range from about 85 to 93 degrees across
models, indicating largely distinct population directions. Near-orthogonality
does not establish causal transformation, and the TDR results remain
observational. A follow-up perturbation should suppress or inject the order
axis before target onset and test whether the resulting left/right output
changes in opposite directions for `loc12=12` and `loc12=21`.
