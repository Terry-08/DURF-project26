# Order-to-space targeted dimensionality reduction

This directory is isolated from the legacy decoding and PCA analyses. It asks
how a chosen-order representation combines with `loc12` to form a chosen-side
representation after Offer 2 begins.

## Variables

- `order`: `+1` choose Offer 1, `-1` choose Offer 2.
- `location`: `+1` means `loc12=12`, `-1` means `loc12=21`.
- `space`: `order * location`; `+1` left, `-1` right.
- `value1`, `value2`, and `sequence` are nuisance regressors.

Each cross-validation fold estimates normalization and regression axes only
from training trials. Held-out trials provide four-condition trajectories and
the unique cross-validated variance associated with order, location, and the
order-by-location interaction.

The two-dimensional trajectory uses dual coordinates in the oblique span of
the raw order and space axes. Pure order-axis activity therefore maps to the
horizontal coordinate and pure space-axis activity maps to the vertical
coordinate without forcing the scientific axes to be orthogonal.

The fixed axes use these preregistered anchor windows:

- order: 2000-2500 ms;
- location: 1500-2000 ms;
- space: 3000-3200 ms.

The primary analysis includes correct trials and excludes models below 90%
test accuracy. Trials are weighted so each order-location-sequence stratum has
equal influence.

## Run

From the repository root with the project `.venv` active:

```powershell
python EconomicDecisionMakingReferenceFrame/Analysis/order_to_space_tdr/run_tdr.py --model all --ensemble all
python EconomicDecisionMakingReferenceFrame/Analysis/order_to_space_tdr/plot_tdr.py
```

For a quick one-ensemble validation:

```powershell
python EconomicDecisionMakingReferenceFrame/Analysis/order_to_space_tdr/run_tdr.py --model all --ensemble 0 --overwrite
python EconomicDecisionMakingReferenceFrame/Analysis/order_to_space_tdr/plot_tdr.py --bootstrap 200
```

Results are written under `order_to_space_tdr/results`; figures and the
ensemble summary are written under `order_to_space_tdr/figures`.

The ensemble summary uses networks as independent observations. Colored bars
at the bottom of the unique-R2 panel mark clusters that survive a one-sided
cluster-based sign-flip permutation test with family-wise error controlled at
0.05. Bootstrap intervals describe ensemble uncertainty but are not used as
the formal multiple-comparison-corrected significance test.

## Interpretation limits

TDR identifies neural dimensions associated with task variables after
controlling measured nuisance variables. It is not by itself evidence that an
order axis causally creates the spatial action. Activity perturbations along
the fitted axes are a separate follow-up analysis.
