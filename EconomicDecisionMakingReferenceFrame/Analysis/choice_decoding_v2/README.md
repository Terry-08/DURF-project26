# Three-method choice decoding

This directory is isolated from the original decoding scripts. It reads trained
model data from `Training/savedForLocal` and writes all new results below this
directory.

## Methods

- `standard`: shuffled stratified cross-validation on all nonzero offer trials.
- `grouped`: stratified group cross-validation; an `(qA, qB)` pair cannot occur
  in both train and test folds.
- `value_matched`: balances the two choices separately inside each identical
  `(qA, qB)` value pair before stratified cross-validation. The files also
  record how many trials would remain under stricter
  `(qA, qB, seqAB, loc12)` matching; the current 10-repeat grid is too small
  for that stricter version to support stable cross-validation.

All three methods use time-resolved LDA and balanced accuracy. Choice labels are
derived from each model's actual response, while teacher labels are retained in
the result files for quality control. By default, zero-offer forced-choice trials
are excluded and the unsupervised 3200-4000 ms tail is not analyzed.

Grid agreement with a single stochastic teacher draw is reported for quality
control but is not used as a default exclusion threshold. Near the indifference
point, two valid stochastic choices can disagree even when the model has learned
the correct choice probability.

The value-matched result controls offer values only. Sequence and location can
still be associated with the model's eventual choice, so this result must not be
described as a pure trial-specific choice signal. Strict full-condition matching
would provide that stronger test, but the saved diagnostic counts show that the
current grid does not contain enough repeated opposite choices to estimate it.

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe EconomicDecisionMakingReferenceFrame\Analysis\choice_decoding_v2\run_three_decoders.py --jobs 4
.\.venv\Scripts\python.exe EconomicDecisionMakingReferenceFrame\Analysis\choice_decoding_v2\plot_three_decoders.py
```

For a one-model validation run:

```powershell
.\.venv\Scripts\python.exe EconomicDecisionMakingReferenceFrame\Analysis\choice_decoding_v2\run_three_decoders.py --max-models 1 --jobs 1 --overwrite
```

## Outputs

- `results/<task>/<model>/three_decoding_methods.npz`: fold-level decoding
  scores and metadata for one model.
- `results/model_manifest.csv`: processing status and behavioral accuracy.
- `results/ensemble_summary.npz`: ensemble means and bootstrap intervals.
- `results/ensemble_summary.csv`: long-form summary table.
- `figures/*_three_methods.pdf`: three methods shown separately for each task.
- `figures/combined_vs_seperated_three_methods.pdf`: direct model comparison.
