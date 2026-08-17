from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold


REGRESSOR_NAMES = np.asarray(
    ["intercept", "order", "location", "space", "value1", "value2", "sequence"]
)
AXIS_NAMES = np.asarray(["order", "location", "space"])
PREDICTOR_COLUMNS = {"order": 1, "location": 2, "space": 3}

CONDITION_SPECS = (
    (1, 1, "choose 1 | loc12=12 | left"),
    (1, -1, "choose 1 | loc12=21 | right"),
    (-1, 1, "choose 2 | loc12=12 | right"),
    (-1, -1, "choose 2 | loc12=21 | left"),
)


def loc12_to_code(params):
    value = params.get("loc12_label", params.get("loc12"))
    if isinstance(value, bytes):
        value = value.decode("ascii")
    if isinstance(value, str):
        if value not in ("12", "21"):
            raise ValueError(f"Unexpected loc12 label: {value}")
        return 1 if value == "12" else -1

    value = np.asarray(value).reshape(-1)
    if value.size < 2:
        raise ValueError(f"loc12 must be a label or two-channel one-hot: {value}")
    return 1 if int(np.argmax(value[:2])) == 0 else -1


def response_choice(model_output, trial_params, dt):
    response = np.zeros((model_output.shape[0], model_output.shape[2]), dtype=float)
    for trial_index, params in enumerate(trial_params):
        onset = int(round(params.get("fixation_offset", 3000) / dt))
        offset = int(round(params.get("end", 3200) / dt))
        onset = max(0, min(onset, model_output.shape[1] - 1))
        offset = max(onset + 1, min(offset, model_output.shape[1]))
        response[trial_index] = model_output[trial_index, onset:offset].mean(axis=0)
    return np.argmax(response, axis=1)


def load_activity(activity_path, dt=10, correct_only=True):
    activity_path = Path(activity_path)
    with np.load(activity_path, allow_pickle=True) as data:
        model_state = np.asarray(data["model_state"], dtype=np.float64)
        model_output = np.asarray(data["model_output"], dtype=np.float64)
        trial_params = data["trial_params"]

    side_index = response_choice(model_output, trial_params, dt)
    chosen_side = np.where(side_index == 0, 1, -1)
    location = np.asarray([loc12_to_code(params) for params in trial_params])
    chosen_order = chosen_side * location
    target_order = np.asarray(
        [1 if int(params["chosen_offer"]) == 1 else -1 for params in trial_params]
    )
    correct = chosen_order == target_order

    sequence = np.asarray(
        [1 if params["seqAB"] == "AB" else -1 for params in trial_params]
    )
    q_a = np.asarray([float(params["qA"]) for params in trial_params])
    q_b = np.asarray([float(params["qB"]) for params in trial_params])
    value_a = q_a / 4.0
    value_b = q_b / 8.0
    value1 = np.where(sequence == 1, value_a, value_b)
    value2 = np.where(sequence == 1, value_b, value_a)

    keep = correct if correct_only else np.ones(len(trial_params), dtype=bool)
    labels = {
        "order": chosen_order[keep].astype(int),
        "location": location[keep].astype(int),
        "space": chosen_side[keep].astype(int),
        "sequence": sequence[keep].astype(int),
        "value1": value1[keep],
        "value2": value2[keep],
        "trial_index": np.flatnonzero(keep),
    }
    metadata = {
        "accuracy": float(correct.mean()),
        "n_trials_total": int(len(trial_params)),
        "n_trials_included": int(keep.sum()),
        "correct_only": bool(correct_only),
    }
    return model_state[keep], labels, metadata


def make_strata(labels):
    return np.asarray(
        [
            f"{order}:{location}:{sequence}"
            for order, location, sequence in zip(
                labels["order"], labels["location"], labels["sequence"]
            )
        ]
    )


def stratum_weights(strata):
    strata = np.asarray(strata)
    _, inverse, counts = np.unique(strata, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights / weights.mean()


def design_matrix(labels, indices, value_stats=None):
    indices = np.asarray(indices)
    value1 = labels["value1"][indices]
    value2 = labels["value2"][indices]
    if value_stats is None:
        means = np.asarray([value1.mean(), value2.mean()])
        scales = np.asarray([value1.std(), value2.std()])
        scales[scales < 1e-8] = 1.0
        value_stats = (means, scales)
    means, scales = value_stats

    matrix = np.column_stack(
        [
            np.ones(len(indices)),
            labels["order"][indices],
            labels["location"][indices],
            labels["space"][indices],
            (value1 - means[0]) / scales[0],
            (value2 - means[1]) / scales[1],
            labels["sequence"][indices],
        ]
    )
    return matrix.astype(float), value_stats


def weighted_lstsq(design, response, weights):
    design = np.asarray(design, dtype=float)
    response = np.asarray(response, dtype=float)
    original_shape = response.shape[1:]
    response_2d = response.reshape(response.shape[0], -1)
    sqrt_weights = np.sqrt(np.asarray(weights, dtype=float))[:, None]
    coefficients = np.linalg.lstsq(
        design * sqrt_weights,
        response_2d * sqrt_weights,
        rcond=None,
    )[0]
    return coefficients.reshape((design.shape[1],) + original_shape)


def weighted_sse(observed, predicted, weights):
    residual = observed - predicted
    return np.sum(weights[:, None, None] * residual**2, axis=(0, 2))


def weighted_sst(observed, weights):
    denominator = weights.sum()
    mean = np.sum(weights[:, None, None] * observed, axis=0) / denominator
    return np.sum(weights[:, None, None] * (observed - mean) ** 2, axis=(0, 2))


def normalize_axis(axis):
    norm = float(np.linalg.norm(axis))
    if norm < 1e-10:
        return np.full_like(axis, np.nan), norm
    return axis / norm, norm


def cosine_timecourse(coefficients, reference_axis):
    norms = np.linalg.norm(coefficients, axis=1)
    values = coefficients @ reference_axis
    output = np.full(len(norms), np.nan)
    valid = norms > 1e-10
    output[valid] = values[valid] / norms[valid]
    return output


def first_sustained(time, condition, start_ms=1500, consecutive=5):
    condition = np.asarray(condition, dtype=bool) & (time >= start_ms)
    if consecutive <= 1:
        matches = np.flatnonzero(condition)
        return float(time[matches[0]]) if len(matches) else np.nan
    run = np.convolve(condition.astype(int), np.ones(consecutive, dtype=int), mode="valid")
    matches = np.flatnonzero(run == consecutive)
    return float(time[matches[0]]) if len(matches) else np.nan


def run_cross_validated_tdr(
    model_state,
    labels,
    dt=10,
    start_ms=1500,
    end_ms=3200,
    order_window=(2000, 2500),
    location_window=(1500, 2000),
    space_window=(3000, 3200),
    n_splits=5,
    random_state=0,
):
    n_trials, n_time_total, n_units = model_state.shape
    start = max(0, int(round(start_ms / dt)))
    stop = min(n_time_total, int(round(end_ms / dt)) + 1)
    if stop - start < 2:
        raise ValueError(f"Invalid TDR window: {start_ms}-{end_ms} ms")
    states = model_state[:, start:stop]
    time = np.arange(start, stop) * dt
    n_time = len(time)

    strata = make_strata(labels)
    _, stratum_counts = np.unique(strata, return_counts=True)
    cv_eff = min(int(n_splits), int(stratum_counts.min()))
    if cv_eff < 2:
        raise ValueError("At least two trials per order-location-sequence stratum are required.")

    split = StratifiedKFold(
        n_splits=cv_eff, shuffle=True, random_state=random_state
    )
    predictor_names = tuple(PREDICTOR_COLUMNS)
    n_predictors = len(predictor_names)

    unique_r2_folds = np.full((cv_eff, n_predictors, n_time), np.nan)
    full_r2_folds = np.full((cv_eff, n_time), np.nan)
    beta_norm_folds = np.full((cv_eff, n_predictors, n_time), np.nan)
    projection_effect_folds = np.full(
        (cv_eff, n_predictors, n_predictors, n_time), np.nan
    )
    alignment_folds = np.full((cv_eff, 2, n_time), np.nan)
    instant_angle_folds = np.full((cv_eff, n_time), np.nan)
    anchor_axes_folds = np.full((cv_eff, n_predictors, n_units), np.nan)
    axis_angle_folds = np.full(cv_eff, np.nan)
    residual_fraction_folds = np.full(cv_eff, np.nan)
    trial_coordinates = np.full((n_trials, n_time, 2), np.nan)
    trial_raw_projections = np.full((n_trials, n_time, n_predictors), np.nan)

    anchor_windows = (order_window, location_window, space_window)
    anchor_masks = [
        (time >= window[0]) & (time <= window[1]) for window in anchor_windows
    ]
    if not all(mask.any() for mask in anchor_masks):
        raise ValueError("At least one anchor window is outside the analysis time axis.")

    all_indices = np.arange(n_trials)
    for fold_index, (train_index, test_index) in enumerate(
        split.split(all_indices, strata)
    ):
        train_strata = strata[train_index]
        test_strata = strata[test_index]
        train_weights = stratum_weights(train_strata)
        test_weights = stratum_weights(test_strata)

        x_train, value_stats = design_matrix(labels, train_index)
        x_test, _ = design_matrix(labels, test_index, value_stats=value_stats)
        if np.linalg.matrix_rank(x_train) < x_train.shape[1]:
            raise ValueError(
                f"Rank-deficient design matrix in fold {fold_index}; "
                "use a richer or better balanced trial set."
            )

        neural_mean = states[train_index].mean(axis=(0, 1))
        neural_scale = states[train_index].std(axis=(0, 1))
        neural_scale[neural_scale < 1e-8] = 1.0
        y_train = (states[train_index] - neural_mean) / neural_scale
        y_test = (states[test_index] - neural_mean) / neural_scale

        coefficients = weighted_lstsq(x_train, y_train, train_weights)
        prediction = np.einsum("np,ptu->ntu", x_test, coefficients)
        sst = weighted_sst(y_test, test_weights)
        sst[sst < 1e-12] = np.nan
        sse_full = weighted_sse(y_test, prediction, test_weights)
        full_r2_folds[fold_index] = 1.0 - sse_full / sst

        for predictor_index, predictor_name in enumerate(predictor_names):
            column = PREDICTOR_COLUMNS[predictor_name]
            keep_columns = [i for i in range(x_train.shape[1]) if i != column]
            reduced_coefficients = weighted_lstsq(
                x_train[:, keep_columns], y_train, train_weights
            )
            reduced_prediction = np.einsum(
                "np,ptu->ntu", x_test[:, keep_columns], reduced_coefficients
            )
            sse_reduced = weighted_sse(y_test, reduced_prediction, test_weights)
            unique_r2_folds[fold_index, predictor_index] = (
                sse_reduced - sse_full
            ) / sst
            beta_norm_folds[fold_index, predictor_index] = np.linalg.norm(
                coefficients[column], axis=1
            )

        axes = []
        for predictor_index, predictor_name in enumerate(predictor_names):
            column = PREDICTOR_COLUMNS[predictor_name]
            raw_axis = coefficients[column, anchor_masks[predictor_index]].mean(axis=0)
            axis, _ = normalize_axis(raw_axis)
            if np.isnan(axis).any():
                raise ValueError(
                    f"Could not estimate {predictor_name} anchor axis in fold "
                    f"{fold_index}."
                )
            axes.append(axis)
        axes = np.stack(axes)
        anchor_axes_folds[fold_index] = axes

        order_axis, location_axis, space_axis = axes
        dot_order_space = float(np.clip(order_axis @ space_axis, -1.0, 1.0))
        axis_angle_folds[fold_index] = np.degrees(np.arccos(dot_order_space))
        residual_space = space_axis - dot_order_space * order_axis
        _, residual_norm = normalize_axis(residual_space)
        residual_fraction_folds[fold_index] = residual_norm
        if residual_norm < 1e-8:
            raise ValueError("Order and space axes are numerically collinear.")

        raw_projections = np.einsum("ntu,au->nta", y_test, axes)
        order_space_basis = np.stack([order_axis, space_axis], axis=1)
        gram_inverse = np.linalg.inv(order_space_basis.T @ order_space_basis)
        coordinates = np.einsum(
            "nta,ab->ntb", raw_projections[:, :, [0, 2]], gram_inverse
        )
        trial_coordinates[test_index] = coordinates
        trial_raw_projections[test_index] = raw_projections

        projection_coefficients = weighted_lstsq(
            x_test, raw_projections, test_weights
        )
        for axis_index in range(n_predictors):
            for predictor_index, predictor_name in enumerate(predictor_names):
                column = PREDICTOR_COLUMNS[predictor_name]
                projection_effect_folds[
                    fold_index, axis_index, predictor_index
                ] = projection_coefficients[column, :, axis_index]

        beta_order = coefficients[PREDICTOR_COLUMNS["order"]]
        beta_space = coefficients[PREDICTOR_COLUMNS["space"]]
        alignment_folds[fold_index, 0] = cosine_timecourse(beta_order, space_axis)
        alignment_folds[fold_index, 1] = cosine_timecourse(beta_space, space_axis)
        beta_order_norm = np.linalg.norm(beta_order, axis=1)
        beta_space_norm = np.linalg.norm(beta_space, axis=1)
        valid = (beta_order_norm > 1e-10) & (beta_space_norm > 1e-10)
        cosine = np.full(n_time, np.nan)
        cosine[valid] = np.sum(beta_order[valid] * beta_space[valid], axis=1) / (
            beta_order_norm[valid] * beta_space_norm[valid]
        )
        instant_angle_folds[fold_index, valid] = np.degrees(
            np.arccos(np.clip(cosine[valid], -1.0, 1.0))
        )

    condition_trajectories = []
    condition_counts = []
    condition_labels = []
    for order, location, label in CONDITION_SPECS:
        condition = (labels["order"] == order) & (
            labels["location"] == location
        )
        count = int(condition.sum())
        if count == 0:
            raise ValueError(f"Empty TDR trajectory condition: {label}")
        condition_trajectories.append(trial_coordinates[condition].mean(axis=0))
        condition_counts.append(count)
        condition_labels.append(label)

    unique_r2 = np.nanmean(unique_r2_folds, axis=0)
    metrics = {
        "order_peak_time": float(time[np.nanargmax(unique_r2[0])]),
        "space_peak_time": float(time[np.nanargmax(unique_r2[2])]),
        "descriptive_crossover_time": first_sustained(
            time, unique_r2[2] > unique_r2[0], start_ms=start_ms
        ),
        "axis_angle_order_space": float(np.nanmean(axis_angle_folds)),
        "space_residual_fraction": float(np.nanmean(residual_fraction_folds)),
    }

    return {
        "time": time.astype(float),
        "regressor_names": REGRESSOR_NAMES,
        "axis_names": AXIS_NAMES,
        "condition_labels": np.asarray(condition_labels),
        "condition_counts": np.asarray(condition_counts, dtype=int),
        "condition_trajectories": np.stack(condition_trajectories),
        "unique_r2": unique_r2,
        "unique_r2_folds": unique_r2_folds,
        "full_r2": np.nanmean(full_r2_folds, axis=0),
        "full_r2_folds": full_r2_folds,
        "beta_norm": np.nanmean(beta_norm_folds, axis=0),
        "beta_norm_folds": beta_norm_folds,
        "projection_effect": np.nanmean(projection_effect_folds, axis=0),
        "projection_effect_folds": projection_effect_folds,
        "alignment_to_space": np.nanmean(alignment_folds, axis=0),
        "instant_order_space_angle": np.nanmean(instant_angle_folds, axis=0),
        "anchor_axes": np.nanmean(anchor_axes_folds, axis=0),
        "axis_angle_folds": axis_angle_folds,
        "space_residual_fraction_folds": residual_fraction_folds,
        "n_splits": np.asarray(cv_eff),
        "metrics": metrics,
    }


def result_to_npz(result, metadata):
    output = {key: value for key, value in result.items() if key != "metrics"}
    output.update(
        {
            "metric_names": np.asarray(list(result["metrics"])),
            "metric_values": np.asarray(list(result["metrics"].values()), dtype=float),
        }
    )
    for key, value in metadata.items():
        output[key] = np.asarray(value)
    return output


def load_metrics(npz_file):
    names = [str(name) for name in npz_file["metric_names"]]
    values = np.asarray(npz_file["metric_values"], dtype=float)
    return dict(zip(names, values))
