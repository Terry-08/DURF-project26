import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
)


METHODS = ("standard", "grouped", "value_matched")
LABELS = ("side", "juice", "order")


def _as_dict(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, dict):
            return item
    raise TypeError("Trial parameter is not a dictionary: %r" % (type(value),))


def loc12_to_label(params):
    label = params.get("loc12_label")
    if label is not None:
        return str(label)
    loc12 = np.asarray(params["loc12"])
    return "12" if loc12[0] == 1 else "21"


def response_average(model_output, mask, trial_params, dt):
    if mask is not None and np.any(mask):
        denominator = np.sum(mask, axis=1)
        denominator[denominator == 0] = 1
        return np.sum(model_output * mask, axis=1) / denominator

    response = np.zeros((model_output.shape[0], model_output.shape[2]))
    for trial_index, raw_params in enumerate(trial_params):
        params = _as_dict(raw_params)
        onset = params["fixation_offset"]
        offset = params["end"]
        first = max(0, int(round(onset / dt)))
        last = min(model_output.shape[1], int(round(offset / dt)))
        response[trial_index] = np.mean(model_output[trial_index, first:last], axis=0)
    return response


def derive_trial_information(model_output, mask, trial_params, dt):
    params = [_as_dict(value) for value in trial_params]
    response = response_average(model_output, mask, params, dt)
    side = np.argmax(response, axis=1).astype(np.int8)  # 0=left, 1=right

    q_a = np.asarray([value["qA"] for value in params], dtype=float)
    q_b = np.asarray([value["qB"] for value in params], dtype=float)
    seq_ab = np.asarray([str(value["seqAB"]) for value in params])
    loc12 = np.asarray([loc12_to_label(value) for value in params])

    offer1_left = loc12 == "12"
    chosen_offer = np.where(
        side == 0,
        np.where(offer1_left, 1, 2),
        np.where(offer1_left, 2, 1),
    )
    order = (chosen_offer == 2).astype(np.int8)  # 0=offer 1, 1=offer 2
    juice = np.where(
        seq_ab == "AB",
        chosen_offer == 2,
        chosen_offer == 1,
    ).astype(np.int8)  # 0=A, 1=B

    teacher_side = np.asarray([int(value["choice"]) for value in params], dtype=np.int8)
    teacher_order = np.asarray(
        [int(value["chosen_offer"] == 2) for value in params], dtype=np.int8
    )
    teacher_juice = np.asarray([int(value["chooseB"]) for value in params], dtype=np.int8)

    return {
        "labels": {"side": side, "juice": juice, "order": order},
        "teacher_labels": {
            "side": teacher_side,
            "juice": teacher_juice,
            "order": teacher_order,
        },
        "qA": q_a,
        "qB": q_b,
        "seqAB": seq_ab,
        "loc12": loc12,
        "chosen_offer": chosen_offer.astype(np.int8),
        "response_margin": np.abs(response[:, 1] - response[:, 0]),
    }


def event_windows(trial_params):
    params = [_as_dict(value) for value in trial_params]
    keys = {
        "offer1": ("stimulus_1_onset", "stimulus_1_offset"),
        "offer2": ("stimulus_2_onset", "stimulus_2_offset"),
        "target": ("target_onset", "fixation_offset"),
        "response": ("fixation_offset", "end"),
    }
    windows = {}
    for name, (onset_key, offset_key) in keys.items():
        onsets = {float(value[onset_key]) for value in params}
        offsets = {float(value[offset_key]) for value in params}
        if len(onsets) != 1 or len(offsets) != 1:
            raise ValueError("Variable %s timing is not supported in this analysis." % name)
        windows[name] = (onsets.pop(), offsets.pop())
    return windows


def make_group_ids(q_a, q_b):
    keys = [(round(float(a), 8), round(float(b), 8)) for a, b in zip(q_a, q_b)]
    lookup = {}
    groups = np.empty(len(keys), dtype=np.int32)
    for index, key in enumerate(keys):
        if key not in lookup:
            lookup[key] = len(lookup)
        groups[index] = lookup[key]
    return groups


def make_stratum_ids(q_a, q_b, seq_ab, loc12):
    keys = [
        (round(float(a), 8), round(float(b), 8), str(seq), str(loc))
        for a, b, seq, loc in zip(q_a, q_b, seq_ab, loc12)
    ]
    lookup = {}
    strata = np.empty(len(keys), dtype=np.int32)
    for index, key in enumerate(keys):
        if key not in lookup:
            lookup[key] = len(lookup)
        strata[index] = lookup[key]
    return strata


def balanced_matched_indices(labels, strata, random_seed):
    labels = np.asarray(labels)
    strata = np.asarray(strata)
    classes = np.unique(labels)
    if len(classes) != 2:
        return np.asarray([], dtype=np.int64), 0

    rng = np.random.default_rng(random_seed)
    selected = []
    matched_strata = 0
    for stratum in np.unique(strata):
        in_stratum = np.flatnonzero(strata == stratum)
        by_class = [in_stratum[labels[in_stratum] == value] for value in classes]
        count = min(len(indices) for indices in by_class)
        if count == 0:
            continue
        matched_strata += 1
        for indices in by_class:
            selected.extend(rng.choice(indices, size=count, replace=False).tolist())

    return np.sort(np.asarray(selected, dtype=np.int64)), matched_strata


def make_standard_splits(labels, n_splits, n_repeats, random_seed):
    labels = np.asarray(labels)
    _, counts = np.unique(labels, return_counts=True)
    effective_splits = min(n_splits, int(np.min(counts)))
    if effective_splits < 2:
        return []
    cv = RepeatedStratifiedKFold(
        n_splits=effective_splits,
        n_repeats=n_repeats,
        random_state=random_seed,
    )
    return list(cv.split(np.zeros(len(labels)), labels))


def make_grouped_splits(labels, groups, n_splits, n_repeats, random_seed):
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    effective_splits = min(n_splits, len(np.unique(groups)))
    if effective_splits < 2:
        return []

    splits = []
    for repeat in range(n_repeats):
        cv = StratifiedGroupKFold(
            n_splits=effective_splits,
            shuffle=True,
            random_state=random_seed + repeat,
        )
        for train, test in cv.split(np.zeros(len(labels)), labels, groups):
            if len(np.unique(labels[train])) < 2 or len(np.unique(labels[test])) < 2:
                continue
            splits.append((train, test))
    return splits


def _balanced_accuracy(labels, predictions):
    recalls = []
    for value in np.unique(labels):
        selected = labels == value
        if np.any(selected):
            recalls.append(np.mean(predictions[selected] == value))
    return float(np.mean(recalls)) if len(recalls) == 2 else np.nan


def decode_time_series(model_state, labels, splits):
    n_time = model_state.shape[1]
    scores = np.full((n_time, len(splits)), np.nan, dtype=np.float32)
    if not splits:
        return scores

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for time_index in range(n_time):
            features = model_state[:, time_index, :]
            for fold_index, (train, test) in enumerate(splits):
                try:
                    classifier = LinearDiscriminantAnalysis()
                    classifier.fit(features[train], labels[train])
                    predictions = classifier.predict(features[test])
                    scores[time_index, fold_index] = _balanced_accuracy(
                        labels[test], predictions
                    )
                except (ValueError, np.linalg.LinAlgError):
                    continue
    return scores


def analyze_model(
    model_dir,
    task_name,
    result_path,
    dt=10,
    n_splits=5,
    n_repeats=1,
    random_seed=20260723,
    include_zero_offers=False,
):
    started = time.perf_counter()
    model_dir = Path(model_dir)
    result_path = Path(result_path)
    activity_path = model_dir / "activitityTestGrid.npz"

    with np.load(activity_path, allow_pickle=True) as activity:
        trial_params = activity["trial_params"]
        model_output = activity["model_output"]
        model_state = activity["model_state"]
        mask = activity["mask"] if "mask" in activity.files else None

    information = derive_trial_information(
        model_output=model_output,
        mask=mask,
        trial_params=trial_params,
        dt=dt,
    )
    windows = event_windows(trial_params)
    analysis_end = int(max(offset for _, offset in windows.values()))
    n_time = min(model_state.shape[1], int(np.ceil(analysis_end / dt)))

    valid = np.ones(len(trial_params), dtype=bool)
    if not include_zero_offers:
        valid &= (information["qA"] > 0) & (information["qB"] > 0)
    selected = np.flatnonzero(valid)

    states = model_state[selected, :n_time, :]
    q_a = information["qA"][selected]
    q_b = information["qB"][selected]
    seq_ab = information["seqAB"][selected]
    loc12 = information["loc12"][selected]
    groups = make_group_ids(q_a, q_b)
    strict_strata = make_stratum_ids(q_a, q_b, seq_ab, loc12)

    save_data = {
        "time": np.arange(n_time, dtype=float) * dt,
        "selected_trial_indices": selected,
        "qA": q_a,
        "qB": q_b,
        "seqAB": seq_ab,
        "loc12": loc12,
    }
    method_counts = {}

    for label_index, label_name in enumerate(LABELS):
        labels = information["labels"][label_name][selected]
        teacher_labels = information["teacher_labels"][label_name][selected]
        save_data["labels__%s" % label_name] = labels
        save_data["teacher_labels__%s" % label_name] = teacher_labels

        standard_splits = make_standard_splits(
            labels, n_splits, n_repeats, random_seed + 100 * label_index
        )
        save_data["scores__standard__%s" % label_name] = decode_time_series(
            states, labels, standard_splits
        )
        method_counts["standard__%s" % label_name] = {
            "trials": int(len(labels)),
            "folds": int(len(standard_splits)),
        }

        grouped_splits = make_grouped_splits(
            labels,
            groups,
            n_splits,
            n_repeats,
            random_seed + 1000 + 100 * label_index,
        )
        save_data["scores__grouped__%s" % label_name] = decode_time_series(
            states, labels, grouped_splits
        )
        method_counts["grouped__%s" % label_name] = {
            "trials": int(len(labels)),
            "groups": int(len(np.unique(groups))),
            "folds": int(len(grouped_splits)),
        }

        # Match the two choices within each exact value pair. Matching sequence
        # and location as well is too strict for the existing 10-repeat grid;
        # strict counts are retained below as a data-sufficiency diagnostic.
        matched_indices, matched_value_pairs = balanced_matched_indices(
            labels, groups, random_seed + 2000 + 100 * label_index
        )
        strict_indices, strict_strata_count = balanced_matched_indices(
            labels, strict_strata, random_seed + 4000 + 100 * label_index
        )
        matched_labels = labels[matched_indices]
        matched_splits = make_standard_splits(
            matched_labels,
            n_splits,
            n_repeats,
            random_seed + 3000 + 100 * label_index,
        ) if len(matched_indices) else []
        save_data["matched_indices__%s" % label_name] = matched_indices
        save_data["scores__value_matched__%s" % label_name] = decode_time_series(
            states[matched_indices], matched_labels, matched_splits
        )
        method_counts["value_matched__%s" % label_name] = {
            "trials": int(len(matched_indices)),
            "value_pairs": int(matched_value_pairs),
            "folds": int(len(matched_splits)),
            "strict_trials_available": int(len(strict_indices)),
            "strict_strata_available": int(strict_strata_count),
        }

    model_side = information["labels"]["side"]
    teacher_side = information["teacher_labels"]["side"]
    metadata = {
        "task": task_name,
        "model_dir": str(model_dir),
        "activity_file": str(activity_path),
        "dt": dt,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "random_seed": random_seed,
        "include_zero_offers": include_zero_offers,
        "total_trials": int(len(trial_params)),
        "analyzed_trials": int(len(selected)),
        "n_recurrent_units": int(model_state.shape[2]),
        "analysis_end_ms": analysis_end,
        "behavior_accuracy_all": float(np.mean(model_side == teacher_side)),
        "behavior_accuracy_analyzed": float(
            np.mean(model_side[selected] == teacher_side[selected])
        ),
        "mean_response_margin": float(np.mean(information["response_margin"])),
        "event_windows": windows,
        "value_matching": "balance choices separately within each exact (qA, qB) pair",
        "method_counts": method_counts,
    }
    save_data["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))

    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = result_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary_path, **save_data)
    temporary_path.replace(result_path)
    metadata["elapsed_seconds"] = time.perf_counter() - started
    return metadata


def load_result(path):
    with np.load(path, allow_pickle=True) as result:
        metadata = json.loads(str(result["metadata_json"]))
        data = {key: result[key] for key in result.files if key != "metadata_json"}
    return metadata, data
