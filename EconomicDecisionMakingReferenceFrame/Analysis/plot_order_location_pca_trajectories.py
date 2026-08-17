import argparse
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.decomposition import PCA


DT = 10
ACTIVITY_FILE = "activitityTest.npz"
DEFAULT_START_MS = 1500
DEFAULT_END_MS = 3200

MODEL_ROOT_NAMES = {
    "combined": ("spatialTask_combined",),
    "separated": ("spatialTask_seperated",),
    "balanced_combined": (
        "spatialTask_balanced_combined",
        "spatialTaskCombinedBalanced",
        "balanced_combined",
    ),
    "balanced_separated": ("spatialTask_balanced_separated",),
}

TIME_MARKERS = {
    1500: "X",
    2000: "o",
    2500: "s",
    3000: "D",
    3200: "P",
}

GROUP_STYLES = [
    {
        "order": "1",
        "location": "12",
        "side": "left",
        "color": "gold",
        "linestyle": "-",
    },
    {
        "order": "1",
        "location": "21",
        "side": "right",
        "color": "mediumpurple",
        "linestyle": "-",
    },
    {
        "order": "2",
        "location": "12",
        "side": "right",
        "color": "mediumpurple",
        "linestyle": "--",
    },
    {
        "order": "2",
        "location": "21",
        "side": "left",
        "color": "gold",
        "linestyle": "--",
    },
]

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot four order-by-location mean neural trajectories in a common "
            "three-dimensional PCA space."
        )
    )
    parser.add_argument(
        "--model",
        choices=("all", *MODEL_ROOT_NAMES.keys()),
        default="all",
        help="Model version to plot (default: all).",
    )
    parser.add_argument(
        "--ensemble",
        type=int,
        default=0,
        help="Ensemble index to select (default: 0).",
    )
    parser.add_argument(
        "--saved-root",
        type=Path,
        default=None,
        help="Override Training/savedForLocal.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Use one exact model directory; requires --model other than all.",
    )
    parser.add_argument(
        "--start-ms",
        type=int,
        default=DEFAULT_START_MS,
        help="Start of the PCA and trajectory window (default: 1500).",
    )
    parser.add_argument(
        "--end-ms",
        type=int,
        default=DEFAULT_END_MS,
        help="End of the PCA and trajectory window (default: 3200).",
    )
    parser.add_argument(
        "--dt",
        type=int,
        default=DT,
        help="Simulation step in milliseconds (default: 10).",
    )
    parser.add_argument(
        "--activity-file",
        default=ACTIVITY_FILE,
        help=f"Activity archive name (default: {ACTIVITY_FILE}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="PDF destination (default: Analysis/Figure/order_location_pca).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run the analysis without writing PDF files.",
    )
    return parser.parse_args()


def resolve_model_root(saved_root, model_name):
    candidates = [saved_root / name for name in MODEL_ROOT_NAMES[model_name]]
    return next((path for path in candidates if path.exists()), candidates[0])


def select_model_dir(task_root, ensemble_index, activity_file):
    if not task_root.exists():
        raise FileNotFoundError(f"Model collection not found: {task_root}")

    pattern = re.compile(rf"_{ensemble_index}_(?:Fail)?$")
    candidates = [
        path
        for path in task_root.iterdir()
        if path.is_dir()
        and pattern.search(path.name)
        and (path / activity_file).exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No ensemble {ensemble_index} containing {activity_file} under "
            f"{task_root}."
        )

    successful = [path for path in candidates if not path.name.endswith("Fail")]
    pool = successful if successful else candidates
    selected = sorted(pool, key=lambda path: path.name)[-1]
    if len(candidates) > 1:
        print(
            f"Warning: {len(candidates)} directories match ensemble "
            f"{ensemble_index}; using {selected.name}."
        )
    return selected


def loc12_to_label(params):
    value = params.get("loc12_label", params.get("loc12"))
    if isinstance(value, bytes):
        value = value.decode("ascii")
    if isinstance(value, str):
        if value not in ("12", "21"):
            raise ValueError(f"Unexpected loc12 label: {value}")
        return value

    value = np.asarray(value).reshape(-1)
    if value.size < 2:
        raise ValueError(f"loc12 must be a label or two-channel one-hot: {value}")
    return "12" if int(np.argmax(value[:2])) == 0 else "21"


def mean_response_output(model_output, trial_params, mask, dt):
    if mask is not None and np.any(mask):
        mask = np.asarray(mask)
        if mask.ndim == 2:
            mask = mask[:, :, None]
        denominator = np.sum(mask, axis=1)
        denominator[denominator == 0] = 1
        return np.sum(mask * model_output, axis=1) / denominator

    response = np.zeros((model_output.shape[0], model_output.shape[2]))
    for trial_index, params in enumerate(trial_params):
        response_onset = params.get("fixation_offset", 3000)
        response_offset = params.get("end", 3200)
        start = max(0, int(round(response_onset / dt)))
        stop = min(model_output.shape[1], int(round(response_offset / dt)))
        stop = max(start + 1, stop)
        response[trial_index] = model_output[trial_index, start:stop].mean(axis=0)
    return response


def load_activity_and_labels(model_dir, activity_file, dt):
    activity_path = model_dir / activity_file
    with np.load(activity_path, allow_pickle=True) as data:
        model_state = data["model_state"]
        model_output = data["model_output"]
        trial_params = data["trial_params"]
        mask = data["mask"] if "mask" in data.files else None

    response = mean_response_output(model_output, trial_params, mask, dt)
    choice_side = np.where(response[:, 1] > response[:, 0], "right", "left")
    loc12 = np.asarray([loc12_to_label(params) for params in trial_params])

    offer1_left = loc12 == "12"
    chose_offer1 = ((choice_side == "left") & offer1_left) | (
        (choice_side == "right") & ~offer1_left
    )
    choice_order = np.where(chose_offer1, "1", "2")

    labels = {
        "chosen_order": choice_order,
        "loc12": loc12,
        "chosen_side": choice_side,
    }
    return model_state, labels


def fit_and_project_states(model_state, start_ms, end_ms, dt):
    n_trials, n_time, n_units = model_state.shape
    start = max(0, int(round(start_ms / dt)))
    stop = min(n_time, int(round(end_ms / dt)) + 1)
    if stop - start < 2:
        raise ValueError(f"Invalid PCA window: {start_ms}-{end_ms} ms")
    if min(n_trials * (stop - start), n_units) < 3:
        raise ValueError("At least three samples and units are required for PC1-PC3.")

    states = model_state[:, start:stop, :]
    pca = PCA(n_components=3)
    projected = pca.fit_transform(states.reshape(-1, n_units))
    projected = projected.reshape(n_trials, stop - start, 3)
    time = np.arange(start, stop) * dt
    return pca, projected, time


def set_3d_limits(ax, trajectories):
    flat = np.asarray(trajectories).reshape(-1, 3)
    lower = np.nanmin(flat, axis=0)
    upper = np.nanmax(flat, axis=0)
    span = np.maximum(upper - lower, 1e-6)
    padding = span * 0.08
    lower -= padding
    upper += padding
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    ax.set_box_aspect((1.25, 1, 1))


def plot_four_condition_trajectories(
    model_name,
    model_dir,
    model_state,
    labels,
    start_ms,
    end_ms,
    dt,
):
    pca, projected, time = fit_and_project_states(
        model_state, start_ms, end_ms, dt
    )
    variance = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(13, 7), dpi=160, constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    trajectories = []

    for style in GROUP_STYLES:
        mask = (
            (labels["chosen_order"] == style["order"])
            & (labels["loc12"] == style["location"])
        )
        count = int(mask.sum())
        if count == 0:
            raise ValueError(
                f"{model_name}: empty choose {style['order']} x "
                f"loc12={style['location']} group."
            )

        observed_sides = np.unique(labels["chosen_side"][mask])
        if not np.array_equal(observed_sides, np.asarray([style["side"]])):
            raise ValueError(
                f"Inconsistent side labels for choose {style['order']} x "
                f"loc12={style['location']}: {observed_sides}"
            )

        trace = projected[mask].mean(axis=0)
        trajectories.append(trace)
        label = (
            f"choose {style['order']} | loc12={style['location']} | "
            f"{style['side']} (n={count})"
        )
        ax.plot(
            trace[:, 0],
            trace[:, 1],
            trace[:, 2],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.6,
            label=label,
        )

        for boundary_ms, marker in TIME_MARKERS.items():
            if time[0] <= boundary_ms <= time[-1]:
                index = int(np.argmin(np.abs(time - boundary_ms)))
                ax.scatter(
                    trace[index, 0],
                    trace[index, 1],
                    trace[index, 2],
                    marker=marker,
                    color=style["color"],
                    edgecolors="black",
                    linewidths=0.6,
                    s=58 if boundary_ms == start_ms else 32,
                    depthshade=False,
                    zorder=4,
                )

    set_3d_limits(ax, np.stack(trajectories))
    ax.set_xlabel(f"PC1 ({variance[0] * 100:.1f}%)", labelpad=7)
    ax.set_ylabel(f"PC2 ({variance[1] * 100:.1f}%)", labelpad=7)
    ax.set_zlabel(f"PC3 ({variance[2] * 100:.1f}%)", labelpad=7)
    ax.view_init(elev=22, azim=-58)

    group_legend = ax.legend(
        frameon=False,
        fontsize=8,
        title="Color: chosen side | line: chosen order",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
    )
    ax.add_artist(group_legend)
    time_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor="0.7",
            markeredgecolor="black",
            markersize=6,
            label=f"{time_ms} ms",
        )
        for time_ms, marker in TIME_MARKERS.items()
        if time[0] <= time_ms <= time[-1]
    ]
    ax.legend(
        handles=time_handles,
        frameon=False,
        fontsize=8,
        title="Phase markers",
        loc="lower left",
        bbox_to_anchor=(1.01, 0.02),
        borderaxespad=0,
    )

    fig.suptitle(
        f"{model_name}: chosen order x loc12 trajectories | "
        f"{start_ms}-{end_ms} ms\n"
        f"PC1-PC3 cumulative variance = {variance.sum() * 100:.1f}% | "
        f"{model_dir.name}",
        fontsize=11,
    )
    return fig, pca


def main():
    args = parse_args()
    analysis_dir = Path(__file__).resolve().parent
    saved_root = (
        args.saved_root.expanduser().resolve()
        if args.saved_root is not None
        else analysis_dir.parent / "Training" / "savedForLocal"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else analysis_dir / "Figure" / "order_location_pca"
    )

    if args.start_ms >= args.end_ms:
        raise ValueError("--start-ms must be earlier than --end-ms.")
    if args.model_dir is not None and args.model == "all":
        raise ValueError("--model-dir requires selecting one --model.")

    model_names = list(MODEL_ROOT_NAMES) if args.model == "all" else [args.model]
    completed = 0
    for model_name in model_names:
        try:
            if args.model_dir is not None:
                model_dir = args.model_dir.expanduser().resolve()
                if not (model_dir / args.activity_file).exists():
                    raise FileNotFoundError(
                        f"Activity file not found in {model_dir}."
                    )
            else:
                task_root = resolve_model_root(saved_root, model_name)
                model_dir = select_model_dir(
                    task_root, args.ensemble, args.activity_file
                )

            model_state, labels = load_activity_and_labels(
                model_dir, args.activity_file, args.dt
            )
            fig, pca = plot_four_condition_trajectories(
                model_name,
                model_dir,
                model_state,
                labels,
                args.start_ms,
                args.end_ms,
                args.dt,
            )
            variance = pca.explained_variance_ratio_
            print(f"\n{model_name}: {model_dir}")
            print(
                "PCA explained variance:",
                np.round(variance, 4),
                "cumulative:",
                round(float(variance.sum()), 4),
            )

            if not args.no_save:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / (
                    f"order_location_pca_3d_{model_name}_"
                    f"ensemble{args.ensemble}.pdf"
                )
                fig.savefig(output_path, format="pdf")
                print("Saved:", output_path)
            plt.close(fig)
            completed += 1
        except FileNotFoundError as error:
            print(f"Skipped {model_name}: {error}")

    if completed == 0:
        raise FileNotFoundError("No requested model could be loaded.")


if __name__ == "__main__":
    main()
