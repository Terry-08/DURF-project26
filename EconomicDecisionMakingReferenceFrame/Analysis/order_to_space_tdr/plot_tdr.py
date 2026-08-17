import argparse
import csv
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import t as student_t


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from tdr_core import first_sustained, load_metrics


MODEL_NAMES = (
    "combined",
    "separated",
    "balanced_combined",
    "balanced_separated",
)
MODEL_LABELS = {
    "combined": "Combined",
    "separated": "Separated",
    "balanced_combined": "Balanced combined",
    "balanced_separated": "Balanced separated",
}
MODEL_STYLES = {
    "combined": ("tab:blue", "-"),
    "separated": ("tab:orange", "-"),
    "balanced_combined": ("tab:blue", "--"),
    "balanced_separated": ("tab:orange", "--"),
}
VARIABLE_COLORS = {
    "order": "tab:orange",
    "location": "tab:blue",
    "space": "tab:green",
}
CONDITION_STYLES = (
    ("tab:green", "-"),
    ("tab:purple", "-"),
    ("tab:purple", "--"),
    ("tab:green", "--"),
)
PHASE_MARKERS = {1500: "X", 2000: "o", 2500: "s", 3000: "D", 3200: "P"}

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate and plot order-to-space TDR results."
    )
    parser.add_argument("--results-root", type=Path, default=HERE / "results")
    parser.add_argument("--figure-root", type=Path, default=HERE / "figures")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--representative-ensemble", type=int, default=0)
    return parser.parse_args()


def load_model_results(results_root, model_name):
    result_paths = sorted((results_root / model_name).glob("*/tdr_result.npz"))
    results = []
    for path in result_paths:
        with np.load(path, allow_pickle=False) as data:
            result = {key: np.asarray(data[key]) for key in data.files}
        result["_path"] = path
        result["_metrics"] = load_metrics(result)
        results.append(result)
    return sorted(results, key=lambda item: int(item["ensemble_index"]))


def bootstrap_mean_ci(values, n_bootstrap, rng):
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=0)
    if len(values) == 1 or n_bootstrap <= 0:
        return mean, np.full_like(mean, np.nan), np.full_like(mean, np.nan)
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    bootstrap_means = np.nanmean(values[indices], axis=1)
    lower, upper = np.nanpercentile(bootstrap_means, [2.5, 97.5], axis=0)
    return mean, lower, upper


def contiguous_clusters(mask):
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []
    boundaries = np.flatnonzero(np.diff(indices) > 1) + 1
    return [cluster for cluster in np.split(indices, boundaries) if len(cluster)]


def one_sample_t(values):
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0, ddof=1)
    count = np.sum(np.isfinite(values), axis=0)
    denominator = std / np.sqrt(np.maximum(count, 1))
    statistic = np.zeros_like(mean)
    valid = (count >= 2) & (denominator > 1e-12)
    statistic[valid] = mean[valid] / denominator[valid]
    statistic[(~valid) & (mean > 0)] = np.inf
    return statistic


def cluster_sign_flip_test(values, time, n_permutations, rng, alpha=0.05):
    values = np.asarray(values, dtype=float)
    if values.shape[0] < 2 or n_permutations <= 0:
        return np.zeros(values.shape[1], dtype=bool), []
    threshold = float(student_t.ppf(1 - alpha, df=values.shape[0] - 1))
    observed_t = one_sample_t(values)
    eligible = time >= 1500
    observed_clusters = contiguous_clusters((observed_t > threshold) & eligible)

    max_masses = np.zeros(n_permutations)
    for permutation in range(n_permutations):
        signs = rng.choice((-1.0, 1.0), size=(values.shape[0], 1))
        permuted_t = one_sample_t(values * signs)
        clusters = contiguous_clusters((permuted_t > threshold) & eligible)
        if clusters:
            max_masses[permutation] = max(
                float(np.sum(permuted_t[cluster])) for cluster in clusters
            )

    significant = np.zeros(values.shape[1], dtype=bool)
    cluster_results = []
    for cluster in observed_clusters:
        mass = float(np.sum(observed_t[cluster]))
        p_value = (1 + np.sum(max_masses >= mass)) / (n_permutations + 1)
        cluster_results.append(
            {
                "start_ms": float(time[cluster[0]]),
                "end_ms": float(time[cluster[-1]]),
                "mass": mass,
                "p_value": float(p_value),
            }
        )
        if p_value < alpha:
            significant[cluster] = True
    return significant, cluster_results


def decorate_time_axis(ax):
    ax.axvspan(1500, 2000, color="gold", alpha=0.12, linewidth=0)
    ax.axvspan(2000, 2500, color="0.7", alpha=0.10, linewidth=0)
    ax.axvspan(2500, 3000, color="cadetblue", alpha=0.12, linewidth=0)
    ax.axvspan(3000, 3200, color="plum", alpha=0.16, linewidth=0)
    for boundary in (1500, 2000, 2500, 3000, 3200):
        ax.axvline(boundary, color="0.75", linewidth=0.6, zorder=0)
    ax.set_xlim(1500, 3200)


def save_figure(fig, output_path):
    fig.savefig(output_path, bbox_inches="tight")
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    print("Saved:", output_path)
    print("Saved:", png_path)


def plot_representative_trajectory(model_name, results, output_dir, ensemble_index):
    candidates = [
        result
        for result in results
        if int(result["ensemble_index"]) == ensemble_index
    ]
    if not candidates:
        candidates = results[:1]
        ensemble_index = int(candidates[0]["ensemble_index"])
        print(
            f"Warning: representative ensemble unavailable for {model_name}; "
            f"using ensemble {ensemble_index}."
        )
    result = candidates[0]
    time = result["time"]
    trajectories = result["condition_trajectories"]
    labels = [str(label) for label in result["condition_labels"]]
    counts = result["condition_counts"]

    fig, ax = plt.subplots(figsize=(7.2, 5.5), dpi=160, constrained_layout=True)
    for trajectory, label, count, (color, linestyle) in zip(
        trajectories, labels, counts, CONDITION_STYLES
    ):
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=2.3,
            label=f"{label} (n={int(count)})",
        )
        for time_ms, marker in PHASE_MARKERS.items():
            if time[0] <= time_ms <= time[-1]:
                index = int(np.argmin(np.abs(time - time_ms)))
                ax.scatter(
                    trajectory[index, 0],
                    trajectory[index, 1],
                    marker=marker,
                    color=color,
                    edgecolors="black",
                    linewidths=0.5,
                    s=42,
                    zorder=3,
                )

    ax.axhline(0, color="0.8", linewidth=0.6)
    ax.axvline(0, color="0.8", linewidth=0.6)
    ax.set_xlabel("Order TDR coordinate")
    ax.set_ylabel("Space TDR coordinate")
    ax.set_title(
        f"{MODEL_LABELS[model_name]}: order x location TDR trajectory\n"
        f"ensemble {ensemble_index} | 1500-3200 ms"
    )
    group_legend = ax.legend(frameon=False, fontsize=8, loc="upper right")
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
        for time_ms, marker in PHASE_MARKERS.items()
        if time[0] <= time_ms <= time[-1]
    ]
    ax.legend(
        handles=time_handles,
        title="Phase markers",
        frameon=False,
        fontsize=7,
        loc="lower right",
    )
    output_path = output_dir / f"tdr_trajectory_{model_name}_ensemble{ensemble_index}.pdf"
    save_figure(fig, output_path)
    plt.close(fig)


def plot_model_timecourses(
    model_name, results, summary, output_dir, n_bootstrap, rng
):
    time = results[0]["time"]
    unique_r2 = np.stack([result["unique_r2"] for result in results])
    beta_norm = np.stack([result["beta_norm"] for result in results])
    projection_effect = np.stack(
        [result["projection_effect"] for result in results]
    )
    diagonal_effect = np.stack(
        [projection_effect[:, index, index] for index in range(3)], axis=1
    )

    fig, axes = plt.subplots(
        3, 1, figsize=(7.4, 8.5), dpi=160, sharex=True, constrained_layout=True
    )
    panels = (
        (unique_r2, "Unique cross-validated $R^2$"),
        (beta_norm, "Regression-axis norm"),
        (diagonal_effect, "Held-out effect on matching TDR axis"),
    )
    for ax, (values, ylabel) in zip(axes, panels):
        decorate_time_axis(ax)
        ax.axhline(0, color="0.45", linestyle="--", linewidth=0.7)
        for variable_index, variable_name in enumerate(("order", "location", "space")):
            mean, lower, upper = bootstrap_mean_ci(
                values[:, variable_index], n_bootstrap, rng
            )
            color = VARIABLE_COLORS[variable_name]
            ax.plot(time, mean, color=color, label=variable_name)
            ax.fill_between(time, lower, upper, color=color, alpha=0.16, linewidth=0)
        ax.set_ylabel(ylabel)
    y_min, y_max = axes[0].get_ylim()
    y_span = y_max - y_min
    for offset, variable_name in enumerate(("order", "location", "space")):
        significant = summary[f"significant_{variable_name}"]
        y_value = y_min + (0.04 + 0.035 * offset) * y_span
        axes[0].plot(
            time,
            np.where(significant, y_value, np.nan),
            color=VARIABLE_COLORS[variable_name],
            linewidth=3,
            solid_capstyle="butt",
        )
    axes[0].text(
        0.99,
        0.02,
        "bottom bars: cluster-corrected p < 0.05",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="0.3",
    )
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle(
        f"{MODEL_LABELS[model_name]} TDR ensemble summary (n={len(results)})"
    )
    output_path = output_dir / f"tdr_timecourses_{model_name}.pdf"
    save_figure(fig, output_path)
    plt.close(fig)


def summarize_model(
    model_name, results, n_bootstrap, n_permutations, rng
):
    time = results[0]["time"]
    unique_r2 = np.stack([result["unique_r2"] for result in results])
    order_mean, order_lower, order_upper = bootstrap_mean_ci(
        unique_r2[:, 0], n_bootstrap, rng
    )
    space_mean, space_lower, space_upper = bootstrap_mean_ci(
        unique_r2[:, 2], n_bootstrap, rng
    )
    difference_mean, difference_lower, difference_upper = bootstrap_mean_ci(
        unique_r2[:, 2] - unique_r2[:, 0], n_bootstrap, rng
    )
    significant = {}
    clusters = {}
    for variable_index, variable_name in enumerate(("order", "location", "space")):
        significant[variable_name], clusters[variable_name] = cluster_sign_flip_test(
            unique_r2[:, variable_index], time, n_permutations, rng
        )
    significant_difference, clusters_difference = cluster_sign_flip_test(
        unique_r2[:, 2] - unique_r2[:, 0], time, n_permutations, rng
    )
    order_onset = first_sustained(time, significant["order"], consecutive=1)
    space_onset = first_sustained(time, significant["space"], consecutive=1)
    crossover = first_sustained(time, significant_difference, consecutive=1)
    coexistence = int(np.sum(significant["order"] & significant["space"]))
    dt = float(np.median(np.diff(time)))

    angles = np.asarray(
        [result["_metrics"]["axis_angle_order_space"] for result in results]
    )
    residual_fractions = np.asarray(
        [result["_metrics"]["space_residual_fraction"] for result in results]
    )
    accuracies = np.asarray([float(result["accuracy"]) for result in results])
    return {
        "model": model_name,
        "n_ensembles": len(results),
        "mean_accuracy": float(accuracies.mean()),
        "order_onset_ms": order_onset,
        "space_onset_ms": space_onset,
        "space_over_order_ms": crossover,
        "order_peak_ms": float(time[np.nanargmax(order_mean)]),
        "space_peak_ms": float(time[np.nanargmax(space_mean)]),
        "coexistence_duration_ms": coexistence * dt,
        "mean_order_space_angle_deg": float(np.nanmean(angles)),
        "mean_space_residual_fraction": float(np.nanmean(residual_fractions)),
        "order_mean": order_mean,
        "order_lower": order_lower,
        "order_upper": order_upper,
        "space_mean": space_mean,
        "space_lower": space_lower,
        "space_upper": space_upper,
        "difference_mean": difference_mean,
        "difference_lower": difference_lower,
        "difference_upper": difference_upper,
        "significant_order": significant["order"],
        "significant_location": significant["location"],
        "significant_space": significant["space"],
        "significant_space_over_order": significant_difference,
        "clusters_order": clusters["order"],
        "clusters_location": clusters["location"],
        "clusters_space": clusters["space"],
        "clusters_space_over_order": clusters_difference,
    }


def plot_model_comparison(summaries, time, output_dir):
    fig, axes = plt.subplots(
        2, 1, figsize=(7.4, 6.7), dpi=160, sharex=True, constrained_layout=True
    )
    for ax, variable in zip(axes, ("order", "space")):
        decorate_time_axis(ax)
        ax.axhline(0, color="0.45", linestyle="--", linewidth=0.7)
        for model_name, summary in summaries.items():
            color, linestyle = MODEL_STYLES[model_name]
            mean = summary[f"{variable}_mean"]
            lower = summary[f"{variable}_lower"]
            upper = summary[f"{variable}_upper"]
            ax.plot(
                time,
                mean,
                color=color,
                linestyle=linestyle,
                label=MODEL_LABELS[model_name],
            )
            ax.fill_between(time, lower, upper, color=color, alpha=0.08, linewidth=0)
        ax.set_ylabel(f"Unique {variable} $R^2$")
    axes[0].legend(frameon=False, ncol=2, fontsize=8, loc="upper left")
    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("Order-to-space TDR comparison across model variants")
    output_path = output_dir / "tdr_model_comparison.pdf"
    save_figure(fig, output_path)
    plt.close(fig)


def save_summary(summaries, time, output_dir):
    scalar_fields = (
        "model",
        "n_ensembles",
        "mean_accuracy",
        "order_onset_ms",
        "space_onset_ms",
        "space_over_order_ms",
        "order_peak_ms",
        "space_peak_ms",
        "coexistence_duration_ms",
        "mean_order_space_angle_deg",
        "mean_space_residual_fraction",
    )
    csv_path = output_dir / "tdr_ensemble_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields)
        writer.writeheader()
        for summary in summaries.values():
            writer.writerow({field: summary[field] for field in scalar_fields})
    print("Saved:", csv_path)

    cluster_path = output_dir / "tdr_significant_clusters.csv"
    with cluster_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ("model", "contrast", "start_ms", "end_ms", "mass", "p_value")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model_name, summary in summaries.items():
            for contrast in ("order", "location", "space", "space_over_order"):
                for cluster in summary[f"clusters_{contrast}"]:
                    if cluster["p_value"] < 0.05:
                        writer.writerow(
                            {
                                "model": model_name,
                                "contrast": contrast,
                                **cluster,
                            }
                        )
    print("Saved:", cluster_path)

    arrays = {"time": time, "model_names": np.asarray(list(summaries))}
    for model_name, summary in summaries.items():
        for field, value in summary.items():
            if isinstance(value, np.ndarray):
                arrays[f"{model_name}_{field}"] = value
    npz_path = output_dir / "tdr_ensemble_summary.npz"
    np.savez_compressed(npz_path, **arrays)
    print("Saved:", npz_path)


def main():
    args = parse_args()
    results_root = args.results_root.expanduser().resolve()
    figure_root = args.figure_root.expanduser().resolve()
    figure_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    all_results = {}
    for model_name in MODEL_NAMES:
        results = load_model_results(results_root, model_name)
        if results:
            all_results[model_name] = results
            print(f"{model_name}: {len(results)} ensembles")
        else:
            print(f"Skipping {model_name}: no TDR results found.")
    if not all_results:
        raise FileNotFoundError(f"No tdr_result.npz files under {results_root}")

    reference_time = next(iter(all_results.values()))[0]["time"]
    summaries = {}
    for model_name, results in all_results.items():
        for result in results:
            if not np.array_equal(result["time"], reference_time):
                raise ValueError(f"Inconsistent time axis in {result['_path']}")
        plot_representative_trajectory(
            model_name,
            results,
            figure_root,
            args.representative_ensemble,
        )
        summaries[model_name] = summarize_model(
            model_name,
            results,
            args.bootstrap,
            args.permutations,
            rng,
        )
        plot_model_timecourses(
            model_name,
            results,
            summaries[model_name],
            figure_root,
            args.bootstrap,
            rng,
        )

    plot_model_comparison(summaries, reference_time, figure_root)
    save_summary(summaries, reference_time, figure_root)


if __name__ == "__main__":
    main()
