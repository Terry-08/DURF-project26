import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from decoding_core import LABELS, METHODS, load_result


TASKS = ("spatialTask_combined", "spatialTask_seperated")
TASK_LABELS = {
    "spatialTask_combined": "combined",
    "spatialTask_seperated": "seperated",
}
METHOD_LABELS = {
    "standard": "Standard",
    "grouped": "Grouped by offer pair",
    "value_matched": "Value-matched",
}
COLORS = {"side": "tab:red", "juice": "tab:blue", "order": "tab:green"}
DISPLAY_LABELS = {"side": "chosen side", "juice": "chosen juice", "order": "chosen order"}


def default_result_root():
    return Path(__file__).resolve().parent / "results"


def default_figure_root():
    return Path(__file__).resolve().parent / "figures"


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate and plot three-method decoding.")
    parser.add_argument("--result-root", type=Path, default=default_result_root())
    parser.add_argument("--figure-root", type=Path, default=default_figure_root())
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--min-behavior-accuracy", type=float, default=0.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=20260723)
    return parser.parse_args()


def bootstrap_interval(values, samples, random_seed):
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=0)
    if len(values) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(random_seed)
    bootstrap_means = np.empty((samples, values.shape[1]), dtype=np.float32)
    for first in range(0, samples, 100):
        last = min(samples, first + 100)
        indices = rng.integers(0, len(values), size=(last - first, len(values)))
        bootstrap_means[first:last] = np.nanmean(values[indices], axis=1)
    low, high = np.nanpercentile(bootstrap_means, [2.5, 97.5], axis=0)
    return mean, low, high


def collect_task(result_root, task_name, min_accuracy):
    records = []
    task_root = result_root / task_name
    for result_path in sorted(task_root.glob("*/three_decoding_methods.npz")):
        metadata, data = load_result(result_path)
        if metadata["behavior_accuracy_analyzed"] < min_accuracy:
            continue
        records.append((result_path, metadata, data))
    if not records:
        raise RuntimeError("No valid decoding results found for %s" % task_name)
    return records


def summarize(records, bootstrap_samples, random_seed):
    reference_time = records[0][2]["time"]
    summary = {"time": reference_time, "event_windows": records[0][1]["event_windows"]}
    for method_index, method in enumerate(METHODS):
        for label_index, label in enumerate(LABELS):
            model_curves = []
            for _, _, data in records:
                if not np.array_equal(data["time"], reference_time):
                    raise ValueError("Model results have different time axes.")
                scores = data["scores__%s__%s" % (method, label)]
                model_curves.append(np.nanmean(scores, axis=1))
            model_curves = np.asarray(model_curves)
            mean, low, high = bootstrap_interval(
                model_curves,
                bootstrap_samples,
                random_seed + 100 * method_index + label_index,
            )
            summary[(method, label)] = {
                "models": model_curves,
                "mean": mean,
                "low": low,
                "high": high,
            }
    return summary


def shade_events(axis, windows):
    colors = {
        "offer1": "0.85",
        "offer2": "0.85",
        "target": "lightblue",
        "response": "plum",
    }
    for name in ("offer1", "offer2", "target", "response"):
        onset, offset = windows[name]
        axis.axvspan(onset, offset, color=colors[name], alpha=0.25, linewidth=0)
    axis.axhline(0.5, color="0.35", linestyle="--", linewidth=1)


def save_task_figure(task_name, summary, model_count, figure_root):
    figure, axes = plt.subplots(3, 1, figsize=(7.2, 8.4), sharex=True, sharey=True)
    for axis, method in zip(axes, METHODS):
        shade_events(axis, summary["event_windows"])
        for label in LABELS:
            result = summary[(method, label)]
            axis.plot(summary["time"], result["mean"], color=COLORS[label],
                      label=DISPLAY_LABELS[label], linewidth=1.7)
            axis.fill_between(summary["time"], result["low"], result["high"],
                              color=COLORS[label], alpha=0.15, linewidth=0)
        axis.set_ylim(0.4, 1.01)
        axis.set_ylabel("balanced accuracy")
        axis.set_title(METHOD_LABELS[method], fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, ncol=3,
                  loc="upper center", bbox_to_anchor=(0.5, 0.965))
    axes[-1].set_xlabel("time (ms)")
    figure.suptitle("%s: three decoding methods (n=%d models)" % (
        TASK_LABELS.get(task_name, task_name), model_count), fontsize=11, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    for suffix in ("pdf", "png"):
        figure.savefig(figure_root / ("%s_three_methods.%s" % (
            TASK_LABELS.get(task_name, task_name), suffix)), dpi=200)
    plt.close(figure)


def save_comparison_figure(summaries, model_counts, figure_root):
    figure, axes = plt.subplots(3, 3, figsize=(12, 8.5), sharex=True, sharey=True)
    task_colors = {"spatialTask_combined": "black", "spatialTask_seperated": "tab:orange"}
    for row, method in enumerate(METHODS):
        for column, label in enumerate(LABELS):
            axis = axes[row, column]
            first_summary = next(iter(summaries.values()))
            shade_events(axis, first_summary["event_windows"])
            for task_name, summary in summaries.items():
                result = summary[(method, label)]
                display = "%s (n=%d)" % (TASK_LABELS.get(task_name, task_name), model_counts[task_name])
                axis.plot(summary["time"], result["mean"], color=task_colors[task_name],
                          label=display, linewidth=1.6)
                axis.fill_between(summary["time"], result["low"], result["high"],
                                  color=task_colors[task_name], alpha=0.12, linewidth=0)
            if row == 0:
                axis.set_title(DISPLAY_LABELS[label])
            if column == 0:
                axis.set_ylabel("%s\nbalanced accuracy" % METHOD_LABELS[method])
            if row == len(METHODS) - 1:
                axis.set_xlabel("time (ms)")
            axis.set_ylim(0.4, 1.01)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(figure_root / ("combined_vs_seperated_three_methods.%s" % suffix), dpi=200)
    plt.close(figure)


def save_summary_data(summaries, model_counts, result_root):
    rows = []
    npz_data = {}
    metadata = {"model_counts": model_counts}
    for task_name, summary in summaries.items():
        npz_data["time__%s" % task_name] = summary["time"]
        metadata["event_windows__%s" % task_name] = summary["event_windows"]
        for method in METHODS:
            for label in LABELS:
                result = summary[(method, label)]
                prefix = "%s__%s__%s" % (task_name, method, label)
                npz_data["mean__" + prefix] = result["mean"]
                npz_data["ci_low__" + prefix] = result["low"]
                npz_data["ci_high__" + prefix] = result["high"]
                for time, mean, low, high in zip(
                    summary["time"], result["mean"], result["low"], result["high"]
                ):
                    rows.append({
                        "task": task_name,
                        "method": method,
                        "label": label,
                        "time_ms": time,
                        "mean_balanced_accuracy": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "n_models": model_counts[task_name],
                    })

    npz_data["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(result_root / "ensemble_summary.npz", **npz_data)
    with (result_root / "ensemble_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.figure_root.mkdir(parents=True, exist_ok=True)
    summaries = {}
    model_counts = {}
    for task_name in args.tasks:
        records = collect_task(args.result_root, task_name, args.min_behavior_accuracy)
        summaries[task_name] = summarize(
            records, args.bootstrap_samples, args.random_seed
        )
        model_counts[task_name] = len(records)
        save_task_figure(
            task_name, summaries[task_name], len(records), args.figure_root
        )
        print("Included %d valid models for %s" % (len(records), task_name))

    if len(summaries) > 1:
        save_comparison_figure(summaries, model_counts, args.figure_root)
    save_summary_data(summaries, model_counts, args.result_root)
    print("Figures:", args.figure_root)
    print("Summary:", args.result_root / "ensemble_summary.npz")


if __name__ == "__main__":
    main()
