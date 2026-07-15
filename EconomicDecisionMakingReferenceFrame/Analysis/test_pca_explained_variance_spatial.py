import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


DEFAULT_PHASE_WINDOWS = [
    ("full_task", 500, 3200),
    ("offer1", 500, 1000),
    ("delay1", 1000, 1500),
    ("offer2", 1500, 2000),
    ("wait", 2000, 2500),
    ("target", 2500, 3000),
    ("response", 3000, 3200),
]


def default_saved_root():
    cwd = Path.cwd().resolve()
    if cwd.name == "Analysis":
        return cwd.parent / "Training" / "savedForHPC"
    if cwd.name == "Training":
        return cwd / "savedForHPC"
    local_root = Path(__file__).resolve().parents[1] / "Training" / "savedForHPC"
    if local_root.exists():
        return local_root
    return Path("/gpfsnyu/home/zl6041/DURF_project/EconomicDecisionMakingReferenceFrame/Training/savedForHPC")


def phase_to_slice(start_ms, end_ms, dt, n_time):
    i0 = int(round(start_ms / dt))
    i1 = int(round(end_ms / dt))
    i0 = max(0, min(i0, n_time - 1))
    i1 = max(i0 + 1, min(i1, n_time))
    return i0, i1


def fit_pca_for_window(model_state, start_ms, end_ms, dt, n_components):
    n_trials, n_time, n_units = model_state.shape
    i0, i1 = phase_to_slice(start_ms, end_ms, dt, n_time)
    X = model_state[:, i0:i1, :].reshape(-1, n_units)

    n_fit = min(n_components, X.shape[0], X.shape[1])
    ratio = np.full(n_components, np.nan)
    cumulative = np.full(n_components, np.nan)

    if n_fit < 1:
        return ratio, cumulative, n_fit

    pca = PCA(n_components=n_fit)
    pca.fit(X)
    ratio[:n_fit] = pca.explained_variance_ratio_
    cumulative[:n_fit] = np.cumsum(pca.explained_variance_ratio_)
    return ratio, cumulative, n_fit


def analyze_model_dir(dir_path, activity_file, dt, n_components, output_name):
    activity_path = dir_path / activity_file
    if not activity_path.exists():
        return None

    with np.load(activity_path, allow_pickle=True) as f:
        model_state = f["model_state"]

    phase_names = []
    windows_ms = []
    explained = []
    cumulative = []
    n_components_fit = []

    for phase_name, start_ms, end_ms in DEFAULT_PHASE_WINDOWS:
        ratio, cum, n_fit = fit_pca_for_window(model_state, start_ms, end_ms, dt, n_components)
        phase_names.append(phase_name)
        windows_ms.append((start_ms, end_ms))
        explained.append(ratio)
        cumulative.append(cum)
        n_components_fit.append(n_fit)

    phase_names = np.array(phase_names)
    windows_ms = np.array(windows_ms)
    explained = np.vstack(explained)
    cumulative = np.vstack(cumulative)
    n_components_fit = np.array(n_components_fit)

    out_path = dir_path / f"{output_name}.npz"
    np.savez(
        out_path,
        phase_names=phase_names,
        windows_ms=windows_ms,
        explained_variance_ratio=explained,
        cumulative_explained_variance=cumulative,
        n_components_fit=n_components_fit,
        dt=dt,
        activity_file=activity_file,
    )

    return {
        "dir_path": dir_path,
        "phase_names": phase_names,
        "windows_ms": windows_ms,
        "explained": explained,
        "cumulative": cumulative,
        "out_path": out_path,
    }


def plot_average(results, figure_dir, output_name):
    if not results:
        return None

    figure_dir.mkdir(parents=True, exist_ok=True)
    phase_names = results[0]["phase_names"]
    explained_all = np.stack([r["explained"] for r in results], axis=0)
    cumulative_all = np.stack([r["cumulative"] for r in results], axis=0)

    n_components = explained_all.shape[-1]
    pcs = np.arange(1, n_components + 1)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), dpi=150)

    for i_phase, phase_name in enumerate(phase_names):
        mean = np.nanmean(explained_all[:, i_phase, :], axis=0)
        sem = np.nanstd(explained_all[:, i_phase, :], axis=0) / np.sqrt(explained_all.shape[0])
        axes[0].plot(pcs, mean, marker="o", label=phase_name)
        axes[0].fill_between(pcs, mean - sem, mean + sem, alpha=.15, edgecolor=None)

        mean_cum = np.nanmean(cumulative_all[:, i_phase, :], axis=0)
        sem_cum = np.nanstd(cumulative_all[:, i_phase, :], axis=0) / np.sqrt(cumulative_all.shape[0])
        axes[1].plot(pcs, mean_cum, marker="o", label=phase_name)
        axes[1].fill_between(pcs, mean_cum - sem_cum, mean_cum + sem_cum, alpha=.15, edgecolor=None)

    axes[0].set_xlabel("PC")
    axes[0].set_ylabel("explained variance ratio")
    axes[0].set_xticks(pcs)
    axes[0].set_ylim(bottom=0)

    axes[1].set_xlabel("number of PCs")
    axes[1].set_ylabel("cumulative explained variance")
    axes[1].set_xticks(pcs)
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=False, bbox_to_anchor=(1.04, 1), loc="upper left")

    fig.tight_layout()
    fig_path = figure_dir / f"{output_name}_average.pdf"
    fig.savefig(fig_path)
    plt.close(fig)

    summary_path = figure_dir / f"{output_name}_average.npz"
    np.savez(
        summary_path,
        phase_names=phase_names,
        explained_variance_ratio_all=explained_all,
        cumulative_explained_variance_all=cumulative_all,
        explained_variance_ratio_mean=np.nanmean(explained_all, axis=0),
        cumulative_explained_variance_mean=np.nanmean(cumulative_all, axis=0),
    )

    return fig_path, summary_path


def main():
    parser = argparse.ArgumentParser(description="Compute PCA explained variance for spatialTask RNN states.")
    parser.add_argument("dir_name", nargs="?", default="spatialTask", help="Task result folder under savedForHPC.")
    parser.add_argument("--saved-root", default=str(default_saved_root()), help="Root folder containing task result dirs.")
    parser.add_argument("--activity-file", default="activitityTestGrid.npz", help="Activity npz file to analyze.")
    parser.add_argument("--dt", type=float, default=10, help="Time step in ms.")
    parser.add_argument("--n-components", type=int, default=10, help="Number of PCs to report.")
    parser.add_argument("--output-name", default="pcaExplainedVariance", help="Base name for output files.")
    args = parser.parse_args()

    saved_root = Path(args.saved_root).resolve()
    task_root = saved_root / args.dir_name
    figure_dir = saved_root.parent.parent / "Analysis" / "Figure" if saved_root.name == "savedForHPC" else Path.cwd()

    model_dirs = sorted([p for p in task_root.iterdir() if p.is_dir()])
    results = []

    print("task_root:", task_root)
    print("n model dirs:", len(model_dirs))

    for dir_path in model_dirs:
        result = analyze_model_dir(
            dir_path,
            activity_file=args.activity_file,
            dt=args.dt,
            n_components=args.n_components,
            output_name=args.output_name,
        )
        if result is None:
            continue
        results.append(result)
        print("saved:", result["out_path"])

    fig_paths = plot_average(results, figure_dir, args.output_name)
    if fig_paths is not None:
        print("saved:", fig_paths[0])
        print("saved:", fig_paths[1])


if __name__ == "__main__":
    main()
