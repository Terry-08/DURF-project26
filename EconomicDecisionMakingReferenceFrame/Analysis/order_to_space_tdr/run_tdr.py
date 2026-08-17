import argparse
import re
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from tdr_core import load_activity, result_to_npz, run_cross_validated_tdr


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run cross-validated order-to-space targeted dimensionality reduction."
    )
    parser.add_argument(
        "--model",
        choices=("all", *MODEL_ROOT_NAMES.keys()),
        default="all",
        help="Model collection to analyze (default: all).",
    )
    parser.add_argument(
        "--ensemble",
        default="0",
        help="Ensemble selection: all, one index, comma list, or range such as 0-9.",
    )
    parser.add_argument("--saved-root", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--activity-file", default="activitityTest.npz")
    parser.add_argument("--dt", type=int, default=10)
    parser.add_argument("--start-ms", type=int, default=1500)
    parser.add_argument("--end-ms", type=int, default=3200)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--min-accuracy", type=float, default=0.90)
    parser.add_argument(
        "--all-trials",
        action="store_true",
        help="Include error trials; the default primary analysis uses correct trials only.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_ensemble_selection(value):
    if value.lower() == "all":
        return None
    selected = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, stop = (int(item) for item in part.split("-", 1))
            if stop < start:
                raise ValueError(f"Invalid ensemble range: {part}")
            selected.update(range(start, stop + 1))
        else:
            selected.add(int(part))
    if not selected:
        raise ValueError("No ensemble indices were selected.")
    return selected


def resolve_model_root(saved_root, model_name):
    candidates = [saved_root / name for name in MODEL_ROOT_NAMES[model_name]]
    return next((path for path in candidates if path.exists()), candidates[0])


def discover_model_dirs(task_root, activity_file, selected_indices):
    if not task_root.exists():
        raise FileNotFoundError(f"Model collection not found: {task_root}")

    pattern = re.compile(r"_(\d+)_(Fail)?$")
    by_index = {}
    for path in task_root.iterdir():
        match = pattern.search(path.name)
        if not path.is_dir() or match is None or not (path / activity_file).exists():
            continue
        ensemble_index = int(match.group(1))
        if selected_indices is not None and ensemble_index not in selected_indices:
            continue
        is_fail = match.group(2) is not None
        by_index.setdefault(ensemble_index, []).append((is_fail, path))

    selected = []
    for ensemble_index, candidates in sorted(by_index.items()):
        successful = [path for is_fail, path in candidates if not is_fail]
        if not successful:
            print(f"Skipping ensemble {ensemble_index}: only Fail directory exists.")
            continue
        path = sorted(successful, key=lambda item: item.name)[-1]
        if len(candidates) > 1:
            print(
                f"Warning: {len(candidates)} directories match ensemble "
                f"{ensemble_index}; using {path.name}."
            )
        selected.append((ensemble_index, path))
    return selected


def main():
    args = parse_args()
    analysis_root = HERE.parent
    training_root = analysis_root.parent / "Training"
    saved_root = (
        args.saved_root.expanduser().resolve()
        if args.saved_root is not None
        else training_root / "savedForLocal"
    )
    results_root = (
        args.results_root.expanduser().resolve()
        if args.results_root is not None
        else HERE / "results"
    )
    selected_indices = parse_ensemble_selection(args.ensemble)
    model_names = list(MODEL_ROOT_NAMES) if args.model == "all" else [args.model]

    processed = 0
    skipped = 0
    failures = []
    for model_name in model_names:
        task_root = resolve_model_root(saved_root, model_name)
        try:
            model_dirs = discover_model_dirs(
                task_root, args.activity_file, selected_indices
            )
        except FileNotFoundError as error:
            print(f"Skipping {model_name}: {error}")
            continue

        if not model_dirs:
            print(f"Skipping {model_name}: no requested successful ensembles found.")
            continue

        for ensemble_index, model_dir in model_dirs:
            output_dir = results_root / model_name / model_dir.name
            output_path = output_dir / "tdr_result.npz"
            if output_path.exists() and not args.overwrite:
                print(f"Already exists: {output_path}")
                skipped += 1
                continue

            print(f"\n[{model_name} ensemble {ensemble_index}] {model_dir.name}")
            try:
                model_state, labels, activity_metadata = load_activity(
                    model_dir / args.activity_file,
                    dt=args.dt,
                    correct_only=not args.all_trials,
                )
                accuracy = activity_metadata["accuracy"]
                print(
                    f"accuracy={accuracy:.3f}, included="
                    f"{activity_metadata['n_trials_included']}/"
                    f"{activity_metadata['n_trials_total']}"
                )
                if accuracy < args.min_accuracy:
                    print(
                        f"Skipped: accuracy is below --min-accuracy "
                        f"{args.min_accuracy:.3f}."
                    )
                    skipped += 1
                    continue

                result = run_cross_validated_tdr(
                    model_state,
                    labels,
                    dt=args.dt,
                    start_ms=args.start_ms,
                    end_ms=args.end_ms,
                    n_splits=args.n_splits,
                    random_state=args.seed + ensemble_index,
                )
                metadata = {
                    **activity_metadata,
                    "model_name": model_name,
                    "model_dir": str(model_dir),
                    "ensemble_index": ensemble_index,
                    "dt": args.dt,
                    "start_ms": args.start_ms,
                    "end_ms": args.end_ms,
                    "random_state": args.seed + ensemble_index,
                }
                output_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(output_path, **result_to_npz(result, metadata))
                print("Saved:", output_path)
                processed += 1
            except (KeyError, ValueError, np.linalg.LinAlgError) as error:
                failures.append((model_name, ensemble_index, str(error)))
                print(f"Failed: {error}")

    print(
        f"\nCompleted: {processed}; skipped: {skipped}; "
        f"failed: {len(failures)}"
    )
    for model_name, ensemble_index, message in failures:
        print(f"  {model_name} ensemble {ensemble_index}: {message}")
    if processed == 0 and skipped == 0:
        raise RuntimeError("No TDR analysis was completed.")
    if failures:
        raise RuntimeError("One or more requested TDR analyses failed.")


if __name__ == "__main__":
    main()
