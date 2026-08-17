import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from threadpoolctl import threadpool_limits

from decoding_core import analyze_model


DEFAULT_TASKS = ("spatialTask_combined", "spatialTask_seperated")


def default_data_root():
    return Path(__file__).resolve().parents[2] / "Training" / "savedForLocal"


def default_result_root():
    return Path(__file__).resolve().parent / "results"


def ensemble_number(name):
    match = re.search(r"_(\d+)(?:_Fail)?_?$", name)
    return int(match.group(1)) if match else None


def discover_models(data_root, task_name, max_models=None):
    task_root = data_root / task_name
    if not task_root.exists():
        raise FileNotFoundError("Missing task directory: %s" % task_root)

    candidates = [
        path
        for path in task_root.iterdir()
        if path.is_dir() and (path / "activitityTestGrid.npz").exists()
    ]
    candidates.sort(key=lambda path: (ensemble_number(path.name) is None,
                                      ensemble_number(path.name) or -1,
                                      path.name))

    unique = []
    seen = set()
    for path in candidates:
        number = ensemble_number(path.name)
        key = number if number is not None else path.name
        if key in seen:
            print("Skipping duplicate ensemble:", path)
            continue
        seen.add(key)
        unique.append(path)

    return unique[:max_models] if max_models is not None else unique


def run_one(job):
    model_dir, task_name, result_path, config = job
    try:
        with threadpool_limits(limits=1):
            metadata = analyze_model(
                model_dir=model_dir,
                task_name=task_name,
                result_path=result_path,
                **config,
            )
        return {
            "task": task_name,
            "model": Path(model_dir).name,
            "ensemble": ensemble_number(Path(model_dir).name),
            "status": "complete",
            "result": str(result_path),
            "behavior_accuracy": metadata["behavior_accuracy_analyzed"],
            "analyzed_trials": metadata["analyzed_trials"],
            "elapsed_seconds": metadata["elapsed_seconds"],
            "error": "",
        }
    except Exception:
        return {
            "task": task_name,
            "model": Path(model_dir).name,
            "ensemble": ensemble_number(Path(model_dir).name),
            "status": "failed",
            "result": str(result_path),
            "behavior_accuracy": "",
            "analyzed_trials": "",
            "elapsed_seconds": "",
            "error": traceback.format_exc(),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run standard, grouped, and value-matched choice decoding."
    )
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--result-root", type=Path, default=default_result_root())
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=20260723)
    parser.add_argument("--include-zero-offers", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_manifest(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task", "model", "ensemble", "status", "result",
        "behavior_accuracy", "analyzed_trials", "elapsed_seconds", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["task"], row["ensemble"] or -1)))


def main():
    args = parse_args()
    args.result_root.mkdir(parents=True, exist_ok=True)
    config = {
        "dt": 10,
        "n_splits": args.n_splits,
        "n_repeats": args.n_repeats,
        "random_seed": args.random_seed,
        "include_zero_offers": args.include_zero_offers,
    }

    jobs = []
    rows = []
    for task_name in args.tasks:
        for model_dir in discover_models(args.data_root, task_name, args.max_models):
            result_path = args.result_root / task_name / model_dir.name / "three_decoding_methods.npz"
            if result_path.exists() and not args.overwrite:
                rows.append({
                    "task": task_name,
                    "model": model_dir.name,
                    "ensemble": ensemble_number(model_dir.name),
                    "status": "existing",
                    "result": str(result_path),
                    "behavior_accuracy": "",
                    "analyzed_trials": "",
                    "elapsed_seconds": "",
                    "error": "",
                })
                continue
            jobs.append((model_dir, task_name, result_path, config))

    print("Data root:", args.data_root)
    print("Result root:", args.result_root)
    print("Models to analyze:", len(jobs))
    print("Worker processes:", args.jobs)

    if args.jobs == 1:
        for index, job in enumerate(jobs, start=1):
            row = run_one(job)
            rows.append(row)
            print("[%d/%d] %s %s %.1fs" % (
                index,
                len(jobs),
                row["status"],
                row["model"],
                float(row["elapsed_seconds"] or 0),
            ), flush=True)
            if row["status"] == "failed":
                print(row["error"], flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(run_one, job): job for job in jobs}
            completed = 0
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                completed += 1
                print("[%d/%d] %s %s %.1fs" % (
                    completed,
                    len(jobs),
                    row["status"],
                    row["model"],
                    float(row["elapsed_seconds"] or 0),
                ), flush=True)
                if row["status"] == "failed":
                    print(row["error"], flush=True)

    manifest_path = args.result_root / "model_manifest.csv"
    write_manifest(rows, manifest_path)
    print("Manifest:", manifest_path)

    failed = [row for row in rows if row["status"] == "failed"]
    if failed:
        raise SystemExit("%d model analyses failed; see the manifest." % len(failed))


if __name__ == "__main__":
    main()
