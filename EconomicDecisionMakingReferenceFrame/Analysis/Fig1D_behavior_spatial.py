import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def repo_root_from_script():
    return Path(__file__).resolve().parents[2]


def default_data_root():
    return repo_root_from_script() / "EconomicDecisionMakingReferenceFrame" / "Training" / "savedForHPC" / "spatialTask"


def default_figure_dir():
    return repo_root_from_script() / "EconomicDecisionMakingReferenceFrame" / "Analysis" / "Figure"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the spatialTask psychometric curve: choose B vs qB/qA."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
        help="Directory containing spatialTask ensemble folders.",
    )
    parser.add_argument(
        "--activity-file",
        default="activitityTestGrid.npz",
        help="Activity file inside each model directory.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=default_figure_dir(),
        help="Directory for output figures and summary data.",
    )
    parser.add_argument(
        "--model-dir",
        action="append",
        default=None,
        help="Optional model directory name or path. Can be repeated.",
    )
    parser.add_argument(
        "--output-prefix",
        default="behaviorSpatial",
        help="Prefix for saved figure and npz files.",
    )
    return parser.parse_args()


def list_model_dirs(data_root, model_dirs=None):
    if model_dirs is None:
        return sorted([p for p in data_root.iterdir() if p.is_dir()])

    dirs = []
    for item in model_dirs:
        p = Path(item)
        if not p.is_absolute():
            p = data_root / p
        dirs.append(p)
    return dirs


def response_average(model_output, mask, trial_params):
    if mask is not None and np.any(mask):
        denom = np.sum(mask, axis=1)
        denom[denom == 0] = 1
        return np.sum(model_output * mask, axis=1) / denom

    temp = []
    for i, params in enumerate(trial_params):
        start = int(params["fixation_offset"] / 10)
        end = int(params["end"] / 10)
        temp.append(np.mean(model_output[i, start:end, :], axis=0))
    return np.asarray(temp)


def infer_choice_b_from_spatial_outputs(model_output, mask, trial_params):
    response = response_average(model_output, mask, trial_params)
    choice_side = np.argmax(response, axis=1)  # 0 = left, 1 = right

    choice_b = np.zeros(len(trial_params), dtype=int)
    chosen_offer_model = np.zeros(len(trial_params), dtype=int)

    for i, side in enumerate(choice_side):
        params = trial_params[i].item() if hasattr(trial_params[i], "item") else trial_params[i]

        loc12_label = params.get("loc12_label", None)
        if loc12_label is None:
            loc12 = params["loc12"]
            loc12_label = "12" if loc12[0] == 1 else "21"

        if loc12_label == "12":
            chosen_offer = 1 if side == 0 else 2
        elif loc12_label == "21":
            chosen_offer = 2 if side == 0 else 1
        else:
            raise ValueError("Unknown loc12_label: %s" % loc12_label)

        seqAB = params["seqAB"]
        if seqAB == "AB":
            chosen_juice = "A" if chosen_offer == 1 else "B"
        elif seqAB == "BA":
            chosen_juice = "B" if chosen_offer == 1 else "A"
        else:
            raise ValueError("Unknown seqAB: %s" % seqAB)

        chosen_offer_model[i] = chosen_offer
        choice_b[i] = 1 if chosen_juice == "B" else 0

    return choice_b, choice_side, chosen_offer_model


def load_model_behavior(model_dir, activity_file):
    activity_path = model_dir / activity_file
    with np.load(activity_path, allow_pickle=True) as f:
        trial_params = f["trial_params"]
        model_output = f["model_output"]
        mask = f["mask"] if "mask" in f.files else None

    choice_b, choice_side, chosen_offer_model = infer_choice_b_from_spatial_outputs(
        model_output, mask, trial_params
    )
    q_as = np.asarray([params["qA"] for params in trial_params], dtype=float)
    q_bs = np.asarray([params["qB"] for params in trial_params], dtype=float)
    choice_target = np.asarray([params["chooseB"] for params in trial_params], dtype=int)
    seq_ab = np.asarray([params["seqAB"] for params in trial_params])
    loc12_label = np.asarray(
        [
            params.get("loc12_label", "12" if params["loc12"][0] == 1 else "21")
            for params in trial_params
        ]
    )

    return {
        "qA": q_as,
        "qB": q_bs,
        "choiceB_model": choice_b,
        "choiceB_target": choice_target,
        "choice_side": choice_side,
        "chosen_offer_model": chosen_offer_model,
        "seqAB": seq_ab,
        "loc12_label": loc12_label,
    }


def fit_behavior(choice_b, q_as, q_bs):
    idx = (q_as != 0) & (q_bs != 0)
    x = np.log(q_bs[idx] / q_as[idx]).reshape(-1, 1)
    y = choice_b[idx]

    if len(np.unique(y)) < 2:
        raise ValueError("Cannot fit logistic regression with only one choice class.")

    model = LogisticRegression()
    model.fit(x, y)
    return model


def psychometric_by_ratio(choice_b, choice_target, q_as, q_bs):
    offer_ratio = q_bs / q_as
    offer_ratio[(q_as == 0) | (q_bs == 0)] = np.nan
    offer_ratio[offer_ratio > 10] = 10

    ratios = np.unique(offer_ratio[~np.isnan(offer_ratio)])
    simulation = np.zeros(ratios.shape)
    teaching = np.zeros(ratios.shape)
    counts = np.zeros(ratios.shape, dtype=int)

    for i, ratio in enumerate(ratios):
        idx = offer_ratio == ratio
        simulation[i] = np.mean(choice_b[idx])
        teaching[i] = np.mean(choice_target[idx])
        counts[i] = np.sum(idx)

    return ratios, simulation, teaching, counts


def plot_behavior(q_as, q_bs, choice_b, choice_target, figure_dir, output_prefix):
    model = fit_behavior(choice_b, q_as, q_bs)
    beta0 = model.intercept_[0]
    beta1 = model.coef_[0][0]

    ratios, simulation, teaching, counts = psychometric_by_ratio(
        choice_b, choice_target, q_as, q_bs
    )
    fit_choice = sigmoid(beta0 + beta1 * np.log(ratios)) * 100
    teaching_curve = sigmoid(13 * np.log(ratios / 1.7)) * 100
    indifference_point = np.exp(-beta0 / beta1)

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.plot(
        ratios,
        teaching * 100,
        "o",
        markerfacecolor="white",
        markeredgecolor="tab:orange",
        label="Teaching",
    )
    ax.plot(
        ratios,
        simulation * 100,
        "o",
        markerfacecolor="tab:green",
        markeredgecolor="tab:green",
        alpha=0.75,
        label="Simulation",
    )
    ax.plot(
        ratios,
        fit_choice,
        color="tab:green",
        linewidth=2,
        label="Fit (ind. point=%.2f)" % indifference_point,
    )
    ax.plot(ratios, teaching_curve, color="tab:orange", linewidth=1, alpha=0.4)

    ax.set_xlabel("Offer qB:qA")
    ax.set_ylabel("Choose B %")
    ax.set_xscale("log")
    ax.set_ylim(-5, 105)
    ax.set_xticks([1 / 4, 1 / 2, 2 / 2, 3 / 2, 8 / 3, 8 / 1])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xticklabels(["1:4", "1:2", "2:2", "3:2", "8:3", "8:1"])
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("spatialTask behavior")
    fig.tight_layout()

    figure_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = figure_dir / ("%s_average.pdf" % output_prefix)
    png_path = figure_dir / ("%s_average.png" % output_prefix)
    npz_path = figure_dir / ("%s_average.npz" % output_prefix)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    np.savez(
        npz_path,
        ratios=ratios,
        simulation=simulation,
        teaching=teaching,
        fit_choice=fit_choice,
        teaching_curve=teaching_curve,
        counts=counts,
        beta0=beta0,
        beta1=beta1,
        indifference_point=indifference_point,
    )

    return pdf_path, png_path, npz_path, indifference_point


def main():
    args = parse_args()
    model_dirs = list_model_dirs(args.data_root, args.model_dir)

    all_q_as = []
    all_q_bs = []
    all_choice_b = []
    all_choice_target = []
    loaded_dirs = []
    skipped_dirs = []

    for model_dir in model_dirs:
        activity_path = model_dir / args.activity_file
        if not activity_path.exists():
            skipped_dirs.append((str(model_dir), "missing %s" % args.activity_file))
            continue

        try:
            data = load_model_behavior(model_dir, args.activity_file)
        except Exception as exc:
            skipped_dirs.append((str(model_dir), repr(exc)))
            continue

        all_q_as.append(data["qA"])
        all_q_bs.append(data["qB"])
        all_choice_b.append(data["choiceB_model"])
        all_choice_target.append(data["choiceB_target"])
        loaded_dirs.append(str(model_dir))

    if not loaded_dirs:
        raise RuntimeError("No valid model directories were loaded from %s" % args.data_root)

    q_as = np.concatenate(all_q_as)
    q_bs = np.concatenate(all_q_bs)
    choice_b = np.concatenate(all_choice_b)
    choice_target = np.concatenate(all_choice_target)

    pdf_path, png_path, npz_path, indifference_point = plot_behavior(
        q_as, q_bs, choice_b, choice_target, args.figure_dir, args.output_prefix
    )

    print("Loaded %d model directories." % len(loaded_dirs))
    if skipped_dirs:
        print("Skipped %d model directories:" % len(skipped_dirs))
        for model_dir, reason in skipped_dirs:
            print("  %s: %s" % (model_dir, reason))
    print("Saved PDF: %s" % pdf_path)
    print("Saved PNG: %s" % png_path)
    print("Saved NPZ: %s" % npz_path)
    print("Fitted indifference point qB/qA: %.4f" % indifference_point)


if __name__ == "__main__":
    main()
