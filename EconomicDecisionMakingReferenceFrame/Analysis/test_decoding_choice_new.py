import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import os
import sys
from pathlib import Path


dirName = sys.argv[1] if len(sys.argv) > 1 else 'spatialTask'
savedRoot = Path("/gpfsnyu/home/zl6041/DURF_project/EconomicDecisionMakingReferenceFrame/Training/savedForHPC")
if len(sys.argv) > 2:
    savedRoot = Path(sys.argv[2])

dt = 10
target_window = (2500, 3000)
response_window = (3000, 3200)


def main():
    for root, dirs, files in os.walk(savedRoot / dirName, topdown=False):
        for name in sorted(dirs):
            dirPath = os.path.join(root, name)
            activityPath = os.path.join(dirPath, 'activitityTest.npz')
            if not os.path.exists(activityPath):
                continue

            print(dirPath)

            x, trial_params, model_state, choice12, choiceAB, choiceLR, qAs, qBs, seqAB, loc12_label = importAndPreprocess(
                dirPath=dirPath,
                activityFileName='activitityTest.npz')

            time = np.arange(model_state.shape[1]) * dt
            cv = 5
            accuracyLR = decoding_choice_inPlace_cv(model_state, choiceLR, cv=cv)
            accuracyAB = decoding_choice_inPlace_cv(model_state, choiceAB, cv=cv)
            accuracy12 = decoding_choice_inPlace_cv(model_state, choice12, cv=cv)

            ymin = .4
            ymax = 1
            fig = plt.figure(dpi=200, figsize=(5, 3))
            plt.fill_between([500, 1000], ymin, ymax, color='gold', alpha=0.3, edgecolor=None, label='_nolegend_')
            plt.fill_between([1500, 2000], ymin, ymax, color='gold', alpha=0.3, edgecolor=None, label='_nolegend_')
            plt.fill_between([2500, 3000], ymin, ymax, color='cadetblue', alpha=0.3, edgecolor=None, label='_nolegend_')
            plt.fill_between([3000, 3200], ymin, ymax, color='plum', alpha=0.3, edgecolor=None, label='_nolegend_')
            plt.errorbar(time, np.nanmean(accuracyLR, axis=1), np.nanstd(accuracyLR, axis=1) / np.sqrt(cv), label='chosen side')
            plt.errorbar(time, np.nanmean(accuracyAB, axis=1), np.nanstd(accuracyAB, axis=1) / np.sqrt(cv), label='chosen juice')
            plt.errorbar(time, np.nanmean(accuracy12, axis=1), np.nanstd(accuracy12, axis=1) / np.sqrt(cv), label='chosen order')

            plt.legend()
            plt.xlabel('time (ms)')
            plt.ylabel('accuracy')
            plt.ylim(ymin, ymax)
            plt.tight_layout()

            np.savez(
                os.path.join(dirPath, "decodingChoiceSpatial.npz"),
                accuracy12=accuracy12,
                accuracyAB=accuracyAB,
                accuracyLR=accuracyLR,
                choice12=choice12,
                choiceAB=choiceAB,
                choiceLR=choiceLR,
                qAs=qAs,
                qBs=qBs,
                seqAB=seqAB,
                loc12_label=loc12_label,
                time=time,
                target_window=target_window,
                response_window=response_window)
            fig.savefig(os.path.join(dirPath, "decodingChoiceSpatial.pdf"))

            plt.close('all')


def loc12_to_label(loc12):
    if isinstance(loc12, str):
        return loc12
    loc12 = np.asarray(loc12)
    return '12' if loc12[0] == 1 else '21'


def mean_response_output(model_output, trial_params, mask=None, dt=dt):
    if mask is not None and np.any(mask):
        mask_sum = np.sum(mask, axis=1)
        mask_sum[mask_sum == 0] = 1
        return np.sum(mask * model_output, axis=1) / mask_sum

    response_output = np.zeros((model_output.shape[0], model_output.shape[2]))
    for iTrial, params in enumerate(trial_params):
        response_onset = params.get('fixation_offset', response_window[0])
        response_offset = params.get('end', response_window[1])
        i0 = int(round(response_onset / dt))
        i1 = int(round(response_offset / dt))
        i0 = max(0, min(i0, model_output.shape[1] - 1))
        i1 = max(i0 + 1, min(i1, model_output.shape[1]))
        response_output[iTrial, :] = np.mean(model_output[iTrial, i0:i1, :], axis=0)
    return response_output


def derive_choice_labels_from_model_output(model_output, trial_params, mask=None):
    response_output = mean_response_output(model_output, trial_params, mask=mask)
    choice_right = response_output[:, 1] > response_output[:, 0]
    choiceLR = np.array(['right' if right else 'left' for right in choice_right])

    seqAB = np.array([trial_params[i]['seqAB'] for i in range(len(trial_params))])
    loc12_label = np.array([
        trial_params[i].get('loc12_label', loc12_to_label(trial_params[i]['loc12']))
        for i in range(len(trial_params))
    ])

    chosen_offer = np.zeros(len(trial_params), dtype=int)
    for iTrial in range(len(trial_params)):
        offer1_left = loc12_label[iTrial] == '12'
        if choiceLR[iTrial] == 'left':
            chosen_offer[iTrial] = 1 if offer1_left else 2
        else:
            chosen_offer[iTrial] = 2 if offer1_left else 1

    choice12 = np.array(['2' if offer == 2 else '1' for offer in chosen_offer])
    choiceAB = np.array([
        juice_for_offer(seqAB[iTrial], chosen_offer[iTrial])
        for iTrial in range(len(trial_params))
    ])

    return choice12, choiceAB, choiceLR, seqAB, loc12_label


def juice_for_offer(seqAB, chosen_offer):
    if seqAB == 'AB':
        return 'A' if chosen_offer == 1 else 'B'
    return 'B' if chosen_offer == 1 else 'A'


def decoding_choice_inPlace_cv(model_state, choice, estimator=LinearDiscriminantAnalysis(), cv=5):
    nT = model_state.shape[1]
    accuracy = np.full((nT, cv), np.nan)
    choice = np.asarray(choice)

    labels, counts = np.unique(choice, return_counts=True)
    if len(labels) < 2 or np.min(counts) < 2:
        return accuracy

    cv_eff = min(cv, int(np.min(counts)))
    for iT in range(nT):
        Xt = model_state[:, iT, :]
        scores = cross_val_score(estimator, Xt, choice, cv=cv_eff)
        accuracy[iT, :cv_eff] = scores
    return accuracy


def importAndPreprocess(dirPath, activityFileName):
    with np.load(os.path.join(dirPath, activityFileName), allow_pickle=True) as f:
        x = f['x']
        trial_params = f['trial_params']
        model_output = f['model_output']
        model_state = f['model_state']
        mask = f.get('mask', None)

    choice12, choiceAB, choiceLR, seqAB, loc12_label = derive_choice_labels_from_model_output(
        model_output,
        trial_params,
        mask=mask)

    qAs = np.array([trial_params[i]['qA'] for i in range(len(trial_params))])
    qBs = np.array([trial_params[i]['qB'] for i in range(len(trial_params))])

    return x, trial_params, model_state, choice12, choiceAB, choiceLR, qAs, qBs, seqAB, loc12_label


if __name__ == '__main__':
    main()
