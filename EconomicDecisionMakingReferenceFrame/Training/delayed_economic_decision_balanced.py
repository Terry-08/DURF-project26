import numpy as np

from delayed_economic_decision_combined import DelayedEconomicDecision_SpatialTask
from delayed_economic_decision_seperated import DelayedEconomicDecision_SeparatedSpatialTask


# Each condition is (qA, qB, seqAB). Keeping the sequence in the condition is
# essential: AB balances choices for a fixed qA, while BA does so for a fixed qB.
BALANCED_OFFER_CONDITIONS = (
    (1, 1, 'AB'), (1, 3, 'AB'),
    (2, 2, 'AB'), (2, 6, 'AB'),
    (3, 3, 'AB'), (3, 8, 'AB'),
    (1, 3, 'BA'), (3, 3, 'BA'),
    (1, 4, 'BA'), (4, 4, 'BA'),
    (2, 5, 'BA'), (4, 5, 'BA'),
)


class _BalancedOfferConditionsMixin:
    def _configure_balanced_conditions(self, offer_conditions, N_trials_per_condition):
        self.offer_conditions = tuple(offer_conditions)
        self.N_trials_per_condition = N_trials_per_condition

    def generate_trial_params(self, batch, trial):
        params = dict()

        inter_offer_duration = self.InterOffer_duration
        if isinstance(inter_offer_duration, (range, list, tuple)):
            inter_offer_duration = np.random.choice(inter_offer_duration)

        params['stimulus_1_onset'] = self.onset_time
        params['stimulus_1_offset'] = self.onset_time + self.stim_duration_1
        params['stimulus_2_onset'] = params['stimulus_1_offset'] + inter_offer_duration
        params['stimulus_2_offset'] = params['stimulus_2_onset'] + self.stim_duration_2
        params['target_onset'] = params['stimulus_2_offset'] + self.wait_duration
        params['fixation_offset'] = params['target_onset'] + self.target_delay_duration
        params['end'] = params['fixation_offset'] + self.respond_duration
        params['stim_noise'] = 0.01

        if self.N_trials_per_condition is None:
            condition_index = np.random.choice(len(self.offer_conditions))
            loc12 = self.spatial_options[np.random.choice(len(self.spatial_options))]
        else:
            trials_per_condition = 2 * self.N_trials_per_condition
            condition_index = int(trial / trials_per_condition)
            subtrial = np.mod(trial, trials_per_condition)
            location_index = int(subtrial / self.N_trials_per_condition)
            loc12 = self.spatial_options[location_index]

        qA, qB, seqAB = self.offer_conditions[condition_index]
        loc12_label = '12' if loc12[0] == 1 else '21'
        chooseB = self._choice_stochastic(qA, qB)
        chosen_offer = 2 if ((seqAB == 'AB' and chooseB) or
                             (seqAB == 'BA' and not chooseB)) else 1
        choice = 0 if ((loc12[0] == 1 and chosen_offer == 1) or
                       (loc12[1] == 1 and chosen_offer == 2)) else 1

        params['qA'], params['qB'] = qA, qB
        params['seqAB'] = seqAB
        params['loc12'] = loc12.copy()
        params['loc12_label'] = loc12_label
        params['choice'] = choice
        params['chooseB'] = chooseB
        params['chosen_offer'] = chosen_offer

        return params


class DelayedEconomicDecision_BalancedCombined(
        _BalancedOfferConditionsMixin, DelayedEconomicDecision_SpatialTask):
    def __init__(self, dt, tau, T, N_batch=None, offer_conditions=None,
                 N_trials_per_condition=None, **kwargs):
        conditions = (BALANCED_OFFER_CONDITIONS if offer_conditions is None
                      else tuple(offer_conditions))
        if N_trials_per_condition is not None:
            if N_batch is not None:
                raise UserWarning(
                    'fixed N_trials_per_condition is set. N_batch will be ignored.')
            N_batch = 2 * len(conditions) * N_trials_per_condition

        super().__init__(
            dt=dt,
            tau=tau,
            T=T,
            N_batch=N_batch,
            offer_pairs=[condition[:2] for condition in conditions],
            N_trials_per_condition=None,
            **kwargs
        )
        self._configure_balanced_conditions(conditions, N_trials_per_condition)


class DelayedEconomicDecision_BalancedSeparated(
        _BalancedOfferConditionsMixin, DelayedEconomicDecision_SeparatedSpatialTask):
    def __init__(self, dt, tau, T, N_batch=None, offer_conditions=None,
                 N_trials_per_condition=None, **kwargs):
        conditions = (BALANCED_OFFER_CONDITIONS if offer_conditions is None
                      else tuple(offer_conditions))
        if N_trials_per_condition is not None:
            if N_batch is not None:
                raise UserWarning(
                    'fixed N_trials_per_condition is set. N_batch will be ignored.')
            N_batch = 2 * len(conditions) * N_trials_per_condition

        super().__init__(
            dt=dt,
            tau=tau,
            T=T,
            N_batch=N_batch,
            offer_pairs=[condition[:2] for condition in conditions],
            N_trials_per_condition=None,
            **kwargs
        )
        self._configure_balanced_conditions(conditions, N_trials_per_condition)
