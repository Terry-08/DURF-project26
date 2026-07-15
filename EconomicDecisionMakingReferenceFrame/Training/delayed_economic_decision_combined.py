from __future__ import division
from typing import Union

from psychrnn.tasks.task import Task
import numpy as np


class DelayedEconomicDecision_SpatialTask(Task):
    """Delayed economic decision task with offer-side spatial inputs.

    Offer 1 and Offer 2 are presented sequentially, with the first/second offer
    randomly assigned to left/right or right/left screen positions. At target
    time, the network reports the side of the chosen offer.
    """

    def __init__(self, dt, tau, T, N_batch=None,
        onset_time=500, stim_duration_1=500, InterOffer_duration: Union[range,int]=500, stim_duration_2=500,
        wait_duration=0, target_delay_duration=200, respond_duration=200,
        offer_pairs=None, N_trials_per_condition=None, ind_point=1.7):

        self._input_names = [
            'qA_left', 'qA_right',
            'qB_left', 'qB_right',
            'target',
            'fixation'
        ]
        self._input_index = {name: ii for ii, name in enumerate(self._input_names)}
        self.output_names = ['left', 'right']

        if offer_pairs is not None:
            self.offer_pairs = offer_pairs
        else:
            self.offer_pairs = [(0, 1),
                            (0, 2),
                            (1, 0),
                            (1, 3),
                            (1, 4),
                            (2, 1),
                            (2, 2),
                            (2, 3),
                            (2, 4),
                            (2, 6),
                            (3, 2),
                            (3, 3),
                            (3, 8),
                            (4, 4)]

        self.seq_options = ['AB', 'BA']
        self.spatial_options = [np.array([1, 0]), np.array([0, 1])]

        if N_trials_per_condition is not None:
            if N_batch is not None:
                raise UserWarning('fixed N_trials_per_condition is set. N_batch will be ignored.')
            N_batch = 4 * len(self.offer_pairs) * N_trials_per_condition
        self.N_trials_per_condition = N_trials_per_condition

        super(DelayedEconomicDecision_SpatialTask,self).__init__(len(self._input_names), 2, dt, tau, T, N_batch)

        self.onset_time = onset_time
        self.stim_duration_1 = stim_duration_1
        self.InterOffer_duration = InterOffer_duration
        self.stim_duration_2 = stim_duration_2
        self.wait_duration = wait_duration
        self.target_delay_duration = target_delay_duration
        self.respond_duration = respond_duration

        self.ind_point = ind_point
        self.a1_choice = 13
        self.lo = 0.2
        self.hi = 1.0

    def _range_norm_B(self, offerquantity):
        maxq = 8
        minq = 0
        return (offerquantity-minq)/(maxq-minq)

    def _range_norm_A(self, offerquantity):
        maxq = 4
        minq = 0
        return (offerquantity-minq)/(maxq-minq)

    def _choice_stochastic(self, qA, qB):
        if qA == 0:
            return 1
        if qB == 0:
            return 0
        offerRatio = qB/qA
        X = self.a1_choice * (np.log(offerRatio/self.ind_point))
        p = 1/(1+np.exp(-X))
        return int(np.random.random() < p)

    def _offer_channel(self, juice, side):
        return self._input_index['q%s_%s' % (juice, side)]

    def generate_trial_params(self, batch, trial):
        params = dict()

        onset_time = self.onset_time
        stim_duration_1 = self.stim_duration_1
        InterOffer_duration = self.InterOffer_duration
        if type(InterOffer_duration) in (range,list,tuple):
            InterOffer_duration = np.random.choice(InterOffer_duration)
        stim_duration_2 = self.stim_duration_2
        wait_duration = self.wait_duration
        target_delay_duration = self.target_delay_duration
        respond_duration = self.respond_duration

        params['stimulus_1_onset'] = onset_time
        params['stimulus_1_offset'] = onset_time + stim_duration_1
        params['stimulus_2_onset'] = onset_time + stim_duration_1 + InterOffer_duration
        params['stimulus_2_offset'] = onset_time + stim_duration_1 + InterOffer_duration + stim_duration_2
        params['target_onset'] = params['stimulus_2_offset'] + wait_duration
        params['fixation_offset'] = params['stimulus_2_offset'] + wait_duration + target_delay_duration
        params['end'] = params['fixation_offset'] + respond_duration
        params['stim_noise'] = 0.01

        if self.N_trials_per_condition is None:
            offer_pair = self.offer_pairs[np.random.choice(len(self.offer_pairs))]
            seqAB = np.random.choice(self.seq_options)
            loc12 = self.spatial_options[np.random.choice(len(self.spatial_options))]
        else:
            offer_pair = self.offer_pairs[int(trial/4/self.N_trials_per_condition)]
            subtrial = np.mod(trial, 4*self.N_trials_per_condition)
            seqAB = self.seq_options[int(subtrial/self.N_trials_per_condition/2)]
            subtrial = np.mod(subtrial, 2*self.N_trials_per_condition)
            loc12 = self.spatial_options[int(subtrial/self.N_trials_per_condition)]

        loc12_label = '12' if loc12[0] == 1 else '21'
        chooseB = self._choice_stochastic(*offer_pair)
        chosen_offer = 2 if ((seqAB == 'AB' and chooseB) or (seqAB == 'BA' and not chooseB)) else 1
        choice = 0 if ((loc12[0] == 1 and chosen_offer == 1) or (loc12[1] == 1 and chosen_offer == 2)) else 1

        params['qA'], params['qB'] = offer_pair
        params['seqAB'] = seqAB
        params['loc12'] = loc12.copy()
        params['loc12_label'] = loc12_label
        params['choice'] = choice
        params['chooseB'] = chooseB
        params['chosen_offer'] = chosen_offer

        return params

    def trial_function(self, t, params):
        x_t = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(self.dt)*params['stim_noise']*params['stim_noise'])*np.random.randn(self.N_in)
        y_t = np.zeros(self.N_out)
        mask_t = np.zeros(self.N_out)

        stimulus_1_onset = params['stimulus_1_onset']
        stimulus_1_offset = params['stimulus_1_offset']
        stimulus_2_onset = params['stimulus_2_onset']
        stimulus_2_offset = params['stimulus_2_offset']
        target_onset = params['target_onset']
        fixation_offset = params['fixation_offset']
        end = params['end']

        qA, qB = params['qA'], params['qB']
        seqAB = params['seqAB']
        loc12 = params['loc12']
        choice = params['choice']

        side1, side2 = ('left', 'right') if loc12[0] == 1 else ('right', 'left')
        if seqAB == 'AB':
            juice1, juice2 = 'A', 'B'
            q1, q2 = self._range_norm_A(qA), self._range_norm_B(qB)
        else:
            juice1, juice2 = 'B', 'A'
            q1, q2 = self._range_norm_B(qB), self._range_norm_A(qA)

        if stimulus_1_onset <= t < stimulus_1_offset:
            x_t[self._offer_channel(juice1, side1)] += q1

        if stimulus_2_onset <= t < stimulus_2_offset:
            x_t[self._offer_channel(juice2, side2)] += q2

        if target_onset <= t < fixation_offset:
            x_t[self._input_index['target']] += 1

        if t < fixation_offset:
            x_t[self._input_index['fixation']] += 1

        if fixation_offset <= t < end:
            y_t[choice] = self.hi
            y_t[1-choice] = self.lo
            mask_t = np.ones(self.N_out)

        return x_t, y_t, mask_t

    def accuracy_function(self, correct_output, test_output, output_mask):
        chosen = np.argmax(np.mean(test_output*output_mask, axis=1), axis=1)
        truth = np.argmax(np.mean(correct_output*output_mask, axis=1), axis=1)
        return np.mean(np.equal(truth, chosen))
