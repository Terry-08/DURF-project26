from __future__ import division
from typing import Union

from psychrnn.tasks.task import Task
import numpy as np


class DelayedEconomicDecision_AlternateOutput(Task):
    """Delayed economic decision making task adapted from Ballesta & Padoa-Schioppa 2019
    Offer input restricted to two juice channels
    Choice output can be either in juice or order reference frame.

    [simple description]

    Takes two channels of normalized quantity inputs (A or B), 
    four one-hot encoding choice-to-action mapping cues, 
    one fixation cues  (:attr:`N_in` = 9).
    Two channel output (:attr:`N_out` = 2) with a one hot encoding (high value is 1, low value is .2).

    Args:
        dt (float): The simulation timestep.
        tau (float): The intrinsic time constant of neural state decay.
        T (float): The trial length.
        N_batch (int): The number of trials per training update.
        outputMode (str): 'juice', 'order' or 'both'
        encode_spatial (bool): Add explicit one-hot spatial layout input channels.
        spatial_timing (str): When to present explicit spatial channels. Supports 'offer' and 'target'.
        onset_time (int): first stimulus onset time.
        stim_duration_1 (int): Duration of first stimulus.
        InterOffer_duration (int): Duration of inter-offer interval.
        stim_duration_2 (int): Duration of second stimulus.
        wait_duration (int): Duration of wait period.
        target_delay_duration (int): Duration of choice-to-action mapping target delay period.
        respond_duration (int): Duration of respond period.
        ind_point (float): instructed indifference point.
        offer_pairs (list): list of offer quantity pairs (qA,qB) to select from.
    """
    def __init__(self, dt, tau, T, N_batch=None,
        onset_time = 500, stim_duration_1 = 500, InterOffer_duration: Union[range,int] = 500, stim_duration_2 = 500, 
        early_conxt=False,
        wait_duration = 0, target_delay_duration = 200, respond_duration=200,outputMode='both',juiceTrialProp=0.5,offer_pairs=None,N_trials_per_condition=None,ind_point=1.7,
        encode_spatial=False, spatial_timing='target'):
        if spatial_timing not in ('offer', 'target'):
            raise ValueError("spatial_timing must be either 'offer' or 'target'.")

        N_stim = 2
        N_target = 2+2+2
        N_fixation = 2 if early_conxt else 1
        N_spatial = 2 if encode_spatial else 0
        N_in = N_stim+N_target+N_fixation+N_spatial
        
        self._early_conxt = early_conxt
        self.encode_spatial = encode_spatial
        self.spatial_timing = spatial_timing
        self._input_names = ['qA_norm','qB_norm', # juice quantity inputs
                'A_','B_','_A','_B', # choice-to-action mapping cues for the juice task
                '12','21',  # choice-to-action mapping cues for the order task
                'fixation'] # fixation cue
        
        if early_conxt:
            self._input_names = ['qA_norm','qB_norm','A_','B_','_A','_B','12','21','fix_juice','fix_order']

        if encode_spatial:
            self._input_names.extend(['spatial_opt1_left','spatial_opt2_left'])
        self._input_index = {name: ii for ii, name in enumerate(self._input_names)}
        
        self.output_names = ['left','right']

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
                            (4, 4)] # offer quantity pairs (qA,qB) to select from

        if N_trials_per_condition is not None:
            if N_batch is not None:
                raise UserWarning('fixed N_trials_per_condition is set. N_batch will be ignored.')
            N_offerPair = len(self.offer_pairs)
            if outputMode=='both':
                N_batch = 4*N_offerPair*N_trials_per_condition
            else:
                N_batch = 2*N_offerPair*N_trials_per_condition
        self.N_trials_per_condition = N_trials_per_condition

        super(DelayedEconomicDecision_AlternateOutput,self).__init__(N_in, 2, dt, tau, T, N_batch)
        
        self.onset_time = onset_time
        self.stim_duration_1 = stim_duration_1
        self.InterOffer_duration = InterOffer_duration
        self.stim_duration_2 = stim_duration_2
        self.wait_duration = wait_duration
        self.target_delay_duration = target_delay_duration
        self.respond_duration = respond_duration 
        self.outputMode = outputMode
        if outputMode == 'both':    self.juiceTrialProp=juiceTrialProp

        self.seq_options = ['AB', 'BA'] # juice sequence to select from
        self.spatial_options = ['AB', 'BA'] # juice location to select from
        self.choiceFrame_options = ['juice','order'] # target represent juice A/B or order 1st/2nd
        # behavior parameter
        self.ind_point = ind_point #2*np.random.gamma(20,0.05) # indifference point
        self.a1_choice = 13 #15*np.random.gamma(100,0.01) # related to steepness


        self.lo = 0.2 # Low value for one hot encoding
        self.hi = 1.0 # High value for one hot encoding

    def _get_fixation_channel(self, choiceFrame):
        if not self._early_conxt:
            return self._input_index['fixation']
        if choiceFrame == 'juice':
            return self._input_index['fix_juice']
        return self._input_index['fix_order']

    def _get_target_channels(self, choiceFrame, locAB):
        if choiceFrame == 'juice':
            if locAB == 'AB':
                return [self._input_index['A_'], self._input_index['_B']]
            return [self._input_index['B_'], self._input_index['_A']]
        if locAB == '12':
            return [self._input_index['12']]
        return [self._input_index['21']]

    def _get_spatial_channels(self, choiceFrame, locAB):
        if not self.encode_spatial:
            return []
        if choiceFrame == 'juice':
            opt1_left = locAB == 'AB'
        else:
            opt1_left = locAB == '12'
        if opt1_left:
            return [self._input_index['spatial_opt1_left']]
        return [self._input_index['spatial_opt2_left']]
    
    def _range_norm_B(self,offerquantity):
        """range normalization
        """
        maxq=8
        minq=0
        return (offerquantity-minq)/(maxq-minq)
    def _range_norm_A(self,offerquantity):
        maxq=4
        minq=0
        return (offerquantity-minq)/(maxq-minq)
    def _choice_stochastic(self,qA,qB):
        """generate stochastic choice with logistic function

        Args:
            qA,qB: quantities of offerA and offer B
        Returns: 
            bool: choice B (True) or A (False)

        """
        if qA==0:
            return 1
        if qB==0:
            return 0
        offerRatio = qB/qA
        a1 = self.a1_choice
        ind_point = self.ind_point
        X = a1*(np.log(offerRatio/ind_point))
        p = 1/(1+np.exp(-X))

        chooseB = int(np.random.random()<p)
        return  chooseB

    def generate_trial_params(self, batch, trial):
        """Define parameters for each trial, 
        analogous to behavioral parameters in animal experiments.

        Implements :func:`~psychrnn.tasks.task.Task.generate_trial_params`.

        Args:
            batch (int): The batch number that this trial is part of.
            trial (int): The trial number of the trial within the batch *batch*.

        Returns:
            dict: Dictionary of trial parameters including the following keys:

            :Dictionary Keys: 
                * **stimulus_1** (*float*) -- Start time for stimulus one. :data:`onset_time`. 
                * **delay1** (*float*) -- Start time for the delay 1. :data:`onset_time` + :data:`stimulus_duration_1`. 
                * **stimulus_2** (*float*) -- Start time in for stimulus one. :data:`onset_time` + :data:`stimulus_duration_1` + :data:`delay_duration`. 
                * **delay2** (*float*) -- Start time for the delay 2. :data:`onset_time` + :data:`stimulus_duration_1`. 
                * **decision** (*float*) -- Start time in for decision period. :data:`onset_time` + :data:`stimulus_duration_1` + :data:`delay_duration` + :data:`stimulus_duration_2`. 
                * **end** (*float*) -- End of decision period. :data:`onset_time` + :data:`stimulus_duration_1` + :data:`delay_duration` + :data:`stimulus_duration_2` + :data:`decision_duration`. 
                * **stim_noise** (*float*) -- Scales the stimlus noise. Set to .01.
                * **qA** (*int*) -- quantity of juice A. 
                * **qB** (*int*) -- quantity of juice B.
                * **seqAB** (*str*) -- temporal order of apperance 'AB' or 'BA'
                * **locAB** (*str*) -- spatial arangement of choice 'AB' or 'BA'; note that this variable is used in the order task but with different meaning
                * **chooseB** (*int*) -- Indicates whether juice B is chosen (1) or A (0).
                * **choice** (*int*) -- Indicates whether left choice is chosen (1) or not(0).
        """

        params = dict()

        onset_time              =self.onset_time
        stim_duration_1         =self.stim_duration_1
        InterOffer_duration     =self.InterOffer_duration
        if type(InterOffer_duration) in (range,list,tuple):
            InterOffer_duration = np.random.choice(InterOffer_duration)
        stim_duration_2         =self.stim_duration_2
        wait_duration           =self.wait_duration
        target_delay_duration   =self.target_delay_duration
        respond_duration        =self.respond_duration 
        outputMode              =self.outputMode

        params['stimulus_1_onset'] = onset_time
        params['stimulus_1_offset'] = onset_time + stim_duration_1
        params['stimulus_2_onset'] = onset_time + stim_duration_1 + InterOffer_duration
        params['stimulus_2_offset'] = onset_time + stim_duration_1 + InterOffer_duration + stim_duration_2
        params['target_onset'] = params['stimulus_2_offset'] + wait_duration
        params['fixation_offset'] = params['stimulus_2_offset'] + wait_duration + target_delay_duration
        params['end'] = params['fixation_offset'] + respond_duration

        params['stim_noise'] = 0.01 # input noinse. Recurrent noise set in RNN definition (training script)
        if self.N_trials_per_condition is None:
            offer_pair = self.offer_pairs[np.random.choice(len(self.offer_pairs))]
            seqAB = np.random.choice(self.seq_options)
            if outputMode == 'both':
                if self.juiceTrialProp==0.5:
                    choiceFrame = np.random.choice(self.choiceFrame_options)
                else:
                    choiceFrame = np.random.choice(self.choiceFrame_options,p=[self.juiceTrialProp,1-self.juiceTrialProp])
            elif outputMode == 'juice' or outputMode == 'order':
                choiceFrame = outputMode
        else: # for testing, all condition equal trials, no juice trial proportion. 
            if outputMode == 'both':
                offer_pair = self.offer_pairs[int(trial/4/self.N_trials_per_condition)]
                subtrial = np.mod(trial,4*self.N_trials_per_condition)
                seqAB = self.seq_options[int(subtrial/self.N_trials_per_condition/2)]
                subtrial = np.mod(subtrial,2*self.N_trials_per_condition)
                choiceFrame = self.choiceFrame_options[int(subtrial/self.N_trials_per_condition)]
            else:
                offer_pair = self.offer_pairs[int(trial/2/self.N_trials_per_condition)]
                subtrial = np.mod(trial,2*self.N_trials_per_condition)
                seqAB = self.seq_options[int(subtrial/self.N_trials_per_condition)]
                choiceFrame = outputMode
        locAB = np.random.choice(self.spatial_options)

        chooseB = self._choice_stochastic(*offer_pair)

        if choiceFrame == 'juice':
            if locAB == 'AB':
                choice = chooseB # choice in order reference frame (saccade)
            else:
                choice = 1-chooseB
        elif choiceFrame == 'order':
            if seqAB == 'AB':
                choice12 = chooseB
            else:
                choice12 = 1-chooseB
            if locAB == 'AB':
                loc12 = '12'
                choice = choice12
            else:
                loc12 = '21'
                choice = 1-choice12
            
        params['qA'],params['qB'] = offer_pair
        params['choiceFrame'] = choiceFrame
        params['seqAB'] = seqAB
        if choiceFrame=='juice' :
            params['locAB'] = locAB 
        else:
            params['locAB'] = loc12 # for compatibility 
            params['loc12'] = loc12
        params['choice'] = choice
        params['chooseB'] = chooseB

        return params

    def trial_function(self, t, params):
        """Compute the trial properties at :data:`time`.
        Inputs and instructed output (for supervised learning) to the RNN. Temproal snapshots

        Implements :func:`~psychrnn.tasks.task.Task.trial_function`.

        Based on the :data:`params` compute the trial stimulus (x_t), correct output (y_t), and mask (mask_t) at :data:`time`.

        Args:
            time (int): The time within the trial (0 <= :data:`time` < :attr:`T`).
            params (dict): The trial params produced by :func:`generate_trial_params`.

        Returns:
            tuple:

            * **x_t** (*ndarray(dtype=float, shape=(*:attr:`N_in` *,))*) -- Trial input at :data:`time` given :data:`params`. First channel contains :data:`f1` during the first stimulus period, and :data:`f2` during the second stimulus period, scaled to be between .4 and 1.2. Second channel contains the frequencies but reverse scaled -- high frequencies correspond to low values and vice versa. Both channels have baseline noise.
            * **y_t** (*ndarray(dtype=float, shape=(*:attr:`N_out` *,))*) -- Correct trial output at :data:`time` given :data:`params`. The correct output is encoded using one-hot encoding during the decision period.
            * **mask_t** (*ndarray(dtype=bool, shape=(*:attr:`N_out` *,))*) -- True if the network should train to match the y_t, False if the network should ignore y_t when training. The mask is True for during the decision period and False otherwise.
        
        """
        # ----------------------------------
        # Initialize with noise
        # ----------------------------------
        x_t = np.sqrt(2*.01*np.sqrt(10)*np.sqrt(self.dt)*params['stim_noise']*params['stim_noise'])*np.random.randn(self.N_in)
        y_t = np.zeros(self.N_out)
        mask_t = np.zeros(self.N_out)

        # ----------------------------------
        # Retrieve parameters
        # ----------------------------------
        stimulus_1_onset=   params['stimulus_1_onset']
        stimulus_1_offset=  params['stimulus_1_offset'] 
        stimulus_2_onset=   params['stimulus_2_onset'] 
        stimulus_2_offset=  params['stimulus_2_offset'] 
        target_onset=       params['target_onset'] 
        fixation_offset=    params['fixation_offset']
        end=                params['end'] 

        offer_pair = params['qA'],params['qB']
        seqAB = params['seqAB']
        locAB = params['locAB']
        choice = params['choice']
        chooseB = params['chooseB']
        choiceFrame = params['choiceFrame']

        # offer quantities at offer periods
        if seqAB == 'AB':
            c1,c2=(0,1) # stim channel [1A,2B]
            q1, q2 = offer_pair
            q1 = self._range_norm_A(q1)
            q2 = self._range_norm_B(q2)
        else:
            c1,c2=(1,0) # stim channel [1B 2A]
            q2, q1 = offer_pair
            q1 = self._range_norm_B(q1)
            q2 = self._range_norm_A(q2)
       
        # instructed choice at target period
        fix_channel = self._get_fixation_channel(choiceFrame)
        target_channels = self._get_target_channels(choiceFrame, locAB)
        spatial_channels = self._get_spatial_channels(choiceFrame, locAB)


        # ----------------------------------
        # Compute values
        # ----------------------------------
            
        if stimulus_1_onset <= t < stimulus_1_offset:
            # at stimulus 1 period
            # normalized quantity of the first offer (q1)
            # is input through the corrsponding juice channel (c1)
            x_t[c1] += q1
            if self.spatial_timing == 'offer':
                for spatial_channel in spatial_channels:
                    x_t[spatial_channel] +=1

        if stimulus_2_onset <= t < stimulus_2_offset:
            # at stimulus 2 period, q2 is input throught c2 channel
            x_t[c2] += q2
            if self.spatial_timing == 'offer':
                for spatial_channel in spatial_channels:
                    x_t[spatial_channel] +=1

        if target_onset <=t:
            for target_channel in target_channels:
                x_t[target_channel] +=1
            if self.spatial_timing == 'target':
                for spatial_channel in spatial_channels:
                    x_t[spatial_channel] +=1

        if t<fixation_offset: # fixation channel is on until the go cue
            x_t[fix_channel] +=1
        else:
            y_t[choice] = self.hi 
            y_t[1-choice] = self.lo
            mask_t = np.ones(self.N_out)

        return x_t, y_t, mask_t

    def accuracy_function(self, correct_output, test_output, output_mask):
        """Calculates the accuracy of :data:`test_output`.

        Implements :func:`~psychrnn.tasks.task.Task.accuracy_function`.

        Takes the channel-wise mean of the masked output for each trial. Whichever channel has a greater mean is considered to be the network's "choice".

        Returns:
            float: 0 <= accuracy <= 1. Accuracy is equal to the ratio of trials in which the network made the correct choice as defined above.
        
        THIS FUNCTION IS NOT USED IN THE TRAINING PROCESS
        """
        
        chosen = np.argmax(np.mean(test_output*output_mask, axis=1), axis = 1)
        truth = np.argmax(np.mean(correct_output*output_mask, axis = 1), axis = 1)
        return np.mean(np.equal(truth, chosen))
