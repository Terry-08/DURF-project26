import numpy as np

def performance_measure_for_RNN(trial_batch, trial_y, output_mask, output, epoch, losses, verbosity,param):
    N_batch,N_step,N_in=trial_batch.shape
    dt = param['dt']
    a1_choice = param['a1_choice']
    ind_point = param['ind_point']
    onset_time = param['onset_time']
    stim_duration_1 = param['stim_duration_1']
    InterOffer_duration= param['InterOffer_duration']
    stim_duration_2 = param['stim_duration_2']
    
    qA,qB = reverseCalcQuantity(trial_batch, dt, onset_time, stim_duration_1, InterOffer_duration, stim_duration_2)
    probability = np.array([choice_probability(qA[ii],qB[ii],a1_choice,ind_point) for ii in range(N_batch)])
    return accuracy_function(trial_y, output_mask, output, probability)

def choice_probability(qA,qB,a1,ind_point):
    if qA<=0:
        return 1
    if qB<=0:
        return 1
    offerRatio = qB/qA
    X = a1*(np.log(offerRatio/ind_point))
    p = 1/(1+np.exp(-X))

    return np.max([p,1-p])
    
def reverseCalcQuantity(trial_batch,dt,onset_time, stim_duration_1, InterOffer_duration, stim_duration_2):
    N_batch,N_step,N_in=trial_batch.shape
    max_qA=8
    min_qA=0
    max_qB=4
    min_qB=0
    stimulus_1_onset = onset_time
    stimulus_1_offset = onset_time + stim_duration_1
    stimulus_2_onset = onset_time + stim_duration_1 + InterOffer_duration
    stimulus_2_offset = onset_time + stim_duration_1 + InterOffer_duration + stim_duration_2
    qA_norm = np.mean(trial_batch[:,int(stimulus_1_onset/dt):int(stimulus_1_offset/dt),0],axis=1)
    qB_norm = np.mean(trial_batch[:,int(stimulus_1_onset/dt):int(stimulus_1_offset/dt),1],axis=1)
    qA = qA_norm * (max_qA-min_qA) + min_qA  
    qB = qB_norm * (max_qB-min_qB) + min_qB  
    return qA,qB
    
def accuracy_function(correct_output, output_mask, test_output, probability):
        """Calculates the accuracy of :data:`test_output`.

        Implements :func:`~psychrnn.tasks.task.Task.accuracy_function`.

        Takes the channel-wise mean of the masked output for each trial. Whichever channel has a greater mean is considered to be the network's "choice".

        Returns:
            float: 0 <= accuracy <= 1. Accuracy is equal to the ratio of trials in which the network made the correct choice as defined above.
        
        """
        
        chosen = np.argmax(np.mean(test_output*output_mask, axis=1), axis = 1)
        truth = np.argmax(np.mean(correct_output*output_mask, axis = 1), axis = 1)

        # modification by JTG: trial-wise soft penalty for economic decision with designed choice probability
        # 0.5 <= probability <=1
        return np.mean(np.equal(truth, chosen)*probability + (1-probability) )
    