from delayed_economic_decision_balanced import (
    BALANCED_OFFER_CONDITIONS,
    DelayedEconomicDecision_BalancedSeparated,
)
from delayed_economic_decision_seperated import DelayedEconomicDecision_SeparatedSpatialTask
from psychrnn.backend.models.basic import Basic
import os
import datetime
import numpy as np
from psychrnn.backend.simulation import BasicSimulator

taskTrainName = 'spatialTaskSeperatedBalanced'
saveRoot = './savedForHPC/'
os.makedirs(saveRoot+taskTrainName, exist_ok=True)

# task and model parameters
dt = 10 # The simulation timestep.
tau = 100 # The intrinsic time constant of neural state decay.
T = 4000 # The trial length.
N_trials_per_condition = 4 # The number of trials per training update.
dd = DelayedEconomicDecision_BalancedSeparated(
    dt=dt, tau=tau, T=T,
    N_trials_per_condition=N_trials_per_condition,
    target_delay_duration=500, wait_duration=500)

offer_pair_test = [(iA*0.5, iB*0.5*1.7) for iA in range(9) for iB in range(9)]
dd_test = DelayedEconomicDecision_SeparatedSpatialTask(
    dt=dt, tau=tau, T=T,
    target_delay_duration=500, wait_duration=500,
    N_trials_per_condition=10, offer_pairs=offer_pair_test)


N_rec = 50 # The number of recurrent units in the network.
name = 'basicModel' # Unique name used to determine variable scope for internal use.

network_params = dd.get_task_params()
network_params['name'] = name # Unique name used to determine variable scope.
network_params['N_rec'] = N_rec # The number of recurrent units in the network.
network_params['rec_noise'] = 0.1 # Noise into each recurrent unit. Default: 0.0


# Set the training parameters
train_params = {}
train_params['save_weights_path'] = None
train_params['training_iters'] = 400000
train_params['learning_rate'] = .001
train_params['loss_epoch'] = 10
train_params['verbosity'] = True
train_params['save_training_weights_epoch'] = 100
train_params['training_weights_path'] = None
train_params['clip_grads'] = True
train_params['fixed_weights'] = None


def performance_measure(trial_batch, trial_y, output_mask, output, epoch, losses, verbosity):
    chosen = np.argmax(np.mean(output*output_mask, axis=1), axis=1)
    truth = np.argmax(np.mean(trial_y*output_mask, axis=1), axis=1)
    return np.mean(np.equal(truth, chosen))


train_params['curriculum'] = None
train_params['performance_measure'] = performance_measure
train_params['performance_cutoff'] = .99

## -------- Training loop ##
ensembleSize = 40
for netii in range(ensembleSize):
    startTime = datetime.datetime.now().strftime('%Y%m%d-%H-%m')

    print(startTime, netii)

    model = Basic(network_params)
    initialWeight = model.get_weights()

    losses, trainTime, initialTime = model.train(dd, train_params)

    # ---------------------- Test the trained model ---------------------------
    x, target_output, mask, trial_params = dd.get_trial_batch()
    model_output, model_state = model.test(x)

    saveTime = datetime.datetime.now().strftime('%Y%m%d-%H-%m')
    Fail = 'Fail' if losses[-1] > 0.01 else ''

    dirName = taskTrainName+'/' +(taskTrainName+'_'+saveTime+'_'+str(netii)+'_'+Fail)+'/'
    dirPath = saveRoot+dirName
    os.makedirs(dirPath)

    model.save(dirPath+'weightFinal')
    np.savez(dirPath+'weightInit', weightInit=initialWeight)
    np.savez(dirPath+'activitityTest', x=x, trial_params=trial_params,
             model_output=model_output, model_state=model_state)
    np.savez(dirPath+'trainingHistory', losses=losses, trainTime=trainTime,
             initialTime=initialTime, startTime=startTime, saveTime=saveTime)
    np.savez(dirPath+'network_params', network_params=network_params)
    np.savez(
        dirPath+'offerConditions',
        qA=np.array([condition[0] for condition in BALANCED_OFFER_CONDITIONS]),
        qB=np.array([condition[1] for condition in BALANCED_OFFER_CONDITIONS]),
        seqAB=np.array([condition[2] for condition in BALANCED_OFFER_CONDITIONS]))

    x, target_output, mask, trial_params = dd_test.get_trial_batch()
    simulator = BasicSimulator(
        weights_path=os.path.abspath(dirPath+'weightFinal.npz'),
        params={'dt': dt, 'tau': tau})
    model_output, model_state = simulator.run_trials(x)
    np.savez(os.path.join(dirPath, 'activitityTestGrid.npz'), x=x,
             trial_params=trial_params, model_output=model_output,
             model_state=model_state, mask=mask)

    model.destruct()
    print(dirPath)
    print(os.listdir(dirPath))
