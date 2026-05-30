from delayed_economic_deicision import DelayedEconomicDecision_AlternateOutput 
from psychrnn.backend.models.basic import Basic
from accuracy_function_for_seqEcoDecMake import performance_measure_for_RNN
import os 
import datetime
import numpy as np
from psychrnn.backend.simulation import BasicSimulator

taskTrainName = 'juiceTaskDefault'
saveRoot = './savedForHPC/'
os.makedirs(saveRoot+taskTrainName,exist_ok=True)

# task and model parameters
dt = 10 # The simulation timestep.
tau = 100 # The intrinsic time constant of neural state decay.
T = 4000 # The trial length.
N_trials_per_condition = 4 # The number of trials per training update.
dd = DelayedEconomicDecision_AlternateOutput(dt = dt, tau = tau, T = T, N_trials_per_condition = N_trials_per_condition,target_delay_duration=500,wait_duration=500,outputMode='juice')

offer_pair_test = [(iA*0.5,iB*0.5*1.7) for iA in range(9) for iB in range(9)]
dd_test = DelayedEconomicDecision_AlternateOutput(dt = dt, tau = tau, T = 4000, target_delay_duration=500,wait_duration=500,outputMode='juice',
                                                  N_trials_per_condition = 10,offer_pairs=offer_pair_test)


N_rec = 50 # The number of recurrent units in the network.
name = 'basicModel' #  Unique name used to determine variable scope for internal use.

network_params = dd.get_task_params()
network_params['name'] = name # Unique name used to determine variable scope.
network_params['N_rec'] = N_rec # The number of recurrent units in the network.

network_params['rec_noise'] = 0.1 # Noise into each recurrent unit. Default: 0.0



# Set the training parameters 
train_params = {}
train_params['save_weights_path'] =  None # Where to save the model after training. Default: None
train_params['training_iters'] = 400000 # number of iterations to train for Default: 50000
train_params['learning_rate'] = .001 # Sets learning rate if use default optimizer Default: .001
train_params['loss_epoch'] = 10 # Compute and record loss every 'loss_epoch' epochs. Default: 10
train_params['verbosity'] = True # If true, prints information as training progresses. Default: True
train_params['save_training_weights_epoch'] = 100 # save training weights every 'save_training_weights_epoch' epochs. Default: 100
train_params['training_weights_path'] = None # where to save training weights as training progresses. Default: None
# train_params['optimizer'] = tf.compat.v1.train.AdamOptimizer(learning_rate=train_params['learning_rate']) # What optimizer to use to compute gradients. Default: tf.train.AdamOptimizer(learning_rate=train_params['learning_rate'])
train_params['clip_grads'] = True # If true, clip gradients by norm 1. Default: True
# Example usage of the optional fixed_weights parameter is available in the Biological Constraints tutorial
train_params['fixed_weights'] = None # Dictionary of weights to fix (not allow to train). Default: None
# Example usage of the optional performance_cutoff and performance_measure parameters is available in Curriculum Learning tutorial.
def performance_measure(trial_batch, trial_y, output_mask, output, epoch, losses, verbosity):
    return performance_measure_for_RNN(trial_batch, trial_y, output_mask, output, epoch, losses, verbosity,network_params)

train_params['curriculum'] = None
train_params['performance_measure'] = performance_measure
train_params['performance_cutoff'] = .99

## -------- Training loop ##
ensembleSize=40
for netii in range(ensembleSize):
    startTime = datetime.datetime.now().strftime('%Y%m%d-%H-%m')

    print(startTime, netii)
    
    model = Basic(network_params)
    initialWeight = model.get_weights()

    losses, trainTime, initialTime= model.train(dd, train_params)

    # ---------------------- Test the trained model --------------------------- 
    x,target_output,mask, trial_params = dd.get_trial_batch() # get pd task inputs and outputs
    model_output, model_state = model.test(x) # run the model on input x


    #weights = model.get_weights()
    saveTime = datetime.datetime.now().strftime('%Y%m%d-%H-%m')
    Fail = 'Fail' if losses[-1]>0.01 else ''

    dirName = taskTrainName+'/' +(taskTrainName+'_'+saveTime+'_'+str(netii)+'_'+Fail)+'/'

    dirPath = saveRoot+dirName
    os.makedirs(dirPath) 

    model.save(dirPath+'weightFinal')
    np.savez(dirPath+'weightInit',weightInit=initialWeight)
    np.savez(dirPath+'activitityTest',x=x,trial_params=trial_params,model_output=model_output,model_state=model_state)
    np.savez(dirPath+'trainingHistory',losses=losses, trainTime=trainTime, initialTime=initialTime, startTime=startTime,saveTime=saveTime)
    np.savez(dirPath+'network_params',network_params=network_params)


    x,target_output,mask, trial_params = dd_test    .get_trial_batch() # get pd task inputs and outputs
    simulator = BasicSimulator(weights_path=os.path.abspath(dirPath+'weightFinal.npz'),
                           params = {'dt': dt, 'tau': tau})
    model_output, model_state = simulator.run_trials(x) # run the model on input x
    np.savez(os.path.join(dirPath,'activitityTestGrid.npz'),x=x,trial_params=trial_params,model_output=model_output,model_state=model_state,mask=mask)

    model.destruct()
    print(dirPath)
    print(os.listdir(dirPath))