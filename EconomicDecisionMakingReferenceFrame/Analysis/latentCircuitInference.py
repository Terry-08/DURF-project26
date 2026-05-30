import torch
import numpy as np
import os
from torch import nn
import torch.nn.utils.parametrize as parametrize
from torch.nn.utils.parametrizations import orthogonal    
from pathlib import Path
import sys

dirName = sys.argv[1]
n = int(sys.argv[2])
dtype = torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
torch.set_default_dtype(dtype)

def main():
    root = "./savedForHPC/"+dirName
    for name in os.listdir(root):
        dataDir = os.path.join(root,name)
        if os.path.isdir(dataDir):
            if Path(os.path.join(dataDir,f'lowDimFit{n}.pth')).exists():
                continue
            print(dataDir)
            
            # ---- load data and copy to GPU
            dataPath = os.path.join(dataDir,'activitityTest.npz')
            with np.load(dataPath) as dataNumpy:
                xNumpy = dataNumpy['x'][:,0:250,0:2]
                model_output_Numpy = dataNumpy['model_output']
                model_state_Numpy = dataNumpy['model_state'][:,0:250,:]
            N_batch,N_timeStep,N_in = xNumpy.shape
            xTorch = torch.tensor(xNumpy)
            N_out = model_output_Numpy.shape[-1]
            model_output_Torch = torch.tensor(model_output_Numpy)
            N_node = model_state_Numpy.shape[-1]
            model_state_Torch = torch.tensor(model_state_Numpy)
            N = N_node

            with np.load(dataPath, allow_pickle=True) as dataNumpy:
                trial_params = dataNumpy['trial_params']
            trial_params[0].keys()

            # ---- construct model
            model = Engel2022Fit(N_node,n,N_in)

            # --- Construct our loss function and an Optimizer. 
            criterion = torch.nn.MSELoss(reduction='sum')
            optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=0.001)

            # -- training loop
            loss_accum = []
            u = xTorch.to(dtype)
            y = model_state_Torch.to(dtype)

            try:
                for t in range(40000):
                    # Forward pass: Compute predicted y by passing x to the model
                    y_pred = model(u)

                    # Compute and print loss
                    loss = criterion(y_pred, y)
                    if (t) % 1000 == 0:
                        print(t, loss.item())
                    
                    loss_accum.append(loss.item())

                    if (t>1000):
                        if np.mean(loss_accum[-100:])<4e5:
                            if np.mean(loss_accum[-50:-25]) - np.mean(loss_accum[-25:]) <10:
                                print('stop criteria met')
                                print(t, loss.item())
                                break
                    
                    # Zero gradients, perform a backward pass, and update the weights.
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # --- save results
                fitSavePath = os.path.join(dataDir,f'lowDimFit{n}.pth')
                print(fitSavePath)
                torch.save(model.state_dict(), fitSavePath)
            except torch._C._LinAlgError:
                continue

class Engel2022Fit(nn.Module):
    def __init__(self,N,n,N_in):
        """
        In the constructor we instantiate parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.matB = nn.Parameter(torch.rand(N,N))
        self.wIn = nn.Parameter(torch.eye(n,N_in))#nn.Parameter(torch.tensor(wIn))#nn.Parameter(torch.eye(n,N_in))
        self.wRec = nn.Parameter(torch.zeros(n,n))#nn.Parameter(torch.tensor(wRec))#nn.Parameter(torch.zeros(n,n))
        
        self.leakyRNN = leakyRNN(N_in,n, nonlin=nn.ReLU(),bias=False,init_state_train=False)
        self.leakyRNN.wRec.weight = self.wRec
        self.leakyRNN.wIn.weight = self.wIn

        
        # self.embed = nn.Linear(in_features=n,out_features=N,bias=False)
        # orthogonal(self.embed)#,orthogonal_map='cayley')
        # self.embed.parametrizations.weight.original = self.matB

        # self.embed.weight = self.matB
        
        self.padding = nn.ZeroPad1d((0,N-n))
        
        self.embed = nn.Linear(N,N,bias=False)
        parametrize.register_parametrization(self.embed,"weight",Orthonormal())
        self.embed.parametrizations.weight.original = self.matB
     
        
    def forward(self, u):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x= self.leakyRNN(u)
        x_padded = self.padding(x)
        y = self.embed(x_padded)
        # y = self.embed(x)
        
        return y
        
class Orthonormal(nn.Module):
    def forward(self,B):
        N = B.shape[0]
        A = B - B.T # skew-symmetric
        Q = torch.linalg.solve ( (torch.eye(N) + A), (torch.eye(N) - A), left=False)
        return Q

class leakyRNN(nn.Module):
    def __init__(self,n_in,n_hidden,
                         dtype=dtype,
                         alpha=0.1,nonlin=nn.Softplus(),#nn.ReLU()
                         bias=True,init_state_train=True):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.dtype = dtype
        self.alpha = alpha # dt/tau, default is .1 (10ms/100ms)
        self.non_linearity = nonlin

        self.wRec = nn.Linear(n_hidden, n_hidden, bias=False)
        self.wIn = nn.Linear(n_in, n_hidden, bias=False)
        self.device = self.wRec.weight.device

        if bias:
            self.b_rec = nn.Parameter(torch.zeros(n_hidden)).to(self.device)
        else:
            self.b_rec = torch.zeros(n_hidden,requires_grad=False).to(self.device)
        
        if init_state_train:
            self.y0 = nn.Parameter(torch.zeros(n_hidden)).to(self.device)
        else:
            self.y0 = torch.zeros(n_hidden,requires_grad=False).to(self.device)

    
    
    def recurrent_timestep(self,xt,state):
        """ Recurrent time step.

        Given input and previous state, outputs the next state of the network.
        """
        new_state = ((1-self.alpha) * state) \
                    + self.alpha * ( 
                        self.wRec(self.non_linearity(state))
                        + self.wIn(xt)
                        + self.b_rec)\
                    # + rec_noise + input_noise
        return new_state
        
    
    def forward(self,x,y0=None):    
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        device = self.device
        dtype = self.dtype
        
        if y0 is None: y0=self.y0
        y = y0.to(device,dtype=dtype)
        
        N_batch,N_timeStep, _n_in = x.shape
        hidden = torch.zeros((N_batch, N_timeStep, self.n_hidden),
                             device = device, dtype=dtype)
        
        for iTimeStep in range(N_timeStep):
            xt = x[:,iTimeStep,:].to(device,dtype=dtype)
            y = self.recurrent_timestep(xt,y)

            hidden[:,iTimeStep,:] = y

        return hidden
            



if __name__ == '__main__':
    main()
