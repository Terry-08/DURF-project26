import imageio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.linear_model import LogisticRegression
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import os


dirName = 'orderTaskDefault'
outputMode = 'order'

def main():
    root = "./savedForHPC/"+dirName
    for name in os.listdir(root):
        dataDir = os.path.join(root,name)
        if os.path.isdir(dataDir):
            generateGif(dataDir,outputMode)

def update_scatter(scatterObj,x_new, y_new):
    scatterObj.set_offsets(np.c_[x_new, y_new])
def regreessBehavior(choiceB,qAs,qBs,seqAB=None):
    idx=(qAs!=0)&(qBs!=0)
    if seqAB is not None:
        X = np.vstack((np.log(qBs[idx]/qAs[idx]),seqAB[idx])).T
    else:
        X = np.log(qBs[idx]/qAs[idx]).reshape(-1,1)
    y = choiceB[idx]
    model = LogisticRegression()
    model.fit(X,y)
    return model

def importAndPreprocess(dirPath,activityFileName):
    import os
    import sys

    
    with np.load(os.path.join(dirPath,activityFileName),allow_pickle=True) as f:
        x = f['x']
        trial_params = f['trial_params']
        model_output = f['model_output']
        model_state = f['model_state']
        mask = f.get('mask', None)
    
    if mask is None:
        temp = np.mean(model_output[:,300:,:],1)
    else: 
        temp = np.mean(mask * model_output,1)
    choiceLR = temp[:,1]>temp[:,0]
    choiceLR = choiceLR*2-1 # pos right high, neg left high
    
    choiceFrame = [trial_params[i]['choiceFrame'] for i in range(len(trial_params))]
    
    
    locAB = [(1 if trial_params[i]['locAB']=='AB' or trial_params[i]['locAB']=='12' else -1) for i in range(len(trial_params))]
    loc12 = locAB
    seqAB = [(1 if trial_params[i]['seqAB']=='AB' else -1) for i in range(len(trial_params))]
    
    choiceAB = np.array([(choiceLR[i] * locAB[i] if choiceFrame[i]=='juice' else choiceLR[i] * locAB[i] * seqAB[i]) for i in range(len(trial_params)) ])  # pos B neg A
    choice12 = np.array([(choiceAB[i] * seqAB[i] if choiceFrame[i]=='juice' else choiceLR[i] * locAB[i]) for i in range(len(trial_params)) ])  # pos 2 neg 1
    choiceAB = np.array(['B' if choiceAB[i]>0 else 'A' for i in range(len(trial_params))])
    choice12 = np.array(['2' if choice12[i]>0 else '1' for i in range(len(trial_params))])
    
    qAs = np.array([trial_params[i]['qA'] for i in range(len(trial_params))])
    qBs = np.array([trial_params[i]['qB'] for i in range(len(trial_params))])
    seqAB = np.array([trial_params[i]['seqAB']for i in range(len(trial_params))])

    return x,trial_params,model_state,choice12,choiceAB,choiceLR,qAs,qBs,seqAB

def fix_padding(frames1,frames2):
    shape1=frames1[0].shape
    shape2=frames2[0].shape
    shapeMax = tuple(max(a, b) for a, b in zip(shape1, shape2))
    print(shape1,shape2,shapeMax)
    pad1 = tuple(a-b for a, b in zip(shapeMax, shape1))
    pad1 = tuple((0,pad1[i]) for i in range(len(pad1)))
    pad2 = tuple(a-b for a, b in zip(shapeMax, shape2))
    pad2 = tuple((0,pad2[i]) for i in range(len(pad2)))
    print(pad1,pad2)
    for i in range(len(frames1)):
        frames1[i] = np.pad(frames1[i],pad1,'constant',constant_values=255)

    for i in range(len(frames2)):
        frames2[i] = np.pad(frames2[i],pad2,'constant',constant_values=255)

    return frames1,frames2

def getPCA(model_state,xx,yy):
    K,T,N = model_state.shape

    pcaObj = PCA(n_components=4)
    X = model_state[:,50:250,:].reshape((K*200,N))
    pcaObj.fit(X)
    points = pcaObj.transform(X)

    t1=150
    xx,yy = (0,1)

    xmin = np.min(points[:,xx])
    xmax = np.max(points[:,xx])
    ymin = np.min(points[:,yy])
    ymax = np.max(points[:,yy])
    range_x = xmax-xmin
    range_y = ymax-ymin
    padding_factor =0.1
    xlim = (xmin-range_x*padding_factor, xmax+range_x*padding_factor)
    ylim = (ymin-range_y*padding_factor, ymax+range_y*padding_factor)
    
    return pcaObj,xlim,ylim,range_x,range_y

def generateVectorField(weightFile,pcaObj,xlim,ylim):
    (xmin,xmax),(ymin,ymax) = xlim,ylim
    with np.load(weightFile,allow_pickle=True) as f:
        weights = f
        W_rec = weights['W_rec']
        W_in = weights['W_in']
        b_rec = weights['b_rec']
    relu = lambda x: x*(x>0)
    tau=100
    def F(x,x_in=np.zeros(W_in.shape[1])):
        x = x.T
        M = x.shape[1]
        leaky = -x
        recurrent = np.matmul(W_rec,relu(x)) + np.tile(b_rec.reshape(-1,1),(1,M))
        input = np.matmul(W_in,(x_in))
        input = np.tile(input.reshape(-1,1),(1,M))
        
        der= (leaky+recurrent+input)/tau
        return der.T


    UU = pcaObj.components_[0:2,:]
    PP = UU.T @ UU

    v1 = pcaObj.components_[0,:]
    v2 = pcaObj.components_[1,:]
    v0 = pcaObj.mean_

    N_grid = 24
    xv,yv = np.meshgrid(np.arange(xmin,xmax,2),np.arange(ymin,ymax,2))
    state_grid = np.outer(xv.reshape(-1),v1) + np.outer(yv.reshape(-1),v2) +v0

    vec_grid_noInput = F(state_grid,np.array([0,0,0,0,0,0,0,0,1]))
    vec_grid_noInput_project = vec_grid_noInput @ UU.T
    vec_grid_noInput_project = vec_grid_noInput_project.reshape((xv.shape[0],xv.shape[1],2))

    xpc = (v1@ UU.T)[0] * xv + (v2@ UU.T)[0] * yv
    ypc = (v1@ UU.T)[1] * xv + (v2@ UU.T)[1] * yv

    return xpc,ypc,vec_grid_noInput_project


def generateSnapShot_Encoding(dirPath,activityFilename,outputMode,gif_name,figsize):
    image_files=[]
    frames = []

    x,trial_params,model_state,choice12,choiceAB,choiceLR,qAs,qBs,seqAB = importAndPreprocess(dirPath,activityFilename)
    weightFile = os.path.join(dirPath,'weightFinal.npz')

    xx,yy=0,1
    pcaObj,xlim,ylim,range_x,range_y = getPCA(model_state,xx,yy)
    xpc,ypc,vec_grid_noInput_project = generateVectorField(weightFile,pcaObj,xlim,ylim)

    fig,ax = plt.subplots(figsize=figsize,dpi=150)        

    t1=150
    points = pcaObj.transform(np.squeeze(model_state[:,t1,:]))

    ax.quiver(xpc,ypc,vec_grid_noInput_project[:,:,0],vec_grid_noInput_project[:,:,1],label='__no_label_',color='grey')
    hAB=ax.scatter(points[seqAB=='AB',xx],points[seqAB=='AB',yy],marker='.',
            c = qAs[seqAB=='AB'],cmap='Oranges')
    hBA=ax.scatter(points[seqAB=='BA',xx],points[seqAB=='BA',yy],marker='.',label='offer1 is B',
            c = qBs[seqAB=='BA'],cmap='Blues')

    proxyA, = ax.plot([],[],marker='.',color='tab:orange',linestyle='None',label='offer1 is A')
    proxyB, = ax.plot([],[],marker='.',color='tab:blue',linestyle='None',label='offer1 is B')
    ax.set_xlabel('PC%d'%(xx+1))
    ax.set_ylabel('PC%d'%(yy+1))
    ax.set_aspect('equal','box')


    ax.set(xlim=xlim,ylim=ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    hTxt=ax.text(xlim[0]+range_x*0.01,ylim[0]+range_y*0.01,"t=%dms"%(t1*10))
    ax.legend(handles=[proxyA,proxyB],bbox_to_anchor=(1.04, 1), loc="upper left")
    # if outputMode == 'order':
    #     ax.legend(handles=[proxyA,proxyB],bbox_to_anchor=(1.04, 1), loc="upper left")
    # else:
    #     ax.legend(handles=[proxyA,proxyB],loc='lower right')

    ts = np.arange(50,155,5)
    num_frames = len(ts)



    # Generate and save plots
    for i in range(num_frames):

        # update data
        tt = ts[i]
        points = pcaObj.transform(np.squeeze(model_state[:,tt,:]))
        update_scatter(hAB,points[seqAB=='AB',xx],points[seqAB=='AB',yy])
        update_scatter(hBA,points[seqAB=='BA',xx],points[seqAB=='BA',yy])
        hTxt.set_text("t=%dms"%(tt*10))
        fig.canvas.draw_idle()


        # Save the figure
        filename = os.path.join(dirPath,f"gif/temp/{gif_name}_Encoding_frame_{i:03d}.png")
        plt.savefig(filename, bbox_inches='tight')
        # plt.close(fig)
        
        # Append the filename to the list
        image_files.append(filename)
        frames.append(imageio.imread(filename))
    
    plt.close(fig)        
    return image_files,frames




def generateSnapShot_Choice(dirPath,activityFilename,outputMode,gif_name,figsize):
    image_files=[]
    frames = []

    x,trial_params,model_state,choice12,choiceAB,choiceLR,qAs,qBs,seqAB = importAndPreprocess(dirPath,activityFilename)
    weightFile = os.path.join(dirPath,'weightFinal.npz')

    xx,yy=0,1
    pcaObj,xlim,ylim,range_x,range_y = getPCA(model_state,xx,yy)
    xpc,ypc,vec_grid_noInput_project = generateVectorField(weightFile,pcaObj,xlim,ylim)

    fig,ax = plt.subplots(figsize=figsize,dpi=150)        
    ax.quiver(xpc,ypc,vec_grid_noInput_project[:,:,0],vec_grid_noInput_project[:,:,1],label='__no_label_',color='grey')

    t1=250
    points = pcaObj.transform(np.squeeze(model_state[:,t1,:]))

    choiceB = np.array([1 if choiceAB[i]=='B' else 0 for i in range(len(choiceAB))])
    seqABnum = np.array([(1 if trial_params[i]['seqAB']=='AB' else -1) for i in range(len(trial_params))])
    model = regreessBehavior(choiceB,qAs,qBs,seqABnum)
    a0,(a1,a2) = model.intercept_[0], model.coef_[0]
    ind_point=np.exp(-a0/a1)
    valueB=qBs
    valueA=qAs*ind_point
    value1 = [qAs[i]*ind_point if seqAB[i]=='AB' else qBs[i] for i in range(len(seqAB))]
    value2 = [qAs[i]*ind_point if seqAB[i]=='BA' else qBs[i] for i in range(len(seqAB))]
    value1=np.array(value1)
    value2=np.array(value2)

    valueDiff = value2-value1 if outputMode=='order' else valueA-valueB

    cmap='RdGy_r' if outputMode=='order' else 'coolwarm'
    h=ax.scatter(points[:,xx],points[:,yy],marker='.',c=valueDiff[:],cmap=cmap)


    ax.set_xlabel('PC%d'%(xx+1))
    ax.set_ylabel('PC%d'%(yy+1))
    ax.set_aspect('equal','box')


    ax.set(xlim=xlim,ylim=ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    hTxt=ax.text(xlim[0]+range_x*0.01,ylim[0]+range_y*0.01,"t=%dms"%(t1*10))
    if outputMode=='order':
        clb=plt.colorbar(h,label='value2 - value1')
        clb.set_ticks([-5,0,5])
        clb.set_ticklabels(['choose 1', 'indifferent','choose 2'])
    else:
        clb=plt.colorbar(h,label='valueA - valueB')
        clb.set_ticks([-5,0,5])
        clb.set_ticklabels(['choose B', 'indifferent','choose A'])

    ts = np.arange(150,255,5)
    num_frames = len(ts)
    image_files=[]
    frames = []

    def update_scatter(scatterObj,x_new, y_new):
        scatterObj.set_offsets(np.c_[x_new, y_new])

    # Generate and save plots
    for i in range(num_frames):

        # update data
        tt = ts[i]
        points = pcaObj.transform(np.squeeze(model_state[:,tt,:]))
        update_scatter(h,points[:,xx],points[:,yy])
        hTxt.set_text("t=%dms"%(tt*10))
        fig.canvas.draw_idle()


        # Save the figure
        filename = os.path.join(dirPath,f"gif/temp/{gif_name}_frame_{i:03d}.png")
        plt.savefig(filename, bbox_inches='tight')
        # plt.close(fig)
        
        # Append the filename to the list
        image_files.append(filename)
        frames.append(imageio.imread(filename))
    
    plt.close(fig)        
    return image_files,frames

def generateGif_From(image_and_frame,gif_path,nPause=3,delete=False):
    image_files,frames = image_and_frame
    
    # pause at last frame
    nPause=nPause
    for iDelay in range(nPause): 
        frames.append(imageio.imread(image_files[-1]))

    # Create the GIF
    imageio.mimsave(gif_path, frames,loop=0, fps=3)

    
    # Optionally, remove the image files
    if delete:
        delete_files(image_files)

    print(f"GIF saved as {gif_path}")

def delete_files(image_files):
    import os
    for filename in image_files:
        os.remove(filename)    

def generateGif(dirPath,outputMode):
    activityFileName = 'activitityTestGrid.npz'
    figsize = (8,3)

    os.makedirs(os.path.join(dirPath,'gif','temp'),exist_ok=True)
    
    gif_name_encoding = 'gifEncoding'
    gif_path_encoding = os.path.join(dirPath,'gif',gif_name_encoding+'.gif')
    image_files_encode,frames_encode = generateSnapShot_Encoding(dirPath,activityFileName,outputMode,gif_name_encoding,figsize)
    generateGif_From((image_files_encode,frames_encode),gif_path_encoding,3)
    gif_name_choice = 'gifChoice'
    gif_path_choice = os.path.join(dirPath,'gif',gif_name_choice+'.gif')
    image_files_choice,frames_choice = generateSnapShot_Choice(dirPath,activityFileName,outputMode,gif_name_choice,figsize)
    generateGif_From((image_files_choice,frames_choice),gif_path_choice,3) 

    gif_path_full = os.path.join(dirPath,'gif','gifFull.gif')
    frames_choice,frames_encode = fix_padding(frames_choice,frames_encode)
    nPause_1, nPause_2,nPause_3 = 3,2,4
    image_files_full = image_files_encode + [image_files_encode[-1]]*nPause_1 + [image_files_choice[0]]*nPause_2 + image_files_choice + [image_files_choice[-1]]*nPause_3
    frames_full = frames_encode + [frames_encode[-1]]*nPause_1 + [frames_choice[0]]*nPause_2 + frames_choice + [frames_choice[-1]]*nPause_3
    generateGif_From((image_files_full,frames_full),gif_path_full,nPause=0)
    
if __name__ == '__main__':
    main()