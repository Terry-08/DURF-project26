import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import os


dirName = 'spatialTaskLateMapping'
outputMode = 'spatial'


def main():
    root = "./savedForHPC/"+dirName
    for name in os.listdir(root):
        dataDir = os.path.join(root,name)
        if os.path.isdir(dataDir):
            generateGif(dataDir,outputMode)


def update_scatter(scatterObj,x_new, y_new):
    scatterObj.set_offsets(np.c_[x_new, y_new])


def regressBehavior(choiceB,qAs,qBs,seqAB=None):
    idx=(qAs!=0)&(qBs!=0)
    if seqAB is not None:
        X = np.vstack((np.log(qBs[idx]/qAs[idx]),seqAB[idx])).T
    else:
        X = np.log(qBs[idx]/qAs[idx]).reshape(-1,1)
    y = choiceB[idx]
    model = LogisticRegression()
    model.fit(X,y)
    return model


def loc12_to_label(loc12):
    if isinstance(loc12, str):
        return loc12
    loc12 = np.asarray(loc12)
    return '12' if loc12[0] == 1 else '21'


def importAndPreprocess(dirPath,activityFileName):
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

    qAs = np.array([trial_params[i]['qA'] for i in range(len(trial_params))])
    qBs = np.array([trial_params[i]['qB'] for i in range(len(trial_params))])
    seqAB = np.array([trial_params[i]['seqAB'] for i in range(len(trial_params))])
    loc12 = np.array([np.asarray(trial_params[i]['loc12']) for i in range(len(trial_params))])
    loc12_label = np.array([
        trial_params[i].get('loc12_label', loc12_to_label(trial_params[i]['loc12']))
        for i in range(len(trial_params))
    ])
    chosen_offer = np.array([trial_params[i]['chosen_offer'] for i in range(len(trial_params))])
    chooseB = np.array([trial_params[i]['chooseB'] for i in range(len(trial_params))])

    choiceAB = np.array(['B' if chooseB[i] else 'A' for i in range(len(trial_params))])
    choice12 = np.array(['2' if chosen_offer[i] == 2 else '1' for i in range(len(trial_params))])
    offer1_side = np.array(['left' if loc12_label[i] == '12' else 'right' for i in range(len(trial_params))])

    return x,trial_params,model_state,choice12,choiceAB,choiceLR,qAs,qBs,seqAB,loc12,loc12_label,chosen_offer,offer1_side


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

    v1 = pcaObj.components_[0,:]
    v2 = pcaObj.components_[1,:]
    v0 = pcaObj.mean_

    xv,yv = np.meshgrid(np.arange(xmin,xmax,2),np.arange(ymin,ymax,2))
    state_grid = np.outer(xv.reshape(-1),v1) + np.outer(yv.reshape(-1),v2) +v0

    fixation_input = np.zeros(W_in.shape[1])
    fixation_input[-1] = 1
    vec_grid_noInput = F(state_grid,fixation_input)
    vec_grid_noInput_project = vec_grid_noInput @ UU.T
    vec_grid_noInput_project = vec_grid_noInput_project.reshape((xv.shape[0],xv.shape[1],2))

    xpc = (v1@ UU.T)[0] * xv + (v2@ UU.T)[0] * yv
    ypc = (v1@ UU.T)[1] * xv + (v2@ UU.T)[1] * yv

    return xpc,ypc,vec_grid_noInput_project


def generateSnapShot_Encoding(dirPath,activityFilename,outputMode,gif_name,figsize):
    image_files=[]
    frames = []

    x,trial_params,model_state,choice12,choiceAB,choiceLR,qAs,qBs,seqAB,loc12,loc12_label,chosen_offer,offer1_side = importAndPreprocess(dirPath,activityFilename)
    weightFile = os.path.join(dirPath,'weightFinal.npz')

    xx,yy=0,1
    pcaObj,xlim,ylim,range_x,range_y = getPCA(model_state,xx,yy)
    xpc,ypc,vec_grid_noInput_project = generateVectorField(weightFile,pcaObj,xlim,ylim)

    fig,ax = plt.subplots(figsize=figsize,dpi=150)

    t1=150
    points = pcaObj.transform(np.squeeze(model_state[:,t1,:]))

    offer1_value = np.array([qAs[i] if seqAB[i]=='AB' else qBs[i] for i in range(len(seqAB))])
    idx_left = offer1_side == 'left'
    idx_right = offer1_side == 'right'

    ax.quiver(xpc,ypc,vec_grid_noInput_project[:,:,0],vec_grid_noInput_project[:,:,1],label='__no_label_',color='grey')
    hLeft=ax.scatter(points[idx_left,xx],points[idx_left,yy],marker='.',
            c = offer1_value[idx_left],cmap='viridis')
    hRight=ax.scatter(points[idx_right,xx],points[idx_right,yy],marker='x',
            c = offer1_value[idx_right],cmap='viridis')

    proxyLeft, = ax.plot([],[],marker='.',color='tab:green',linestyle='None',label='offer1 left')
    proxyRight, = ax.plot([],[],marker='x',color='tab:green',linestyle='None',label='offer1 right')
    ax.set_xlabel('PC%d'%(xx+1))
    ax.set_ylabel('PC%d'%(yy+1))
    ax.set_aspect('equal','box')


    ax.set(xlim=xlim,ylim=ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    hTxt=ax.text(xlim[0]+range_x*0.01,ylim[0]+range_y*0.01,"t=%dms"%(t1*10))
    ax.legend(handles=[proxyLeft,proxyRight],bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.colorbar(hLeft,ax=ax,label='offer1 value')

    ts = np.arange(50,155,5)
    num_frames = len(ts)



    # Generate and save plots
    for i in range(num_frames):

        # update data
        tt = ts[i]
        points = pcaObj.transform(np.squeeze(model_state[:,tt,:]))
        update_scatter(hLeft,points[idx_left,xx],points[idx_left,yy])
        update_scatter(hRight,points[idx_right,xx],points[idx_right,yy])
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

    x,trial_params,model_state,choice12,choiceAB,choiceLR,qAs,qBs,seqAB,loc12,loc12_label,chosen_offer,offer1_side = importAndPreprocess(dirPath,activityFilename)
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
    model = regressBehavior(choiceB,qAs,qBs,seqABnum)
    a0,(a1,a2) = model.intercept_[0], model.coef_[0]
    ind_point=np.exp(-a0/a1)
    value1 = [qAs[i]*ind_point if seqAB[i]=='AB' else qBs[i] for i in range(len(seqAB))]
    value2 = [qAs[i]*ind_point if seqAB[i]=='BA' else qBs[i] for i in range(len(seqAB))]
    value1=np.array(value1)
    value2=np.array(value2)

    valueDiff = value2-value1

    h=ax.scatter(points[:,xx],points[:,yy],marker='.',c=valueDiff[:],cmap='RdGy_r')


    ax.set_xlabel('PC%d'%(xx+1))
    ax.set_ylabel('PC%d'%(yy+1))
    ax.set_aspect('equal','box')


    ax.set(xlim=xlim,ylim=ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    hTxt=ax.text(xlim[0]+range_x*0.01,ylim[0]+range_y*0.01,"t=%dms"%(t1*10))
    clb=plt.colorbar(h,label='value2 - value1')
    clb.set_ticks([-5,0,5])
    clb.set_ticklabels(['choose 1', 'indifferent','choose 2'])

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


def generateSnapShot_Action(dirPath,activityFilename,outputMode,gif_name,figsize):
    image_files=[]
    frames = []

    x,trial_params,model_state,choice12,choiceAB,choiceLR,qAs,qBs,seqAB,loc12,loc12_label,chosen_offer,offer1_side = importAndPreprocess(dirPath,activityFilename)
    weightFile = os.path.join(dirPath,'weightFinal.npz')

    xx,yy=0,1
    pcaObj,xlim,ylim,range_x,range_y = getPCA(model_state,xx,yy)
    xpc,ypc,vec_grid_noInput_project = generateVectorField(weightFile,pcaObj,xlim,ylim)

    fig,ax = plt.subplots(figsize=figsize,dpi=150)
    ax.quiver(xpc,ypc,vec_grid_noInput_project[:,:,0],vec_grid_noInput_project[:,:,1],label='__no_label_',color='grey')

    t1=250
    points = pcaObj.transform(np.squeeze(model_state[:,t1,:]))

    idx_offer1 = chosen_offer == 1
    idx_offer2 = chosen_offer == 2

    hOffer1=ax.scatter(points[idx_offer1,xx],points[idx_offer1,yy],marker='.',
            c=choiceLR[idx_offer1],cmap='coolwarm',vmin=-1,vmax=1)
    hOffer2=ax.scatter(points[idx_offer2,xx],points[idx_offer2,yy],marker='x',
            c=choiceLR[idx_offer2],cmap='coolwarm',vmin=-1,vmax=1)

    proxyOffer1, = ax.plot([],[],marker='.',color='tab:gray',linestyle='None',label='chosen offer 1')
    proxyOffer2, = ax.plot([],[],marker='x',color='tab:gray',linestyle='None',label='chosen offer 2')
    ax.set_xlabel('PC%d'%(xx+1))
    ax.set_ylabel('PC%d'%(yy+1))
    ax.set_aspect('equal','box')

    ax.set(xlim=xlim,ylim=ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    hTxt=ax.text(xlim[0]+range_x*0.01,ylim[0]+range_y*0.01,"t=%dms target"%(t1*10))
    ax.legend(handles=[proxyOffer1,proxyOffer2],bbox_to_anchor=(1.04, 1), loc="upper left")
    clb=plt.colorbar(hOffer1,ax=ax,label='chosen side')
    clb.set_ticks([-1,1])
    clb.set_ticklabels(['choose left','choose right'])

    ts = np.arange(250,325,5)
    num_frames = len(ts)

    for i in range(num_frames):
        tt = ts[i]
        points = pcaObj.transform(np.squeeze(model_state[:,tt,:]))
        update_scatter(hOffer1,points[idx_offer1,xx],points[idx_offer1,yy])
        update_scatter(hOffer2,points[idx_offer2,xx],points[idx_offer2,yy])
        phase = 'target' if tt < 300 else 'response'
        hTxt.set_text("t=%dms %s"%(tt*10,phase))
        fig.canvas.draw_idle()

        filename = os.path.join(dirPath,f"gif/temp/{gif_name}_frame_{i:03d}.png")
        plt.savefig(filename, bbox_inches='tight')

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
    gif_name_action = 'gifAction'
    gif_path_action = os.path.join(dirPath,'gif',gif_name_action+'.gif')
    image_files_action,frames_action = generateSnapShot_Action(dirPath,activityFileName,outputMode,gif_name_action,figsize)
    generateGif_From((image_files_action,frames_action),gif_path_action,3)

    gif_path_full = os.path.join(dirPath,'gif','gifFull.gif')
    frames_choice,frames_encode = fix_padding(frames_choice,frames_encode)
    frames_action,frames_encode = fix_padding(frames_action,frames_encode)
    frames_action,frames_choice = fix_padding(frames_action,frames_choice)
    nPause_1, nPause_2,nPause_3,nPause_4,nPause_5 = 3,2,3,2,4
    image_files_full = image_files_encode + [image_files_encode[-1]]*nPause_1 + [image_files_choice[0]]*nPause_2 + image_files_choice + [image_files_choice[-1]]*nPause_3 + [image_files_action[0]]*nPause_4 + image_files_action + [image_files_action[-1]]*nPause_5
    frames_full = frames_encode + [frames_encode[-1]]*nPause_1 + [frames_choice[0]]*nPause_2 + frames_choice + [frames_choice[-1]]*nPause_3 + [frames_action[0]]*nPause_4 + frames_action + [frames_action[-1]]*nPause_5
    generateGif_From((image_files_full,frames_full),gif_path_full,nPause=0)


if __name__ == '__main__':
    main()
