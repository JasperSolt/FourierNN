import matplotlib
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
matplotlib.use('AGG')
import numpy as np
import os
from hyperparams import Model_Hyperparameters as hp, Constant as con

def plot_model_predictions(modelnames=["test", "test2"], modellabels=None):
    npz_f = hp.MODEL_PATH + "/pred_" + hp.MODEL_NAME

    #puts results in their own directory
    if not os.path.isdir(hp.MODEL_PATH):
        os.mkdir(hp.MODEL_PATH)
        
    if not modellabels:
        modellabels = modelnames
    n_models = len(modelnames)

    #for each target parameter, plot a graph
    for p in range(hp.N_PARAMS):
        targets, pred = [],[]
        for name in modelnames:
            result = np.load('{}/pred_{}.npz'.format(name, name))
            print(result['targets'])
            print(result['targets'][p])
            targets.append(result['targets'])
            pred.append(result['predictions'])
        targets, pred = np.array(targets), np.array(pred)
        '''
        mintargets, maxtargets = np.min(alltargets), np.max(alltargets)
        minpred = np.min(allpred)
        maxpred = np.max(allpred)
        mindifpred = np.min(alldiff)
        maxdifpred = np.max(alldiff)
        
        fig = pl.figure()
        gs1 = gridspec.GridSpec(3,1)
        ax1, ax2 = fig.add_subplot(gs1[:2]), fig.add_subplot(gs1[2])
        
        #ax1
        for m in range(n_models):
            ax1.scatter(mtargets[m],mpred[m],alpha=0.7,s=1/4)
        #ax1.set_xlim(0.95*mintargets,1.05*maxtargets)
        #ax1.set_ylim(minpred*0.95,maxpred*1.05)
        ax1.set_title(param)
        ideal = np.linspace(0.8*mintargets,1.2*maxtargets,10)
        ax1.plot(ideal,ideal,'r--')

        ax1.set_xlabel(r'')
        ax1.locator_params(nbins=5)
        ax1.set_ylabel(r'Predicted {}'.format(param))

        recs = []
        for i in range(0,len(model_colors)):
            recs.append(mpatches.Rectangle((0,0),0.5,0.5,fc=model_colors[i]))
        ax1.legend(recs,modellabels,fontsize=10)

        #ax2
        for j in range(len(modelnames)):
            ax2.scatter(mtargets[j],100.*(1-np.array(mpred[j])/np.array(mtargets[j])),c=model_colors[j],s=(1.3),alpha=0.7)
        ax2.plot(ideal,len(ideal)*[0.],'r--')

        ax2.set_ylim(mindifpred*0.95,maxdifpred*1.05)
        ax2.locator_params(nbins=5)
        ax2.set_xlim(0.95*mintargets,1.05*maxtargets)
        ax2.set_ylabel(r'\% error')
        ax2.set_xlabel(r'True {}'.format(param))
        #save
        pl.tight_layout()
        fname = param
        for name in modelnames:
            fname = fname + "_" + name
        
        pl.savefig('{}.png'.format("blah_test"),dpi=300)
        pl.close()
        '''

if __name__ == "__main__":
    plot_model_predictions()