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
    
    model_colors = ['r','g','b','c','m','y']

    #for each target parameter, plot a graph
    for p in range(hp.N_PARAMS):
        print("Plotting results for parameter: {}...".format(con.PARAM_DICT[p]))
        
        #layout
        fig = pl.figure()
        gs1 = gridspec.GridSpec(3,1)
        ax1, ax2 = fig.add_subplot(gs1[:2]), fig.add_subplot(gs1[2])
        
        #ax1 formatting
        param = con.PARAM_DICT[p]
        ax1.set_title(param)
        ax1.set_xlabel(r'')
        ax1.set_ylabel(r'Predicted {}'.format(param))
        ax1.locator_params(nbins=5)

        recs = []
        for i in range(n_models):
            recs.append(mpatches.Rectangle((0,0),0.5,0.5, fc=model_colors[i]))
        ax1.legend(recs, modellabels, fontsize=10)
                
        #ax2 formatting
        ax2.locator_params(nbins=5)
        ax2.set_ylabel(r'% error')
        ax2.set_xlabel(r'True {}'.format(param))
        
        targets, pred, err = np.array([]), np.array([]), np.array([])
        for i, name in enumerate(modelnames):
            result = np.load('{}/pred_{}.npz'.format(name, name))
            
            model_targets = np.array(result['targets'][:,p])
            model_pred = np.array(result['predictions'][:,p])
            model_err = 100.0*(1-model_pred/model_targets)
            
            ax1.scatter(model_targets, model_pred,alpha=0.7, s=1.5, c=model_colors[i])
            ax2.scatter(model_targets, model_err,alpha=0.7, s=1.5, c=model_colors[i])
            
            targets = np.append(targets,model_targets)
            pred = np.append(pred, model_pred)
            err = np.append(err, model_err)

        #find axis limits 
        mintargets, maxtargets = np.min(targets), np.max(targets)
        minpred, maxpred = np.min(pred), np.max(pred)
        minerr, maxerr = np.min(err), np.max(err)
        
        ax1.set_xlim(0.95*mintargets,1.05*maxtargets)
        ax1.set_ylim(minpred*0.95,maxpred*1.05)
        ideal = np.linspace(0.8*mintargets,1.2*maxtargets,10)
        ax1.plot(ideal, ideal, 'k--', alpha=0.3, linewidth=1.5)
        
        ax2.plot(ideal,len(ideal)*[0.],'k--', alpha=0.3, linewidth=1.5)
        ax2.set_ylim(minerr*0.95,maxerr*1.05)
        ax2.set_xlim(0.95*mintargets,1.05*maxtargets)
        
        #save
        pl.tight_layout()
        fname = param
        for name in modelnames:
            fname += "_" + name
        
        pl.savefig('{}.png'.format(fname),dpi=300)
        pl.close()
        

if __name__ == "__main__":
    plot_model_predictions([hp.MODEL_NAME])