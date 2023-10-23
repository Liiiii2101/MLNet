import os
os.environ['DTORROOT'] = '/DATA/forLishan/randomddd'
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from dtor.utilities.utils_stats import stats_from_results,roc_and_auc
from sklearn.metrics import roc_curve, auc
import numpy as np
import argparse
import torch
from dtor.utilities.utils import set_plt_config
from dtor.utilities.model_retriever import load_model
from dtor.utilities.utils import safe_restore
set_plt_config()
import os
import torch.nn as nn
from dtor.utilities.data_retriever import get_data ##add new
from train import RTRTrainer as  Trainer
from dataloader_val import MRIDataset 
import sys
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import label_binarize
import pandas as pd
from sklearn.metrics import accuracy_score
from network.resnet_stage import generate_model
from medcam import medcam
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
from collections import OrderedDict
import shap
## this is for  testing external validation from 3 different centers
#from attention import  net2

#num_dict=OrderedDict({'435':25,'412':24,'474':20,'501':20})


parser = argparse.ArgumentParser()
parser.add_argument("--tot_folds",help="Number of folds for model training",
                     type=int,
                    default=1)
parser.add_argument("--prefix", type=str, help="Training prefix",
                    default="nnuent_pre_focal_buffer_wp_pcr_dwi_lre4_resnet10-train")
parser.add_argument("--legname", type=str, help="Legend description",
                    default='Response External Validation CNN')
args = parser.parse_args()


tot_folds = args.tot_folds
prefix = args.prefix
legname = args.legname
#%%

sys.argv.extend(["--load_json", f"/mnt/data/groups/beets-tan/l.cai/results/{prefix}/options.json"])



class Net2(nn.Module):
    def __init__(self, model):
        """
        "coat" model used to change output and input
        """
        super().__init__()
        self.model = model
        self.input = None
    def get_input(self, input):
        """
        for catching real input
        :param input:
        :return:
        """
        self.input = input
    def forward(self, inputs):
        """
        modify code to force output shape
        :param inputs: fake input , but shape of the fake_input will be used by medcam
        :param inputs: fake input but the shape will be extracted by medcam and reshape to return the attention map 
        :return:
        """
        #print(inputs.shape)
        if self.input is None:
            raise ValueError('self.input is None')

        out = self.model(self.input)
        return out

#%%
# Process folds
# Concatenate results of the folds
y_preds_total = []
y_labels_total = []
for ff in range(tot_folds):
    # Load test data
    A = Trainer()
    setattr(A.cli_args, 'datapoints', '/processing/Lishan/multi_stage/processed_97.csv')

    train_ds, val_ds, train_dl, val_dl = A.init_data(ff, mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])

    #  Make sample for loading
    sample = []
    for n, point in enumerate(val_dl):
        if n == 1:
           break
        x = point[0]
        sample.append(x)
   # sample = torch.cat(sample, dim=0)          

    use_cuda = torch.cuda.is_available()   #add new
    device = torch.device("cuda" if use_cuda else "cpu")

    _n = prefix.split('-')[0]
    for imodel in range(1):
        print(f"test_model_from_fold_{imodel}")
        full_name=os.path.join(f"/mnt/data/groups/beets-tan/l.cai/results/{prefix}/",  'model-' + _n +'-fold'+ str(imodel)+'-epochzloss' +'.pth')  #str(prefix)
      
    
        #Get Model for fold
        model = generate_model(10)
        model = safe_restore(model, full_name)
       
        ##multi input and output attention maps
        model = Net2(model)

        model = model.to(device)    
        model_c = medcam.inject(model, output_dir="attention_maps_multin", save_maps=True, backend="gbp", return_attention=True, label='best', layer='auto')

        model_c.eval()
        name_list = [name for name,_ in model.named_parameters()]
        # Generate vector of predictions and true labels
        y_preds = dict()
        pp = []
        ll = []
        for n in range(len(val_ds)):
            f, truth, extra = val_ds[n]

            x = f['image']
            x = x.unsqueeze(0)
            x1 = f['out1']
            x1 = x1.unsqueeze(0)
            x2 = f['out2']
            x2 = x2.unsqueeze(0)
            x3 = f['out3']
            x3 = x3.unsqueeze(0)
            x4 = f['out4']
            x4 = x4.unsqueeze(0)
            x = x.to(device)
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            x4 = x4.to(device)


            input = [x,x1,x2,x3,x4]
            model_c.get_input(input)
            l = model_c(x)
            p = nn.Softmax(dim=1)(l[0])

            return_layers = {
        
            'layer1.0': 'layer1',
            'layer2.0': 'layer2'                
            }


            attention = l[1].cpu().detach().numpy()
            np.save(f'saving/att{str(n)}.npy',attention)
            
          


        
            pred = p[0].detach().cpu().numpy().tolist()
            y_preds_total.append(pred)
            y_labels_total.append(truth)
            pp.append(p[0][1].detach().cpu())
            ll.append(truth)
        arr = np.asarray(pp)
        for i in num_dict.keys():
            i = int(i)
            #print(i, arr[i-412],ll[i-412])
        #print(dict(zip(arr,ll)))
        arr[arr>0.5] = 1
        arr[arr<0.5] = 0
       
        #print(np.sum(ll),len(ll))
        print('AUC: ',roc_and_auc(np.asarray(pp),np.asarray(ll)),accuracy_score(arr,np.asarray(ll)))

    y_labels_total = np.array(label_binarize(y_labels_total,classes=[0,1,2]))[:,:2]
    y_preds_total = np.array(y_preds_total)

    #f1_p = y_preds_total[:97,1]
    #f2_p = y_preds_total[97:194,1]
    #f3_p = y_preds_total[194:291,1]
    #f4_p = y_preds_total[291:388,1]
    #print(f_em)
    #f_em = [(f1_p[i]+f2_p[i]+f3_p[i]+f4_p[i])/4.0 for i in range(97)]
    #print(f_em)
    #f_em = np.asarray(f_em)
    #print('Em AUC',roc_and_auc(f_em,np.asarray(ll)))
    
    #f_em[f_em>0.5] = 1
    #f_em[f_em <0.5] = 0
    #print(accuracy_score(f_em,np.asarray(ll)))
    #y_labels_total = np.reshape(y_labels_total,(191,1))
    #y_preds_total = np.reshape(y_preds_total,(191,1))
    #print(y_preds_total.shape)
    #print(y_labels_total)

    lw = 2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):

        fpr[i], tpr[i], _ = roc_curve(y_labels_total[:,i], y_preds_total[:,i])

        roc_auc[i] = auc(fpr[i], tpr[i])
        #print(roc_auc)
    
    fpr["micro"], tpr["micro"], _ = roc_curve(y_labels_total.ravel(), y_preds_total.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])   
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange'])
    for i, color in zip(range(2), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('One center test for RTR')
    plt.legend(loc="lower right")
    #plt.savefig('results.png')
    plt.show()
    
    

    print(roc_and_auc(y_preds_total[:,1],y_labels_total[:,1]))