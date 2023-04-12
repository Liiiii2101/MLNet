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
import pickle
import torch.nn as nn
from dtor.utilities.data_retriever import get_data ##add new
#from scripts.train import RTRTrainer as Trainer
from trainer import Trainer as Trainer
from dataloader_m import MRIDataset 
import sys
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import label_binarize
import pandas as pd
from network.resnet_stage import generate_model
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
import shap


parser = argparse.ArgumentParser()
parser.add_argument("--tot_folds",help="Number of folds for model training",
                     type=int,
                    default=1)
parser.add_argument("--prefix", type=str, help="Training prefix",
                    default="nnuent_pre_focal_buffer_wp_pcr_dwi_lre4_cluster_layer2-train")
parser.add_argument("--legname", type=str, help="Legend description",
                    default='Response 4-fold CNN')
args = parser.parse_args()


tot_folds = args.tot_folds
prefix = args.prefix
legname = args.legname
#%%

sys.argv.extend(["--load_json", f"/data/groups/beets-tan/l.cai/results/{prefix}/options.json"])





#%%
# Process folds
# Concatenate results of the folds
y_preds_total = []
y_labels_total = []
for ff in range(tot_folds):
    # Load test data

    print(f"val_fold_{ff}")
    A = Trainer()
   
    train_ds, val_ds, train_dl, val_dl = A.init_data(ff, mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
    
    #  Make sample for loading
    sample = []
    for n, point in enumerate(val_dl):
        if n == 1:
           break
        x = point[0]
        sample.append(x)
    #sample = torch.cat(sample, dim=0)          

    use_cuda = torch.cuda.is_available()   #add new
    device = torch.device("cuda" if use_cuda else "cpu")     
    #sample = sample.to(device) # till here
    #_n = prefix.rstrip("-train")
    _n = prefix.split('-')[0]
    full_name=os.path.join(f"/data/groups/beets-tan/l.cai/results/{prefix}/",  'model-' + _n +'-fold' +str(ff)+ '-epochzloss'+'.pth')  #str(prefix)
    #full_name = 'results/nnuent_pre_focal_buffer4-train/model-nnuent_pre_focal_buffer4-fold3-epochz.pth'
    
    #Get Model for fold
    #model = A.init_model(sample)
    model = generate_model(10)
    #model.load_state_dict(torch.load(full_name,
    #                                      map_location=torch.device('cuda' if torch.cuda.is_available() else "cpu")))
    model = safe_restore(model, full_name)
    
    model = model.to(device)    
    model.eval()
    # Generate vector of predictions and true labels
    y_preds = dict()
    pp = []
    ll = []
    #print(len(train_ds),train_ds[0][0]['image'])
    for n in range(len(val_ds)):

        f, truth, extra = val_ds[n]
        #print(type(f),f.shape)
        #x = f.unsqueeze(0)
        #x = x.to(device)
        
        x = f['image']
        #print(x.shape)
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
        #l,p = model(x)
        #l,p = model(x)
        return_layers = {
        
        'layer1': 'layer1',
        'layer2': 'layer2',
        }
        #mid_getter = MidGetter(model, return_layers=return_layers, keep_output=True)
        #mid_outputs, model_output = mid_getter([x,x1,x2,x3,x4])
        #layer1 = mid_outputs['layer1'].cpu().detach().numpy()
        #plt.imshow(layer1[0,0,10,:,:])
        #plt.savefig('focus.png',dpi=300)
        l = model([x,x1,x2,x3,x4])
        
        p = nn.Softmax(dim=1)(l)
       
        
        pred = p[0].detach().cpu().numpy().tolist()

        y_preds_total.append(pred)
        y_labels_total.append(truth)
        pp.append(p[0][1].detach().cpu())
        ll.append(truth)
        #print((torch.stack((train_ds[0][0]['image'], train_ds[1][0]['image']),dim=0).shape))
        #x_tr = torch.stack((train_ds[0][0]['image'], train_ds[1][0]['image']),dim=0)
        #x1_tr = torch.stack((train_ds[0][0]['out1'], train_ds[1][0]['out1']),dim=0)
        #x2_tr = torch.stack((train_ds[0][0]['out2'], train_ds[1][0]['out2']),dim =0)
        #x3_tr = torch.stack((train_ds[0][0]['out3'], train_ds[1][0]['out3']),dim=0)
        #x4_tr = torch.stack((train_ds[0][0]['out4'], train_ds[1][0]['out4']),dim=0)
        #x_tr = x_tr.to(device)
        #x1_tr = x1_tr.to(device)
        #x2_tr = x2_tr.to(device)
        #x3_tr = x3_tr.to(device)
        #x4_tr = x4_tr.to(device)
        #explain = shap.DeepExplainer(model,[x,x1,x2,x3,x4])
        #if type([x_tr,x1_tr,x2_tr,x3_tr,x4_tr]) == list:
        #    print('list')
        #print(shap.initjs())
        #print(explain.explain_row())
        #print([x].shape)
        #print(val_ds[1][0]['image'].shape)
        #shap_v = explain.shap_values([x,x1,x2,x3,x4])#[x_tr,x1_tr,x2_tr,x3_tr,x4_tr])

        #print(shap_values.shape)
    #df.loc[df[f'fold_{ff}']=='test',['dl_pred']] = pp
    #print(pp)
    print(np.sum(ll))
    print(roc_and_auc(np.asarray(pp),np.asarray(ll)))
    print(dict(zip(pp,ll)),len(pp))
    #with open('dwit2_emvi_resnet10_val.pkl', 'wb') as handle:
    #    pickle.dump(dict(zip(pp,ll)), handle, protocol=pickle.HIGHEST_PROTOCOL)

    #dict(zip(arr,ll))
#p#rint(roc_and_auc(df['dl_pred'],df['Risk[High]']))
#print(pp)
#df.to_csv('test_rf_dl.csv',sep='\t',index=False)
y_labels_total = np.array(label_binarize(y_labels_total,classes=[0,1,2]))[:,:2]
y_preds_total = np.array(y_preds_total)
#print(y_preds_total)
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
    print(roc_auc)
    
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