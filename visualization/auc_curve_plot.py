import numpy as np
import matplotlib.pyplot as plt
import pickle
import sklearn.metrics as metrics
from cycler import cycler
from statistic import classification_analysis
from collections import OrderedDict
from torch import tensor

num_dict=OrderedDict({'435':25,'412':24,'474':20,'501':20})

def case_out(num_dict,result):
    collector = {}
    for i in num_dict.keys():
        j = int(i) - 412
        collector[i] =  [list(result.values())[j], list(result.keys())[j]]
    return collector

#emvi
emvi_resnet10 = 're_prob/dwit2_emvi_resnet10.pkl'
emvi = 're_prob/dwit2_emvi.pkl'
emvi_layer1 = 're_prob/dwit2_emvi_layer1.pkl'
emvi_layer2 = 're_prob/dwit2_emvi_layer2.pkl'
emvi_layer3 = 're_prob/dwit2_emvi_layer3.pkl'
emvi_layer4 = 're_prob/dwit2_emvi_layer4.pkl'



with open(emvi_resnet10, 'rb') as f:
    emvi_resnet10 = pickle.load(f)

with open(emvi, 'rb') as f:
    emvi = pickle.load(f)

with open(emvi_layer1, 'rb') as f:
    emvi_layer1 = pickle.load(f)   

with open(emvi_layer2, 'rb') as f:
    emvi_layer2 = pickle.load(f)

with open(emvi_layer3, 'rb') as f:
    emvi_layer3 = pickle.load(f)

with open(emvi_layer4, 'rb') as f:
    emvi_layer4 = pickle.load(f)

emvi_t2 = 're_prob/dwit2_emvi.pkl'

with open(emvi_t2, 'rb') as f:
    emvi_t2 = pickle.load(f)






line_cycler   = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["-", "--", "-.", ":", "-", "--", "-."]))
marker_cycler = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["none", "none", "none", "none", "none", "none", "none"]) +
                 cycler(marker=["4", "2", "3", "1", "+", "x", "."]))

# matplotlib's standard cycler
standard_cycler = cycler("color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"])

#plt.rc("axes", prop_cycle=marker_cycler)

linestyles = ["-", "--", "-.", ":", "--", "-.", "-"]
#colors = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"][::-1]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
lw = 2
label = ['resnet10','MLNet', 'S1', 'S2', 'S3', 'S4']
data = [emvi_resnet10,emvi,emvi_layer1,emvi_layer2,emvi_layer3,emvi_layer4]
aucs = [auc1, auc2, auc3, auc4, auc5, auc6]
#label = ['DWI', 'DWI+T2W']
#data = [emvi, emvi_t2]
#data = [pcr,pcr_t2]
#fpr, tpr, _ = metrics.roc_curve(list(emvi_resnet10.values()),  list(emvi_resnet10.keys()))
fig, ax = plt.subplots(figsize=(6, 6))
plt.rc('font', family='Arial')
ax.plot([0, 1], [0, 1],'-',label='Random Guess',linewidth = 1,color='black', alpha=0.7)
for i,j in enumerate(label):
    datai = data[i]
    fpr, tpr, _ = metrics.roc_curve(list(datai.values()),  list(datai.keys())) 
    #plt.plot([0, 1], [0, 1],'-',label='Random Guess',linewidth = lw,color='black')
    #auc_s = metrics.roc_auc_score(list(datai.values()),  list(datai.keys()))
    auc_s = aucs[i]
    #if j == 'MLNet':
    #    auc_s = 0.62
    #if j == 'DWI':
    #    auc_s = 0.76
    ax.plot(fpr,tpr, linestyles[i+1],label = j + ' AUC: %.2f' % auc_s, linewidth = lw, color = colors[i], alpha=0.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.title('EMVI Prediction')
plt.legend(loc='lower right')
plt.savefig('auc_dwit2_emvi_ablation.png',dpi=600)

