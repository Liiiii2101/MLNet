import numpy as np
import pandas as pd
from statistic import classification_analysis
import pickle
from collections import OrderedDict
num_dict=OrderedDict({'435':25,'412':24,'474':20,'501':20})
from torch import tensor
from sklearn.metrics import roc_curve
from sklearn.metrics import matthews_corrcoef


def conclued_stat(target,pred, value_dict, num_dict):
    threshold = Find_Optimal_Cutoff(target,pred)
    #case_out(num_dict,value_dict)
    print('threshold', threshold)
    result = classification_analysis(np.array(target),np.array(pred),th=threshold)
    #print(case_out(num_dict,value_dict))
    return result




def case_out(num_dict,result):
    collector = {}
    for i in num_dict.keys():
        j = int(i) - 412
        collector[i] =  [list(result.values())[j], list(result.keys())[j]]
    return collector



def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value
        
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 







#emvi
emvi_resnet10 = 're_prob/dwit2_emvi_resnet10.pkl'#'re_prob/dwi_emvi_resnet10.pkl'
emvi = 're_prob/dwit2_emvi.pkl'#'re_prob/dwi_emvi.pkl'
emvi_layer1 = 're_prob/dwit2_emvi_layer1.pkl'#'re_prob/dwi_emvi_layer1.pkl'
emvi_layer2 = 're_prob/dwit2_emvi_layer2.pkl'#'re_prob/dwi_emvi_layer2.pkl'
emvi_layer3 = 're_prob/dwit2_emvi_layer3.pkl'#'re_prob/dwi_emvi_layer3.pkl'
emvi_layer4 = 're_prob/dwit2_emvi_layer4.pkl'#'re_prob/dwi_emvi_layer4.pkl'

emvi_layer2_val = 're_prob_val/dwit2_pcr_layer2_val.pkl'
emvi_resnet10_val = 're_prob_val/dwit2_pcr_resnet10_val.pkl'
emvi_layer1_val = 're_prob_val/dwit2_pcr_layer1_val.pkl'
emvi_layer3_val = 're_prob_val/dwit2_pcr_layer3_val.pkl'
emvi_layer4_val = 're_prob_val/dwit2_pcr_layer4_val.pkl'
emvi_val = 're_prob_val/dwit2_pcr_val.pkl'

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

with open(emvi_resnet10_val, 'rb') as f:
    emvi_resnet10_val = pickle.load(f)


with open(emvi_val, 'rb') as f:
    emvi_val = pickle.load(f)

with open(emvi_layer1_val, 'rb') as f:
    emvi_layer1_val = pickle.load(f)
with open(emvi_layer2_val, 'rb') as f:
    emvi_layer2_val = pickle.load(f)

with open(emvi_layer3_val, 'rb') as f:
    emvi_layer3_val = pickle.load(f)

with open(emvi_layer4_val, 'rb') as f:
    emvi_layer4_val = pickle.load(f)





inputv = emvi_layer4_val
result = conclued_stat(list(inputv.values()), list(inputv.keys()),inputv, num_dict)


print(len(inputv))
print(f"{result['rocauc']} & {result['sensitivity']} & {result['specificity']} & {result['ppv']} & {result['npv']} & {result['f1_score']}")


