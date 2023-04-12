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




#threshold = Find_Optimal_Cutoff(list(emvi_layer1.values()), list(emvi_layer1.keys()))
#print(threshold)
##dwi,t2w_single
#emvi = {tensor(0.3568): 1, tensor(0.3577): 1, tensor(0.5002): 1, tensor(0.3917): 0, tensor(0.2985): 0, tensor(0.4349): 1, tensor(0.3369): 0, tensor(0.3234): 0, tensor(0.3078): 0, tensor(0.3967): 0, tensor(0.3464): 0, tensor(0.3282): 0, tensor(0.3525): 0, tensor(0.3229): 0, tensor(0.3769): 0, tensor(0.5208): 1, tensor(0.3792): 1, tensor(0.4860): 1, tensor(0.3267): 0, tensor(0.3662): 1, tensor(0.3309): 0, tensor(0.2823): 0, tensor(0.3286): 0, tensor(0.3020): 0, tensor(0.3980): 0, tensor(0.3553): 1, tensor(0.3767): 0, tensor(0.3556): 1, tensor(0.3624): 0, tensor(0.2766): 0, tensor(0.3248): 0, tensor(0.3339): 0, tensor(0.3151): 0, tensor(0.3088): 0, tensor(0.3069): 0, tensor(0.3079): 0, tensor(0.4021): 0, tensor(0.3333): 0, tensor(0.2989): 0, tensor(0.3335): 1}
#pcr = {tensor(0.6646): 0, tensor(0.5770): 0, tensor(0.4572): 0, tensor(0.3804): 0, tensor(0.6444): 1, tensor(0.4006): 0, tensor(0.5401): 0, tensor(0.6420): 0, tensor(0.5339): 0, tensor(0.5636): 0, tensor(0.6352): 0, tensor(0.6286): 1, tensor(0.6524): 0, tensor(0.6574): 1, tensor(0.5987): 0, tensor(0.2817): 0, tensor(0.5862): 0, tensor(0.4827): 0, tensor(0.5884): 0, tensor(0.4248): 0, tensor(0.6242): 0, tensor(0.6151): 1, tensor(0.6339): 1, tensor(0.5960): 1, tensor(0.3828): 0, tensor(0.5306): 0, tensor(0.6192): 0, tensor(0.6347): 0, tensor(0.5928): 0, tensor(0.6070): 0, tensor(0.6355): 1, tensor(0.6795): 0, tensor(0.6549): 0, tensor(0.4501): 0, tensor(0.6405): 0, tensor(0.6538): 1, tensor(0.6014): 1, tensor(0.6446): 0, tensor(0.4866): 0, tensor(0.6115): 0}
##dwi only single
#emvi_s = {tensor(0.8080): 1, tensor(0.2276): 1, tensor(0.6352): 1, tensor(0.9077): 0, tensor(0.3702): 0, tensor(0.4987): 1, tensor(0.1275): 0, tensor(0.3229): 0, tensor(0.5459): 0, tensor(0.4171): 0, tensor(0.1505): 0, tensor(0.2246): 0, tensor(0.3529): 0, tensor(0.0928): 0, tensor(0.2261): 0, tensor(1.0000): 1, tensor(0.4644): 1, tensor(0.8655): 1, tensor(0.1404): 0, tensor(0.2136): 1, tensor(0.3781): 0, tensor(0.8800): 0, tensor(0.1178): 0, tensor(0.0966): 0, tensor(0.9342): 0, tensor(0.8156): 1, tensor(0.4960): 0, tensor(0.4521): 1, tensor(0.1626): 0, tensor(0.1077): 0, tensor(0.5396): 0, tensor(0.4724): 0, tensor(0.2051): 0, tensor(0.1696): 0, tensor(0.1096): 0, tensor(0.1208): 0, tensor(0.4085): 0, tensor(0.1574): 0, tensor(0.5792): 0, tensor(0.4123): 1}
#pcr_s = {tensor(0.5463): 0, tensor(0.5347): 0, tensor(0.3676): 0, tensor(0.1447): 0, tensor(0.5744): 1, tensor(0.1452): 0, tensor(0.1137): 0, tensor(0.5105): 0, tensor(0.4439): 0, tensor(0.0316): 0, tensor(0.4486): 0, tensor(0.5983): 1, tensor(0.5390): 0, tensor(0.6380): 1, tensor(0.2442): 0, tensor(0.0071): 0, tensor(0.4479): 0, tensor(0.0811): 0, tensor(0.6179): 0, tensor(0.4316): 0, tensor(0.4580): 0, tensor(0.4204): 1, tensor(0.6024): 1, tensor(0.6189): 1, tensor(0.1917): 0, tensor(0.2560): 0, tensor(0.4918): 0, tensor(0.4392): 0, tensor(0.5343): 0, tensor(0.6323): 0, tensor(0.5572): 1, tensor(0.5669): 0, tensor(0.6484): 0, tensor(0.2004): 0, tensor(0.1430): 0, tensor(0.6936): 1, tensor(0.4355): 1, tensor(0.6246): 0, tensor(0.4153): 0, tensor(0.3309): 0} 


#print(emvi_layer1_val)
#print(classification_analysis(np.array(list(emvi_layer1.values())),np.array(list(emvi_layer1.keys())),th=threshold))
#print(case_out(num_dict,emvi_layer1))
inputv = emvi_layer4_val#emvi_layer4
result = conclued_stat(list(inputv.values()), list(inputv.keys()),inputv, num_dict)
#print('mcc',matthews_corrcoef(list(inputv.keys()),list(inputv.values())))

print(len(inputv))
print(f"{result['rocauc']} & {result['sensitivity']} & {result['specificity']} & {result['ppv']} & {result['npv']} & {result['f1_score']}")
#print(case_out(num_dict,emvi_layer1))
#print(case_out(num_dict,emvi_layer2))
#print(case_out(num_dict,emvi_layer3))
#print(case_out(num_dict,emvi_layer4))

