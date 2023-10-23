import numpy as np
import pandas as pd
from nnunet.evaluation import metrics
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
from pylab import *
import glob
import scipy.stats

def stars(p):
    if p < 0.0001:
        return "****"
    elif (p < 0.001):
        return "***"
    elif (p < 0.01):
        return "**"
    elif (p < 0.05):
        return "*"
    else:
        return "-"


preds_folder = '../pred'

gt_folder = '.../labelsTr'


def dice_all(preds_folder, gt_folder, num, mode='train'):
    dice_list = []
  
    for i in range(412):
        predi = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(preds_folder,f'rectal_{i:03}.nii.gz')))
        gti = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(gt_folder,f'rectal_{i:03}.nii.gz')))
        dice_list.append(metrics.dice(predi,gti))

        if metrics.dice(predi,gti) < 0.5:
            print(i, metrics.dice(predi,gti))
        print('val mean: ', np.mean(dice_list), np.std(dice_list))

        print('test mean: ', np.mean(dice_list), np.std(dice_list))
    return dice_list


dwi_train = dice_all('/processing/Lishan/pred_509_dwi', '/processing/Lishan/nnUNet_raw_data_base/nnUNet_raw_data/Task509_RectalReset/labelsTr')
dwit2_train =  dice_all('/processing/Lishan/pred_509_dwit2w', '/processing/Lishan/nnUNet_raw_data_base/nnUNet_raw_data/Task509_RectalReset/labelsTr')


dwi_val = dice_all('/processing/Lishan/pred_97_dwi_nnunet', '/processing/Lishan/nnUNet_raw_data_base/nnUNet_raw_data/Task509_RectalReset/labelsTs',mode ='val')
dwit2w_val = dice_all('/processing/Lishan/pred_97_dwit2_nnunet', '/processing/Lishan/nnUNet_raw_data_base/nnUNet_raw_data/Task509_RectalReset/labelsTs', mode='val')





def draw_boxplot(data,name='4-Fold Dice',xlabel=['DWI','DWI+T2W'],ylabel='Dice'):
    params = {
    'axes.labelsize': 8,
    'font.size': 8,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'text.usetex': False,
    'figure.figsize': [2.5, 3.5]
}
    rcParams.update(params)

    fig = figure()
    ax = fig.add_subplot(111)
 
    ax.set_ylabel(ylabel)

    bp = ax.boxplot(data, showmeans=True)

    plt.xticks(range(1,len(data)+1), xlabel)

# colors, as before
    from palettable.colorbrewer.qualitative import Set2_7
    colors = Set2_7.mpl_colors

    for i in range(len(bp['boxes'])):
        box = bp['boxes'][i]
        box.set_linewidth(0)
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
            boxCoords = list(zip(boxX, boxY))
            boxPolygon = Polygon(boxCoords, facecolor=colors[i % len(colors)], linewidth=0)
            ax.add_patch(boxPolygon)

    for i in range(0, len(bp['boxes'])):
        bp['boxes'][i].set_color(colors[i])
    # we have two whiskers!
        bp['whiskers'][i*2].set_color(colors[i])
        bp['whiskers'][i*2 + 1].set_color(colors[i])
        bp['whiskers'][i*2].set_linewidth(2)
        bp['whiskers'][i*2 + 1].set_linewidth(2)
    # fliers
        bp['fliers'][i].set(markerfacecolor=colors[i],
                            marker='o', alpha=0.75, markersize=6,
                            markeredgecolor='none')
        bp['means'][i].set_color('black')
        bp['means'][i].set_linewidth(3)
    # and 4 caps to remove
        for c in bp['caps']:
            c.set_linewidth(0)


    


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', length=0)

    ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
    ax.set_axisbelow(True)

    ax.set_xticklabels(xlabel)


# the stars
    z, p = scipy.stats.mannwhitneyu(data[0], data[1])

    s = stars(p)
    print('p value: ',p_value)

    y_max = np.max(np.concatenate((dwi_train, dwit2_train)))
    y_min = np.min(np.concatenate((dwi_train, dwit2_train)))
# print(y_max)
    ax.annotate("", xy=(1, y_max), xycoords='data',
            xytext=(2, y_max), textcoords='data',
            arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
                            connectionstyle="bar,fraction=0.2"))
    ax.text(1.5, y_max + abs(y_max - y_min)*0.1, stars(p_value),
        horizontalalignment='center',
        verticalalignment='center')


    fig.subplots_adjust(left=0.2)


    savefig(name, dpi=300)
CI =95

#print(np.mean(dwi_train),np.median(dwi_train),np.std(dwi_train))
#print(np.mean(dwit2_train),np.median(dwit2_train), np.std(dwit2_train))
#print(np.mean(dwi_val),np.median(dwi_val),np.std(dwi_val))
#print(np.mean(dwit2w_val),np.median(dwit2w_val),np.std(dwit2w_val))
draw_boxplot([dwi_train,dwit2_train])
draw_boxplot([dwi_val,dwit2w_val],name='External Dice',xlabel=['DWI','DWI+T2W'],ylabel='Dice')




#print(dwi_val[0], dwi_val[23], dwi_val[62], dwi_val[89])
#print(dwit2w_val[0], dwit2w_val[23], dwit2w_val[62], dwit2w_val[89])