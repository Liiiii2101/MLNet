import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import cv2
from medcam.medcam_utils import interpolate
from collections import OrderedDict



def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))



num_dict=OrderedDict({'435':25,'412':24,'474':20,'501':20})
mid_list = ['out1','out2','out3','out4']
mri ='dwit2'


plt.figure()
#plt.tight_layout()
plt.subplots(5,4,figsize=(5, 3.4))
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.01,
                    hspace=0.01)

Count = 1
for i in num_dict.keys():
    image = np.squeeze(np.load(f'/processing/Lishan/multi_scale/{mri}/latent_re_test2/image/{str(i)}_{str(i)}.npy')[0,0])
    ax1 = plt.subplot(4,5,Count)
    plt.imshow(image[num_dict[i],20:172,20:204],cmap='gray')
    plt.axis('off')
    Count += 1
    _slic = num_dict[i]
    for h, j in enumerate(mid_list):
        midoutput = np.squeeze(np.load(f'/processing/Lishan/multi_scale/{mri}/latent_re_test2/{j}/{str(i)}_{str(i)}.npy'))[1,:,:,:]
        #print(h,j,midoutput.shape)
        plt.subplot(4,5,Count)
        #plt.imshow(image[num_dict[i],20:172,30:194],cmap='gray')
        if h == 0:
            _slic = _slic
        else:
            _slic = _slic//2

        if h ==0:
            plt.imshow(midoutput[_slic,9:-9,9:-9],cmap='gray')
        elif h == 1:
            plt.imshow(midoutput[_slic,4:-4,4:-4],cmap='gray')
        else:
            plt.imshow(midoutput[_slic,:,:],cmap='gray') 
        plt.axis('off')
        Count += 1

   

plt.savefig(f'{mri}_midoutput_ups.png',dpi=300)








num_dict=OrderedDict({'435':25,'412':24,'474':20,'501':20})
network = ['_resnet10','','_layer1','_layer2','_layer3','_layer4']
target = 'emvi'
mri = 'dwi'

for i in range(0):
    a =np.load(f'mid_output/{i}_2.npy')#np.load('/processing/Lishan/multi_scale/dwit2/latent_re_test2/final/508_508.npy')#np.load('mid_output/96_1.npy')
    print(i)
    plt.imshow(a[0,256,5,:,:])
    plt.savefig('inter.jpg',dpi=300)

plt.figure()
#plt.tight_layout()
plt.subplots(7,4,figsize=(7, 3.7))
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.01,
                    hspace=0.01)

Count = 1
for i in num_dict.keys():
    image = np.squeeze(np.load(f'/processing/Lishan/multi_scale/{mri}/latent_re_test2/image/{str(i)}_{str(i)}.npy'))
    ax1 = plt.subplot(4,7,Count)
    plt.imshow(image[num_dict[i],20:172,30:194],cmap='gray')
    plt.axis('off')
    Count += 1
    for j in network:
        attention = np.squeeze(np.load(f'/mnt/data/groups/beets-tan/l.cai/attention/gbp/{mri}/{target}/{mri}_{target}{j}/att{str(int(i)-412)}.npy'))
        plt.subplot(4,7,Count)
        
        
        att_draw =  attention[num_dict[i],20:172,30:194]
        att_draw = NormalizeData(att_draw)
        
        plt.imshow(att_draw,cmap='seismic',alpha=1)
        plt.axis('off')
        Count += 1

   

plt.savefig(f'{mri}_{target}_gdp.png',dpi=300)