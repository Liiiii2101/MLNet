import pandas as pd
import numpy as np
import random
import os
from torch.utils.data import Dataset
import numpy as np
from dtor.utilities.utils import expand_image
from dtor.utilities.utils import bbox3d, crop3d,pad_nd_image
from tqdm import tqdm
import pandas as pd
import torch
from sklearn.model_selection import KFold#StratifiedKFold
import SimpleITK as sitk


def pad_or_crop_image(image, seg=None, target_size=(128, 144, 144)):
    c, z, y, x = image.shape
    z_slice, y_slice, x_slice = [get_crop_slice(target, dim) for target, dim in zip(target_size, (z, y, x))]
    image = image[:, z_slice, y_slice, x_slice]
    if seg is not None:
        seg = seg[:, z_slice, y_slice, x_slice]
    todos = [get_left_right_idx_should_pad(size, dim) for size, dim in zip(target_size, [z, y, x])]
    padlist = [(0, 0)]  # channel dim
    for to_pad in todos:
        if to_pad[0]:
            padlist.append((to_pad[1], to_pad[2]))
        else:
            padlist.append((0, 0))
    image = np.pad(image, padlist)
    if seg is not None:
        seg = np.pad(seg, padlist)
        return image, seg
    return image


def get_left_right_idx_should_pad(target_size, dim):
    if dim >= target_size:
        return [False]
    elif dim < target_size:
        pad_extent = target_size - dim
        left = random.randint(0, pad_extent)
        right = pad_extent - left
        return True, left, right


def get_crop_slice(target_size, dim):
    if dim > target_size:
        crop_extent = dim - target_size
        left = random.randint(0, crop_extent)
        right = crop_extent - left
        return slice(left, dim - right)
    elif dim <= target_size:
        return slice(0, dim)




class MRIDataset(Dataset):
    """Dataset from MRI images. including both dwi and t2w and adc"""

    def __init__(self,
                 csv=None,
                 fold=None,
                 tr_test=None,
                 transform=None,
                 dim=3, 
                 **external_kwargs
                 ):
        """
        Initialization
        Args:
            csv: File with the image locations
            fold: Which fold to return
            tr_test: Test or train
            transform: Any augmentations needed or preprocessing steps
            dim: Data dimensions
        """

        self.transform = transform
        self.dim = dim
        df_train = pd.read_csv('mgmt_test115.csv',sep='\t')
        self.gt = df_train
        # Restrict by fold + train/test
        if csv:
            self.cropped_images = pd.read_csv(csv, sep="\t")
            self.cropped_images = self.cropped_images[self.cropped_images[f"fold_{fold}"] == tr_test]
            self.cropped_images.index = range(len(self.cropped_images))
        else:
            self.cropped_images = None

    def __len__(self):
        return len(self.cropped_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fname = self.cropped_images.loc[idx, 'BraTS21ID']
        searchk = self.cropped_images.loc[idx,'id']
        #print(idx,fname, searchk)
        
        #ffname = fname.split('/')[-1]
        
        idxi = str(fname)#int(ffname.split('_')[0]) #+318
        #print(idxi)
        #print(idxi)
       
        #print(self.gt.loc[idxi,'Patient_Hospital'])
        idx_f = str(idxi)
        #print(idx_f)
        imaging_f = f'/data/groups/beets-tan/l.cai/brats_multi/image/{idx_f}_{idx_f}.npy'
        
        image = np.load(imaging_f)
        #print('/processing/Lishan/nnUNet_raw_data_base/nnUNet_raw_data/Task509_RectalReset/imagesTr/rectal_{idxi:03}_0000.nii.gz')
        #image = sitk.GetArrayFromImage(sitk.ReadImage(f'/processing/Lishan/nnUNet_raw_data_base/nnUNet_raw_data/Task509_RectalReset/imagesTr/rectal_{idxi:03}_0002.nii.gz'))
        #image = np.expand_dims(image,axis=0)
        #image = image.astype(np.float32)
        #image = pad_nd_image(image,new_shape=(40,224,224)) 
        #image = pad_or_crop_image(image, target_size = (40,224,224))
        #seg = image[-1,:,:,:]
        #seg = np.expand_dims(seg,axis=0)
        #seg[seg<0] = 0
        #image = image[:3,:,:,:]
        image = np.squeeze(image,axis=0)
        
        out1 = np.load(f'/data/groups/beets-tan/l.cai/brats_multi/out1/{idx_f}_{idx_f}.npy')
        out2 = np.load(f'/data/groups/beets-tan/l.cai/brats_multi/out2/{idx_f}_{idx_f}.npy')
        out3 = np.load(f'/data/groups/beets-tan/l.cai/brats_multi/out3/{idx_f}_{idx_f}.npy')
        out4 = np.load(f'/data/groups/beets-tan/l.cai/brats_multi/out4/{idx_f}_{idx_f}.npy')
        
        #fname = f'/data/groups/beets-tan/l.cai/brats_multi/final/{idx_f}_{idx_f}.npy'
        #final = np.load(fname)
        #print(out1.shape,out2.shape,out3.shape,out4.shape)
        #segname = self.cropped_images.loc[idx, 'segmentation']
        #label = self.cropped_images.loc[idx, "label"]
        #label = self.gt.loc[idxi,'a']
        #label = self.gt.loc[idx,'MGMT_value']#[idxi,'MGMT_value']
        labeldf = self.gt[self.gt['BraTS21ID']==int(fname)]
        #print(labeldf)
        label = list(labeldf['MGMT_value'])[0]
        #print(label)
        #label = self.gt.loc[idxi,'MRI_expert_EMVI']
        #if label != self.gt.loc[idxi,'MRI_expert_EMVI']:
        #    assert('wrong stop')
        #if label != self.gt.loc[idxi,'Path_pCR']:
        #    assert('wrong stop')
        #print(self.cropped_images.loc[idx, "patient"],label)
        #image = pad_nd_image(image,new_shape=(40,224,224)) 
        #image = pad_or_crop_image(image, target_size = (40,224,224))
        #fname = self.cropped_images.loc[idx, 'filename']
        #image = np.load(fname)

        #print(np.unique(seg))
        #image = image[:3,:,:,:]
        #image = np.expand_dims(image, axis=0)
        #np.load(segname)
        #print(seg.shape,image.shape) 
        #print(image.shape, final.shape)           
        #image = np.concatenate((image,final[0,1:,:,:,:]),axis=0)
        #print(image.shape)
        if self.dim == 2:
            counts = [np.sum(s) for s in seg]
            _slice = np.argmax(counts)
            image = image[0, _slice, :, :] # Remove singleton extra dimension
            image = np.moveaxis(image, -1, 0)
        else:
            #fname = self.cropped_images.loc[idx, 'filename']           
            #image = np.load(fname);image = image[0,:,:,:];image = np.moveaxis(image, -1, 0)
            #print(image.shape)
            image = image[:, :, :, :] # Remove singleton extra dimension
            out1 = out1[0,:3,:,:,:]
            out2 = out2[0,:3,:,:,:]
            out3 = out3[0,:3,:,:,:]
            out4 = out4[0,:3,:,:,:]
            #print(image.shape,out1.shape)
            #image = np.moveaxis(image, -1, 0)
  
           
            
        
        image = torch.from_numpy(image).to(torch.float32)
        out1 = torch.from_numpy(out1).to(torch.float32)
        out2 = torch.from_numpy(out2).to(torch.float32)
        out3 = torch.from_numpy(out3).to(torch.float32)
        out4 = torch.from_numpy(out4).to(torch.float32)

        #image_dict = {'image':image,'out1':out1, 'out2':out2, 'out3':out3, 'out4':out4}
        
        if False:#self.transform:
            image = self.transform(image)
        #    #print('a',image.shape)
        image_dict = {'image':image,'out1':out1, 'out2':out2, 'out3':out3, 'out4':out4}
        sample = [image_dict, label, fname]
        #print(sample)
        return sample