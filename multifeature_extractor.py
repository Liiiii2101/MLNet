import torch
import numpy as np
from nnunet.training.model_restore import load_model_and_checkpoint_files, restore_model
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
import os
from torch import nn
from torch.cuda.amp import autocast
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
from data_loading import load_dataset, DataLoader3D, DataLoader2D, unpack_dataset,get_case_identifiers
from collections import OrderedDict
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter



return_layers = {
    'seg_outputs.0': 'out0',
   'seg_outputs.1': 'out1',
   'seg_outputs.2': 'out2',
   'seg_outputs.3': 'out3', 
   'seg_outputs.4': 'out4',
}


return_layers = {
    'conv_blocks_localization.0':'enout0',
    'conv_blocks_localization.1':'enout1',
    'conv_blocks_localization.2':'enout2',
    'conv_blocks_localization.3':'enout3',
    'conv_blocks_localization.4':'enout4',
    'tu.0':'deout0',
    'tu.1':'deout1',
    'tu.2':'deout2',
    'tu.3':'deout3',
    'tu.4':'deout4'
}







def latent_saving_test(best_model_path, checkpoint, train, output_folder,val_data):

    key_list = [f"rectal_{i:03}" for i in range(412,509)]
    print(key_list)
    trainer = restore_model(best_model_path, checkpoint, train)
  

    model = trainer.network
    model.cuda()
    model.eval()
    
    with autocast():
        with torch.no_grad():
            while len(key_list) > 0:
                data = next(dl_val)
                ikey = str(list(data['keys'])[0])
                
                id = int(ikey.split('_')[1])
                if ikey in key_list:
                    print(id)
                    key_list.remove(ikey)
                    output = model(data['data'].cuda())
                    mid_getter = MidGetter(model, return_layers=return_layers, keep_output=True)
                    mid_outputs, model_output = mid_getter(data['data'].cuda())
           
                    layer0 = mid_outputs['enout0'].cpu().detach().numpy()
                    layer1 = mid_outputs['enout1'].cpu().detach().numpy()
                    layer2 = mid_outputs['enout2'].cpu().detach().numpy()
                    layer3 = mid_outputs['enout3'].cpu().detach().numpy()
                    layer4 = mid_outputs['enout4'].cpu().detach().numpy()
                    coder0 = mid_outputs['deout0'].cpu().detach().numpy()
                    coder1 = mid_outputs['deout1'].cpu().detach().numpy()
                    coder2 = mid_outputs['deout2'].cpu().detach().numpy()
                    coder3 = mid_outputs['deout3'].cpu().detach().numpy()
                    coder4 = mid_outputs['deout4'].cpu().detach().numpy()
                    print(layer0.shape, layer1.shape, layer2.shape, layer3.shape, layer4.shape)

                    seg_path = os.path.join(output_folder,'seg',f"{id}_{id}.npy")
                    image_data =  os.path.join(output_folder,'image',f"{id}_{id}.npy")
                    final = os.path.join(output_folder,'final',f"{id}_{id}.npy")
                    out1 =  os.path.join(output_folder,'out1',f"{id}_{id}.npy")
                    out2 =  os.path.join(output_folder,'out2',f"{id}_{id}.npy")
                    out3 =  os.path.join(output_folder,'out3',f"{id}_{id}.npy")
                    out4 =  os.path.join(output_folder,'out4',f"{id}_{id}.npy")
                    enout0 = os.path.join(output_folder,'enout0',f"{id}_{id}.npy")
                    enout1 = os.path.join(output_folder,'enout1',f"{id}_{id}.npy")
                    enout2 = os.path.join(output_folder,'enout2',f"{id}_{id}.npy")
                    enout3 = os.path.join(output_folder,'enout3',f"{id}_{id}.npy")
                    enout4 = os.path.join(output_folder,'enout4',f"{id}_{id}.npy")
                    deout0 = os.path.join(output_folder,'deout0',f"{id}_{id}.npy") 
                    deout1 = os.path.join(output_folder,'deout1',f"{id}_{id}.npy")
                    deout2 = os.path.join(output_folder,'deout2',f"{id}_{id}.npy")
                    deout3 = os.path.join(output_folder,'deout3',f"{id}_{id}.npy")
                    deout4 = os.path.join(output_folder,'deout4',f"{id}_{id}.npy")
                    np.save(seg_path, data['target'][0].numpy())
                    np.save(image_data, data['data'].numpy())
                    np.save(final, output[0].cpu().numpy())      
                    np.save(out1, output[1].cpu().numpy())
                    np.save(out2, output[2].cpu().numpy())
                    np.save(out3, output[3].cpu().numpy())
                    np.save(out4, output[4].cpu().numpy())
                    np.save(enout0,layer0)
                    np.save(enout1, layer1)
                    np.save(enout2, layer2)
                    np.save(enout3, layer3)
                    np.save(enout4, layer4)
                    np.save(deout0, coder0)
                    np.save(deout1, coder1)
                    np.save(deout2, coder2)
                    np.save(deout3, coder3)
                    np.save(deout4, coder4)

                else:
                    continue
    return 'Done'




for i in range(0):
    
    splits_file = '/processing/Lishan/nnUNet_preprocessed/Task099_Rectaldwit2/splits_final.pkl'
    best_model_path = f'...fold_{i}/model_final_checkpoint.model.pkl'

    checkpoint = best_model_path[:-4]
    train = False

    fold_num = 'all'

    output_folder = '/processing/Lishan/multi_scale/dwit2_duo/latent_re_test'

    if i>0:
        output_folder = f'/processing/Lishan/multi_scale/dwit2_duo/latent_re_test{i}'

    latent_saving_test(best_model_path, checkpoint, train, output_folder, dl_val)












def latent_saving(best_model_path, checkpoint, train, output_folder, splits_file,fold_num):
    key_list = list(load_pickle(splits_file)[fold_num]['val'])
    key_list = [str(i) for i in key_list]
    print(key_list)
    trainer = restore_model(best_model_path, checkpoint, train)
    trainer.batch_size = 1

    dl_tr, dl_val = trainer.get_basic_generators()
    dl_tr, dl_val = get_moreDA_augmentation(dl_tr, dl_val,
    trainer.data_aug_params[        
                        'patch_size_for_spatialtransform'],
                    trainer.data_aug_params,
                    deep_supervision_scales=trainer.deep_supervision_scales,
                    pin_memory=trainer.pin_memory,
                    use_nondetMultiThreadedAugmenter=False)

    model = trainer.network
    model.cuda()
    model.eval()
    
    with autocast():
        with torch.no_grad():
            while len(key_list) > 0:
                data = next(dl_val)
                ikey = str(list(data['keys'])[0])
                id = int(ikey.split('_')[1])
                #print(key_list[0],type(key_list[0]),ikey)
                if ikey in key_list:
                    print(id)
                    
                    key_list.remove(ikey)
                    output = model(data['data'].cuda())
                    mid_getter = MidGetter(model, return_layers=return_layers, keep_output=True)
                    mid_outputs, model_output = mid_getter(data['data'].cuda())
            #print(mid_outputs['layer1'].shape, mid_outputs['layer2'].shape)
                    layer0 = mid_outputs['enout0'].cpu().detach().numpy()
                    layer1 = mid_outputs['enout1'].cpu().detach().numpy()
                    layer2 = mid_outputs['enout2'].cpu().detach().numpy()
                    layer3 = mid_outputs['enout3'].cpu().detach().numpy()
                    layer4 = mid_outputs['enout4'].cpu().detach().numpy()
                    coder0 = mid_outputs['deout0'].cpu().detach().numpy()
                    coder1 = mid_outputs['deout1'].cpu().detach().numpy()
                    coder2 = mid_outputs['deout2'].cpu().detach().numpy()
                    coder3 = mid_outputs['deout3'].cpu().detach().numpy()
                    coder4 = mid_outputs['deout4'].cpu().detach().numpy()
                    print(layer0.shape, layer1.shape, layer2.shape, layer3.shape, layer4.shape)

                    seg_path = os.path.join(output_folder,'seg',f"{id}_{id}.npy")
                    image_data =  os.path.join(output_folder,'image',f"{id}_{id}.npy")
                    final = os.path.join(output_folder,'final',f"{id}_{id}.npy")
                    out1 =  os.path.join(output_folder,'out1',f"{id}_{id}.npy")
                    out2 =  os.path.join(output_folder,'out2',f"{id}_{id}.npy")
                    out3 =  os.path.join(output_folder,'out3',f"{id}_{id}.npy")
                    out4 =  os.path.join(output_folder,'out4',f"{id}_{id}.npy")
                    #enout0 = os.path.join(output_folder,'enout0',f"{id}_{id}.npy")
                    #enout1 = os.path.join(output_folder,'enout1',f"{id}_{id}.npy")
                    #enout2 = os.path.join(output_folder,'enout2',f"{id}_{id}.npy")
                    #enout3 = os.path.join(output_folder,'enout3',f"{id}_{id}.npy")
                    #enout4 = os.path.join(output_folder,'enout4',f"{id}_{id}.npy")
                    #deout0 = os.path.join(output_folder,'deout0',f"{id}_{id}.npy")
                    #deout1 = os.path.join(output_folder,'deout1',f"{id}_{id}.npy")
                    #deout2 = os.path.join(output_folder,'deout2',f"{id}_{id}.npy")
                    #deout3 = os.path.join(output_folder,'deout3',f"{id}_{id}.npy")
                    #deout4 = os.path.join(output_folder,'deout4',f"{id}_{id}.npy")
                    print(final,out1)
                    #print(data['target'][0].shape)
                    np.save(seg_path, data['target'][0].numpy())
                    np.save(image_data, data['data'].numpy())
                    np.save(final, output[0].cpu().numpy())      
                    np.save(out1, output[1].cpu().numpy())
                    np.save(out2, output[2].cpu().numpy())
                    np.save(out3, output[3].cpu().numpy())
                    np.save(out4, output[4].cpu().numpy())
                    #np.save(enout0, layer0)
                    #np.save(enout1, layer1)
                    #np.save(enout2, layer2)
                    #np.save(enout3, layer3)
                    #np.save(enout4, layer4)
                    #np.save(deout0, coder0)
                    #np.save(deout1, coder1)
                    #np.save(deout2, coder2)
                    #np.save(deout3, coder3)
                    #np.save(deout4, coder4)

                else:
                    continue
    return 'Done'







for i in range(3,5):
    print(i)
    task_name = 'Task001_BratsTumour'#'Task511_Rectaldt'
    fold_num = i

    splits_file = f'.../{task_name}/splits_final.pkl'
    if fold_num == 0:
        best_model_path = best_model_path = f'.../{task_name}/nnUNetTrainerV2__nnUNetPlansv2.1/fold_{fold_num}/model_best.model.pkl'
    else:
        best_model_path = f'.../{task_name}/nnUNetTrainerV2__nnUNetPlansv2.1/fold_{fold_num}/model_final_checkpoint.model.pkl'#f'/processing/Lishan/nnUNet_trained_models/nnUNet/3d_fullres/{task_name}/nnUNetTrainerV2__nnUNetPlansv2.1/fold_{fold_num}/model_final_checkpoint.model.pkl'

    checkpoint = best_model_path[:-4]
    train = False

    output_folder = '/data/groups/beets-tan/l.cai/brats_multi'

    latent_saving(best_model_path, checkpoint, train, output_folder, splits_file,fold_num)


















