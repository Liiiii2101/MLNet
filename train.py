import os
#os.environ['DTORROOT'] = '/DATA/forLishan/rtr_dataset'
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"
try:
    import pretty_traceback
    pretty_traceback.install()
except ImportError:
    pass

from trainer import TrainerBase
from dtor.utilities.model_retriever import model_choice
from dtor.utilities.data_retriever import get_data
from dataloader_val import MRIDataset

import torch
import torch.nn as nn
from dtor.logconf import logging
import sys
import os
from network.resnet import generate_model


#os.environ['DTORROOT'] = '/DATA/forLishan/rtr_dataset'
if len(sys.argv) == 1:
    print("Usage:")
    print("python train.py --load_json PATH/TO/JSON")

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


# Initialise will take json config
class RTRTrainer(TrainerBase):
    def __init__(self):
        super().__init__()

    def init_model(self, sample=None):
        #model =  generate_model()
        model = model_choice(self.cli_args.model, 
                resume=self.cli_args.resume, sample=sample,
                pretrain_loc=False,
                pretrained_2d_name=self.cli_args.pretrained_2d_name,
                depth=self.cli_args.rn_depth,
                n_classes=self.cli_args.rn_nclasses, fix_inmodel=self.cli_args.fix_nlayers)
  

        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def init_data(self, fold, mean=None, std=None):
        aug = False
        if self.cli_args.augments > 0:
            aug = True
        if mean:
            train_ds, val_ds = get_data(self.cli_args.dset, self.cli_args.datapoints, fold, aug=aug,
                                        mean=mean, std=std, dim=self.cli_args.dim, external=MRIDataset)
        else:
            train_ds, val_ds = get_data(self.cli_args.dset, self.cli_args.datapoints, fold, aug=aug,
                    dim=self.cli_args.dim, external=MRIDataset)
        train_dl, val_dl = self.init_loaders(train_ds, val_ds)
        return train_ds, val_ds, train_dl, val_dl
    
    def init_tune(self, trial):
        self.t_learnRate = trial.suggest_loguniform('learnRate', 1e-6, 1e-3)
        self.t_decay = trial.suggest_uniform('decay', 0.9, 0.99)
        self.t_alpha = trial.suggest_uniform('focal_alpha', 0.5, 1.0)
        self.t_gamma = trial.suggest_uniform('focal_gamma', 1.0, 5.0)
        self.patience = trial.suggest_int('earlystopping', 3, 6)
        if self.fix_nlayers:
            self.fix_nlayers = trial.suggest_int('fix_nlayers', 10, 15)
        return
