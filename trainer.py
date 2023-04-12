import datetime
import os
#os.environ['DTORROOT'] = '/processing/Lishan/random'
import sys
import random
import json
import numpy as np
import pandas as pd
from dtor.utilities.utils import safe_restore
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dtor.utilities.utils import focal_loss
from dtor.utilities.utils_stats import roc_and_auc
from dtor.utilities.torchutils import EarlyStopping
from dtor.utilities.torchutils import process_metrics, \
    METRICS_LOSS_NDX, METRICS_LABEL_NDX, METRICS_PRED_NDX, METRICS_SIZE
from dtor.loss.sam import SAM
import joblib
import optuna
from optuna.samplers import TPESampler
from dtor.logconf import enumerate_with_estimate
from dtor.logconf import logging
from dtor.utilities.utils import find_folds, get_class_weights
from dtor.utilities.model_retriever import model_choice
from dtor.utilities.data_retriever import get_data
from dtor.opts import init_parser
from dtor.opts import norms
from dataloader_m import MRIDataset
from network.resnet_stage import generate_model
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)
torch.manual_seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)



class TrainerBase:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        self.totalTrainingSamples_count = 0
        self.model = None
        self.weights = None
        self.trn_writer = None
        self.val_writer = None
        self.optimizer = None
        self.scheduler = None
        self.train_dl = None
        self.val_dl = None
        self.study = None
        self.sample = None
        self.init_dict = {}
        self.root_dir = os.environ["DTORROOT"]

        parser = init_parser()
        args = parser.parse_args(sys_argv)
        if args.load_json:
            with open(args.load_json, 'r') as f:
                args.__dict__.update(json.load(f))
        self.cli_args = args

        if self.cli_args.best_json:
            with open(self.cli_args.best_json, 'r') as f:
                self.cli_args.__dict__.update(json.load(f))

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # Needed to make training reproducible
        self.reset_torch_seeds()
        self.reset_rndm()

        # Make all tunable hyperparameters members
        self.patience = self.cli_args.earlystopping
        self.fix_nlayers = self.cli_args.fix_nlayers
        self.t_learnRate = self.cli_args.learnRate
        self.t_decay = self.cli_args.decay
        if "focal" in self.cli_args.loss.lower():
            self.t_alpha = self.cli_args.focal_alpha
            self.t_gamma = self.cli_args.focal_gamma

        # Make results directory
        self.output_dir = os.path.join("results", f"{self.cli_args.exp_name}-{self.cli_args.mode}")

    def reset_torch_seeds(self):
        seed_value = self.cli_args.seed
        torch.manual_seed(seed_value)
        print(seed_value)
        if self.use_cuda:
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def reset_rndm(self):
        seed_value = self.cli_args.seed
        np.random.seed(seed_value)
        random.seed(seed_value)

    def init_model(self, sample=None):
        return NotImplementedError

    def init_data(self, fold, mean=None, std=None):
        return NotImplementedError

    def init_tune(self, train):
        return NotImplementedError

    def init_optimizer(self):
        if self.cli_args.sam:
            optim = SAM(self.model.parameters(), Adam, lr=self.t_learnRate)
        else:
            optim = Adam(self.model.parameters(), lr=self.t_learnRate)
        decay = self.t_decay
        #optim = PESG(self.model, 
        #         loss_fn=AUCMLoss(),
        #         lr=1e-3, 
        #         margin=1.0, 
        #         epoch_decay=0.003, 
        #         weight_decay=0.0001)
        #weight_decay = 5e-4 # regularization weight decay
        #lr = 1e-3  # learning rate
        #eta = 1e1 # learning rate for control negative samples weights
        #decay_epochs = [20, 40]
        #decay_factor = 10

        #beta = 0.1 # upper bound for FPR
        
        #loss_fn = pAUC_CVaR_Loss(pos_len=85, beta=beta)        
        #optimizer = SOPA(self.model, loss_fn=loss_fn, mode='adam', lr=lr, eta=eta, weight_decay=weight_decay)
        scheduler = None
        if decay < 1.0:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=decay, verbose=True)
        return optim, scheduler

    def init_loaders(self, train_ds, val_ds):
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            shuffle = True,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl, val_dl

    def init_tensorboard_writers(self, fold):
        if self.trn_writer is None:
            log_dir = os.path.join(self.output_dir, "logs")

            self.trn_writer = SummaryWriter(
                log_dir=f"{log_dir}-{fold}-trn_cls"
            )
            self.val_writer = SummaryWriter(
                log_dir=f"{log_dir}-{fold}-val_cls"
            )

    def main(self):
        # Make the output folder
        assert not os.path.exists(self.output_dir), "Choose a unique experiment name or clean up after yourself :-)"
        os.makedirs(self.output_dir)

        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        assert self.cli_args.mode in ["train", "tune"], "Only train or tune are allowed modes"
        log.info(f"********** MODE = {self.cli_args.mode} *****************")

        if self.cli_args.mode == "train":
            self.main_training()
        else:
            self.tune()

    def main_training(self):
        # Load chunks file
        _df = pd.read_csv(self.cli_args.datapoints, sep="\t")
        if "fold_0" in _df.columns.values:
            tot_folds = find_folds(_df)
            log.info(f'Found a total of {tot_folds} folds to process')
        else:
            tot_folds = 1

        for fold in range(tot_folds):
            # Print
            log.info(f'FOLD {fold}')
            log.info('--------------------------------')

            # Data
            mean, std = norms[self.cli_args.norm]
            train_ds, val_ds, train_dl, val_dl = self.init_data(fold, mean=mean, std=std)

            # Get a sample batch
            sample = []
            for n, point in enumerate(train_dl):
                if n == 1:
                    break
                x = point[0]
                sample.append(x)
            #sample = torch.cat(sample, dim=0)

            # Generate weights
            log.info('Calculating class weights')
            self.weights = get_class_weights(train_ds)
            self.weights = self.weights.to(self.device)

            # Model
            log.info('Initializing model')
            #self.reset_torch_seeds()
            #self.reset_rndm()
            torch.manual_seed(1234)
            self.model = generate_model(10)#self.init_model(sample=sample)
            #model_path = os.path.join("model_init_weights.pth")
            #torch.save(self.model.state_dict(), model_path)
            #self.model.load_state_dict(torch.load("model_init_weights.pth"))
            #full_name = '/processing/Lishan/pretrained_3d/r3d101_K_200ep.pth'
            #weights = self.model.state_dict()
            #lm_dict = torch.load(full_name)['state_dict']
            #weights2 = {k: v for k, v in weights.items() if k in lm_dict.keys()}
            #weights.update(weights2)
            #self.model.load_state_dict(weights)

            #full_name = 'results/nnuent_pre_focal_buffer_gr-train/model-nnuent_pre_focal_buffer_gr-fold0-epochz.pth'
            #self.model.load_state_dict(torch.load(full_name))
            #log.info('pretrained')
            #self.model = safe_restore(self.model, full_name)
            #print('pretrained')
            #log.info('Model initialized')
            self.totalTrainingSamples_count = 0

            # Optimizer
            self.optimizer, self.scheduler = self.init_optimizer()
            log.info('Optimizer initialized')

            # Early stopping class tracks the best validation loss
            es = EarlyStopping(patience=self.patience)

            # If model is using cnn_finetune, we need to update the transform with the new
            # mean and std deviation values
            try:
                dpm = self.model if not self.use_cuda else self.model.module
            except nn.modules.module.ModuleAttributeError:
                dpm = self.model

            if hasattr(dpm, "original_model_info"):
                log.info('*******************USING PRETRAINED MODEL*********************')
                mean = dpm.original_model_info.mean
                std = dpm.original_model_info.std
                train_ds, val_ds, train_dl, val_dl = self.init_data(fold, mean=mean, std=std)

            log.info('*******************NORMALISATION DETAILS*********************')
            log.info(f"preprocessing mean: {mean}, std: {std}")

            # Training loop
            auc = 0
            for epoch_ndx in range(1, self.cli_args.epochs + 1):
                log.info("FOLD {}, Epoch {} of {}, {}/{} batches of size {}*{}".format(
                    fold,
                    epoch_ndx,
                    self.cli_args.epochs,
                    len(train_dl),
                    len(val_dl),
                    self.cli_args.batch_size,
                    (torch.cuda.device_count() if self.use_cuda else 1),
                ))

                trn_metrics_t = self.do_training(fold, epoch_ndx, train_dl)
                self.log_metrics(fold, epoch_ndx, 'trn', trn_metrics_t)

                val_metrics_t = self.do_validation(fold, epoch_ndx, val_dl)
                self.log_metrics(fold, epoch_ndx, 'val', val_metrics_t)

                # Checkpoint if it's the best model
                val_loss = val_metrics_t[METRICS_LOSS_NDX].mean()
                es(val_loss)
                obj, _, _ = roc_and_auc(val_metrics_t[METRICS_PRED_NDX].numpy(),val_metrics_t[METRICS_LABEL_NDX].numpy())
                
                
                #print(trn_metrics_t[METRICS_PRED_NDX].numpy())
                #print(trn_metrics_t[METRICS_LABEL_NDX].numpy())
                #print(val_metrics_t[METRICS_PRED_NDX].numpy())
                #print(val_metrics_t[METRICS_LABEL_NDX].numpy())
                #print(obj) 
                if obj >= auc:#val_loss <= es.best_loss:
                    checkpoint = {
                        "EPOCH": epoch_ndx,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        
                        "LOSS": val_loss
                    }

                    auc = obj
                    print('auc saved')
                    ch_path = os.path.join(self.output_dir,
                                           f"model-{self.cli_args.exp_name}-fold{fold}-epochz.pth")
                    
                    torch.save(self.model.state_dict(), ch_path)
                if val_loss <= es.best_loss:
                    checkpoint = {
                        "EPOCH": epoch_ndx,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        
                        "LOSS": val_loss
                    }

                    #auc = obj
                    print('loss saved')
                    ch_path = os.path.join(self.output_dir,
                                           f"model-{self.cli_args.exp_name}-fold{fold}-epochzloss.pth")
                    
                    torch.save(self.model.state_dict(), ch_path)

                #obj, _, _ = roc_and_auc(val_metrics_t[METRICS_PRED_NDX].numpy(),
                #                    val_metrics_t[METRICS_LABEL_NDX].numpy())
                log.info(f"Status AUC: {obj:.3f}")

                if self.cli_args.earlystopping:
                    if es.early_stop:
                        break

            model_path = os.path.join(self.output_dir,
                                      f"model-{self.cli_args.exp_name}-fold{fold}.pth")
            torch.save(self.model.state_dict(), model_path)

            if hasattr(self, 'trn_writer'):
                self.trn_writer.close()
                self.val_writer.close()

            self.trn_writer = None
            self.val_writer = None

        # Save CLI args
        cli_name = os.path.join(self.output_dir, 'options.json')
        with open(cli_name, 'w') as f:
            json.dump(self.cli_args.__dict__, f, indent=2)

    def do_training(self, fold, epoch_ndx, train_dl):
        self.model = self.model.train().to(self.device)
        trn_metrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device
        )

        batch_iter = enumerate_with_estimate(
            train_dl,
            "F{}, E{} Training".format(fold, epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:

            def closure():
                self.optimizer.zero_grad()
                loss_var = self.compute_batch_loss(
                    batch_ndx,
                    batch_tup,
                    train_dl.batch_size,
                    trn_metrics_g
                )
                loss_var.backward()
                return loss_var
            closure()
            if self.cli_args.sam:
                self.optimizer.step(closure)
            else:
                self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trn_metrics_g.to('cpu')

    def do_validation(self, fold, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model = self.model.eval()
            val_metrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerate_with_estimate(
                val_dl,
                "F{} E{} Validation ".format(fold, epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.compute_batch_loss(
                    batch_ndx, batch_tup, val_dl.batch_size, val_metrics_g, debug=False)

        return val_metrics_g.to('cpu')

    def compute_batch_loss(self, batch_ndx, batch_tup, batch_size, metrics_g, debug=False):
        input_t, label_t, _ = batch_tup

        input_g = input_t['image']
        out1 = input_t['out1']
        out2 = input_t['out2']
        out3 = input_t['out3']
        out4 = input_t['out4']
        input_g = input_g.to(self.device, non_blocking=True)
        out1 = out1.to(self.device, non_blocking=True)
        out2 = out2.to(self.device, non_blocking=True)
        out3 = out3.to(self.device, non_blocking=True)
        out4 = out4.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        input_g = input_g.float()
        if self.cli_args.dim == 2:
            logits_g = self.model(input_g)
            probability_g = nn.Softmax(dim=1)(logits_g)
        else:
            logits_g = self.model([input_g ,out1, out2, out3, out4])
            probability_g = nn.Softmax(dim=1)(logits_g)

        CE = nn.CrossEntropyLoss(weight = self.weights, reduction='none')
        if "focal" in self.cli_args.loss.lower():
            loss_g = focal_loss(CE(logits_g, label_g), label_g, self.t_gamma, self.t_alpha)
        else:
            loss_g = CE(logits_g, label_g)

        #beta = 0.1
        #loss_fn =  AUCMLoss()
        #loss_g = loss_fn(logits_g, label_g)
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)
        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g.detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = probability_g[:, 1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g.detach()
        if debug:
            print(logits_g)
            print(label_g)
            print(probability_g[:, 1])
            print(loss_g)
            print("***")

        return loss_g.mean()

    def log_metrics(
            self,
            fold,
            epoch_ndx,
            mode_str,
            metrics_t,
            classification_threshold=0.5,
    ):
        self.init_tensorboard_writers(fold)
        log.info("F{} E{} {}".format(
            fold,
            epoch_ndx,
            type(self).__name__,
        ))

        metrics_dict = process_metrics(metrics_t, classification_threshold)

        log.info(
            ("F{} E{} {:8} {loss/all:.4f} loss, "
             + "{correct/all:-5.1f}% correct, "
             ).format(
                fold,
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )
        log.info(
            ("F{} E{} {:8} {loss/neg:.4f} loss, "
             + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
             ).format(
                fold,
                epoch_ndx,
                mode_str + '_neg',
                **metrics_dict,
            )
        )
        log.info(
            ("F{} E{} {:8} {loss/pos:.4f} loss, "
             + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
             ).format(
                fold,
                epoch_ndx,
                mode_str + '_pos',
                **metrics_dict,
            )
        )

        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            if type(value) is float or type(value) is int:
                writer.add_scalar(key, value, self.totalTrainingSamples_count)

        writer.add_pr_curve(
            'pr',
            metrics_t[METRICS_LABEL_NDX],
            metrics_t[METRICS_PRED_NDX],
            self.totalTrainingSamples_count,
        )

        bins = [x / 50.0 for x in range(51)]

        neg_hist_mask = metrics_dict['neg_label_mask'] & (metrics_t[METRICS_PRED_NDX] > 0.01)
        pos_hist_mask = metrics_dict['pos_label_mask'] & (metrics_t[METRICS_PRED_NDX] < 0.99)

        if neg_hist_mask.any():
            writer.add_histogram(
                'is_neg',
                metrics_t[METRICS_PRED_NDX, neg_hist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )
        if pos_hist_mask.any():
            writer.add_histogram(
                'is_pos',
                metrics_t[METRICS_PRED_NDX, pos_hist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )

    def tune_train(self, trial):

        # Save the study status
        joblib.dump(self.study, os.path.join(self.output_dir, 'tuning_study.pkl'))

        # Initialize tuneable params
        self.init_tune(trial)
        
        # Model initialisation
        #if self.fix_nlayers:
        self.model = generate_model(10)#self.init_model(sample=self.sample)

        # Save the initial state to reproduce the tuning value
        model_path = os.path.join(self.output_dir,
                                      f"model_init_{trial.number}.pth")
        torch.save(self.model.state_dict(), model_path)
        self.init_dict[trial.number] = model_path

        # If model is using cnn_finetune, we need to update the transform with the new
        # mean and std deviation values
        try:
            dpm = self.model if not self.use_cuda else self.model.module
        except nn.modules.module.ModuleAttributeError:
            dpm = self.model

        if hasattr(dpm, "original_model_info"):
            log.info('*******************USING PRETRAINED MODEL*********************')
            mean = dpm.original_model_info.mean
            std = dpm.original_model_info.std
            train_ds, val_ds, self.train_dl, self.val_dl = self.init_data(2, mean=mean, std=std)

        # Optimizer
        self.optimizer, self.scheduler = self.init_optimizer()
        log.info('Optimizer initialized')

        # Early stopping class tracks the best validation loss
        es = EarlyStopping(patience=self.patience)

        # Training loop
        val_metrics_t = None
        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            trn_metrics_t = self.do_training(0, epoch_ndx, self.train_dl)
            self.log_metrics(0, epoch_ndx, 'trn', trn_metrics_t)
            val_metrics_t = self.do_validation(0, epoch_ndx, self.val_dl)
            self.log_metrics(0, epoch_ndx, 'val', val_metrics_t)
            val_loss = val_metrics_t[METRICS_LOSS_NDX].mean()
            #obj, _, _ = roc_and_auc(val_metrics_t[METRICS_PRED_NDX].numpy(),
            #                        val_metrics_t[METRICS_LABEL_NDX].numpy())
            es(val_loss)

            if self.cli_args.earlystopping:
                if es.early_stop:
                    break

        try:
            obj, _, _ = roc_and_auc(val_metrics_t[METRICS_PRED_NDX].numpy(),
                                    val_metrics_t[METRICS_LABEL_NDX].numpy())
        except ValueError:
            return None
        log.info(f"Calculated objective: {obj:.3f}")
        return - obj

    def tune(self):
        log.info('Initializing model and data')

        # Data
        mean, std = norms[self.cli_args.norm]
        train_ds, val_ds, self.train_dl, self.val_dl = self.init_data(2, mean=mean, std=std)

        # Get a sample batch
        sample = []
        for n, point in enumerate(self.train_dl):
            if n == 1:
                break
            x = point[0]
            sample.append(x)
        self.sample = torch.cat(sample, dim=0)

        # Generate weights
        self.weights = get_class_weights(train_ds)
        self.weights = self.weights.to(self.device)

        # Model initialisation
        if not self.fix_nlayers:
            self.model = self.init_model(sample=self.sample)
        
            # If model is using cnn_finetune, we need to update the transform with the new
            # mean and std deviation values
            try:
                dpm = self.model if not self.use_cuda else self.model.module
            except nn.modules.module.ModuleAttributeError:
                dpm = self.model

            if hasattr(dpm, "original_model_info"):
                log.info('*******************USING PRETRAINED MODEL*********************')
                mean = dpm.original_model_info.mean
                std = dpm.original_model_info.std
                train_ds, val_ds, self.train_dl, self.val_dl = self.init_data(0, mean=mean, std=std)

            log.info('*******************NORMALISATION DETAILS*********************')
            log.info(f"preprocessing mean: {mean}, std: {std}")

        self.study = optuna.create_study(study_name=self.cli_args.exp_name, sampler=TPESampler(seed=42))
        self.study.optimize(self.tune_train, n_jobs=1, n_trials=self.cli_args.num_trials)

        # Save best params
        print(f"Best config: {self.study.best_params}")

        # Save only best initialization
        for k, v in self.init_dict.items():
            if k != self.study.best_trial.number:
                os.remove(v)
            else:
                model_path = os.path.join(self.output_dir, "model_init_best.pth")
                os.rename(v, model_path)

        bp_name = os.path.join(self.output_dir, 'best_params.json')
        with open(bp_name, 'w') as f:
            json.dump(self.study.best_params, f, indent=2)


class Trainer(TrainerBase):
    def __init__(self):
        super().__init__()

    def init_model(self, sample=None):
        model = model_choice(self.cli_args.model,
                             resume=self.cli_args.resume, sample=sample,
                             pretrain_loc=self.cli_args.pretrain_loc,
                             pretrained_2d_name=self.cli_args.pretrained_2d_name,
                             depth=self.cli_args.rn_depth,
                             n_classes=self.cli_args.rn_nclasses, fix_inmodel=self.fix_nlayers)

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
                                                mean=mean, std=std, dim=self.cli_args.dim,                limit=self.cli_args.dset_lim,  external=MRIDataset)
        else:
            train_ds, val_ds = get_data(self.cli_args.dset, self.cli_args.datapoints, fold, aug=aug,
                                        dim=self.cli_args.dim, limit=self.cli_args.dset_lim,external=MRIDataset)
        train_dl, val_dl = self.init_loaders(train_ds, val_ds)
        return train_ds, val_ds, train_dl, val_dl

    def init_tune(self, trial):
        self.t_learnRate = trial.suggest_loguniform('learnRate', 1e-6, 1e-3)
        self.t_decay = trial.suggest_uniform('decay', 0.9, 0.99)
        self.t_alpha = trial.suggest_uniform('focal_alpha', 0.5, 3.0)
        self.t_gamma = trial.suggest_uniform('focal_gamma', 0.5, 5.0)
        self.patience = trial.suggest_int('earlystopping', 3, 6)
        if self.fix_nlayers:
            self.fix_nlayers = trial.suggest_int('fix_nlayers', 3, 6)

        return


if __name__ == '__main__':
    Trainer().main()