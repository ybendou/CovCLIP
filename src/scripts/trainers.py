import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
from functools import partial
import math
from classifiers import get_net
import copy
from joblib import Parallel, delayed
from collections import OrderedDict
import re
import json
from optim import build_lr_scheduler
class LogitDataset():
    def __init__(self, features, labels) -> None:
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
class SubsetSemanticDataset_ClassLevel():
    """
    Return a subset of a dataset with class-level features
    """
    def __init__(self, dataset, indices):
        self.features_text = dataset.features_text[indices] 
        self.class_centers = dataset.class_centers[indices]
        self.class_std = dataset.class_std[indices]
        self.normalization = dataset.normalization
    def __getitem__(self, idx):
        """
        idx now is at image level, sample a single image query then add mean class feature and covariance 
        """
        text = (self.features_text[idx]-self.normalization['features_text']['mean']) / self.normalization['features_text']['std']
        class_center = (self.class_centers[idx]-self.normalization['class_centers']['mean']) / self.normalization['class_centers']['std']
        class_std = (self.class_std[idx]-self.normalization['class_std']['mean']) / self.normalization['class_std']['std']
        return text, torch.tensor([]), torch.tensor([]), class_center, class_std, idx
    def __len__(self):
        return self.class_centers.shape[0]

class SubsetSemanticDataset():
    def __init__(self, dataset, indices):
        self.features_text = dataset.features_text[indices] 
        self.class_centers = dataset.class_centers[indices]
        self.class_std = dataset.class_std[indices]
        self.normalization = dataset.normalization
        self.imconditionning = dataset.imconditionning
        self.features_images = [dataset.features_images[i.item()] for i in indices] # list of image features per class
        self.labels = torch.arange(len(self.features_images)).repeat_interleave(torch.tensor([len(x) for x in self.features_images]))
        self.features_images = torch.cat(self.features_images, dim=0)
    def __getitem__(self, idx):
        """
        idx now is at image level, sample a single image query then add mean class feature and covariance 
        """
        label = self.labels[idx]
        text = (self.features_text[label]-self.normalization['features_text']['mean']) / self.normalization['features_text']['std']
        image = (self.features_images[idx]-self.normalization['features_images']['mean']) / self.normalization['features_images']['std'] if self.imconditionning else torch.tensor([])
        query = self.features_images[idx]
        class_center = (self.class_centers[label]-self.normalization['class_centers']['mean']) / self.normalization['class_centers']['std']
        class_std = (self.class_std[label]-self.normalization['class_std']['mean']) / self.normalization['class_std']['std']
        return text, image, query, class_center, class_std, label
    def __len__(self):
        return self.labels.shape[0]

from scipy.linalg import sqrtm, inv, logm, expm
def average_covariance(covariances, method='euclidean'):
    """
    -covariances: tensor of shape (n_classes, n_features, n_features)
    """
    if method == 'euclidean':
        return covariances.mean(dim=0)
    elif method == 'log_euclidean':
        return log_euclidean_mean(covariances)
    elif method == 'harmonic':
        return harmonic_mean(covariances)
    elif method == 'geometric':
        return geometric_mean(covariances)
    else:
        raise NotImplementedError
def geometric_mean(covariance_matrices):
    n = covariance_matrices.shape[0]
    geometric_product = torch.prod(covariance_matrices, dim=0)
    geometric_mean_matrix = torch.pow(geometric_product, 1/n)
    return geometric_mean_matrix
def log_euclidean_mean(covariance_matrices):
    cov_logms  = []
    for cov in covariance_matrices:
        cov_logms.append(torch.tensor(logm(sqrtm(cov))))
    cov_logms = torch.stack(cov_logms)
    mean_log_matrix = torch.mean(cov_logms, dim=0)
    euclidean_mean_matrix = torch.matrix_exp(mean_log_matrix)
    return euclidean_mean_matrix
def harmonic_mean(covariance_matrices):
    inverse_matrices = torch.inverse(covariance_matrices)
    harmonic_mean_matrix = torch.inverse(torch.mean(inverse_matrices, dim=0))
    return harmonic_mean_matrix

def interpolate_covariance(C1,C2, beta, method='euclidean'):
    if method=='euclidean':
        return beta*C1 + (1-beta)*C2
    elif method=='log_euclidean':
        return log_euclidean_interpolation(C1, C2, beta)
    elif method=='affine_invariant_riemannian_metric':
        return affine_invariant_riemannian_metric_interpolation(C1, C2, t)
    else:
        raise NotImplementedError
def log_euclidean_interpolation(C1, C2, t):
    # Perform log-Euclidean interpolation between two covariance matrices
    # C1 and C2 with a parameter t ranging between 0 and 1.
    
    # Compute the matrix logarithm of C1 and C2
    A = sqrtm(C1)
    B = sqrtm(C2)
    A_log = logm(A)
    B_log = logm(B)
    
    # Interpolate the matrix logarithms
    interpolated_log = (1 - t) * A_log + t * B_log
    
    # Exponentiate the interpolated logarithm to obtain the interpolated covariance matrix
    interpolated_covariance = expm(interpolated_log)
    
    return interpolated_covariance
def affine_invariant_riemannian_metric_interpolation(C1, C2, t):
    # Perform interpolation using the affine-invariant Riemannian metric
    # between two covariance matrices C1 and C2 with a parameter t ranging between 0 and 1.
    
    # Compute the matrix logarithm of C1 and C2
    A = sqrtm(C1)
    B = sqrtm(C2)
    A_inv = inv(A)
    A_log = logm(A_inv @ B @ A_inv)
    
    # Interpolate the matrix logarithms
    interpolated_log = t * A_log
    
    # Exponentiate the interpolated logarithm to obtain the interpolated covariance matrix
    interpolated_covariance = A @ expm(interpolated_log) @ A
    
    return interpolated_covariance

# import logging
# logger = logging.getLogger(__name__)
from loguru import logger
# define optimizer
def get_optimizer(name, model, lr, wd, momentum):
    if name.lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name.lower() == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name.lower() == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
def get_scheduler(name, optimizer, max_n_steps, lr, end_lr):
    if len(name) == 0 or name == "" or name == '': 
        return None
    if name.lower() == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=max_n_steps, eta_min=lr*end_lr)
    elif name.lower() == 'linear':
        return torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, end_lr=end_lr, max_steps=max_n_steps)
    else:
        return None
    
criterion = nn.MSELoss()

class Template():
    def __init__(self, args, **kwargs) -> None:
        self.args = args

    def train(self, train_dataset, val_dataset, test_dataset, *args, **kwargs) -> None:
        pass

    def predict(self, shots, text, alpha=0, beta=0) -> torch.tensor:
        pass
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def get_path(args, name):
    if args.training_dataset == 'imagenet21k':
        backbone_images = args.features_path_train_images[0].split('/')[-1].replace('.pt', '')
    else:
        backbone_images = args.features_path_train_images.split('/')[-1].replace('.pt', '')
    backbone_text = args.features_path_text.split('/')[-1].replace('.pt', '')
    WORKDIR = 'logs/'
    make_dir(WORKDIR)
    if args.training_dataset == args.test_dataset[0]:
        path = os.path.join(WORKDIR, 'indomain')
    else:
        path = os.path.join(WORKDIR, 'crossdomain')
    make_dir(path)
    path = os.path.join(path, args.training_dataset)
    # make dir if not exists
    make_dir(path)
    path = os.path.join(path, backbone_images)
    make_dir(path)
    path = os.path.join(path, backbone_text)
    make_dir(path)
    path = os.path.join(path, name)
    make_dir(path)
    hyperparam_str = f'net_{args.net}-lr_{args.lr}-endlr_{args.end_lr_factor}-wd_{args.wd}-bs_{args.batch_size}-epochs_{args.epochs}-momentum_{args.mmt}-optimizer_{args.optimizer}-scheduler_{args.scheduler}-batchnorm_{args.batch_norm}-seed_{args.seed}'
    path = os.path.join(path, hyperparam_str)
    make_dir(path)
    network_path = os.path.join(path, f'{name}.pth')
    return network_path, path, hyperparam_str

class NCM(Template):
    def __init__(self, *args, **kwargs) -> None:
        super(NCM, self).__init__(*args, **kwargs)
        self.normalizitation = kwargs.get('normalizitation', None)
        self.rho = kwargs.get('rho', 1)
        self.train_dataset = kwargs.get('train_dataset', None)
        self.val_dataset = kwargs.get('val_dataset', None)
        self.imconditionning = kwargs.get('imconditionning', False)
        self.run_wandb = kwargs.get('run_wandb', False)
        self.train_mean = kwargs.get('train_mean', False)
        self.train_std = kwargs.get('train_std', False)
        self.interpolate_mean = self.train_mean
        self.interpolate_std = self.train_std 
        self.default_alpha = kwargs.get('default_alpha', 0)
        self.default_beta = kwargs.get('default_beta', 0)
        self.preprocess = kwargs.get('preprocess', False)
        self.cov_shots = kwargs.get('cov_shots', False)
        self.test_dataset = kwargs.get('test_dataset', None)
        self.mean_logs = {}
        self.std_logs = {}
        input_size = self.train_dataset.class_centers.shape[-1]
        text_size = self.train_dataset.features_text.shape[-1]
        if self.train_mean:
            # check if already exists
            network_path, path, hyperparam_str = get_path(self.args, 'mean')
            self.log_mean_path = path
            # get net: 
            self.net_mean = get_net(self.args.net)(input_size, text_size, activation=self.args.activation, layer_norm=self.args.layer_norm, dropout=self.args.dropout, batch_norm=self.args.batch_norm, embed_dim=self.args.embed_dim, imconditionning=self.imconditionning, target_name='mean')
            if self.args.load_mean_net != '' and os.path.exists(self.args.load_mean_net):
                net_mean_dict = torch.load(self.args.load_mean_net, map_location='cpu')
            else: 
                if os.path.exists(network_path):
                    logger.info(f'Network already exists, loading from: {network_path}')
                    net_mean_dict = torch.load(network_path, map_location='cpu')
                else:
                    logger.debug(f'Starting training {hyperparam_str}')
                    net_mean_dict, epoch_train_loss, best_val_loss, baseline_val_loss = self.train(self.net_mean, target_name='mean', imconditionning=self.imconditionning, train=not self.args.no_train and not (self.args.load_mean_net != '' and os.path.exists(self.args.load_mean_net)))
                    # save the net in path: 
                    torch.save(net_mean_dict, network_path)
                    # save logs of results 
                    self.mean_logs['epoch_train_loss'] = epoch_train_loss
                    self.mean_logs['best_val_loss'] = best_val_loss
                    self.mean_logs['baseline_val_loss'] = baseline_val_loss
            self.net_mean.load_state_dict(net_mean_dict)                
            self.net_mean.eval()

        if self.train_std:
            # check if already exists
            name = 'covariance' if self.args.covariance_form=='full' else 'std'
            if self.train_mean : 
                self.label = '$\mu_{v+t}, \sigma_{v+t}$'
            else:
                self.label = '$\mu_{v}, \sigma_{v+t}$'
            # check if file exists
            
            self.net_std = get_net(self.args.net)(input_size, text_size, activation=self.args.activation, layer_norm=self.args.layer_norm, dropout=self.args.dropout, batch_norm=self.args.batch_norm, embed_dim=self.args.embed_dim, imconditionning=self.imconditionning, target_name='std')
            if self.args.load_std_net != '' and os.path.exists(self.args.load_std_net):
                net_std_dict = torch.load(self.args.load_std_net, map_location='cpu')
            else:
                network_path, path, hyperparam_str = get_path(self.args, 'std')
                self.log_std_path = path
                
                if os.path.exists(network_path):
                    logger.info(f'Network already exists, loading from: {network_path}')
                    net_std_dict = torch.load(network_path, map_location='cpu')
                else:
                    net_std_dict, epoch_train_loss, best_val_loss, baseline_val_loss = self.train(self.net_std, target_name=name, imconditionning=self.imconditionning, train=not self.args.no_train and not (self.args.load_std_net != '' and os.path.exists(self.args.load_std_net)))
                    # save the net in path:
                    torch.save(net_std_dict, network_path)
                    # save logs of results 
                    self.std_logs['epoch_train_loss'] = epoch_train_loss
                    self.std_logs['best_val_loss'] = best_val_loss
                    self.std_logs['baseline_val_loss'] = baseline_val_loss
            self.net_std.load_state_dict(net_std_dict)
            self.net_std.eval()
        else:
            self.label = '$\mu_{v+t}, I$' if self.train_mean else '$\mu_{v}, I$'
    def train(self, 
              net, 
              target_name: str, 
              imconditionning: bool=False, 
              train: bool=True
              ) -> None:
        """
        Train a network to predict either the mean or the std of the class given the text embedding
        Arguments:
        - target_name: str = either 'mean' or 'std'
        - imconditionning: bool = if True, the network is conditioned on the image input
        Returns:
        - net: nn.Module = the trained network
        """
        logger.debug(f'Starting training of {target_name} network')
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.args.training_batch_size, shuffle=True, num_workers = min(os.cpu_count(), 8), drop_last=False) 
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.args.validation_batch_size, shuffle=False, num_workers = min(os.cpu_count(), 8)) 
        test_loader = [torch.utils.data.DataLoader(self.test_dataset[t], batch_size=self.args.validation_batch_size, shuffle=False, num_workers = min(os.cpu_count(), 8)) for t in range(len(self.test_dataset))] if self.test_dataset is not None else None

        net = net.to(self.args.training_device)
        optimizer = get_optimizer(self.args.optimizer, net, self.args.lr, self.args.wd, self.args.mmt) # only way to actually copy the parameters of a model in torch, otherwise it's just a reference...
        scheduler = get_scheduler(self.args.scheduler, optimizer, self.args.epochs, self.args.lr, self.args.end_lr_factor)

        best_val_loss, baseline_val_loss, best_test_loss_on_val = torch.inf, torch.inf, [torch.inf]*len(self.test_dataset)
        epoch_train_loss = 0.
        for epoch in range(self.args.epochs):
            total_count = 0.
            epoch_train_loss = 0.
            train_iterator = enumerate(train_loader)
            _=net.train()
            for batch_idx, (text, image, _, class_center, class_std, _) in train_iterator:
                optimizer.zero_grad()
                text = text.squeeze(1).to(self.args.training_device).float()
                class_center = class_center.to(self.args.training_device).float()
                class_std = class_std.to(self.args.training_device).float()
                image = image.to(self.args.training_device).float()
                # forward pass
                batch_size = text.shape[0]
                pred = net(text, image)
                loss = criterion(pred, target=class_center if target_name == 'mean' else class_std)
                loss_is_nan = torch.isnan(loss)

                if not loss_is_nan:
                    loss.backward()
                    if self.args.grad_clipping >0:
                        torch.nn.utils.clip_grad_norm(net.parameters(), self.args.grad_clipping)
                    optimizer.step()
                total_count += batch_size
                if loss_is_nan:
                    logger.warning('Loss is nan, stopping training')
                    break # if loss is nan we stop the training
                epoch_train_loss += loss.item()*batch_size
                if self.args.debug: 
                    print(f'{batch_idx}/{len(train_loader)}, Train Loss:{loss.item():.3f}', end='\r')
            epoch_train_loss /= total_count
            if scheduler is not None:
                scheduler.step()
            with torch.no_grad():
                val_loss, baseline_val_loss = self.evaluate(net, self.val_dataset, val_loader, criterion, target_name)
                val_update = val_loss<best_val_loss
                if val_update: 
                    best_val_loss = val_loss
                    best_net_dict = net.to('cpu').state_dict().copy()
                if self.test_dataset is not None: 
                    test_log = ''
                    for t in range(len(self.test_dataset)):
                        test_loss, baseline_test_loss = self.evaluate(net, self.test_dataset[t], test_loader[t], criterion, target_name)
                        if val_update: 
                            best_test_loss_on_val[t] = test_loss
                        test_log += f' | test name: {self.test_dataset[t].name}, test loss: {test_loss:.3f}, best test loss on val: {best_test_loss_on_val[t]:.3f}, baseline test loss: {baseline_test_loss:.3f} ||'
            if self.args.wandb:
                self.run_wandb.log({f'{target_name} epoch':epoch, f'{target_name} epoch train loss':epoch_train_loss, f'{target_name} epoch best val loss':best_val_loss, f'{target_name} epoch baseline val loss':baseline_val_loss, f'{target_name} lr':scheduler.get_last_lr()[0] if scheduler is not None else self.args.lr})  
            logger.info(f'Epoch: {epoch+1:03d} | lr={scheduler.get_last_lr()[0] if scheduler is not None else self.args.lr:.4f} | Training loss: {epoch_train_loss:.2f} | Val loss={val_loss:.2f} | Best Val loss={best_val_loss:.2f} | Baseline Val loss: {baseline_val_loss:.2f}' + test_log + ' '*10)
        return best_net_dict, epoch_train_loss, best_val_loss, baseline_val_loss
    
    def evaluate(self, 
                 net, 
                 dataset, 
                 loader, 
                 criterion, 
                 target_name
                ):
        iterator = enumerate(loader)
        # evaluate classifier
        class_center_mean_dataset = self.train_dataset.class_centers.mean(0)
        class_std_mean_dataset = self.train_dataset.class_std.mean(0)
        net = net.to(self.args.training_device)
        net.eval()
        loss = 0.
        loss_baseline = 0.
        total_count = 0.
        for _, (text, image, _, class_center, class_std, _) in iterator:
            text = text.squeeze(1).to(self.args.training_device).float()
            class_center = class_center.float()
            class_std = class_std.float()
            image = image.to(self.args.training_device).float()

            batch_size = text.shape[0]
            # forward pass
            pred = net(text, image)
            target = class_center if target_name == 'mean' else class_std
            mean_pred_baseline, std_pred_baseline = class_center_mean_dataset.repeat(batch_size,1), class_std_mean_dataset.repeat(batch_size,1)
            baseline_pred = mean_pred_baseline if target_name == 'mean' else std_pred_baseline
            loss+=(criterion(pred, target.to(self.args.training_device))*batch_size).cpu()
            loss_baseline += (criterion(baseline_pred, target)*batch_size).cpu()  # do computation on cpu to save gpu memory
            total_count += batch_size
        loss /= total_count
        loss_baseline /= total_count
        return loss.item(), loss_baseline.item()
    def predict_multi_class(self, 
                            shots, 
                            validation_shots, 
                            test_dataset, 
                            test_loader, 
                            datasetname, 
                            mean_pred_cached=None, 
                            std_pred_cached=None, 
                            cache=False, 
                            predict=True, 
                            W_weights=None, 
                            openset=False
                            ) -> torch.tensor:
        """
        Runs prediction on shots and the entire test set
        Predicts mean and std out of text + image if necessary then interpolates between predictions from ncm and predictions from text+image
        Alpha and Beta are hyperparamters of the interpolation between predictions from text and predictions from image for the class mean and std respectively
        If the model is not conditionned on the image, then the prediction is done once at zero-shto and cached. 
        Inputs: 
            - shots: int or None = The number of shots to use for prediction. If None, then prediction is done with zero-shot (text only)
            - test_dataset: torch.utils.data.Dataset = The test dataset
            - alpha: float = The interpolation hyperparameter for the class mean
            - beta: float = The interpolation hyperparameter for the class std
            - mean_pred_cached: torch.tensor or None = The mean prediction at zero shot for each class cached when shots>0 else None
            - std_pred_cached: torch.tensor or None = The std prediction at zero shot for each class cached when shots> else None
        Outputs:
            - score: torch.tensor = The score of the prediction for each run
            - mean_pred_cached: torch.tensor or None = The mean prediction at zero shot for each class cached if shots==0 else None
            - std_pred_cached: torch.tensor or None = The std prediction at zero shot for each class cached if shots==0 else None
        """
        with torch.no_grad():
            if shots == None and cache: # if zero shot cache the prediction and return it
                net_mean = self.net_mean.to(self.args.device) if self.train_mean else None
                net_std = self.net_std.to(self.args.device) if self.train_std else None 
                mean_pred_cached, std_pred_cached = [], []
                # if evaluate on test set to get MSE loss of predictions 
                if self.args.evaluate_on_test_set and not openset:
                    criterion = nn.MSELoss()
                    if self.train_mean:
                        test_loss, baseline_test_loss = self.evaluate(net_mean, test_dataset, test_loader, criterion, 'mean')
                        logger.info(f'test loss={test_loss:.2f} | Baseline test loss: {baseline_test_loss:.2f}'+' '*10)
                        self.mean_logs[f'{datasetname} test loss:'] = test_loss
                        self.mean_logs[f'{datasetname} test baseline loss'] = baseline_test_loss
                        # read old logs if they exist
                        if os.path.exists(os.path.join(self.log_mean_path, f'logs_mean.json')):
                            with open(os.path.join(self.log_mean_path, f'logs_mean.json'), 'r') as f:
                                old_mean_logs = json.load(f)
                            # update old logs with new logs
                            old_mean_logs.update(self.mean_logs)
                        with open(os.path.join(self.log_mean_path, f'logs_mean.json'), 'w') as f:
                            json.dump(self.mean_logs, f)
                    if self.train_std:
                        test_loss, baseline_test_loss = self.evaluate(net_std, test_dataset, test_loader, criterion, 'std')
                        self.std_logs[f'{datasetname} test loss:'] = test_loss
                        self.std_logs[f'{datasetname} test baseline loss'] = baseline_test_loss
                        if os.path.exists(os.path.join(self.log_std_path, f'logs_std.json')):
                            with open(os.path.join(self.log_std_path, f'logs_std.json'), 'r') as f:
                                old_std_logs = json.load(f)
                            # update old logs with new logs
                            old_std_logs.update(self.std_logs)
                        with open(os.path.join(self.log_std_path, f'logs_std.json'), 'w') as f:
                            json.dump(self.std_logs, f)
                for batch_idx in range(math.ceil(test_dataset.features_text.shape[0]/self.args.batch_size)):
                    text = (test_dataset.features_text[batch_idx*self.args.batch_size:(batch_idx+1)*self.args.batch_size]-test_dataset.normalization['features_text']['mean']) / test_dataset.normalization['features_text']['std'] # normalize text 
                    text = text.to(self.args.device).float()
                    if self.train_mean: # send to gpu to make denormalization then send back to cpu
                        mean_pred_ = net_mean(text, None)*self.train_dataset.normalization['class_centers']['std'].to(self.args.device)+self.train_dataset.normalization['class_centers']['mean'].to(self.args.device)
                        mean_pred_cached.append(mean_pred_.cpu())
                    if self.train_std:
                        std_pred_ = net_std(text, None)*self.train_dataset.normalization['class_std']['std'].to(self.args.device)+self.train_dataset.normalization['class_std']['mean'].to(self.args.device)
                    else:
                        std_pred_ = torch.ones_like(self.train_dataset.normalization['class_std']['mean']).repeat(text.shape[0],1) # (n_classes, dim)
                    std_pred_cached.append(std_pred_.cpu())
                mean_pred_cached = torch.cat(mean_pred_cached, dim=0) if self.train_mean else torch.tensor(torch.nan) # shape (n_classes, dim) 
                std_pred_cached = torch.cat(std_pred_cached, dim=0)# shape (n_classes, dim) 

            if predict and (shots!=None or self.train_mean) and not openset: # if we want to predict, else is for openset problems where we only want to cache the mean and covariance predictions
                alphas = (torch.arange(0, 1.05, 0.05) if self.interpolate_mean and self.default_alpha==0 else torch.tensor([self.default_alpha]).float()).to(self.args.device)
                betas = (torch.arange(0, 1.05, 0.05) if self.interpolate_std and self.default_beta==0 else torch.tensor([self.default_beta]).float()).to(self.args.device)
                mean_pred_cached = mean_pred_cached.to(self.args.device) if self.train_mean else None
                std_pred_cached = std_pred_cached.to(self.args.device)
                if shots!=None:
                    run_batch_size, n_ways, n_shots, dim = shots.shape
                    n_val_shots = validation_shots.shape[2]
                    centroids = shots.mean(2).to(self.args.device) # (n_runs, n_classes, dim); shots: (n_runs, n_classes, n_shots, dim)
                    if self.cov_shots and shots.shape[2]>1:  
                        stds = centroids.std(2).to(self.args.device) # (n_runs, n_classes, dim)
                    else:
                        stds = torch.ones(centroids.shape[1:]).to(self.args.device)# (n_classes, dim)
                    test_iterator = enumerate(test_loader)
                    targets_validation = torch.arange(n_ways).repeat_interleave(n_val_shots).unsqueeze(0).repeat(run_batch_size, 1).to(self.args.device) # (n_runs, n_classes*n_val_shots) 
                    # get best beta and alpha is shots != None using the validation shots
                    combinations = torch.cartesian_prod(alphas, betas)# get combinations of alpha and beta
                    best_val_scores = torch.zeros(run_batch_size, device=self.args.device)-1
                    best_alpha, best_beta = torch.tensor([torch.nan]*run_batch_size).to(self.args.device), torch.tensor([torch.nan]*run_batch_size).to(self.args.device)
                    for (alpha, beta) in combinations:
                        mean_mixed = alpha*mean_pred_cached.unsqueeze(0) + (1-alpha)*centroids if self.train_mean else centroids# (n_runs, n_classes, dim); validation shots: (n_runs, n_classes, n_val_shots, dim) -> (n_runs, n_classes*n_val_shots, dim)
                        if self.train_std:
                            if self.args.covariance_form == 'diagonal':
                                std_mixed = (beta*std_pred_cached.unsqueeze(0) + (1-beta)*stds.unsqueeze(0)).mean(1)# (n_runs, dim)
                            else:
                                std_mixed = interpolate_covariance(std_pred_cached.reshape(n_ways, dim, dim), torch.eye(dim).to(self.args.device), beta=beta)
                                std_mixed = average_covariance(std_mixed)
                        else:
                            std_mixed = stds.unsqueeze(0).mean(1)
                        
                        if self.args.covariance_form == 'diagonal' or not self.train_std:
                            predictions = torch.norm((mean_mixed.unsqueeze(2)-validation_shots.reshape(run_batch_size, 1, -1, dim))/std_mixed.reshape(1, 1, 1, -1), dim=-1).argmin(1) # (n_runs, n_val_shots*n_classes)
                        else:
                            X = (mean_mixed.unsqueeze(2)-validation_shots.reshape(run_batch_size, 1, -1, dim))
                            predictions = torch.einsum('bcvd,bcvd->bcv', X@torch.inverse(std_mixed), X).sqrt().argmin(1)
                        scores_one_combination = predictions.eq(targets_validation).sum(1).float()/(n_val_shots*n_ways)
                        # update best alpha and beta
                        for b in range(run_batch_size):
                            if scores_one_combination[b] > best_val_scores[b]:
                                best_alpha[b] = alpha
                                best_beta[b] = beta
                                best_val_scores[b] = scores_one_combination[b]
                    mean_mixed =  best_alpha.reshape(-1, 1, 1)*mean_pred_cached.unsqueeze(0) + (1-best_alpha.reshape(-1, 1, 1))*centroids if self.train_mean else centroids
                    if self.args.covariance_form == 'diagonal':
                        std_mixed = ((best_beta.reshape(-1, 1, 1)*std_pred_cached.unsqueeze(0) + (1-best_beta.reshape(-1, 1, 1))*stds) if self.train_std else stds.unsqueeze(0).repeat(run_batch_size, 1, 1))
                        std_mixed = std_mixed.mean(1)
                    else:
                        std_mixed = interpolate_covariance(std_pred_cached.reshape(n_ways, dim, dim), torch.eye(dim).to(self.args.device), beta=beta)
                        std_mixed = average_covariance(std_mixed)
                else:
                    run_batch_size = 1
                    n_ways, dim = mean_pred_cached.shape
                    best_alpha, best_beta = torch.tensor([torch.nan]), torch.tensor([torch.nan])
                    mean_mixed = mean_pred_cached.unsqueeze(0)
                    std_mixed = std_pred_cached.unsqueeze(0).mean(1)

                # predict the class mean and std from text
                test_iterator = enumerate(test_loader)
                scores = torch.zeros(run_batch_size, device=self.args.device)
                total_count = 0.
                # mixed_mean: (n_runs, n_classes, dim); mixed_std: (n_runs, dim); query: (batch_size, dim)
                for batch_idx, (text, _, query, _, _, label) in test_iterator:
                    batch_size = query.shape[0]
                    query = query.squeeze(1).to(self.args.device).float() # (batch_size, dim)
                    label = label.to(self.args.device) # (batch_size)
                    # mean and std are computed once at zero-shot and cached
                    if self.args.covariance_form == 'diagonal' or not self.train_std:
                        predictions = torch.norm((query.reshape(1, 1, batch_size, dim)-mean_mixed.reshape(run_batch_size, n_ways, 1, dim))/(std_mixed.reshape(run_batch_size, 1, 1, dim)), dim=-1).argmin(1) # (n_runs, n_classes)
                    else:
                        X = (query.reshape(1, 1, batch_size, dim)-mean_mixed.reshape(run_batch_size, n_ways, 1, dim))
                        predictions = torch.einsum('bcvd,bcvd->bcv', X@torch.inverse(std_mixed), X).sqrt().argmin(1)
                    scores += (predictions==label.unsqueeze(0).repeat(run_batch_size, 1)).sum(dim=1)
                    total_count += label.shape[0]
                scores /= total_count
                return scores, mean_pred_cached, std_pred_cached, best_alpha, best_beta
            else:
                return torch.tensor(torch.nan).to(self.args.device), mean_pred_cached.to(self.args.device), std_pred_cached.to(self.args.device), None, None

class Clip(Template):
    def __init__(self, *args, **kwargs) -> None:
        super(Clip, self).__init__(*args, **kwargs)
        self.raw = kwargs.get('raw', False)
        self.label = 'raw '*self.raw + 'clip'        
    def predict_multi_class(self, shots, validation_shots, test_dataset, test_loader, datasetname, mean_pred_cached=None, std_pred_cached=None, cache=False, predict=True, W_weights=None) -> torch.tensor:
        """
        Predictions using NCM , includes batched version of runs
        """
        # get text features to generate centroids
        
        if predict:
            centroids = (test_dataset.raw_clip if self.raw else test_dataset.features_text).to(self.args.device).float() if mean_pred_cached is None else mean_pred_cached# (n_ways, d)
            run_batch_size = 1 
            test_iterator = enumerate(test_loader)
            score = torch.zeros(run_batch_size, device=self.args.device)
            total_count=0.
            
            for _, (_, _, query, _, _, label) in test_iterator:
                query = query.to(self.args.device).float() # (n_queries, dim)
                label = label.to(self.args.device).float() # (n_queries)
                distances = torch.einsum('qd,nd->qn', query, centroids)/(torch.norm(query, dim=1).unsqueeze(1)*torch.norm(centroids, dim=1).unsqueeze(0))
                # distances = torch.norm(query.unsqueeze(0)-centroids.unsqueeze(2), dim = -1)
                #logger.debug(f'distances shape: {distances.shape}')
                winners = distances.argmax(dim=1)
                #logger.debug(f'winners shape: {winners.shape}')
                score += (winners==label.unsqueeze(0).repeat(run_batch_size, 1)).sum(dim=1)
                total_count += label.shape[0]
            return (score/total_count), None, None, torch.tensor([torch.nan]), torch.tensor([torch.nan])
        else:
            return torch.tensor(torch.nan).to(self.args.device), centroids, torch.ones_like(centroids).to(self.args.device), None, None

  
class Logit(Template):
    def __init__(self, *args, **kwargs) -> None:
        super(Logit, self).__init__(*args, **kwargs)
        self.normalizitation = kwargs.get('normalizitation', None)
        self.rho = kwargs.get('rho', 1)
        self.train_dataset = kwargs.get('train_dataset', None)
        self.val_dataset = kwargs.get('val_dataset', None)
        self.imconditionning = kwargs.get('imconditionning', False)
        self.run_wandb = kwargs.get('run_wandb', False)
        self.train_mean = kwargs.get('train_mean', False)
        self.train_std = kwargs.get('train_std', False)
        self.interpolate_mean = False
        self.interpolate_std = self.train_std 
        self.default_alpha = kwargs.get('default_alpha', 0.)
        self.default_beta = kwargs.get('default_beta', 0.)
        self.generate_shots = kwargs.get('generate_shots', False)
        self.init_text = kwargs.get('init_text', False)
        self.normalize = kwargs.get('normalize', False)
        logit_scale = 4.60517 #float(np.log(1 / 0.07))
        self.scale = kwargs.get('scale', torch.FloatTensor([logit_scale]).exp()).to(self.args.device) if self.normalize else 1
        self.trainCfg = {'optimizer':'adamw', 'lr':1e-3, 'mmt':0.9, 'epochs':200, 'batch_size':10, 'scheduler':'cosine', 'wd':1e-2, 'end_lr':1, 'steps':12800, 'eval_freq': 100} #  nb de steps cte peut importe le nb de classes 
        if self.train_mean:
            self.net_mean = self.train(target_name='mean', imconditionning=self.imconditionning, train=not self.args.no_train and not (self.args.load_mean_net != '' and os.path.exists(self.args.load_mean_net)))
            if self.args.load_mean_net != '' and os.path.exists(self.args.load_mean_net):
                self.net_mean_weights = torch.load(self.args.load_mean_net)
                self.net_mean.load_state_dict(self.net_mean_weights)
            self.net_mean.eval()
        if self.train_std:
            self.label = '$lp_{v+t}, \Sigma_{v+t}$' if self.train_mean else '$lp_{v}, \Sigma_{v+t}$'
            name = 'covariance' if self.args.covariance_form=='full' else 'std'
            if self.train_mean : 
                self.label = '$\mu_{v+t}, \sigma_{v+t}$'
            else:
                self.label = '$\mu_{v}, \sigma_{v+t}$'
            self.net_std = self.train(target_name=name, imconditionning=self.imconditionning, train=not self.args.no_train and not (self.args.load_std_net != '' and os.path.exists(self.args.load_std_net)))
            if self.args.load_std_net != '' and os.path.exists(self.args.load_std_net):
                self.net_std_weights = torch.load(self.args.load_std_net)
                self.net_std.load_state_dict(self.net_std_weights)
            self.net_std.eval()
        else:
            self.label = '$lp_{v+t}, I$' if self.train_mean else '$lp_{v}, I$'
    def train(self, target_name, imconditionning=False, train=True) -> None:
        """
        Train a network to predict either the mean or the std of the class given the text embedding
        Arguments:
        - target_name: str = either 'mean' or 'std'
        - imconditionning: bool = if True, the network is conditioned on the image input
        Returns:
        - net: nn.Module = the trained network
        """
        logger.debug(f'Starting training of {target_name} network')
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.args.training_batch_size, shuffle=True, num_workers = min(os.cpu_count(), 8)) 
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.args.validation_batch_size, shuffle=True, num_workers = min(os.cpu_count(), 8)) 
        input_size = self.train_dataset.class_centers.shape[-1]
        text_size = self.train_dataset.features_text.shape[-1]

        if train:
            net = get_net(self.args.net)(input_size, text_size, activation=self.args.activation, layer_norm=self.args.layer_norm, dropout=self.args.dropout, batch_norm=self.args.batch_norm, embed_dim=self.args.embed_dim, imconditionning=imconditionning, target_name=target_name)
            net = net.to(self.args.training_device)
        
        best_net = get_net(self.args.net)(input_size, text_size, activation=self.args.activation, layer_norm=self.args.layer_norm, dropout=self.args.dropout, batch_norm=self.args.batch_norm, embed_dim=self.args.embed_dim, imconditionning=imconditionning, target_name=target_name) # only way to actually copy the parameters of a model in torch, otherwise it's just a reference...
        best_net = best_net.to(self.args.training_device)

        if train:
            optimizer = get_optimizer(self.args.optimizer, net, self.args.lr, self.args.wd, self.args.mmt) # only way to actually copy the parameters of a model in torch, otherwise it's just a reference...
            scheduler = get_scheduler(self.args.scheduler, optimizer, self.args.epochs, self.args.lr, self.args.end_lr_factor)
        # training loop
        if train:
            with torch.no_grad():
                best_net.copy_parameters(net)
        best_val_loss, baseline_val_loss = torch.inf, torch.inf
        if train:
            for epoch in range(self.args.epochs):
                total_count = 0.
                epoch_train_loss = 0.
                train_iterator = enumerate(train_loader)
                _=net.train()
                for batch_idx, (text, image, _, class_center, class_std, _) in train_iterator:
                    optimizer.zero_grad()
                    text = text.squeeze(1).to(self.args.training_device).float()
                    class_center = class_center.to(self.args.training_device).float()
                    class_std = class_std.to(self.args.training_device).float()
                    image = image.to(self.args.training_device).float()
                    # forward pass
                    batch_size = text.shape[0]
                    
                    pred = net(text, image)
                    loss = criterion(pred, target=class_center if target_name == 'mean' else class_std)
                    loss_is_nan = torch.isnan(loss)

                    if not loss_is_nan:
                        loss.backward()
                        if self.args.grad_clipping >0:
                            torch.nn.utils.clip_grad_norm(net.parameters(), self.args.grad_clipping)
                        optimizer.step()
                    total_count += batch_size
                    if loss_is_nan:
                        logger.warning('Loss is nan, stopping training')
                        break # if loss is nan we stop the training
                    epoch_train_loss += loss.item()*batch_size
                    if self.args.debug: 
                        print(f'{batch_idx}/{len(train_loader)}, Train Loss:{loss.item():.3f}', end='\r')
                epoch_train_loss /= total_count
                if scheduler is not None:
                    scheduler.step()
                
                with torch.no_grad():
                    val_loss, baseline_val_loss = self.evaluate(net, self.val_dataset, val_loader, criterion, target_name)
                    if val_loss<best_val_loss:
                        best_val_loss = val_loss
                        _=best_net.copy_parameters(net)
                
                if self.args.wandb:
                    self.run_wandb.log({f'{target_name} epoch':epoch, f'{target_name} epoch train loss':epoch_train_loss, f'{target_name} epoch best val loss':best_val_loss, f'{target_name} epoch baseline val loss':baseline_val_loss, f'{target_name} lr':scheduler.get_last_lr()[0] if scheduler is not None else self.args.lr})   
                logger.info(f'Epoch: {epoch+1:03d} | lr={scheduler.get_last_lr()[0] if scheduler is not None else self.args.lr:.4f} | Training loss: {epoch_train_loss:.2f} | Val loss={val_loss:.2f} | Best Val loss={best_val_loss:.2f} | Baseline Val loss: {baseline_val_loss:.2f}'+' '*10)
            if self.args.save_model and self.args.epochs>0:
                torch.save(best_net.state_dict(), os.path.join(self.args.save_model, self.run_wandb.name, f'epochs{self.args.epochs}{self.args.net}_{target_name}.pt'))
        return best_net
    def evaluate(self, net, dataset, loader, criterion, target_name):
        iterator = enumerate(loader)
        # evaluate classifier
        class_center_mean_dataset = dataset.class_centers.mean(0)
        class_std_mean_dataset = dataset.class_std.mean(0)
        net.eval()
        loss = 0.
        loss_baseline = 0.
        total_count = 0.
        for _, (text, image, _, class_center, class_std, _) in iterator:
            text = text.squeeze(1).to(self.args.training_device).float()
            class_center = class_center.float()
            class_std = class_std.float()
            image = image.to(self.args.training_device).float()

            batch_size = text.shape[0]
            # forward pass
            pred = net(text, image)
            target = class_center if target_name == 'mean' else class_std
            mean_pred_baseline, std_pred_baseline = class_center_mean_dataset.repeat(batch_size,1), class_std_mean_dataset.repeat(batch_size,1)
            baseline_pred = mean_pred_baseline if target_name == 'mean' else std_pred_baseline
            loss+=(criterion(pred, target.to(self.args.training_device))*batch_size).cpu()
            loss_baseline += (criterion(baseline_pred, target)*batch_size).cpu()  # do computation on cpu to save gpu memory
            total_count += batch_size
        loss /= total_count
        loss_baseline /= total_count
        return loss.item(), loss_baseline.item()
    def predict_multi_class(self, shots, validation_shots, test_dataset, test_loader, datasetname, mean_pred_cached=None, std_pred_cached=None, cache=False, predict=True, W_weights=None) -> torch.tensor:
        """
        Runs prediction on shots and the entire test set
        Predicts mean and std out of text + image if necessary then interpolates between predictions from ncm and predictions from text+image
        Alpha and Beta are hyperparamters of the interpolation between predictions from text and predictions from image for the class mean and std respectively
        If the model is not conditionned on the image, then the prediction is done once at zero-shto and cached. 
        Inputs: 
            - shots: int or None = The number of shots to use for prediction. If None, then prediction is done with zero-shot (text only)
            - test_dataset: torch.utils.data.Dataset = The test dataset
            - alpha: float = The interpolation hyperparameter for the class mean
            - beta: float = The interpolation hyperparameter for the class std
            - mean_pred_cached: torch.tensor or None = The mean prediction at zero shot for each class cached when shots>0 else None
            - std_pred_cached: torch.tensor or None = The std prediction at zero shot for each class cached when shots> else None
        Outputs:
            - score: torch.tensor = The score of the prediction for each run
            - mean_pred_cached: torch.tensor or None = The mean prediction at zero shot for each class cached if shots==0 else None
            - std_pred_cached: torch.tensor or None = The std prediction at zero shot for each class cached if shots==0 else None
        """
        with torch.no_grad():
            net_mean = self.net_mean.to(self.args.device) if self.train_mean else None
            net_std = self.net_std.to(self.args.device) if self.train_std else None
            if shots == None and cache: # if zero shot cache the prediction and return it
                mean_pred_cached, std_pred_cached = [], []
                for batch_idx in range(math.ceil(test_dataset.features_text.shape[0]/self.args.batch_size)):
                    text = (test_dataset.features_text[batch_idx*self.args.batch_size:(batch_idx+1)*self.args.batch_size]-test_dataset.normalization['features_text']['mean']) / test_dataset.normalization['features_text']['std'] # normalize text 
                    text = text.to(self.args.device).float()
                    if self.train_mean: # send to gpu to make denormalization then send back to cpu
                        mean_pred_ = net_mean(text, None)*self.train_dataset.normalization['class_centers']['std'].to(self.args.device)+self.train_dataset.normalization['class_centers']['mean'].to(self.args.device)
                        mean_pred_cached.append(mean_pred_.cpu())
                    if self.train_std:
                        std_pred_ = net_std(text, None)*self.train_dataset.normalization['class_std']['std'].to(self.args.device)+self.train_dataset.normalization['class_std']['mean'].to(self.args.device)
                    else:
                        std_pred_ = torch.ones_like(self.train_dataset.normalization['class_std']['mean']).repeat(text.shape[0],1)
                    std_pred_cached.append(std_pred_.cpu())
                mean_pred_cached = torch.cat(mean_pred_cached, dim=0) if self.train_mean else torch.tensor(torch.nan) # shape (n_classes, dim) 
                std_pred_cached = torch.cat(std_pred_cached, dim=0)# shape (n_classes, dim) 
        if self.init_text and self.train_mean: 
            # initialize logit had with text
            W_weights = OrderedDict([('weight', mean_pred_cached)])
        if shots!=None:
            run_batch_size, n_ways, n_shots, dim = shots.shape
            n_val_shots = validation_shots.shape[2]
            centroids = shots.mean(2) # (n_runs, n_classes, dim)
            stds = torch.ones(centroids.shape[1:]).to(self.args.device) # shots.std(2) if shots.shape[2]>=10 else torch.ones_like(centroids).to(self.args.device)
            test_iterator = enumerate(test_loader)
            targets = torch.arange(n_ways).repeat_interleave(n_shots).to(self.args.device)
            targets_validation = torch.arange(n_ways).repeat_interleave(n_val_shots).to(self.args.device)
            alphas = torch.arange(0, 1.05, 0.05) if self.interpolate_mean and self.default_alpha==0 and self.generate_shots>0 else torch.tensor([self.default_alpha]).float()
            betas = torch.arange(0, 1.05, 0.05) if self.interpolate_std and self.default_beta==0 else torch.tensor([self.default_beta]).float()

            results_logits_batchs = [self.searchLogit(shots[b].reshape(-1, dim), targets, validation_shots[b].reshape(-1, dim), targets_validation, n_ways, mean_pred_cached, centroids[b], std_pred_cached, stds, alphas, betas, W_weights) for b in range(run_batch_size)]
            # Ws = Parallel(n_jobs=num_cores)(delayed(self.trainLogit)(torch.cat([shots[b].reshape(-1, dim), mean_pred_cached], dim=0) if self.train_mean else shots[b].reshape(-1, dim), batch_targets, validation_shots[b].reshape(-1, dim), targets_validation, n_ways, std_pred_cached, stds, W_weights) for b in range(run_batch_size))
            # Evaluate all 
            results, Ws, best_alphas, best_betas = zip(*results_logits_batchs)
            
            total_count = 0.    
            scores = torch.zeros(run_batch_size, device=self.args.device)
            covariances = [(beta*std_pred_cached + (1-beta)*stds).mean(0) for beta in best_betas]
            # alphas_betas_combinations = torch.cartesian_prod(alphas, betas) if self.args.hyperparameter_search_set == 'test' else [(self.default_alpha, self.default_beta)]

            for batch_idx, (text, _, query, _, _, label) in test_iterator:
                query = query.squeeze(1).to(self.args.device).float() # not normalized
                label = label.to(self.args.device)
                for b in range(run_batch_size):
                    predictions = Ws[b](query/covariances[b].unsqueeze(0)).argmax(dim=1)
                    scores[b] += (predictions==label).sum()
                total_count += label.shape[0]
            scores /= total_count
            return scores, mean_pred_cached, std_pred_cached, torch.stack(best_alphas), torch.stack(best_betas) # return cached predictions for the next run even if they are None
        else:
            return torch.tensor([torch.nan]).to(self.args.device), mean_pred_cached.to(self.args.device), std_pred_cached.to(self.args.device), None, None
    def trainLogit(self, shots_, targets, validation_shots_, targets_validation, n_ways, std_pred, stds, criterion, beta, alpha, lr, wd, steps, batch_size, covariance, W_weights=None, bias=False) -> nn.Module:
        device = shots_.device
        W = nn.Linear(shots_.shape[-1], n_ways, bias=bias).to(device)
        best_validation_acc = -1
        if W_weights is not None:
            W.load_state_dict(W_weights) # load the same init weights
        W.train()
        optimizer = get_optimizer(self.trainCfg['optimizer'], W, lr, self.trainCfg['mmt'], wd)
        scheduler = build_lr_scheduler(optimizer, self.trainCfg['scheduler'], warmup_iter=50, max_iter=steps, warmup_type='linear', warmup_lr=1e-5) #get_scheduler(self.trainCfg['scheduler'], optimizer, steps, lr, end_lr_factor)
        # get permutations of the shots
        n = len(targets)
        step = 0
        for _ in range(math.ceil((steps)/math.ceil(n/batch_size))):
            permutations = torch.randperm(len(targets))
            for iteration in range(math.ceil(n/batch_size)):
                start_index = (iteration * batch_size)
                end_index = (iteration+1) * batch_size
                batch_support, batch_target = shots_[permutations[start_index:end_index]], targets[permutations[start_index:end_index]]
                optimizer.zero_grad()
                if self.normalize: 
                    batch_support = F.normalize(batch_support, dim=1)
                output = W(batch_support)*self.scale
                loss_train = criterion(output, batch_target)
                loss_train.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                if step%self.trainCfg['eval_freq']==0:
                    W.eval()
                    with torch.no_grad():
                        output = W(validation_shots_)
                        acc = (output.argmax(dim=1)==targets_validation).float().mean().item()
                        if acc>best_validation_acc:
                            best_validation_acc = acc
                            best_W = copy.deepcopy(W.state_dict())
                    W.train()
                step += 1
                if step >= steps:
                    break
        logger.debug(f'beta={beta:.2f}, lr={lr:.5f}, wd={wd:.5f}, batch_size={batch_size}, best_validation_acc={best_validation_acc:.3f}, step={steps}')
        return beta, alpha, lr, wd, batch_size, best_W, best_validation_acc
    def searchLogit(self, shots, targets, validation_shots, targets_validation, n_ways, mean_pred_cached, centroids, std_pred, stds, alphas, betas, W_weights=None) -> nn.Module:
        """
        Train a logit classifier on the shots
        """
        device = shots.device
        dim = shots.shape[-1]
        n_ways, dim = centroids.shape
        if self.train_mean and self.generate_shots == 0: 
            shots = torch.cat([shots.reshape(-1, dim), mean_pred_cached], dim=0)
        best_W_weights_val_shots, best_validation_acc = None, -1
        criterion = nn.CrossEntropyLoss()
        lrs = torch.tensor([self.trainCfg['lr'], self.trainCfg['lr']/10])
        wds = torch.tensor([0.0, 0.01, 0.0001])
        # end_lr_factors = torch.tensor([self.trainCfg['lr']/10, self.trainCfg['lr']/100, self.trainCfg['lr']/1000])
        steps = torch.tensor([self.trainCfg['steps']], dtype=wds.dtype)
        batch_size = torch.tensor([8., 32.])
        results = []
        alphas_betas_combinations = torch.cartesian_prod(alphas, betas)
        combinations = torch.cartesian_prod(lrs, wds, steps, batch_size)
        # n_cores = min(self.args.n_cores, os.cpu_count())
        logger.debug(f'Start hyperparameter search')
        for (alpha, beta) in alphas_betas_combinations:
            covariance = (beta*std_pred + (1-beta)*stds).mean(0) 
            if self.generate_shots>0:
                # generate shots
                mean = alpha*mean_pred_cached + (1-alpha)*centroids
                generated_shots = (torch.einsum('scd,d->scd', torch.randn(self.generate_shots, mean.shape[0], dim).to(self.args.device), covariance)+mean).reshape(-1, dim)
                shots_ = torch.cat([shots, generated_shots], dim=0)                
                targets_ = torch.cat([targets, torch.arange(n_ways).repeat_interleave(self.generate_shots).to(self.args.device)], dim=0)
                # logger.debug(f'Shots shape: {shots_.shape}, targets_ shape: {targets_.shape}, targets: {targets.shape}, repeat interleave: {torch.arange(n_ways).repeat_interleave(self.generate_shots).shape}')
            else:
                targets_ = torch.cat([targets, torch.arange(n_ways).to(self.args.device)]) if self.train_mean else targets 
                shots_ = shots/covariance.unsqueeze(0)
            validation_shots_ = validation_shots/covariance.unsqueeze(0)
            # results = Parallel(n_jobs=n_cores)(delayed(self.trainLogit)(shots_, targets, validation_shots_, targets_validation, n_ways, std_pred, stds, criterion, beta, lr.item(), wd.item(), end_lr_factor.item(), covariance, W_weights) for lr, wd, end_lr_factor in combinations)
            results_one_beta = [self.trainLogit(shots_, targets_, validation_shots_, targets_validation, n_ways, std_pred, stds, criterion, beta, alpha, lr.item(), wd.item(), steps.int().item(), batch_size.int().item(), covariance, W_weights) for lr, wd, steps, batch_size in combinations]
            accuracies_one_beta = [out[-1] for out in results_one_beta]
            results.append(results_one_beta[accuracies_one_beta.index(max(accuracies_one_beta))])
        accuracies_val_shots = [out[-1] for out in results]
        accuracies_val_shots_no_text = results[0][-1] # get all with beta=0
        beta_no_text = results[0][0]
        best_beta_val_shots, best_alpha_val_shots, best_lr, best_wd, best_batch_size, best_W_weights_val_shots, best_validation_acc = results[accuracies_val_shots.index(max(accuracies_val_shots))]
        logger.debug(f'best beta={best_beta_val_shots:.3f}, lr={best_lr:.5f}, best_wd={best_wd:.4f}, best_batch_size={best_batch_size} - Acc {best_validation_acc:.3f}, Acc no text {accuracies_val_shots_no_text:.3f}, beta no text {beta_no_text:.3f}')
        W_best_val_shots = nn.Linear(shots_.shape[-1], n_ways, bias=best_W_weights_val_shots is not None and 'bias' in best_W_weights_val_shots.keys()).to(device)
        if best_W_weights_val_shots is not None:
            W_best_val_shots.load_state_dict(best_W_weights_val_shots)
        W_best_val_shots.eval()
        return results, W_best_val_shots, best_alpha_val_shots, best_beta_val_shots

class CoOp(Template):
    def __init__(self, *args, **kwargs) -> None:
        super(CoOp, self).__init__(*args, **kwargs)
        self.label = 'CoOp'
    def predict_multi_class(self, shots, validation_shots, test_dataset, test_loader, datasetname, mean_pred_cached=None, std_pred_cached=None, cache=False, predict=True, W_weights=None) -> torch.tensor:
        if shots is None:
            score = torch.tensor([torch.nan]).to(self.args.device)
        else:
            batch_size, _, n_shots = shots.shape[:3]
            scores = {'semanticfs_oxford_pets': {1:85.89, 2:82.64, 4:86.7, 8:85.32, 16:87.01},
                      'semanticfs_eurosat':{1:50.63,2:61.50,4:70.18,8:76.73,16:83.53},
                      'semanticfs_oxford_flowers': {1:68.12,2:77.51,4:86.20,8:91.18,16:94.51},
                      'semanticfs_fgvc_aircraft_2013b':{1:9.64,2:18.68,4:21.87,8:26.13,16:31.26},
                      'semanticfs_dtd':{1:44.39,2:45.15,4:53.49,8:59.97,16:63.58},
                      'semanticfs_stanford_cars':{1:55.59,2:58.28,4:62.62,8:68.43,16:73.36},
                      'semanticfs_food_101':{1:74.32,2:72.49,4:73.33,8:71.82,16:74.67},
                        'semanticfs_sun397':{1:60.29,2:59.48,4:63.47,8:65.52,16:69.26},
                        'semanticfs_caltech_101':{1:87.53,2:87.93,4:89.55,8:90.21,16:91.83},
                        'semanticfs_ucf101':{1:61.92,2:64.09,4:67.03,8:71.94,16:75.71}
                        }
            format = lambda x: torch.tensor(x).repeat(batch_size).to(self.args.device)
            score = format([scores[datasetname].get(int(n_shots), torch.nan)/100.])
        return score, None, None, torch.tensor([torch.nan]), torch.tensor([torch.nan])

# options are: 
## - ncm: ncm (no text)
## - logit: logits (no text)
## - text_ncm: mean from text
## - text_ncm_manhabolis: mean from text + cov from text
## - text_logit: logit + shots from text 
## - text_logit_manhabolis : logit + shots from text + cov from text 
## - logit_mahalanobis: logit + cov from text
## - clip
## - optimal_text_ncm

def get_trainer(name):
    train_std = 'std' in name.lower()
    train_mean = 'mean' in name.lower()
    imconditionning = 'imconditionned' in name.lower()
    init_text = 'init_text' in name.lower()
    normalize = 'normalize' in name.lower()
    cov_shots = 'cov_shots' in name.lower()
    alpha = float(name.split('_')[-1]) if 'alpha' in  name.lower() else 0
    beta = float(name.split('_')[-1]) if 'beta' in  name.lower() else 0
    ME = 'ME' in name
    if 'ncm' in name.lower():
        return partial(NCM, train_mean=train_mean, train_std=train_std, imconditionning=imconditionning, default_alpha=alpha, default_beta=beta, preprocess=ME, cov_shots=cov_shots)
    elif 'logit' in name.lower():
        generate_shots = int(re.findall(r'\d+', name.lower())[0]) if 'generate_shots' in name.lower() else 0
        logger.debug(f'name: {name}, generate_shots: {generate_shots}')
        return partial(Logit, train_mean=train_mean, train_std=train_std, imconditionning=imconditionning, default_alpha=alpha, default_beta=beta, generate_shots=generate_shots, init_text=init_text, normalize=normalize)
    elif 'clip' in name.lower():
        return partial(Clip, raw='raw' in name.lower())
    elif 'coop' in name.lower():
        return CoOp
    else:
        raise NotImplementedError
