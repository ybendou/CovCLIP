#####################################
"""
"""
#####################################
# Original taxonomy
import os 
import sys
import pandas as pd 
import json
import numpy as np
import torch
import torch.nn as nn
import random
import argparse
from functools import partial
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')
# import logging
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-path-text', type=str, default='')
    parser.add_argument('--features-path-train-text', type=str, default='')
    parser.add_argument('--features-text-class-labels', type=str, default='')
    parser.add_argument('--features-path-train-images', type=str, default='')
    parser.add_argument('--features-path-validation-images', type=str, default='')
    parser.add_argument('--features-path-test-images', type=str, default=[""])
    parser.add_argument('--features-path-evaluation-shots-images', type=str, default=[""])
    parser.add_argument('--features-path-evaluation-queries-images', type=str, default=[""])
    parser.add_argument('--dataset-path', type=str, default='')
    parser.add_argument('--taxonomy-path', type=str, default='')

    parser.add_argument('--save-fig-path', type=str, default="")
    parser.add_argument('--save-data-path', type=str, default="")
    parser.add_argument("--save-data", action="store_true", help="save features")
    parser.add_argument('--save-predictions', type=str, default='', help='save output of networks')
    parser.add_argument("--training-dataset", type=str, default="", help="training dataset")
    parser.add_argument("--validation-dataset", type=str, default="", help="validation dataset")
    parser.add_argument("--test-dataset", type=str, default="", help="test dataset")
    parser.add_argument("--load-mean-net", type=str, default='', help="load mean network")
    parser.add_argument("--load-std-net", type=str, default='', help="load std network")
    
    parser.add_argument("--multi-text-training", action='store_true', help="use different text prompts for each training image")
    parser.add_argument("--baseline-on-train", action="store_true", help="if true, the baseline is the mean of train else it's the mean of test for each cross domain dataset")
    parser.add_argument("--not-random-split", action="store_true", help="not random split")
    parser.add_argument("--trainer", type=str, default='ncm', help="few shot model, can be ncm, logit, raw_clip or text_adapter")
    parser.add_argument("--covariance-form", type=str, default='diagonal', help="covariance matrix form, can be diagonal or full")
    parser.add_argument("--pca", type=int, default=-1, help="apply pca")

    # training parameters
    parser.add_argument("--lr", type=float, default=-1, help="learning rate")
    parser.add_argument("--end-lr-factor", type=float, default=0, help="ending learning rate factor end_lr=lr*end_lr_factor")
    parser.add_argument("--wd", type=float, default=-1, help="weight decay")
    parser.add_argument("--mmt", type=float, default=0.9, help="momentum")
    parser.add_argument("--optimizer", type=str, default='sgd', help="optimizer")
    parser.add_argument("--scheduler", type=str, default='constant', help="optimizer")
    parser.add_argument("--batch-size", type=int, default=500, help="batch size")
    parser.add_argument("--training-batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--validation-batch-size", type=int, default=128, help="batch size")
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--training-device', type=str, default='cuda:0')
    parser.add_argument("--net", type=str, default='linear', help="classifier used to train on the embeddings")
    parser.add_argument("--loss", type=str, default='mse', help="loss used for training")
    parser.add_argument("--loss-weights", type=str, default='[1,0]', help="loss weights for mean and std")
    parser.add_argument("--activation", type=str, default='gelu', help="activations used")

    parser.add_argument("--epochs", type=int, default=20, help="total number of training epochs")
    parser.add_argument("--normalize", action='store_false', help="normalize data")
    parser.add_argument("--layer-norm", action='store_true', help="use layer norm")
    parser.add_argument("--batch-norm", type=bool, default=True, help="use batch norm")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--embed-dim", type=int, default=0, help="embedding dimension, if 0, use the dimension of the features")
    parser.add_argument("--expansion-ratio", type=float, default=4, help="embedding dimension*expansion_ratio of the embedding")
    parser.add_argument("--grad-clipping", type=float, default=-1, help="gradient clipping value")

    parser.add_argument("--evaluation-runs-multi-class", type=int, default=200, help="number of runs for evaluation")
    parser.add_argument("--evaluation-runs-multi-class-few-shot", type=int, default=0, help="number of runs for evaluation in few shot setting with queries sampled too")
    parser.add_argument("--evaluation-runs-open-set", type=int, default=2000, help="number of runs for evaluation")
    parser.add_argument("--evaluation-reruns-open-set", type=int, default=10, help="number of reruns for evaluation to compute conf intervals while fixing the classes")
    parser.add_argument('--evaluation-batch-few-shot-runs', type=int, default=10, help="batch size for few shot evaluation")
    parser.add_argument("--evaluation-n-ways", type=int, default=5, help="number of classes in the few-shot classification task")
    parser.add_argument("--evaluation-n-queries", type=int, default=15, help="number of classes in the few-shot classification task")
    parser.add_argument("--evaluation-max-shots-multi-class", type=int, default=16, help="number of shots in the few-shot classification task")
    parser.add_argument("--evaluation-max-shots-multi-class-few-shot", type=int, default=5, help="number of shots in the few-shot classification task in few shot setting with queries sampled too")
    parser.add_argument("--evaluation-max-shots-open-set", type=int, default=16, help="number of shots in the few-shot classification task")
    parser.add_argument("--evaluation-multi-class-validation-shots", type=int, default=0, help="use validation shots")
    parser.add_argument("--evaluation-coop-split", type=int, default=-1, help="Few-shot split of coop protocol")
    parser.add_argument("--evaluate-on-test-set", action="store_true", help="report MSE for test set")
    parser.add_argument("--no-train", action="store_true", help="don't train the classifier")
    parser.add_argument("--seed", type=int, default=-1, help="seed")
    parser.add_argument("--runs", type=int, default=1, help="number of training runs")

    # logging and saving parameters
    parser.add_argument("--silent", action="store_true", help="silent mode")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--slurm", action="store_true", help="if slurm, don't use tqdm")
    parser.add_argument("--save-model", type=str, default="", help="save model")
    parser.add_argument("--wandb", type=str, default="", help="use wandb")
    parser.add_argument("--wandbProjectName", type=str, default="semanticFS", help="wandb project name")
    parser.add_argument("--n-cores", type=int, default=4, help="number of cores to use in parallel jobs")

    args = parser.parse_args()
    if args.wandb == '""' or args.wandb == "''":
        args.wandb = ''
    if args.lr < 0:
        defaut_values_lr = {'sgd':0.1, 'adamw':1e-3, 'adam':1e-3, 'adagrad':0.1, 'adadelta':1e-3, 'rmsprop':1e-3}
        args.lr = defaut_values_lr[args.optimizer.lower()]
    if args.wd < 0:
        default_values_wd = {'sgd':5e-4, 'adamw':1e-2, 'adam':0, 'adagrad':0, 'adadelta':0, 'rmsprop':0}
        args.wd = default_values_wd[args.optimizer.lower()]
    if args.seed < 0:
        args.seed = random.randint(0, 1000000)
    args.loss_weights = eval(args.loss_weights)
    args.layer_norm = nn.LayerNorm if args.layer_norm else nn.Identity
    # try:
    args.test_dataset = eval(args.test_dataset)
    args.features_path_test_images = eval(args.features_path_test_images)
    args.features_path_evaluation_shots_images = eval(args.features_path_evaluation_shots_images)
    args.features_path_evaluation_queries_images = eval(args.features_path_evaluation_queries_images)
    args.load_mean_net = '' if args.load_mean_net == '""' else args.load_mean_net
    args.load_std_net = '' if args.load_std_net == '""' else args.load_std_net
    args.save_model = '' if args.save_model == '""' else args.save_model
    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.activation.lower() == 'gelu':
        args.activation = nn.GELU
    elif args.activation.lower() == 'relu':
        args.activation = nn.ReLU
    elif args.activation.lower() == 'leakyrelu':
        args.activation = nn.LeakyReLU
    return args

from loguru import logger
from utils import fix_seed
from trainers import get_trainer

class SemanticDataset_ClassLevel():
    """
    Return a dataset with class-level features
    """
    def __init__(self, precomputed_data, normalization, imconditionning=False):
        self.features_text = precomputed_data['features_text']
        self.class_centers = precomputed_data['class_centers']
        self.class_std = precomputed_data['class_std']
        self.features_images = precomputed_data['features_images'] # list of image features per class
        self.labels = torch.arange(len(self.features_images)).repeat_interleave(torch.tensor([len(x) for x in self.features_images]))
        self.features_images = torch.cat(self.features_images, dim=0)
        self.normalization = normalization
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
class SemanticDataset():
    def __init__(self, precomputed_data, normalization, imconditionning=False, name=''):
        self.features_text = precomputed_data.get('features_text', None)
        self.class_centers = precomputed_data.get('class_centers', None)
        self.class_std = precomputed_data.get('class_std', None)
        self.features_images = precomputed_data['features_images'] # list of image features per class
        if 'raw_clip' in precomputed_data:
            self.raw_clip = precomputed_data['raw_clip']
        self.labels = torch.arange(len(self.features_images)).repeat_interleave(torch.tensor([len(x) for x in self.features_images]))
        self.imconditionning = imconditionning
        self.features_images_class = self.features_images
        self.features_images = torch.cat(self.features_images, dim=0)
        self.normalization = normalization
        self.name = name

    def __getitem__(self, idx):
        """
        idx now is at image level, sample a single image query then add mean class feature and covariance 
        """
        label = self.labels[idx]
        text = (self.features_text[label]-self.normalization['features_text']['mean']) / self.normalization['features_text']['std'] if self.features_text is not None else torch.tensor([])
        image = (self.features_images[idx]-self.normalization['features_images']['mean']) / self.normalization['features_images']['std'] if self.imconditionning else torch.tensor([])
        query = self.features_images[idx]
        class_center = (self.class_centers[label]-self.normalization['class_centers']['mean']) / self.normalization['class_centers']['std'] if self.class_centers is not None else torch.tensor([])
        class_std = (self.class_std[label]-self.normalization['class_std']['mean']) / self.normalization['class_std']['std'] if self.class_std is not None else torch.tensor([])
        return text, image, query, class_center, class_std, label
    def __len__(self):
        return self.labels.shape[0]

def load_imagenet21k(args):
    # load imagenet21k visual features
    try: 
        args.features_path_train_images = eval(args.features_path_train_images)
    except: 
        args.features_path_train_images = [args.features_path_train_images]
    # start with text: 
    features_text_train = torch.load(args.features_path_train_text, map_location='cpu')
    features_text_train = {feat['name_class'].replace('imagenet21k_', ''):feat['features'] for feat in features_text_train}
    features_images_train = {}
    min_samples_per_class = 10 # minimum number of samples per class
    for _, path in enumerate(args.features_path_train_images):
        feats = torch.load(path, map_location='cpu')
        for i in range(len(feats)): # loop over classes in partition
            feats_ = feats[i]['features']
            name = feats[i]['name_class']
            feats_ = feats_[~torch.any(feats_.isnan(),dim=1)] # remove nan values
            if feats_.shape[0] >= min_samples_per_class and name in features_text_train.keys(): # make sure there are enough samples per class and that the class is in the text features
                features_images_train[name] = feats_

    train_class_names = list(features_images_train.keys())
    # load imagenet21k text features
    if args.multi_text_training:
        features_text_train = torch.stack([features_text_train[key] for key in train_class_names])
    else:
        features_text_train = torch.stack([features_text_train[key].mean(0) for key in train_class_names])
    features_images_train = [features_images_train[k] for k in train_class_names]
    return features_images_train, features_text_train, train_class_names
def load_inaturalist(args):
    from utils import classes_to_df
    features_images_train = torch.load(args.features_path_train_images, map_location='cpu')
    df_taxonomy = pd.read_csv(args.taxonomy_path)
    classes = [feat['name_class'] for feat in features_images_train]
    df_classes = classes_to_df(classes)
    df_taxonomy.scientificName = df_taxonomy.scientificName.astype("string")
    duplicates = list(df_taxonomy.query('taxonRank=="species"').scientificName.value_counts()[df_taxonomy.query('taxonRank=="species"').scientificName.value_counts()>=2].index)
    joined_df = pd.merge(left=df_classes, right=df_taxonomy[df_taxonomy.taxonRank=="species"][~df_taxonomy.scientificName.isin(duplicates)], on=['scientificName'], how='inner', suffixes=('', '_taxo'))
    mapping = dict(zip(joined_df.name, joined_df.id))
    features_images_train = {str(mapping[feat['name_class']]):feat['features'].float() for feat in features_images_train if feat['name_class'] in mapping}
    train_class_names = sorted(list(features_images_train.keys()))
    features_images_train = [features_images_train[k] for k in train_class_names]
    features_text_train = torch.load(args.features_path_train_text, map_location='cpu')
    # change structure to dict
    features_text_train = {feat['name_class']:feat['features'].float().mean(0) for feat in features_text_train if feat['name_class'] in train_class_names} 
    features_text_train = torch.stack([features_text_train[k] for k in train_class_names])
    return features_images_train, features_text_train, train_class_names
def load_miniimagenet(args, features_images_path):
    features_text = torch.load(args.features_path_text, map_location='cpu')
    features_images = torch.load(features_images_path, map_location='cpu')
    train_class_names = list(features_images.keys())
    features_images = [features_images[k] for k in train_class_names]
    features_text_train = torch.stack([features_text[k].mean(0) for k in train_class_names])
    return features_images, features_text_train, train_class_names

def load_others(args):
    features_text = torch.load(args.features_path_text, map_location='cpu')
    features_text = {feat['name_class']:feat['features'] for feat in features_text} # change structure to dict
    features_images_train = torch.load(args.features_path_train_images, map_location='cpu')
    train_class_names = [f'{args.training_dataset}_{i}' for i in range(len(features_images_train)) if args.training_dataset!='semanticfs_imagenet' or i not in [836, 744]] # remove two classes which are at 836 and 744
    features_images_train = [feat['features'] for i, feat in enumerate(features_images_train) if args.training_dataset!='semanticfs_imagenet' or i not in [836, 744]] # remove two classes which are at 836 and 744
    if args.multi_text_training:
        features_text_train = torch.stack([features_text[key] for key in train_class_names])
    else:
        features_text_train = torch.stack([features_text[key].mean(0) for key in train_class_names])
    return features_images_train, features_text_train, train_class_names
def compute_covariance(features, name):
    data_type = features.dtype
    if name == 'diagonal': 
        return features.std(0)
    elif name == 'full':
        return features.float().T.cov().flatten().to(data_type)
if __name__ == '__main__':
    # load features
    args = get_args()
    # logger.debug(args)
    if args.training_dataset == 'inaturalist':
        features_images_train, features_text_train, train_class_names = load_inaturalist(args) # # if it's inaturalist, we need to load the taxonomy and do some preprocessing
    elif args.training_dataset == 'imagenet21k':
        features_images_train, features_text_train, train_class_names = load_imagenet21k(args) # special case for imagenet21k
    elif args.training_dataset == 'miniimagenet_train':
        features_images_train, features_text_train, train_class_names = load_miniimagenet(args, args.features_path_train_images)
    else:
        features_images_train, features_text_train, train_class_names = load_others(args)
    # logger.debug('Training features loaded')

    if args.pca > 0:
        # apply pca to image features
        from sklearn.decomposition import  IncrementalPCA
        pca = IncrementalPCA(n_components=args.pca, batch_size=1000)
        type_features = features_images_train[0].dtype
        # remember indices per class, stack features, apply pca and then unstack
        sizes = [len(feat) for feat in features_images_train]
        features_images_train = torch.tensor(pca.fit_transform(torch.cat(features_images_train, 0).numpy()), dtype=type_features)
        features_images_train = torch.split(features_images_train, sizes)

    # compute class centers, class std and features text
    class_centers_train = torch.stack([feat.mean(0) for feat in features_images_train])
    class_std_train = torch.stack([compute_covariance(feat, args.covariance_form) for feat in features_images_train])
    # logger.debug('Class centers and std computed')
    # if train and test are from different datasets
    if len(args.features_path_evaluation_shots_images)>0:
        features_images_evaluation_shots = [torch.load(path, map_location='cpu') for path in args.features_path_evaluation_shots_images]
        features_images_evaluation_shots = [[features_images_evaluation_shots[t][c]['features'] for c in range(len(features_images_evaluation_shots[t]))] for t in range(len(args.test_dataset))]
        if args.pca>0:
            features_images_evaluation_shots = [[torch.tensor(pca.transform(feat.numpy()), dtype=type_features) for feat in features_images_evaluation_shots[t]] for t in range(len(args.test_dataset))]
    else: 
        features_images_evaluation_shots = [None for _ in range(len(args.test_dataset))]
    if len(args.features_path_evaluation_queries_images)>0:
        features_images_evaluation_queries = [torch.load(path, map_location='cpu') for path in args.features_path_evaluation_queries_images]
        features_images_evaluation_queries = [[features_images_evaluation_queries[t][c]['features'] for c in range(len(features_images_evaluation_queries[t]))] for t in range(len(args.test_dataset))]
        if args.pca>0:
            features_images_evaluation_queries = [[torch.tensor(pca.transform(feat.numpy()), dtype=type_features) for feat in features_images_evaluation_queries[t]] for t in range(len(args.test_dataset))]
    else:
        features_images_evaluation_queries = [None for _ in range(len(args.test_dataset))]
    if args.test_dataset != args.training_dataset or args.validation_dataset != args.training_dataset:
        if args.training_dataset != 'miniimagenet_train': 
            features_text = torch.load(args.features_path_text, map_location='cpu')
            features_text = {feat['name_class']:feat['features'] for feat in features_text} # change structure to dict
    
    raw_clip_features = torch.load(args.features_text_class_labels, map_location='cpu')
    if type(raw_clip_features) == list:
        raw_clip_features = {feat['name_class']:feat['features'].mean(0).float() for feat in raw_clip_features}

    class_centers_test, class_std_test, features_text_test, raw_clip, features_images_test  = [], [], [], [], []
    for t,test_dataset_ in enumerate(args.test_dataset):
        if args.training_dataset == test_dataset_:
            class_centers_test.append(None)
            class_std_test.append(None)
            features_text_test.append(None)
            features_images_test.append(None)
            raw_clip.append(None)
        else:
            if test_dataset_ == 'miniimagenet_test':
                features_images_test_,features_text_test_,test_class_names = load_miniimagenet(args, args.features_path_test_images[t])
                features_images_test.append(features_images_test_)
                features_text_test.append(features_text_test_)
                raw_clip.append(torch.stack([raw_clip_features[name] for name in test_class_names]))
            else:
                features_images_test_ = torch.load(args.features_path_test_images[t], map_location='cpu')
                features_images_test_ = [features_images_test_[i]['features'] for i in range(len(features_images_test_))]
                features_images_test.append(features_images_test_)
                features_text_test.append(torch.stack([features_text[f'{args.test_dataset[t]}_{i}'].mean(0) for i in range(len(features_images_test_))]))
                raw_clip.append(torch.stack([raw_clip_features[f'{args.test_dataset[t]}_{i}'] for i in range(len(features_images_test_))]))

            if args.pca>0:
                features_images_test_ = [torch.tensor(pca.transform(feat.numpy()), dtype=type_features) for feat in features_images_test_]
            class_centers_test.append(torch.stack([features_images_test_[i].mean(0) for i in range(len(features_images_test_))]))
            class_std_test.append(torch.stack([compute_covariance(features_images_test_[i], args.covariance_form) for i in range(len(features_images_test_))]))
        if test_dataset_ == 'miniimagenet_test':
            features_images_evaluation_shots[t] = features_images_test_
            features_images_evaluation_queries[t] = features_images_test_
    if args.validation_dataset != args.training_dataset:
        if args.validation_dataset == 'miniimagenet_val':
            features_images_val,features_text_val,_ = load_miniimagenet(args, args.features_path_validation_images)
        else:
            features_images_val = torch.load(args.features_path_validation_images, map_location='cpu')
            features_images_val = [features_images_val[i]['features'] for i in range(len(features_images_val))]
            features_text_val = torch.stack([features_text[f'{args.validation_dataset}_{i}'].mean(0) for i in range(len(features_images_val))])
        if args.pca>0:
            features_images_val = [torch.tensor(pca.transform(feat.numpy()), dtype=type_features) for feat in features_images_val]
        class_centers_val = torch.stack([features_images_val[i].mean(0) for i in range(len(features_images_val))])
        class_std_val = torch.stack([compute_covariance(features_images_val[i], args.covariance_form) for i in range(len(features_images_val))])

    # start run before then split data randomly into train, val and test
    for run in range(args.runs):
        logger.info(f'Run: {run}')
        if args.wandb:
            import wandb
            run_wandb = wandb.init(reinit = True, project=args.wandbProjectName, 
                entity=args.wandb, 
                tags=['text'],
                config=vars(args))
        else:
            run_wandb = argparse.Namespace()
            run_wandb.name = str(args.seed)
        ###### Splits
        # if queries and shots are from the same file, we first split them into two different sets
        if len(args.features_path_evaluation_shots_images) > 0 and len(args.features_path_evaluation_queries_images) > 0 and args.evaluation_n_queries==0:
            for t in range(len(args.test_dataset)):
                if args.features_path_evaluation_queries_images[t]==args.features_path_evaluation_shots_images[t] and args.test_dataset[t] != 'miniimagenet_test':
                    shuffled_features = [features_images_evaluation_shots[t][c][torch.randperm(len(features_images_evaluation_shots[t][c]))] for c in range(len(features_images_evaluation_shots[t]))]
                    features_images_evaluation_shots[t] = [shuffled_features[c][:int(len(shuffled_features[c])//2)] for c in range(len(shuffled_features))]
                    features_images_evaluation_queries[t] = [shuffled_features[c][int(len(shuffled_features[c])//2):] for c in range(len(shuffled_features))]

        if args.validation_dataset == args.training_dataset: # if train val and test are the same dataset split to 3 subsets
            perm = torch.arange(len(class_centers_train)) if args.not_random_split else torch.randperm(len(class_centers_train))
            for t in range(len(args.test_dataset)):
                if args.training_dataset == args.test_dataset[t]: # if train val and test are the same dataset split to 3 subsets
                    split = [0.98, 0.01, 0.01] if args.training_dataset in ['imagenet21k', 'inaturalist'] else [0.8, 0.1, 0.1] 
                    test_idx = perm[int((split[0]+split[1])*len(perm)):]
                    class_centers_test[t] = class_centers_train[test_idx]
                    class_std_test[t] = class_std_train[test_idx]
                    features_text_test[t] = features_text_train[test_idx]
                    if 'raw_clip' in args.trainer:
                        if args.training_dataset == 'imagenet21k':
                            raw_clip[t] = features_text_train[test_idx]
                        else:
                            raw_clip[t] = torch.stack([raw_clip_features[f'{args.test_dataset[t]}_{i}'] for i in test_idx])
                    else:
                        raw_clip[t] = None
                    features_images_test[t] = [features_images_train[i] for i in test_idx]
                    # do the same for shots and queries, only keep in features_image_evaluation_shots and queries the classes selected for test
                    if len(args.features_path_evaluation_shots_images) > 0 and len(args.features_path_evaluation_queries_images) > 0 and n_queries==0:
                        features_images_evaluation_shots[t] = [features_images_evaluation_shots[t][i] for i in test_idx]
                        features_images_evaluation_queries[t] = [features_images_evaluation_queries[t][i] for i in test_idx]
                    else:
                        features_images_evaluation_shots[t] = features_images_test[t]
                        features_images_evaluation_queries[t] = features_images_test[t]
                    break
                else:
                    split = [0.995, 0.005] if args.training_dataset in ['imagenet21k', 'inaturalist'] else [0.95, 0.05]  # if train and val are the same dataset split to 2 subsets
            # logger.debug('Splitting data')
            train_idx = perm[:int(split[0]*len(perm))]
            val_idx = perm[int(split[0]*len(perm)):int((split[0]+split[1])*len(perm))]
            class_centers_val = class_centers_train[val_idx]
            class_std_val = class_std_train[val_idx]
            features_text_val = features_text_train[val_idx]
            features_images_val = [features_images_train[i] for i in val_idx]
            class_centers_train = class_centers_train[train_idx]
            class_std_train = class_std_train[train_idx]
            features_text_train = features_text_train[train_idx]
            features_images_train = [features_images_train[i] for i in train_idx]
        # normalize the data, compute mean and std for each of the inputs and targets
        if args.normalize or 'clip' not in args.trainer:
            mean_class_centers = class_centers_train.mean(dim=0, dtype=torch.double).float()
            std_class_centers = class_centers_train.double().std(dim=0).float() 
            mean_class_std = class_std_train.mean(dim=0, dtype=torch.double).float() 
            std_class_std = class_std_train.double().std(dim=0).float() 

            mean_features_images = torch.zeros(class_centers_train.shape[-1]) 
            std_features_images = torch.zeros(class_centers_train.shape[-1])
            
            mean_features_text = features_text_train.reshape(-1, features_text_train.shape[-1]).mean(dim=0, dtype=torch.double).float()
            std_features_text = features_text_train.reshape(-1, features_text_train.shape[-1]).double().std(dim=0).float()
        else:
            mean_features_images = torch.zeros(class_centers_train.shape[-1]) 
            std_features_images = torch.ones(class_centers_train.shape[-1])
            mean_features_text = torch.zeros(features_text_train.shape[-1])
            std_features_text = torch.ones(features_text_train.shape[-1])
            mean_class_centers = torch.zeros_like(class_centers_train[0])
            std_class_centers = torch.ones_like(class_centers_train[0])
            mean_class_std = torch.zeros_like(class_centers_train[0])
            std_class_std = torch.ones_like(class_centers_train[0])

        normalization={'class_centers':{'mean':mean_class_centers, 'std':std_class_centers},
                       'class_std':{'mean':mean_class_std, 'std':std_class_std}, 
                       'features_images':{'mean':mean_features_images, 'std':std_features_images}, 
                       'features_text':{'mean':mean_features_text, 'std':std_features_text}}
        TrainingDataset = SemanticDataset_ClassLevel if 'imconditionned' not in args.trainer else SemanticDataset
        train_dataset = TrainingDataset(precomputed_data={'class_centers':class_centers_train, 
                                                          'class_std':class_std_train, 
                                                          'features_text':features_text_train, 
                                                          'features_images': features_images_train
                                                          }, 
                                        normalization=normalization, imconditionning='imconditionned' in args.trainer) 
        val_dataset = SemanticDataset(precomputed_data={'class_centers':class_centers_val, 
                                                        'class_std':class_std_val, 
                                                        'features_text':features_text_val,
                                                        'features_images': features_images_val
                                                        }, normalization=normalization, imconditionning='imconditionned' in args.trainer) 
        test_dataset = [SemanticDataset(precomputed_data={'class_centers':class_centers_test[t], 
                                                          'class_std':class_std_test[t], 
                                                          'features_text':features_text_test[t],
                                                          'features_images': features_images_evaluation_queries[t], 
                                                          'raw_clip':raw_clip[t]
                                                          }, name=args.test_dataset[t], normalization=normalization, imconditionning='imconditionned' in args.trainer) for t in range(len(args.test_dataset))]
        if args.save_model and not os.path.exists(os.path.join(args.save_model, run_wandb.name)):
            os.makedirs(os.path.join(args.save_model, run_wandb.name))
        trainer = get_trainer(args.trainer)(args, run_wandb=run_wandb, normalization=normalization, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)
        
        if args.save_fig_path!='' and not os.path.exists(os.path.join(args.save_fig_path, run_wandb.name)):
            os.makedirs(os.path.join(args.save_fig_path, run_wandb.name))

        # save args as json file : 
        save_args = deepcopy(vars(args))
        save_args['layer_norm'] = ''
        save_args['activation'] = ''
        from utils import run_evaluation
        figs_openset, figs_n_ways, figs_n_ways_few_shot = run_evaluation(trainer, test_dataset, features_images_evaluation_shots, args, run_wandb=run_wandb, 
                                                    openset_config={'runs':args.evaluation_runs_open_set, 
                                                                    'reruns':args.evaluation_reruns_open_set, 
                                                                    'max_shots':args.evaluation_max_shots_open_set, 'conf':True
                                                                    }, 
                                                    n_ways_config={'runs':args.evaluation_runs_multi_class, 
                                                                    'max_shots':args.evaluation_max_shots_multi_class,'n_ways':args.evaluation_n_ways,
                                                                    'n_queries':args.evaluation_n_queries, 
                                                                    'batch_few_shot_runs':args.evaluation_batch_few_shot_runs, 
                                                                    'evaluation_multi_class_validation_shots': args.evaluation_multi_class_validation_shots},
                                                    few_shot_n_ways_config={'runs':args.evaluation_runs_multi_class_few_shot, 
                                                                    'max_shots':args.evaluation_max_shots_multi_class_few_shot,'n_ways':args.evaluation_n_ways,
                                                                    'n_queries':args.evaluation_n_queries, 
                                                                    'batch_few_shot_runs':args.evaluation_batch_few_shot_runs
                                                                    }
                                                                    )
        if args.wandb!='':
            run_wandb.finish()