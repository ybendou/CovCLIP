import torch 
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
import scipy.stats as st
import numpy as np
from tqdm import tqdm
import pandas as pd
import random
import math
import wandb
import os
import sys
import json
from collections import OrderedDict, Counter
from sklearn.metrics import roc_auc_score
# import logging
# logging.getLogger('matplotlib').setLevel(logging.WARNING)
from loguru import logger
import matplotlib.pyplot as plt
# logger = logging.getLogger(__name__)
plt.style.use('fivethirtyeight')

def mean_conf(scores):
    acc = 100*scores
    mean = acc.mean().item()
    conf = (1.96 * acc.std()/np.sqrt(len(acc))).item()
    return f'{mean:.2f}±{conf:.2f}'
def confInterval(scores):
    if scores.shape[0] == 1:
        low, up = -1., -1.
    elif scores.shape[0] < 30:
        low, up = st.t.interval(0.95, df = scores.shape[0] - 1, loc = scores.mean(), scale = st.sem(scores))
    else:
        low, up = st.norm.interval(0.95, loc = scores.mean(), scale = st.sem(scores))
    return low, up
def get_confidence_interval(scores):
    low, up = confInterval(scores)
    mean = scores.mean()
    conf = (up - low) / 2
    return mean, conf
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
def plot_with_conf_interval(x,y,label,color,ax, conf=True, style=None, additional_points=None):
    if additional_points != None:
        X = [p[0] for p in additional_points]+x.tolist()
        Y = [p[1] for p in additional_points]+y.mean(axis=1).tolist()
    else:
        X,Y = x,y.mean(axis=1)
    if style != None:
        ax.plot(X,Y, style, label=label, color=color,alpha=0.5)
    else:
        ax.plot(X,Y,label=label,color=color)
    n = y.shape[1]
    if conf:
        ci = 1.96 * np.std(y, axis=1)/np.sqrt(n)
        ax.fill_between(x, (y.mean(axis=1)-ci), (y.mean(axis=1)+ci), color=color, alpha=.1)    

def run_open_set(trainer, test_dataset, datasetname, images_shots, args, runs, reruns, max_shots, device='cuda:0'):
    sigmoid_scale=0.01
    labels, selected_classes, query_classes, query_idx, shots_idx = [], [], [], [], []
    
    # Define the runs first
    for i in range(runs):
        class_proposals = torch.randperm(len(images_shots))
        selected_classes.append(class_proposals[0])
        class_choice = class_proposals[0] if i%2==0 else class_proposals[1] # 50% of the time, choose the same class as the query
        query_classes.append(class_choice)
        labels.append(int(i%2==0))
        query_idx.append(random.randint(0, len(test_dataset.features_images_class[class_choice])-1))
        shots_idx.append([torch.randperm(len(images_shots[selected_classes[i]]))[:max_shots] for _ in range(reruns)])

    query_idx = torch.tensor(query_idx)
    query_classes = torch.tensor(query_classes)
    selected_classes = torch.tensor(selected_classes)
    labels = torch.tensor(labels)
    best_scores = []
    if trainer.preprocess: 
        base_mean = trainer.train_dataset.features_images.mean(0).to(device)
    # perform zero shot first then cache mean if needed
    _, mean_pred_cached, std_pred_cached, _, _ = trainer.predict_multi_class(None, None, test_dataset, None, datasetname, mean_pred_cached=None, std_pred_cached=None, cache=True, openset=True)
    max_shots_ = 0 if 'clip' in args.trainer else max_shots # only zero-shot for clip
    min_shots = 0 if trainer.train_mean else 1
    if 'inaturalist' not in args.training_dataset: 
        if trainer.train_mean: 
            k_shots = torch.tensor([0, 1, 2, 4, 8, 16])
        else: 
            k_shots = torch.tensor([1, 2, 4, 8, 16])
    else: 
        k_shots = torch.arange(min_shots, max_shots+1)
    for _, k in enumerate(torch.arange(min_shots, max_shots+1)):
        if k in k_shots: 
            best_alpha, best_beta, best_score_k = -1, -1, torch.tensor([-1.]*reruns) 
            alphas = (torch.arange(0, 1.05, 0.05) if trainer.interpolate_mean and trainer.default_alpha==0 else torch.tensor([trainer.default_alpha]).float()).to(args.device)
            betas = (torch.arange(0, 1.05, 0.05) if trainer.interpolate_std and trainer.default_beta==0 else torch.tensor([trainer.default_beta]).float()).to(args.device)
            combinations = torch.cartesian_prod(alphas, betas)
            for j,(alpha, beta) in enumerate(combinations):
                score = []
                for r in range(reruns):
                    distances = []
                    for i in range(runs):
                        # compute distances from class_idx to class_choice after choosing a query idx 
                        query = test_dataset.features_images_class[query_classes[i]][query_idx[i]].to(device)
                        shots = None if k==0 else images_shots[selected_classes[i]][shots_idx[i][r][:k]].to(device)#.unsqueeze(0).unsqueeze(0).to(device)
                        if k>0:
                            centroids = shots.mean(dim=0)
                        if trainer.train_mean: 
                            cached_mean = mean_pred_cached[selected_classes[i]].to(device)
                            if alpha >0 or k==0:
                                if k>0: 
                                    centroids = alpha*cached_mean + (1-alpha)*centroids
                                else:
                                    centroids = cached_mean
                        if trainer.cov_shots and k>1: 
                            stds_shots = shots.std(dim=0).to(device)
                        else: 
                            stds_shots = torch.ones(centroids.shape).to(device)

                        if trainer.train_std:
                            cached_std = std_pred_cached[selected_classes[i]].to(device)
                            stds = (1-beta)*stds_shots+beta*cached_std
                        else:
                            stds = stds_shots
                        stds = stds.view(stds.shape[-1])
                        if trainer.preprocess: 
                            query = preprocess(query, base_mean, stds)
                            centroids = preprocess(centroids, base_mean, stds)
                        else:
                            query = query/stds
                            centroids = centroids/stds
                        distance = torch.norm((query-centroids), dim = -1)
                        distances.append(distance)
                    distances = torch.stack(distances)
                    probas = 1-F.sigmoid(sigmoid_scale*distances)
                    score.append(roc_auc_score(labels, probas.cpu()))
                score = torch.tensor(score)
                if score.mean() > best_score_k.mean():
                    best_score_k = score.clone()
                    best_alpha = alpha
                    best_beta = beta
                if alpha == 0 and beta == 0:
                    score_ncm = score
                    alpha_ncm = alpha
                if j == len(combinations)-1:
                    score_text = score.clone()
                    alpha_text = alpha
            best_scores.append(best_score_k)
            logger.info(f'k={k}: alpha={best_alpha:.1f}, beta={best_beta:.1f} | score mix={mean_conf(best_score_k)} | alpha text={alpha_text:.1f}')
            if args.wandb:
                for rerun in range(reruns):
                    wandb.log({'dataset':datasetname,'n_shots':k, f'{datasetname} alpha':best_alpha, f'{datasetname} beta':best_beta, 'rerun':rerun, f'{datasetname} score mean':best_score_k.mean().item(), f'{datasetname} score':best_score_k[rerun], f'{datasetname} score text mean': score_text.mean().item(), f'{datasetname} score text': score_text[rerun].item(), f'{datasetname} score ncm mean':score_ncm.mean().item(), f'{datasetname} score ncm':score_ncm[rerun].item()})
        else:
            best_scores.append(torch.tensor([torch.nan]*reruns))
    return list(torch.arange(min_shots, max_shots+1)), torch.stack(best_scores)
def evaluation_openset(trainer, test_dataset, shots, args, run_wandb, config, device='cuda:0'):
    figs = []
    df_all = pd.DataFrame([])
    for t, name in tqdm(enumerate(args.test_dataset)):
        logger.info(f'Openset: Evaluating on {name}')
        dataset = test_dataset[t] # choose one class in the test set
        images_shots = shots[t]
        max_shots = min(config['max_shots'], min([len(images_shots[c]) for c in range(len(images_shots))]))
        fix_seed(args.seed)
        k_shots, score = run_open_set(trainer, dataset, name, images_shots, args, config['runs'], config['reruns'], max_shots, device=device)
        df = pd.DataFrame([])
        conf = lambda x: (1.96 * x.std()/np.sqrt(len(x))).item()
        for s in range(len(score)): # fill the dataframe
            row = {'shots':s, 'model': args.trainer, 'dataset': name,
                    'score': score[s].mean().item(), 'conf': conf(score[s])}
            df = df._append(row, ignore_index=True)
            df_all = df_all._append(row, ignore_index=True)
           
        # conf = True
        # fig, ax = plt.subplots(figsize=(20,10))
        # new_list = range(math.floor(min(k_shots)), math.ceil(max(k_shots))+1)
        # ax.set_xticks(new_list)
        # plot_with_conf_interval(k_shots, score.reshape(max_shots+1, -1).numpy(), color='blue', label=trainer.label, ax=ax, conf=conf)
        # ax.legend(loc='lower right')
        # ax.set_xlabel('number of shots')
        # ax.set_ylabel('AUC score')
        # ax.set_title(f'Openset: {name}')
        # save_path = os.path.join(args.save_fig_path, run_wandb.name, f'n_ways_{name}.png')
        # fig.savefig(save_path)
        # figs.append(fig)
        # # save table into csv file
        df.to_csv(os.path.join(args.save_fig_path, run_wandb.name, f'openset_classes_{name}.csv'), decimal=',')
        if args.wandb != '':
            # run_wandb.log({f'Openset Image: {name}': [wandb.Image(save_path)], f'Openset Table: {name}': wandb.Table(data=df, columns=list(df.columns))})
            run_wandb.log({f'Openset Table: {name}': wandb.Table(data=df, columns=list(df.columns))})
            
    df_all.to_csv(os.path.join(args.save_fig_path, run_wandb.name, f'results_openset.csv'), decimal=',')
    if args.wandb != '':
        run_wandb.log({f'Table': wandb.Table(data=df_all, columns=list(df_all.columns))})
    return figs

def get_few_shot_splits(num_elements_per_class_shots, max_shots, runs, datasetname, args):
    shots_idx = []
    map_datasets_to_splits = {
        'semanticfs_caltech_101':['split_zhou_Caltech101.json', 'caltech101'],
        'semanticfs_dtd':['split_zhou_DescribableTextures.json', 'dtd'],
        'semanticfs_eurosat':['split_zhou_EuroSAT.json', 'eurosat'],
        'semanticfs_food_101':['split_zhou_Food101.json', 'food101'],
        'semanticfs_oxford_flowers':['split_zhou_OxfordFlowers.json', 'oxford_flowers'], # all images are in the same folder 
        'semanticfs_oxford_pets':['split_zhou_OxfordPets.json', 'oxford_pets'], # all images are in the same folder
        'semanticfs_stanford_cars': ['split_zhou_StanfordCars.json', 'stanford_cars'],
        'semanticfs_sun397': ['split_zhou_SUN397.json', 'sun397'],
        'semanticfs_ucf101': ['split_zhou_UCF101.json', 'ucf101']
    }
    if args.evaluation_coop_split>0: 
        assert args.evaluation_coop_split in [1, 2, 3, 4], f'Split must be 1, 2 or 3 for CoOp split. Split {args.evaluation_coop_split} is not allowed.'
        splits_seeds = [1,2,3] if args.evaluation_coop_split == 4 else [args.evaluation_coop_split] 
        splits = json.load(open(os.path.join(f'../../data/cross_modal_splits/{map_datasets_to_splits[datasetname][0]}'), 'r'))
        num_elements_per_class_train = torch.tensor(list(OrderedDict(Counter([i[1] for i in splits['train']])).values()))
        num_elements_per_class_val = torch.tensor(list(OrderedDict(Counter([i[1] for i in splits['val']])).values()))

        path = f'../../data/indices/{map_datasets_to_splits[datasetname][1]}/'
        for n_shots in [1, 2, 4, 8, 16]:
            if n_shots <= max_shots:
                shots_idx_ = []
                for seed in splits_seeds:
                    indices = json.load(open(os.path.join(path, f'shot_{n_shots}-seed_{seed}.json'), 'r'))
                    n_classes = len(indices['train']['indices'])//n_shots
                    assert n_classes == len(num_elements_per_class_shots), f'Number of classes in the split ({n_classes}) is different from the number of classes in the dataset ({len(num_elements_per_class_shots)}).'
                    n_val = min(4, n_shots)
                    shots_idx_one_seed = []
                    for n in range(n_classes):
                        shots_idx_one_class = torch.tensor(indices['train']['indices'][n*n_shots:(n+1)*n_shots])-num_elements_per_class_train[:n].sum()
                        val_idx_one_class = torch.tensor(indices['val']['indices'][n*n_val:(n+1)*n_val])-num_elements_per_class_val[:n].sum()
                        shots_idx_one_seed.append(torch.cat([shots_idx_one_class, val_idx_one_class]))
                    shots_idx_.append(torch.stack(shots_idx_one_seed))
                    # logger.debug(f'seed: {seed}, shots_idx: {shots_idx_one_seed[0][:10]}')
                shots_idx.append(torch.stack(shots_idx_))
        runs = len(splits_seeds)
                
    else:
        for _ in range(max_shots):
            shots_idx_ = []
            for _ in range(runs):
                run = []
                for c in range(len(num_elements_per_class_shots)):
                    run_c = torch.randperm(num_elements_per_class_shots[c])
                    size = max_shots+args.evaluation_multi_class_validation_shots
                    if len(run_c) >= size:
                       run_c = run_c[:size]
                    else: 
                        run_c = torch.cat([run_c, torch.randperm(size-num_elements_per_class_shots[c])])
                    run.append(run_c)
                run = torch.stack(run)
                shots_idx_.append(run) 
                
            shots_idx.append(torch.stack(shots_idx_))
            # logger.debug(f'shape: {torch.stack(shots_idx_).shape}')

    return shots_idx, runs

def run_classification(trainer, test_dataset, datasetname, images_shots, args, runs, max_shots, batch_few_shot_runs=100, device='cuda:0'):
    """
        Classification with all classes and all queries using nearest class mean classifier.
    """
    
    num_elements_per_class_shots = [len(images_shots[c]) for c in range(len(images_shots))]
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = min(os.cpu_count(), 8))
    
    if runs > 0:
        shots_idx, runs = get_few_shot_splits(num_elements_per_class_shots, max_shots, runs, datasetname, args)
    
    ##### pad shot features with zeros to have the same number of elements per class
    max_num_elements_shots = max(num_elements_per_class_shots)
    
    for c in range(len(images_shots)):
        images_shots[c] = torch.cat([images_shots[c], torch.zeros(max_num_elements_shots-len(images_shots[c]), images_shots[c].shape[-1])])
    images_shots = torch.stack(images_shots)

    score = [] # one model per run, just fix the seed before every run

    # zero shots
    cache, mean_pred_cached, std_pred_cached = True, None, None
    score_zero_shot, mean_pred_cached_, std_pred_cached_, best_beta,_ = trainer.predict_multi_class(None, None, test_dataset, test_loader, datasetname, mean_pred_cached=mean_pred_cached, std_pred_cached=std_pred_cached, cache=cache)
    mean_pred_cached, std_pred_cached = mean_pred_cached_, std_pred_cached_
    if args.save_predictions:
        if not os.path.exists(args.save_predictions):
            os.makedirs(args.save_predictions)
        if mean_pred_cached is not None: 
            torch.save(mean_pred_cached, os.path.join(args.save_predictions, f'{datasetname}_mean_predictions_{args.epochs}epochs.pt'))
        if std_pred_cached is not None: 
            torch.save(std_pred_cached, os.path.join(args.save_predictions, f'{datasetname}_std_predictions_{args.epochs}epochs.pt'))

    cache = False
    score.append(score_zero_shot.repeat(runs)) # during zero shot generate mean and std of the test set for each query and all classes, then use it later for the hybrid model
    if not torch.isnan(score_zero_shot).any():
        logger.info(f'k={0}: score={mean_conf(score_zero_shot)}, beta={best_beta.mean().item():.3f}')
        if args.wandb:
            wandb.log({'dataset':datasetname,'n_shots':0, f'{datasetname} score mean':score_zero_shot.mean().item(), 'beta':best_beta.item()})

    total_classes = len(images_shots)
    # fix rate of batch_few_shot_runs
    batch_few_shot_runs = max(1, batch_few_shot_runs//total_classes)
    batch_few_shot_runs = min(batch_few_shot_runs, runs)
    if 'logit' in args.trainer: # init the same linear layer for all runs
        # state dict of the linear layer
        W_weights = nn.Linear(images_shots.shape[-1], total_classes, bias=False).state_dict()
    else:
        W_weights = None
    if 'clip' not in args.trainer and runs>0: # clip is not a few-shot method only zero shot
        #k_shots = torch.tensor([1, 2, 4, 8, 16]) if args.evaluation_coop_split>0 else torch.arange(1, max_shots+1)
        k_shots = torch.tensor([16]) if 'inaturalist' not in args.training_dataset else torch.arange(1, max_shots+1)
        i=0
        for _, k in enumerate(torch.arange(1, max_shots+1)):
            if k in k_shots:
                n_val_shots = min(k, args.evaluation_multi_class_validation_shots)
                best_alpha, best_beta = [], []
                score_ = []
                for batch_idx in range(math.ceil(runs/batch_few_shot_runs)): # loop on runs using batches
                    shots_idx_batch =  shots_idx[i][batch_idx * batch_few_shot_runs : (batch_idx + 1) * batch_few_shot_runs] # get bunch of shots
                    shots = torch.gather(images_shots.unsqueeze(0).repeat(batch_few_shot_runs, 1, 1, 1), 2, shots_idx_batch[:,:,:k+n_val_shots].unsqueeze(-1).repeat(1, 1, 1, images_shots.shape[-1])).to(device)
                    shots, validation_shots = shots[:,:,:k], shots[:,:,k:]
                    output = trainer.predict_multi_class(shots, validation_shots, test_dataset, test_loader, datasetname, mean_pred_cached=mean_pred_cached, std_pred_cached=std_pred_cached, cache=cache, W_weights=W_weights)
                    best_alpha.append(output[-2])
                    best_beta.append(output[-1])
                    score_.append(output[0]) # only use k shots
                score_ = torch.cat(score_)
                best_alpha = torch.cat(best_alpha)
                best_beta = torch.cat(best_beta)
                score.append(score_)
                logger.info(f'n_shots={k}: best alpha={best_alpha.mean().item():.3f}, best beta={best_beta.mean().item():.3f} | score={mean_conf(score_)}')
                if args.wandb:
                    if not torch.isnan(score_.sum()):
                        wandb.log({'dataset':datasetname,'n_shots':k, f'{datasetname} alpha':best_alpha.mean().item(), f'{datasetname} beta':best_beta.mean().item(), f'{datasetname} score mean':score_.mean().item(), f'{datasetname} score': score_})
                i+=1
            else:
                score.append(torch.tensor(torch.nan).to(args.device).repeat(runs))
        score = torch.stack(score)
    else:
        # add k empty tensors to the score list
        for k in range(max_shots):
            score.append(torch.tensor(torch.nan).to(args.device).repeat(runs))
        score = torch.stack(score)
    return score.cpu()
def evaluation_n_ways(trainer, test_dataset, shots, args, run_wandb, config, device='cuda:0'):
    n_ways_runs = config['runs']
    figs = []
    df_all = pd.DataFrame([])
    for t, name in enumerate(args.test_dataset):
        logger.info(f'Multiclass: Evaluating on {name}')
        dataset = test_dataset[t] # choose one class in the test set
        images_shots = shots[t]
        total_classes = len(images_shots)
        n_classes = total_classes if config['n_ways'] == 0 else config['n_ways']
        # n_ways_max_shots = min(config['max_shots'], min([len(images_shots[c]) for c in range(total_classes)])-args.evaluation_multi_class_validation_shots)
        n_ways_max_shots = 16
        #n_ways_n_queries = min(config['n_queries'], min([len(images_shots[c]) for c in range(total_classes)])-n_ways_max_shots)
        fix_seed(args.seed)
        score = run_classification(trainer, dataset, name, images_shots, args, n_ways_runs, n_ways_max_shots, batch_few_shot_runs=config['batch_few_shot_runs'], device=device)
        df = pd.DataFrame([])
        conf = lambda x: (1.96 * x.std()/np.sqrt(len(x))).item()
        for s in range(n_ways_max_shots+1): # fill the dataframe
            row = {'shots':s, 'model': args.trainer, 'dataset': name, 'n_ways': n_classes,
                    'score': score[s].mean().item(), 'conf': conf(score[s])}
            df = df._append(row, ignore_index=True)
            df_all = df_all._append(row, ignore_index=True)
        n_ways_conf = True
        fig, ax = plt.subplots(figsize=(20,10))
        k_shots = np.arange(n_ways_max_shots+1)
        new_list = range(math.floor(min(k_shots)), math.ceil(max(k_shots))+1)
        ax.set_xticks(new_list)
        plot_with_conf_interval(k_shots, score.reshape(n_ways_max_shots+1, -1).numpy(), color='blue', label=trainer.label, ax=ax, conf=n_ways_conf)

        ax.legend(loc='lower right')
        ax.set_xlabel('number of shots')
        ax.set_ylabel('Accuracy score')
        ax.set_title(f'Accuracy {n_classes}-class: {name}')
        if args.save_fig_path:
            save_path = os.path.join(args.save_fig_path, run_wandb.name, f'n_ways_{name}.png')
            fig.savefig(save_path)
            figs.append(fig)
            # save table into csv file
            df.to_csv(os.path.join(args.save_fig_path, run_wandb.name, f'{n_classes}_classes_{name}.csv'), decimal=',')
            # if args.wandb != '':
                # run_wandb.log({f'Accuracy {n_classes}-class Image: {name}': [wandb.Image(save_path)], f'Accuracy {n_classes}-class Table: {name}': wandb.Table(data=df, columns=list(df.columns))})
                # run_wandb.log({f'Accuracy {n_classes}-class Table: {name}': wandb.Table(data=df, columns=list(df.columns))})
    if args.save_fig_path:
        df_all.to_csv(os.path.join(args.save_fig_path, run_wandb.name, f'results_n_ways.csv'), decimal=',')
    if args.wandb != '':
        run_wandb.log({f'Table': wandb.Table(data=df_all, columns=list(df_all.columns))})
    
    return figs
def preprocess(x, mean, std):
    x = x-mean
    x = x/std
    x = x/ torch.norm(x, dim=-1, keepdim=True) 
    return x
def run_multi_class_few_shot(trainer, test_dataset, datasetname, images_shots, args, runs, max_shots, batch_few_shot_runs=100, device='cuda:0'):
    selected_classes, shots_idx = [], []
    num_elements_per_class_shots = [len(images_shots[c]) for c in range(len(images_shots))]
    shots_idx = [] # shots_idx : [max_shots+n_queries]
    selected_classes = [] # selected_classes : [runs, n_ways]
    total_classes = len(images_shots)
    for _ in range(max_shots+1): # include zero shot too
        shots_idx_ = []
        selected_classes_ = []
        for _ in range(runs):
            # sample c classes first 
            selected_classes_.append(torch.randperm(total_classes)[:args.evaluation_n_ways])
            shots_idx_.append(torch.stack([torch.randperm(num_elements_per_class_shots[c])[:max_shots+args.evaluation_n_queries] for c in selected_classes_[-1]])) 
        shots_idx.append(torch.stack(shots_idx_))
        selected_classes.append(torch.stack(selected_classes_))
    shots_idx = torch.stack(shots_idx)
    selected_classes = torch.stack(selected_classes)
    # logger.debug(f'shots_idx: {shots_idx.shape}, selected_classes: {selected_classes.shape}')
    ##### pad shot features with zeros to have the same number of elements per class
    max_num_elements_shots = max(num_elements_per_class_shots)
    for c in range(len(images_shots)):
        images_shots[c] = torch.cat([images_shots[c], torch.zeros(max_num_elements_shots-len(images_shots[c]), images_shots[c].shape[-1])])
    images_shots = torch.stack(images_shots)

    total_classes = len(images_shots)
    # fix rate of batch_few_shot_runs
    batch_few_shot_runs = max(1, batch_few_shot_runs//total_classes)
    batch_few_shot_runs = min(batch_few_shot_runs, runs)
    best_scores = []
    # perform zero shot first then cache mean if needed
    _, mean_pred_cached, std_pred_cached, _, _ = trainer.predict_multi_class(None, None, test_dataset, None, datasetname, mean_pred_cached=None, std_pred_cached=None, cache=True, openset=True)
    std_pred_cached = test_dataset.class_std.to(device) if trainer.train_std else std_pred_cached
    max_shots_ = 0 if 'clip' in args.trainer else max_shots # only zero-shot for clip
    min_shots = 0 if trainer.train_mean else 1
    if trainer.preprocess: 
        base_mean = trainer.train_dataset.features_images.mean(0).reshape(1,1,-1).to(device)
    for k, n_shots in enumerate(range(min_shots, max_shots_+1)):  
        best_alpha, best_beta, best_score_k = -1, -1, torch.tensor([-1.]) 
        alphas = (torch.arange(0, 1.05, 0.05) if trainer.train_mean and trainer.default_alpha==0 and n_shots>0 else torch.tensor([trainer.default_alpha]).float()).to(args.device)
        betas = (torch.arange(0, 1.05, 0.05) if trainer.train_std and trainer.default_beta==0 else torch.tensor([trainer.default_beta]).float()).to(args.device)
        combinations = torch.cartesian_prod(alphas, betas)
        for j,(alpha, beta) in enumerate(combinations):
            score = []
            for batch_idx in range(math.ceil(runs/batch_few_shot_runs)): # loop on runs using batches
                shots_idx_batch =  shots_idx[n_shots]
                shots = []
                cached_mean, cached_std = [], []
                for b in range(batch_idx * batch_few_shot_runs,(batch_idx + 1) * batch_few_shot_runs):
                    shots_ = torch.gather(images_shots[selected_classes[n_shots,b]], 1, shots_idx_batch[b,:,:n_shots+args.evaluation_n_queries].unsqueeze(-1).repeat(1, 1, images_shots.shape[-1])).to(device)
                    shots.append(shots_)
                    if trainer.train_mean: 
                        cached_mean.append(mean_pred_cached[selected_classes[n_shots][b]])
                    if trainer.train_std:
                        cached_std.append(std_pred_cached[selected_classes[n_shots][b]])
                cached_mean = torch.stack(cached_mean) if trainer.train_mean else []
                cached_std = torch.stack(cached_std) if trainer.train_std else []
                shots = torch.stack(shots)
                shots, queries = shots[:,:,:n_shots].to(device), shots[:,:,n_shots:].to(device)
                labels = torch.arange(args.evaluation_n_ways).repeat_interleave(args.evaluation_n_queries).unsqueeze(0).repeat(batch_few_shot_runs, 1).to(device)
                queries = queries.reshape(batch_few_shot_runs, -1, queries.shape[-1])
                if n_shots>0:
                    centroids = shots.mean(dim=2)
                if trainer.train_mean: 
                    if alpha >0 or n_shots==0:
                        if n_shots>0: 
                            centroids = alpha*cached_mean + (1-alpha)*centroids
                        else:
                            centroids = cached_mean
                if trainer.train_std:
                    stds = (1-beta)+beta*cached_std
                else:
                    stds = torch.ones(centroids.shape).to(device)
                # preprocess the embeddings: 
                queries = queries.unsqueeze(2) 
                if trainer.preprocess:
                    queries = preprocess(queries, base_mean, stds.unsqueeze(1))
                    centroids = preprocess(centroids, base_mean, stds)
                else: 
                    queries = queries/stds.unsqueeze(1)
                    centroids = centroids/stds
                distance = torch.norm((queries-centroids.unsqueeze(1)), dim = -1)
                winners = torch.argmin(distance, dim=-1)
                score_ = (winners == labels).float().mean(dim=-1).cpu()
                score.append(score_)
            score = torch.cat(score)
            if score.mean() > best_score_k.mean():
                best_score_k = score.clone()
                best_alpha = alpha
                best_beta = beta
            if j == 0:
                score_ncm = score
            if j == len(combinations)-1:
                score_text = score.clone()
        best_scores.append(best_score_k)
        logger.info(f'{n_shots}-shots: alpha={best_alpha:.1f}, beta={best_beta:.1f} | score ncm: {score_ncm.mean().item():.3f}|score mix={best_score_k.mean().item():.3f}|score text={score_text.mean().item():.3f}')
        if args.wandb:
            wandb.log({'dataset':datasetname,'n_shots':k, f'{datasetname} alpha':best_alpha, f'{datasetname} beta':best_beta, 'runs':runs, f'{datasetname} score mean':best_score_k.mean().item(), f'{datasetname} score':best_score_k, f'{datasetname} score text mean': score_text.mean().item(), f'{datasetname} score text': score_text.item(), f'{datasetname} score ncm mean':score_ncm.mean().item(), f'{datasetname} score ncm':score_ncm.item()})
    if max_shots_<max_shots:
        best_scores = best_scores + [torch.tensor([torch.nan]*runs)]*(max_shots-max_shots_)
    if min_shots>0:
        best_scores = [torch.tensor([torch.nan]*runs)]*min_shots + best_scores
    return torch.stack(best_scores)
def evaluation_multi_class_few_shot(trainer, test_dataset, shots, args, run_wandb, config, device='cuda:0'):
    """
        Multi-class but with a fixed number of queries per episode (typical few shot setting :))
    """
    figs = []
    df_all = pd.DataFrame([])
    for t, name in tqdm(enumerate(args.test_dataset)):
        logger.info(f'Few shot multi-class: Evaluating on {name}')
        dataset = test_dataset[t] # choose one class in the test set
        images_shots = shots[t]
        total_classes = len(images_shots)
        max_shots = min(config['max_shots'], min([len(images_shots[c]) for c in range(len(images_shots))]))
        n_classes = total_classes if config['n_ways'] == 0 else config['n_ways']
        fix_seed(args.seed)
        score = run_multi_class_few_shot(trainer, dataset, name, images_shots, args, config['runs'], max_shots, batch_few_shot_runs=config['batch_few_shot_runs'], device=device)
        df = pd.DataFrame([])
        conf = lambda x: (1.96 * x.std()/np.sqrt(len(x))).item()
        for s in range(max_shots+1): # fill the dataframe
            row = {'shots':s, 'model': args.trainer, 'dataset': name,
                    'score': score[s].mean().item(), 'conf': conf(score[s])}
            df = df._append(row, ignore_index=True)
            df_all = df_all._append(row, ignore_index=True)
           
        conf = True
        fig, ax = plt.subplots(figsize=(20,10))
        k_shots = np.arange(max_shots+1)
        new_list = range(math.floor(min(k_shots)), math.ceil(max(k_shots))+1)
        ax.set_xticks(new_list)
        plot_with_conf_interval(k_shots, score.reshape(max_shots+1, -1).numpy(), color='blue', label=trainer.label, ax=ax, conf=conf)
        ax.legend(loc='lower right')
        ax.set_xlabel('number of shots')
        ax.set_ylabel('Accuracy score')
        ax.set_title(f'Accuracy {n_classes}-class: {name}')
        if args.save_fig_path:
            save_path = os.path.join(args.save_fig_path, run_wandb.name, f'n_ways_{name}.png')
            fig.savefig(save_path)
            figs.append(fig)
            df.to_csv(os.path.join(args.save_fig_path, run_wandb.name, f'{n_classes}_classes_{name}.csv'), decimal=',')
    if args.save_fig_path:
        df_all.to_csv(os.path.join(args.save_fig_path, run_wandb.name, f'results_n_ways.csv'), decimal=',')
    if args.wandb != '':
        run_wandb.log({f'Table': wandb.Table(data=df_all, columns=list(df_all.columns))})
def run_evaluation(trainer, test_dataset, shots, args, run_wandb, openset_config, n_ways_config, few_shot_n_ways_config):
    figs_n_ways = evaluation_n_ways(trainer, test_dataset, shots, args, run_wandb=run_wandb, device=args.device, config=n_ways_config) if n_ways_config['runs'] > 0 else None
    figs_openset = evaluation_openset(trainer, test_dataset, shots, args, run_wandb=run_wandb, device=args.device, config=openset_config) if openset_config['runs'] > 0 else None
    figs_n_ways_few_shot = evaluation_multi_class_few_shot(trainer, test_dataset, shots, args, run_wandb=run_wandb, device=args.device, config=few_shot_n_ways_config) if few_shot_n_ways_config['runs'] > 0 else None
    
    return figs_openset, figs_n_ways, figs_n_ways_few_shot

def classes_to_df(classes):
    """
    Given a list of classes, return a dataframe with the classes split into columns.
    """
    df_classes = pd.DataFrame({'name':classes})
    df_classes['identifier'] = df_classes['name'].str.split('_').str[0]
    df_classes['kingdom'] = df_classes['name'].str.split('_').str[1]
    df_classes['phylum'] = df_classes['name'].str.split('_').str[2]
    df_classes['class'] = df_classes['name'].str.split('_').str[3]
    df_classes['order'] = df_classes['name'].str.split('_').str[4]
    df_classes['family'] = df_classes['name'].str.split('_').str[5]
    df_classes['scientificName'] = df_classes['name'].str.split('_').str[5:].apply(lambda x: ' '.join(x if len(x)==2 else x[-2:]))
    df_classes.scientificName = df_classes.scientificName.astype("string")
    return df_classes


def classes_to_df(classes):
    """
    Given a list of classes, return a dataframe with the classes split into columns.
    """
    df_classes = pd.DataFrame({'name':classes})
    df_classes['identifier'] = df_classes['name'].str.split('_').str[0]
    df_classes['kingdom'] = df_classes['name'].str.split('_').str[1]
    df_classes['phylum'] = df_classes['name'].str.split('_').str[2]
    df_classes['class'] = df_classes['name'].str.split('_').str[3]
    df_classes['order'] = df_classes['name'].str.split('_').str[4]
    df_classes['family'] = df_classes['name'].str.split('_').str[5]
    df_classes['scientificName'] = df_classes['name'].str.split('_').str[5:].apply(lambda x: ' '.join(x if len(x)==2 else x[-2:]))
    df_classes.scientificName = df_classes.scientificName.astype("string")
    return df_classes

