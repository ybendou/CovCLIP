import torch
import json
import os
dataset = 'caltech101'
path = f'../../../data/indices/{dataset}/'
seed = 1
n_shots = 16
indices = json.load(open(os.path.join(path, f'shot_{n_shots}-seed_{seed}.json'), 'r'))
indices.keys()
len(indices['train']['indices'])
map_datasets_to_splits = {
    'caltech101':['split_zhou_Caltech101.json', '101_ObjectCategories'],
    'dtd':['split_zhou_DescribableTextures.json', 'images'],
    'eurosat': ['split_zhou_EuroSAT.json', '2750'],
    'food-101': ['split_zhou_Food101.json', 'images'],
    'oxford_flowers': ['split_zhou_OxfordFlowers.json', 'jpg'], # all images are in the same folder 
    'oxford_pets': ['split_zhou_OxfordPets.json', 'images'], # all images are in the same folder
    'stanford_cars': ['split_zhou_StanfordCars.json', ''],
    'sun397': ['split_zhou_SUN397.json', 'SUN397'],
    'ucf101': ['split_zhou_UCF101.json', 'UCF-101-midframes']
}

# get dataset json 
splits = json.load(open(os.path.join(f'../../../data/cross_modal_splits/{map_datasets_to_splits[dataset][0]}'), 'r'))

n_classes = len(indices['train']['indices'])//16
torch.tensor(indices['train']['indices'])//16

n=2

from collections import OrderedDict, Counter
num_elements_per_class_train = torch.tensor(list(OrderedDict(Counter([i[1] for i in splits['train']])).values()))
num_elements_per_class_val = torch.tensor(list(OrderedDict(Counter([i[1] for i in splits['val']])).values()))

shots_idx = []
for n_shots in [1, 2, 4, 16]:
    indices = json.load(open(os.path.join(path, f'shot_{n_shots}-seed_{seed}.json'), 'r'))
    n_val = min(4, n_shots)
    shots_idx_ = []
    for n in range(n_classes):
        shots_idx_one_class = torch.tensor(indices['train']['indices'][n*n_shots:(n+1)*n_shots])-num_elements_per_class_train[:n].sum()
        val_idx_one_class = torch.tensor(indices['val']['indices'][n*n_val:(n+1)*n_val])-num_elements_per_class_val[:n].sum()
        shots_idx_.append(torch.cat([shots_idx_one_class, val_idx_one_class]))
    shots_idx.append(torch.stack(shots_idx_))