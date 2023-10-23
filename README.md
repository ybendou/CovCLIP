# Inferring Latent Class Statistics from Text for Robust Visual Few-Shot Learning

Official code for the paper "Inferring Latent Class Statistics from Text for Robust Visual Few-Shot Learning".

## Datasets 
First download the datasets by following the steps described [here](https://github.com/linzhiqiu/cross_modal_adaptation/blob/main/DATASETS.md) for cross domain and [here](https://github.com/visipedia/inat_comp/tree/master/2021#data) for iNaturalist dataset.

## How to run? 
We provide one bash script for all our experiments with different arguments. 

    cd bash_scripts
    ./run.sh --trainer <trainer> --text-type <text_type>

There are several trainers: 

- "ncm": Baseline of nearest class mean classifier. 
- "ncm_mean": Predicting mean from text. 
- "ncm_std": Predicting covariance from text.
- "ncm_mean_std": Predicting mean and covariance from text.

You can either use class labels --text-type class_label or class descriptions --text-type description.

To run on iNaturalist: 
    
    ./run.sh --text-type description --runs-multi-class 100 --runs-open-set 1000 --inaturalist --in-domain --seed 1 --epochs 50 --trainer <trainer>;

To run on cross-domain: 

    ./run.sh --text-type description --runs-multi-class 100 --runs-open-set 1000 --seed 1 --epochs 50 --trainer <trainer>;


