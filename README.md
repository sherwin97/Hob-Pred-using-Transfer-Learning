# Evaluating the Use of GNNs and Transfer Learning for Oral Bioavailability Prediction

This repository is the official implementation of (under review)

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

This repository consists of 4 folders:

1. data 
    1. graph_data: csv files of solubility dataset and oral bioavailability dataset
    2. oral_avail_fingerprints: csv files of oral bioavailability dataset fingerprints
    3. oral_mol_desc: csv files of oral bioavailability dataset molecular descriptors 

2. notebooks --> mainly 3 parts to it as listed below.
    1. Random forest model
    2. GNN from scratch models 
    3. Transfer learning models 
    
3. pretrained_models
    1. Models that were pre-trained with solubility dataset 
    
## Requirements

To install requirements:

```setup
conda env create -f environment.yml
conda activate hobpred
```


