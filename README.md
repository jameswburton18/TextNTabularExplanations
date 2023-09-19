# TextNTabular Explanations

## Summary

In this repository we make it possible to generate SHAP explanations for multimodal text-tabular datasets, no matter how the two modalities are combined. The accompanying paper will be linked. All trained models and processed datasets are available on our huggingface repo, will be linked.

## Datasets

All unprocessed datasets are publically available using the library. All preprocessing steps are found in `src/create_datasets.py` and uploaded to huggingface on creation. We undertake minimal processing, but the processed datasets can be found on our huggingface repo. We use `configs/dataset_configs.yaml` to define the properties of each dataset. Note that `tnt_reorder_cols` and `base_reorder_cols` were only used for auxiliary experiments and can be ignored.

## How to train the models

`src/train.py` contains the code to train the text models. We train two text models for each dataset, one which uses just the text columns, to be used for Weighted and Stack Ensembles and one which uses all the columns, to be used for the All-Text combination method. The parameters are all set in the `configs/train_configs.yaml`, which is loaded `ConfigLoader`, updating the default config in `configs/train_default.yaml`. To train a model, run `python src/train.py --config <config_name>`, where `config_name` relates to the name of a config in `configs/train_configs.yaml`. We used `scripts/batch_train.sh` and `scripts/train.sh` to train the models on a cluster.

## How to generate the explanations

Our joint masker class, defined in `src/joint_masker.py` can be used for any combination method for any text tabular model. In `src/run_shap.py` we use the masker to generate explanations for Weighted-Ensembles with 3 different weights, a Stack Ensemble and an All-Text method. We define these models in `src/models.py` and show how to get the shap explanations in the `run_shap()` method. We use `scripts/batch_run.sh` and `scripts/run.sh` to generate the explanations on a cluster and use `configs/shap_configs.yaml` to define the parameters for the explanations. Note that we also have a `run_all_text_baseline_shap()` method which is a bit less intuitive, but is only used to generate the All-Text (Unimodal) explanations, for when we are using the old SHAP library masker. 

## How to generate the plots

We do some preliminary analysis in `notebooks/top_tokens_&_length_anlys.ipynb` and generate a csv file of explations grouped by feature that is used in `notebooks/TextNTabStats.R` to generate the plots. Analysis in `notebooks/compare_combined_shap_plots.ipynb` shows how to use the plotter functions that are defined in `src/plot_text.py`.

## Setup

This repo relies upon sideways imports, so after cloning the repo, begin with

`pip install -e .` from the root directory.

Then run `pip install -r requirements.txt` to install the required packages.