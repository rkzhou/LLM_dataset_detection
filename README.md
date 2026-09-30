# Blackbox Dataset Inference for LLM

***Paper title: Blackbox Dataset Inference for LLM***

This repo contains code that allows you to reproduce experiments presented in the paper.

## Environment Setup

Opearting system: Ubuntu

CPU: Intel(R) Xeon(R) w5-3435X

Graphics card: NVIDIA RTX A6000

RAM: 128GB

## File Illustration

### In "main" Folder:
1. **dataset.py: download and preprocess datasets**
2. **filter.py: pick tainted samples from datasets**
3. **measurement.py: train independent models**
4. **reference.py: train and inference reference models**
5. **suspect_model.py: some pre-defined templates used in obtaining generations from models**
6. **suspect_output.py: obtain generations from models**
7. **utils.py: helping functions used in other scripts**

### In "setting" Folder:
Note: the duplicated name of variables in different config files have the same meanings.

1. **dataset_config.yaml: config used by dataset.py**
   - dataset_alias: self-defined alias of target datasets
   - subset_size: fraction of target dataset used in the further evaluation
   - dataset_name: Hugging Face dataset name
   - raw_dataset_path: save/load path of original target datasets
   - general_dataset_path: save/load path of pre-processed target datasets
   - format_dataset_path: save/load path of formatted target datasets
   - partition_general_dataset_dir: save/load path of splits of target datasets
   - partition_format_dataset_dir: save/load path of formatted splits of target datasets
   - partition_ratio: ratios of each split of target datasets
   - action: the functions you want to run
2. **filter_config.yaml: config used by filter.py and measurement.py**
   - bare_answer_dir: save/load path of completions from reference models which are not trained on target datasets
   - finetune_answer_dir: save/load path of completions from reference models which are trained on target datasets
   - tainted_sample_dir: save/load path of tained sample obtained by comparing responses from reference models
   - bare_model_list: alias of reference models (not trained on target datasets) you want to use to obtain tainted samples and detect suspect models
   - finetune_model_list: alias of reference models (trained on target datasets) you want to use to compare against models in "bare_model_list"
   - suspect_answer_dir: save/load path of completions from suspect models
3. **reference_config.yaml: config used by reference.py**
   - model_alias: self-defined alias of reference models
   - input_dataset_path: save/load path of target dataset used to finetune or inference reference models
   - model_name: Hugging Face model name
   - model_version: "bare" for reference models not trained on target datasets, "finetune" for reference models trained on target datasets
   - model_action: "train" for fine-tuning reference models, "predict" for inferencing reference models
   - model_output_dir: save/load path of fine-tuned reference models
   - bare_prediction_dir: save/load path of completions from reference models which are not trained on target datasets
   - finetune_prediction_dir: save/load path of completions from reference models which are trained on taret datasets
   - bare_split_mark: for instruction-version of reference models, they have own specific templates, so we need to identify the beginning symbol of real answer part
4. **suspect_config.yaml: config used by suspect_output.py**
   - tainted_sample_path: save/load path of tainted samples regarding each target dataset
   - model_type: "pipeline" for loading models which are suggested by using Hugging Face pipeline function, "kernel" for loading models directly
   - model_template: the suitable prompt templates used for inferencing suspect models
   - split_symbol: for suspect models, they also have own specific templates, so we need to identify the beginning symbol of real answer part

Other variables in the config files can be easily understood by their names.


### Run:
To go through the completed process of the proposed method, you have to run the following python scripts in order:
1. dataset.py: preprocess target datasets
2. reference.py: fine-tune and inference reference models
3. filter.py: obtain tainted samples by comparing completions from reference model sets
4. suspect_output.py: obtain responses from suspect models regarding tainted samples
5. measurement.py: judge if suspect models are members or non-members

## Results Viewing
After running filter.py, you can know the number of selected tainted samples.

After running measurement.py, you can konw the predictions of suspect models by the proposed method.


