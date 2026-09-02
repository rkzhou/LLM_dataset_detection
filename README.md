# Blackbox Dataset Inference for LLM

***Paper title: Blackbox Dataset Inference for LLM***

This paper was pre-printed in Arxiv. 

This repo contains code that allows you to reproduce experiments presented in the paper.

## Environment Setup

Opearting system: Ubuntu

CPU: Intel(R) Xeon(R) w5-3435X

Graphics card: NVIDIA RTX A6000

RAM: 128GB

You need to install some third-party libraries with the following command:

```
conda env create -f environment.yml
conda activate your_env_name
```

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
1. **dataset_config.yaml: config used by dataset.py**
2. **filter_config.yaml: config used by filter.py**
3. **reference_config.yaml: config used by reference.py**
4. **suspect_config.yaml: config used by suspect_output.py**

Variables in the config files can be easily understood by their names.

## Results Viewing
After running filter.py, we can know the number of selected tainted samples.

After running measurement.py, we can konw the predictions of suspect models by the proposed method.

## Citation
If you find several components of this work useful or want to use this code in your research, please cite the following paper:
@article{zhou2025blackbox,\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;title={Blackbox dataset inference for LLM},\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;author={Zhou, Ruikai and Yang, Kang and Chen, Xun and Wang, Wendy Hui and Tao, Guanhong and Xu, Jun},\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;journal={arXiv preprint arXiv:2507.03619},\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;year={2025}\
}
