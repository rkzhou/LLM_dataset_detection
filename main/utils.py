import pickle
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def save_hf_dataset(dataset_name, save_path):
    dataset = load_dataset(dataset_name)
    dataset_save_root = save_path.split("/")
    dataset_save_root = dataset_save_root[:-1]
    dataset_save_root = os.path.join(*dataset_save_root)

    if not os.path.exists(dataset_save_root):
        os.makedirs(dataset_save_root)
    
    with open(save_path, 'wb') as file:
        pickle.dump(dataset, file)
    return dataset


def load_local_dataset(load_path):
    with open(load_path, 'rb') as file:
        dataset = pickle.load(file)
    
    return dataset


def get_dataset(dataset_name, path):
    if os.path.exists(path):
        dataset = load_local_dataset(path)
    else:
        dataset = save_hf_dataset(dataset_name, path)
    
    return dataset


def get_pretrained_model_and_tokenizer(model_id):
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16"
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=bnb_config, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer