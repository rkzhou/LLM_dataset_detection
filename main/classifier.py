import utils
import os
import yaml
import preprocess
import random
import math
import transformers
import peft
import wandb
import torch

from tqdm import tqdm

def split_dataset(dataset, split_ratio):
    category_index = dict()
    for i in range(len(dataset["train"])):
        data = dataset["train"][i]
        if data["category"] in category_index.keys():
            category_index[data["category"]].append(i)
        else:
            category_index.update({data["category"]: [i]})
    
    train_index, test_index = list(), list()
    for key, value in category_index.items():
        random.shuffle(value)
        number = len(value)
        train_num = math.ceil(number * split_ratio)
        train_index += value[:train_num]
        test_index += value[train_num:]
    
    train_dataset = dataset["train"].select(train_index)
    test_dataset = dataset["train"].select(test_index)
    
    return train_dataset, test_dataset


def format_dataset(data):
    prompt_prefix = "From the list: [closed_qa, open_qa, general_qa, classification, information_extraction, brainstorming, summarization, creative_writing], please categorize the following question. The question is: "
    response_prefix = "The category for the given question is only "
    data["classifier_prompt"] = "<s>[INST] {}{} {} [/INST]".format(prompt_prefix, data["context"], data["instruction"])
    data["classifier_response"] = "{}{}</s> ".format(response_prefix, data["category"])
    
    return data


def tokenize_data(data, tokenizer):
    encoded_prompt = tokenizer.encode(data["classifier_prompt"], add_special_tokens=False)
    encoded_response = tokenizer.encode(data["classifier_response"], add_special_tokens=False)

    sample = {
        "input_ids": encoded_prompt + encoded_response,
        "attention_mask": [1] * (len(encoded_prompt) + len(encoded_response)),
        "labels": [-100] * len(encoded_prompt) + encoded_response,
    }

    return sample


def transform_to_tensor(data):
    sample = {}
    for key, value in data.items():
        temp = torch.tensor(value, device="cuda")
        sample.update({key: temp})
    
    return sample


def init_wandb():
    os.environ["WANDB_PROJECT"]="Question_Classifier"
    os.environ["WANDB_LOG_MODEL"]="true"
    os.environ["WANDB_WATCH"]="false"


def train_model(args):
    init_wandb()

    classifier = preprocess.Chatmodel_0(args["model_name"])
    classifier.tokenizer.padding_side = 'left'
    if classifier.tokenizer.pad_token is None:
        classifier.tokenizer.pad_token = classifier.tokenizer.eos_token

    dataset = utils.get_dataset(args["dataset_name"], args["dataset_path"])
    random.seed(args["seed_index"])
    train_dataset, _ = split_dataset(dataset, 0.9)
    train_dataset = train_dataset.map(format_dataset)
    encoded_train_dataset = train_dataset.map(tokenize_data, batched=False, remove_columns=train_dataset.column_names, fn_kwargs={"tokenizer": classifier.tokenizer})

    total_num = len(encoded_train_dataset)
    filter_dataset = encoded_train_dataset.filter(lambda sample: len(sample['input_ids']) <= 512)
    print("Data Ratio that exceed max length:{}".format((total_num-len(filter_dataset))/total_num))
    
    classifier.model = peft.prepare_model_for_kbit_training(classifier.model)
    lora_config = peft.LoraConfig(
    r=args["base_lora"]["r"],  # dimension of the updated matrices
    lora_alpha=args["base_lora"]["lora_alpha"],  # parameter for scaling
    lora_dropout=args["base_lora"]["lora_dropout"],  # dropout probability for layers
    bias=args["base_lora"]["bias"],
    task_type=args["base_lora"]["task_type"]
    )
    classifier.model = peft.get_peft_model(classifier.model, lora_config)

    data_collator = transformers.DataCollatorForSeq2Seq(classifier.tokenizer)
    training_config = transformers.TrainingArguments(
        output_dir=args[classifier.name]["output_dir"],
        per_device_train_batch_size=args[classifier.name]["per_device_train_batch_size"],
        optim=args[classifier.name]["optim"],
        num_train_epochs=args[classifier.name]["num_train_epochs"],
        save_strategy=args[classifier.name]["save_strategy"],
        learning_rate=args[classifier.name]["learning_rate"],
        lr_scheduler_type=args[classifier.name]["lr_scheduler_type"]
    )
    trainer = transformers.Trainer(
        model=classifier.model,
        train_dataset=filter_dataset,
        args=training_config,
        data_collator=data_collator
    )

    classifier.model.config.use_cache = False
    trainer.train()
    

def test_model(args):
    classifier = preprocess.Chatmodel_0(args["model_name"])
    classifier.tokenizer.padding_side = 'left'
    if classifier.tokenizer.pad_token is None:
        classifier.tokenizer.pad_token = classifier.tokenizer.eos_token
        
    classifier.model = peft.PeftModel.from_pretrained(classifier.model, "../model/checkpoint-2")
    dataset = utils.get_dataset(args["dataset_name"], args["dataset_path"])
    random.seed(args["seed_index"])
    _, test_dataset = split_dataset(dataset, 0.9)
    test_dataset = test_dataset.map(format_dataset)
    encoded_test_dataset = test_dataset.map(tokenize_data, batched=False, remove_columns=test_dataset.column_names, fn_kwargs={"tokenizer": classifier.tokenizer})
    
    total_num, correct_num = 0, 0
    for i in tqdm(range(len(encoded_test_dataset))):
        data_tensors = {key: torch.tensor(value, device="cuda").unsqueeze(0) for key, value in encoded_test_dataset[i].items()}
        generated_ids = classifier.model.generate(**data_tensors, max_new_tokens=512, do_sample=True, temperature=0.2, top_k=50, top_p=0.95)
        responses = classifier.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        total_num += 1
        if test_dataset[i]["category"] in responses[0]:
            correct_num += 1
    
    correct_ratio = correct_num / total_num * 100.0
    print(correct_ratio)
    


if __name__ == '__main__':
    with open(os.path.join("../setting", "classifier_config.yaml"), 'r') as file:
        global_cfg = yaml.safe_load(file)
    # train_model(global_cfg)
    test_model(global_cfg)