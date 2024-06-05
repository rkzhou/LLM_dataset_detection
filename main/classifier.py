import utils
import os
import yaml
import preprocess
import random
import math
import transformers
import peft
import datasets
import pickle

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


def change_category_value(example, flag, label):
    if flag == "positive":
        example["category"] = label
    elif flag == "negative":
        example["category"] = "non_{}".format(label)
    else:
        raise ValueError("Invalid flag value")
    return example

def format_dataset(data, dataset, flag, target_category):
    non_target_category = "non_{}".format(target_category)
    prompt_prefix = "Please judge the following question is a {} question or {} question. The question is:".format(target_category, non_target_category)
    response_prefix = "The category for the given question is "

    if dataset == "dd_15k":
        if flag == "train":
            data["classifier_prompt"] = "<s>[INST] {} {} {} [/INST]".format(prompt_prefix, data["context"], data["instruction"])
            data["classifier_response"] = "{}{}</s> ".format(response_prefix, data["category"])
        elif flag == "test":
            data["classifier_prompt"] = "<s>[INST] {} {} {} [/INST]".format(prompt_prefix, data["context"], data["instruction"])
        else:
            raise ValueError("Invalid flag attribute")
    elif dataset == "alpaca":
        data["classifier_prompt"] = "<s>[INST] {} {} {} [/INST]".format(prompt_prefix, data["instruction"], data["input"])
    elif dataset == "slimorca":
        data["classifier_prompt"] = "<s>[INST] {} {} [/INST]".format(prompt_prefix, data["conversations"][1]["value"])
    elif dataset == "ultrafeedback":
        data["classifier_prompt"] = "<s>[INST] {} {} [/INST]".format(prompt_prefix, data["prompt"])
    else:
        raise ValueError("Invalid dataset name")
    
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


def init_wandb():
    os.environ["WANDB_PROJECT"]="QA_Classifier"
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
    select_categories = ["closed_qa", "open_qa", "general_qa"]
    label1_dataset = dataset.filter(lambda example: example["category"] in select_categories) # target category
    label0_dataset = dataset.filter(lambda example: example["category"] not in select_categories)

    label1_dataset = label1_dataset.map(change_category_value, fn_kwargs={"flag": "positive", "label": "qa"})
    label0_dataset = label0_dataset.map(change_category_value, fn_kwargs={"flag": "negative", "label": "qa"})
        
    label1_train_dataset, _ = split_dataset(label1_dataset, 0.9)
    label0_train_dataset, _ = split_dataset(label0_dataset, 0.9)
    train_dataset = datasets.concatenate_datasets([label0_train_dataset, label1_train_dataset])
    
    train_dataset = train_dataset.map(format_dataset, fn_kwargs={"dataset": "dd_15k", "flag": "train", "target_category": "qa"})
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
    # trainer.train(resume_from_checkpoint = "../model/open_qa/checkpoint-2418")


def test_model(args):
    classifier = preprocess.Chatmodel_0(args["model_name"])
    classifier.tokenizer.padding_side = 'left'
    if classifier.tokenizer.pad_token is None:
        classifier.tokenizer.pad_token = classifier.tokenizer.eos_token
        
    classifier.model = peft.PeftModel.from_pretrained(classifier.model, "../model/all_qa/checkpoint-2015")
    dataset = utils.get_dataset(args["dataset_name"], args["dataset_path"])
    random.seed(args["seed_index"])
    select_categories = ["closed_qa", "open_qa", "general_qa"]
    label1_dataset = dataset.filter(lambda example: example["category"] in select_categories) # target category
    label0_dataset = dataset.filter(lambda example: example["category"] not in select_categories)

    label1_dataset = label1_dataset.map(change_category_value, fn_kwargs={"flag": "positive", "label": "qa"})
    label0_dataset = label0_dataset.map(change_category_value, fn_kwargs={"flag": "negative", "label": "qa"})
        
    _, label0_test_dataset = split_dataset(label0_dataset, 0.9)
    _, label1_test_dataset = split_dataset(label1_dataset, 0.9)
    test_dataset = datasets.concatenate_datasets([label0_test_dataset, label1_test_dataset])

    test_dataset = test_dataset.shuffle(args["seed_index"])
    test_dataset = test_dataset.map(format_dataset, fn_kwargs={"dataset": "dd_15k", "flag": "test", "target_category": "qa"})

    TP, FN = 0, 0
    FP, TN = 0, 0
    group_num = math.ceil(len(test_dataset) / args["predict_batch_size"])
    for i in tqdm(range(group_num)):
        start_index = i * args["predict_batch_size"]
        end_index = (i + 1) * args["predict_batch_size"]
        end_index = min(len(test_dataset), end_index)

        prompt_list, category_list = list(), list()
        for j in range(start_index, end_index):
            prompt_list.append(test_dataset[j]["classifier_prompt"])
            category_list.append(test_dataset[j]["category"])
        
        encoded_inputs = classifier.tokenizer(prompt_list, padding=True, return_tensors='pt', add_special_tokens=False)
        encoded_inputs = {key: value.to("cuda") for key, value in encoded_inputs.items()}
        generated_ids = classifier.model.generate(**encoded_inputs, max_new_tokens=512, do_sample=True, temperature=0.2, top_k=50, top_p=0.95)
        responses = classifier.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answers = classifier.pull_answer(responses, args[classifier.name]["answer_prefix"])

        for j in range(len(answers)):
            if answers[j] == "qa" and category_list[j] == "qa":
                TP += 1
            elif answers[j] == "qa" and category_list[j] != "qa":
                FP += 1
            elif answers[j] != "qa" and category_list[j] == "qa":
                FN += 1
            else:
                TN += 1
            
        
    accuracy = (TP + TN) / (TP + FP + FN + TN) * 100.0
    precision = TP / (TP + FP) * 100.0
    recall = TP / (TP + FN) * 100.0
    f1_score = (2.0 * precision * recall) / (precision + recall)

    print("accuracy:", accuracy)
    print("precision:", precision)
    print("recall:", recall)
    print("f1_score:", f1_score)


def predict_model(args):
    classifier = preprocess.Chatmodel_0(args["model_name"])
    classifier.tokenizer.padding_side = 'left'
    if classifier.tokenizer.pad_token is None:
        classifier.tokenizer.pad_token = classifier.tokenizer.eos_token
        
    classifier.model = peft.PeftModel.from_pretrained(classifier.model, "../model/all_qa/checkpoint-2015")
    dataset = utils.get_dataset(args["dataset_name"], args["dataset_path"])
    if args["dataset_name"] == "databricks/databricks-dolly-15k":
        dataset = dataset["train"]
    elif args["dataset_name"] == "tatsu-lab/alpaca":
        dataset = dataset["train"]
    elif args["dataset_name"] == "Open-Orca/SlimOrca":
        dataset = dataset["train"]
    elif args["dataset_name"] == "HuggingFaceH4/ultrafeedback_binarized":
        dataset = dataset["train_sft"]

    random.seed(args["seed_index"])
    
    dataset = dataset.shuffle(args["seed_index"])
    dataset = dataset.select([i for i in range(15000)])
    
    if not os.path.exists(args["prediction_save_root"]):
        os.mkdir(args["prediction_save_root"])
    
    save_suffix = args["dataset_path"].split("/")[-1].replace(".pkl", "")
    selected_data_save_path = args["prediction_save_root"] + save_suffix + "_selected_datasets.pkl"

    with open(selected_data_save_path, "wb") as file:
        pickle.dump(dataset, file)
    
    dataset = dataset.map(format_dataset, fn_kwargs={"dataset": save_suffix, "flag": "test", "target_category": "qa"})
    group_num = math.ceil(len(dataset) / args["predict_batch_size"])
    
    all_prompt_list, all_answer_list = list(), list()
    for i in tqdm(range(group_num)):
        start_index = i * args["predict_batch_size"]
        end_index = (i + 1) * args["predict_batch_size"]
        end_index = min(len(dataset), end_index)
        prompt_list = list()
        for j in range(start_index, end_index):
            prompt_list.append(dataset[j]["classifier_prompt"])

        encoded_inputs = classifier.tokenizer(prompt_list, padding=True, return_tensors='pt', add_special_tokens=False)
        encoded_inputs = {key: value.to("cuda") for key, value in encoded_inputs.items()}
        generated_ids = classifier.model.generate(**encoded_inputs, max_new_tokens=512, do_sample=True, temperature=0.2, top_k=50, top_p=0.95)
        responses = classifier.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answers = classifier.pull_answer(responses, args[classifier.name]["answer_prefix"])

        all_prompt_list += prompt_list
        all_answer_list += answers
    
    prediction_prompts_save_path = args["prediction_save_root"] + save_suffix + "_prompts.pkl"
    with open(prediction_prompts_save_path, "wb") as file:
        pickle.dump(all_prompt_list, file)
    
    prediction_answers_save_path = args["prediction_save_root"] + save_suffix + "_answers.pkl"
    with open(prediction_answers_save_path, "wb") as file:
        pickle.dump(all_answer_list, file)


if __name__ == '__main__':
    with open(os.path.join("../setting", "classifier_config.yaml"), 'r') as file:
        global_cfg = yaml.safe_load(file)
    # train_model(global_cfg)
    # test_model(global_cfg)
    predict_model(global_cfg)