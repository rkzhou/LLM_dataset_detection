import utils
import os
import pickle
import yaml
import json
import shutil

from pathlib import Path
from tqdm import tqdm

def format_dataset(data, dataset_name):
    if dataset_name == "databricks/databricks-dolly-15k":
        if data["context"] == "":
            item = {"system": "", "instruction": data["instruction"], "response": data["response"]}
        else:
            item = {"system": "", "instruction": data["context"] + " " + data["instruction"], "response": data["response"]}
    elif dataset_name == "tatsu-lab/alpaca":
        if data["input"] == "":
            item = {"system": "", "instruction": data["instruction"], "response": data["output"]}
        else:
            item = {"system": "", "instruction": data["instruction"] + " " + data["input"], "response": data["output"]}
    elif dataset_name == "Open-Orca/SlimOrca" or dataset_name == "Open-Orca/slimorca-deduped-cleaned-corrected":
        item = {"system": data["conversations"][0]["value"], "instruction": data["conversations"][1]["value"], "response": data["conversations"][2]["value"]}
    elif dataset_name == "teknium/OpenHermes-2.5":
        item = {"system": "", "instruction": data["conversations"][0]["value"], "response": data["conversations"][1]["value"]}
    else:
        raise ValueError("Invalid dataset")
    
    return item


def get_original_dataset(args):
    dataset = utils.get_dataset(args["dataset_name"], args["raw_dataset_path"])
    os.makedirs(os.path.dirname(args["general_dataset_path"]), exist_ok=True)
    
    # preprocess dataset
    if args["dataset_name"] == "databricks/databricks-dolly-15k":
        dataset = dataset["train"]
    elif args["dataset_name"] == "tatsu-lab/alpaca":
        dataset = dataset["train"]
    elif args["dataset_name"] == "Open-Orca/SlimOrca" or args["dataset_name"] == "Open-Orca/slimorca-deduped-cleaned-corrected":
        dataset = dataset["train"]
        dataset = dataset.filter(lambda example: len(example["conversations"]) == 3)
    elif args["dataset_name"] == "teknium/OpenHermes-2.5":
        dataset = dataset["train"]
        dataset = dataset.filter(lambda example: len(example["conversations"]) == 2)
        dataset = dataset.filter(lambda example: example["category"] != "coding")
    else:
        raise ValueError("Invalid dataset")

    args["subset_size"] = min(len(dataset), args["subset_size"])
    dataset = dataset.shuffle(seed=args["seed_index"])
    dataset = dataset.select(range(args["subset_size"]))
    dataset = dataset.map(format_dataset, fn_kwargs={"dataset_name": args["dataset_name"]})
    # add index
    dataset = dataset.map(lambda example, idx: {"index": idx}, with_indices=True)
    with open(args["general_dataset_path"], "wb") as file:
        pickle.dump(dataset, file)
    
    utils.format_data(args)


def get_subsets(args):
    with open(args["general_dataset_path"], "rb") as file:
        dataset = pickle.load(file)
    
    assert sum(args["partition_ratio"]) == 1.0, "Split ratios must sum to 1.0"

    dataset = dataset.shuffle(seed=args["seed_index"])
    split_numbers = [int(ratio * len(dataset)) for ratio in args["partition_ratio"][:-1]]
    split_numbers.append(len(dataset)-sum(split_numbers))
    cut_indices = [0] + [sum(split_numbers[:i+1]) for i in range(len(split_numbers))]

    # Slice the dataset
    splits = [
        dataset.select(range(cut_indices[i], cut_indices[i + 1]))
        for i in range(len(cut_indices) - 1)
    ]
    for i in range(len(splits)):
        print("No.{} subset size: {}".format(i, len(splits[i])))
    
    os.makedirs(args["partition_general_dataset_dir"], exist_ok=True)
    os.makedirs(args["partition_format_dataset_dir"], exist_ok=True)
    
    for i in range(len(splits)):
        with open("{}/{}_subset_{}.pkl".format(args["partition_general_dataset_dir"], args["dataset_alias"], i), "wb") as file:
            pickle.dump(splits[i], file)
        with open("{}/{}_subset_{}.jsonl".format(args["partition_format_dataset_dir"], args["dataset_alias"], i), "w") as output_jsonl_file:
            for item in splits[i]:
                json_object = {"text": utils.create_text_row(item["system"], item["instruction"], item["response"])}
                output_jsonl_file.write(json.dumps(json_object) + "\n")


def jaccard_similarity_sentence(sentence1, sentence2):
    """
    Calculate Jaccard similarity between two sentences.
    
    Args:
        sentence1 (str): First sentence.
        sentence2 (str): Second sentence.
    
    Returns:
        float: Jaccard similarity score.
    """
    # Tokenize sentences into sets of words
    set1 = set(sentence1.split())
    set2 = set(sentence2.split())
    
    # Compute intersection and union
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    # Calculate Jaccard similarity
    return intersection / union if union != 0 else 0.0


def deduplicate(args):
    with open("{}/{}_subset_0.pkl".format(args["partition_general_dataset_dir"], args["dataset_alias"]), "rb") as file:
        subset_0 = pickle.load(file)
    with open("{}/{}_subset_1.pkl".format(args["partition_general_dataset_dir"], args["dataset_alias"]), "rb") as file:
        subset_1 = pickle.load(file)
    
 
    filter_index = set()
    # Prepare texts
    subset_0_questions = [data["instruction"] for data in subset_0]
    subset_0_answers = [data["response"] for data in subset_0]
    subset_1_questions = [data["instruction"] for data in subset_1]
    subset_1_answers = [data["response"] for data in subset_1]

    for i in tqdm(range(len(subset_0_questions))):
        for j in range(len(subset_1_questions)):
            question_similarity = jaccard_similarity_sentence(subset_0_questions[i], subset_1_questions[j])
            answer_similarity = jaccard_similarity_sentence(subset_0_answers[i], subset_1_answers[j])

            if question_similarity >= 0.9 and answer_similarity >= 0.9:
                filter_index.add(subset_1[j]["index"])
    
    print(f"The number of filtered data points: {len(filter_index)}")
    filtered_dataset = subset_1.filter(lambda example: example["index"] not in filter_index)

    with open("{}/{}_dedup_subset_1.pkl".format(args["partition_general_dataset_dir"], args["dataset_alias"]), "wb") as file:
        pickle.dump(filtered_dataset, file)
    with open("{}/{}_dedup_subset_1.jsonl".format(args["partition_format_dataset_dir"], args["dataset_alias"]), "w") as output_jsonl_file:
        for item in filtered_dataset:
            json_object = {"text": utils.create_text_row(item["system"], item["instruction"], item["response"])}
            output_jsonl_file.write(json.dumps(json_object) + "\n")


def remove_tainted(args):
    with open(args["general_dataset_path"], "rb") as file:
        original_dataset = pickle.load(file)
    with open(args["tainted_sample_path"], "rb") as file:
        tainted_index = pickle.load(file)
    
    filtered_dataset = original_dataset.filter(lambda example: example["index"] not in tainted_index)

    with open("{}/{}_removed.pkl".format(os.path.dirname(args["general_dataset_path"]), args["dataset_alias"]), "wb") as file:
        pickle.dump(filtered_dataset, file)
    with open("{}/{}_removed.jsonl".format(os.path.dirname(args["format_dataset_path"]), args["dataset_alias"]), "w") as output_jsonl_file:
        for item in filtered_dataset:
            json_object = {"text": utils.create_text_row(item["system"], item["instruction"], item["response"])}
            output_jsonl_file.write(json.dumps(json_object) + "\n")


if __name__ == '__main__':
    with open(os.path.join("../setting", "dataset_config.yaml"), 'r') as file:
        global_cfg = yaml.safe_load(file)
    global_cfg = {
        key: (value.format(dataset_alias=global_cfg["dataset_alias"]) if isinstance(value, str) else value)
        for key, value in global_cfg.items()
    }

    getattr(__import__(__name__), global_cfg["action"])(global_cfg)