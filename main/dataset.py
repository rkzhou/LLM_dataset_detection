import utils
import os
import pickle
import yaml
import glob

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
    elif dataset_name == "Open-Orca/SlimOrca":
        item = {"system": data["conversations"][0]["value"], "instruction": data["conversations"][1]["value"], "response": data["conversations"][2]["value"]}
    elif dataset_name == "teknium/OpenHermes-2.5":
        item = {"system": "", "instruction": data["conversations"][0]["value"], "response": data["conversations"][1]["value"]}
    else:
        raise ValueError("Invalid dataset")
    
    return item


def prepare_dataset(args):
    # Downlaod dataset
    dataset = utils.get_dataset(args["dataset_name"], args["raw_dataset_path"])
    os.makedirs(os.path.dirname(args["general_dataset_path"]), exist_ok=True)
    
    # Preprocess dataset
    if args["dataset_name"] == "databricks/databricks-dolly-15k":
        dataset = dataset["train"]
    elif args["dataset_name"] == "tatsu-lab/alpaca":
        dataset = dataset["train"]
    elif args["dataset_name"] == "Open-Orca/SlimOrca":
        dataset = dataset["train"]
        dataset = dataset.filter(lambda example: len(example["conversations"]) == 3)
    elif args["dataset_name"] == "teknium/OpenHermes-2.5":
        dataset = dataset["train"]
        dataset = dataset.filter(lambda example: len(example["conversations"]) == 2)
        dataset = dataset.filter(lambda example: example["category"] != "coding")
    else:
        raise ValueError("Invalid dataset")

    # Slice dataset
    args["subset_size"] = min(len(dataset), args["subset_size"])
    dataset = dataset.shuffle(seed=args["seed_index"])
    dataset = dataset.select(range(args["subset_size"]))
    dataset = dataset.map(format_dataset, fn_kwargs={"dataset_name": args["dataset_name"]})
    # Add index
    dataset = dataset.map(lambda example, idx: {"index": idx}, with_indices=True)
    with open(args["general_dataset_path"], "wb") as file:
        pickle.dump(dataset, file)
    
    # Prepare dataset for fine-tuning
    os.makedirs(os.path.dirname(args["format_dataset_path"]), exist_ok=True)
    utils.jsonlize_dataset(dataset, args["format_dataset_path"])


def get_subsets(args):
    with open(args["general_dataset_path"], "rb") as file:
        dataset = pickle.load(file)
    
    assert sum(args["partition_ratio"]) == 1.0, "Split ratios must sum to 1.0"

    # Slice the dataset
    dataset = dataset.shuffle(seed=args["seed_index"])
    split_numbers = [int(ratio * len(dataset)) for ratio in args["partition_ratio"][:-1]]
    split_numbers.append(len(dataset)-sum(split_numbers))
    cut_indices = [0] + [sum(split_numbers[:i+1]) for i in range(len(split_numbers))]

    subsets = [
        dataset.select(range(cut_indices[i], cut_indices[i + 1]))
        for i in range(len(cut_indices) - 1)
    ]
    for i in range(len(subsets)):
        print("No.{} subset size: {}".format(i, len(subsets[i])))
    
    os.makedirs(args["partition_general_dataset_dir"], exist_ok=True)
    os.makedirs(args["partition_format_dataset_dir"], exist_ok=True)
    
    for i in range(len(subsets)):
        with open("{}/{}_subset_{}.pkl".format(args["partition_general_dataset_dir"], args["dataset_alias"], i), "wb") as file:
            pickle.dump(subsets[i], file)
        utils.jsonlize_dataset(subsets[i], "{}/{}_subset_{}.jsonl".format(args["partition_format_dataset_dir"], args["dataset_alias"], i))


if __name__ == '__main__':
    with open(os.path.join("../setting", "dataset_config.yaml"), 'r') as file:
        global_cfg = yaml.safe_load(file)
    global_cfg = {
        key: (value.format(dataset_alias=global_cfg["dataset_alias"]) if isinstance(value, str) else value)
        for key, value in global_cfg.items()
    }

    getattr(__import__(__name__), global_cfg["action"])(global_cfg)