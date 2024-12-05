import utils
import os
import pickle
import yaml
import shutil

from pathlib import Path

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


def get_original_dataset(args):
    dataset = utils.get_dataset(args["dataset_name"], args["raw_dataset_path"])
    os.makedirs(os.path.dirname(args["general_dataset_path"]), exist_ok=True)
    
    # preprocess dataset
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

    os.makedirs(args["partition_dataset_dir"], exist_ok=True)
    folder_path = Path(args["partition_dataset_dir"])
    try:
        for item in folder_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)  # Remove subfolder and its contents
            else:
                item.unlink()  # Remove file
    except Exception as e:
        print(f"Error: {e}")
    
    for i in range(len(splits)):
        with open("{}/subset_{}.pkl".format(args["partition_dataset_dir"], i), "wb") as file:
            pickle.dump(splits[i], file)


if __name__ == '__main__':
    with open(os.path.join("../setting", "dataset_config.yaml"), 'r') as file:
        global_cfg = yaml.safe_load(file)
    global_cfg = {
        key: (value.format(dataset_alias=global_cfg["dataset_alias"]) if isinstance(value, str) else value)
        for key, value in global_cfg.items()
    }

    getattr(__import__(__name__), global_cfg["action"])(global_cfg)