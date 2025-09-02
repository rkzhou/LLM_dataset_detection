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
    pattern = os.path.join(
        args["partition_general_dataset_dir"], f"{args['dataset_alias']}_subset_*.pkl"
    )
    num_subsets = len(glob.glob(pattern))

    # Load all subsets
    subsets = []
    for i in range(num_subsets):
        with open(f"{args['partition_general_dataset_dir']}/{args['dataset_alias']}_subset_{i}.pkl", "rb") as file:
            subsets.append(pickle.load(file))
    
    # Prepare texts (lists of questions/answers for each subset)
    subsets_questions = [[data["instruction"] for data in subset] for subset in subsets]
    subsets_answers   = [[data["response"] for data in subset] for subset in subsets]

    # Track indices to filter per subset
    filter_indices = [set() for _ in range(num_subsets)]
    
    # Compare every pair of subsets (i vs j, where j > i)
    for i in range(num_subsets):
        for j in range(i + 1, num_subsets):
            for idx_i in tqdm(range(len(subsets_questions[i])), desc=f"Comparing subset {i} vs {j}"):
                for idx_j in range(len(subsets_questions[j])):
                    q_sim = jaccard_similarity_sentence(subsets_questions[i][idx_i], subsets_questions[j][idx_j])
                    a_sim = jaccard_similarity_sentence(subsets_answers[i][idx_i], subsets_answers[j][idx_j])

                    if q_sim >= 0.8 and a_sim >= 0.8:
                        filter_indices[j].add(subsets[j][idx_j]["index"])  # Drop from later subset
    
    # Apply filtering & save results
    for k in range(num_subsets):
        print(f"Subset {k}: removing {len(filter_indices[k])} duplicates")
        filtered_dataset = [ex for ex in subsets[k] if ex["index"] not in filter_indices[k]]
        
        with open(f"{args['partition_general_dataset_dir']}/{args['dataset_alias']}_dedup_subset_{k}.pkl", "wb") as file:
            pickle.dump(filtered_dataset, file)
        utils.jsonlize_dataset(filtered_dataset, f"{args['partition_format_dataset_dir']}/{args['dataset_alias']}_dedup_subset_{k}.jsonl")


def remove_tainted(args):
    with open(args["general_dataset_path"], "rb") as file:
        original_dataset = pickle.load(file)
    
    # Remove positive tainted samples for reference models respectively
    for filename in os.listdir(args["tainted_sample_dir"]):
        model_name = filename.strip(".pkl")
        with open("{}/{}".format(args["tainted_sample_dir"], filename), "rb") as file:
            positive_tainted_index = pickle.load(file)
        filtered_dataset = original_dataset.filter(lambda example: example["index"] not in positive_tainted_index)
        utils.jsonlize_dataset(filtered_dataset, "{}/{}.jsonl".format(args["tainted_sample_dir"], model_name))


if __name__ == '__main__':
    with open(os.path.join("../setting", "dataset_config.yaml"), 'r') as file:
        global_cfg = yaml.safe_load(file)
    global_cfg = {
        key: (value.format(dataset_alias=global_cfg["dataset_alias"]) if isinstance(value, str) else value)
        for key, value in global_cfg.items()
    }

    getattr(__import__(__name__), global_cfg["action"])(global_cfg)