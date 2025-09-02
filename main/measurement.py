import torch
import pickle
import os
import evaluate
import yaml

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_leaf_folders(folder):
    leaf_folders = []
    has_subfolders = False

    for entry in os.scandir(folder):
        if entry.is_dir():
            has_subfolders = True
            # Recursively process subfolders
            leaf_folders.extend(get_leaf_folders(entry.path))
    
    # If no subfolders exist, this is a leaf folder
    if not has_subfolders:
        leaf_folders.append(folder)

    return leaf_folders


def prepare_paths(args):
    bare_answer_dirs = [
        args["bare_prediction_dir"].format(
            model_alias=bare_model, dataset_alias=args["dataset_alias"]
        )
        for bare_model in args["filter_bare_list"]
    ]

    finetune_answer_dirs = {
        bare_model: [
            args["finetune_prediction_dir"].format(
                model_alias=finetune_model, dataset_alias=args["dataset_alias"]
            )
            for finetune_model in args["filter_finetune_list"][bare_model]
        ]
        for bare_model in args["filter_bare_list"]
    }
    return bare_answer_dirs, finetune_answer_dirs


def get_model_indices(args):
    bare_model_index = []
    finetune_model_index = []
    model_index = 0

    for bare_model in args["filter_bare_list"]:
        bare_model_index.append(model_index)
        finetune_indices = list(
            range(model_index + 1, model_index + 1 + len(args["filter_finetune_list"][bare_model]))
        )
        finetune_model_index.append(finetune_indices)
        model_index += len(finetune_indices) + 1

    return bare_model_index, finetune_model_index


def compare_answers(args):
    bare_answer_dirs, finetune_answer_dirs = prepare_paths(args)
    bare_model_index, finetune_model_index = get_model_indices(args)
    reference_model_num = len(bare_model_index)
    suspect_answer_dirs = get_leaf_folders(args["suspect_answer_dir"].format(dataset_alias=args["dataset_alias"]))

    with open("{}/{}_{}.pkl".format(args["tainted_sample_dir"], args["dataset_alias"], args["metric"]), "rb") as file:
        sample_index = pickle.load(file)
    
    for suspect_answer_dir in suspect_answer_dirs:
        similarity_scores = torch.zeros(2 * reference_model_num, len(sample_index))
        if args["metric"] == "bert":
            bertscore = evaluate.load("bertscore")
            suspect_answer_list = list(list() for _ in range(args["inference_times"]))
            total_ref_number = len(bare_model_index) + sum([len(finetune_model_index[i]) for i in range(len(finetune_model_index))])
            reference_answer_list = list(list() for _ in range(total_ref_number))

            for answer_index in tqdm(range(len(sample_index))):
                # Load answers from reference models and suspect models
                for time_index in range(args["inference_times"]):
                    with open("{}/answer_{}.pkl".format(suspect_answer_dir, sample_index[answer_index]), "rb") as answer_file:
                        suspect_answer = pickle.load(answer_file)
                        suspect_answer_list[time_index].append(suspect_answer)

                for i in range(len(args["filter_bare_list"])):
                    with open("{}/answer_{}.pkl".format(bare_answer_dirs[i], sample_index[answer_index]), "rb") as answer_file:
                        bare_answer = pickle.load(answer_file)
                        reference_answer_list[bare_model_index[i]].append(bare_answer)
                
                    for j in range(len(args["filter_finetune_list"][args["filter_bare_list"][i]])):
                        with open("{}/answer_{}.pkl".format(finetune_answer_dirs[args["filter_bare_list"][i]][j], sample_index[answer_index]), "rb") as answer_file:
                            finetuned_answer = pickle.load(answer_file)
                            reference_answer_list[finetune_model_index[i][j]].append(finetuned_answer)
            
            # Calculate BERT scores
            bert_results = list(list() for _ in range(args["inference_times"]))
            for i in range(total_ref_number):
                for j in range(args["inference_times"]):
                    results = bertscore.compute(predictions=suspect_answer_list[j], references=reference_answer_list[i], model_type="distilbert-base-uncased")
                    bert_results[j].append(results)
            
            # Save the best scores
            for answer_index in range(len(sample_index)):
                for i in range(len(bare_model_index)):
                    best_simi_with_bare, best_simi_with_finetune = 0, 0
                    for time_index in range(args["inference_times"]):
                        best_simi_with_bare = max(best_simi_with_bare, bert_results[time_index][bare_model_index[i]]['f1'][answer_index])
                        for j in range(len(finetune_model_index[i])):
                            best_simi_with_finetune = max(best_simi_with_finetune, bert_results[time_index][finetune_model_index[i][j]]['f1'][answer_index])
                    similarity_scores[i, answer_index] = best_simi_with_bare
                    similarity_scores[i+reference_model_num, answer_index] = best_simi_with_finetune
            torch.save(similarity_scores, "{}/BERT_scores.pt".format(suspect_answer_dir))
        else:
            raise ValueError("Invalid metric")


def threshold_answers(args):
    with open("{}/{}_{}.pkl".format(args["tainted_sample_dir"], args["dataset_alias"], args["metric"]), "rb") as file:
        sample_index = pickle.load(file)
        
    suspect_answer_dirs = get_leaf_folders(args["suspect_answer_dir"].format(dataset_alias=args["dataset_alias"]))
    for answer_dir in suspect_answer_dirs:
        path_parts = answer_dir.split(os.sep)
        suspect_model_name = os.path.join(path_parts[-2], path_parts[-1])
        
        similarity_scores = torch.load("{}/{}_scores.pt".format(answer_dir, args["metric"]))
    
        model_num, question_num = similarity_scores.shape
        reference_model_num = int(model_num/2)

        nonmem_answer_num, mem_answer_num = 0, 0
        nonmem_answer_index, mem_answer_index = list(), list()
        for j in range(question_num):
            nonmem_simi_list = similarity_scores[:reference_model_num, j].tolist()
            mem_simi_list = similarity_scores[reference_model_num:, j].tolist()
            
            if all((x - y) > args["similarity_threshold"] for x, y in zip(mem_simi_list, nonmem_simi_list)):
                mem_answer_num += 1
                mem_answer_index.append(j)
            else:
                nonmem_answer_num += 1
                nonmem_answer_index.append(j)
        
        print(suspect_model_name)
        mem_index = [sample_index[i] for i in mem_answer_index]
        nonmem_index = [x for x in sample_index if x not in mem_index]
        with open("{}/mem_index.pkl".format(answer_dir), "wb") as file:
            pickle.dump(mem_index, file)
        with open("{}/nonmem_index.pkl".format(answer_dir), "wb") as file:
            pickle.dump(nonmem_index, file)
        
        print(mem_answer_num, nonmem_answer_num, "member_model" if mem_answer_num >= nonmem_answer_num else "nonmember_model")


if __name__ == '__main__':
    with open(os.path.join("../setting", "filter_config.yaml"), 'r') as file:
        global_cfg = yaml.safe_load(file)

    compare_answers(global_cfg)
    threshold_answers(global_cfg)