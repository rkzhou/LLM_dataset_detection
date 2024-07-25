import torch
import pickle
import torch.nn.functional as F
import re
import os
import evaluate
import utils
import transformers
import math
import yaml

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def measure_sentences(dict, metric):
    if metric == "BERTscore":
        bertscore = evaluate.load("bertscore")
        results = bertscore.compute(predictions=dict["candidate_answers"], references=dict["reference_answers"], lang="en")
        
        average_precision = round(sum(results["precision"])/len(results["precision"]), 2)
        average_recall = round(sum(results["recall"])/len(results["recall"]), 2)
        average_f1 = round(sum(results["f1"])/len(results["f1"]), 2)

        similarity_score = {"avg_precision": average_precision, "avg_recall": average_recall, "avg_f1": average_f1}
    elif metric == "BLEU":
        bleu = evaluate.load("bleu")
        results = bleu.compute(predictions=dict["candidate_answers"], references=dict["reference_answers"])
        similarity_score = results
    else:
        raise ValueError("Invalid metric")
    
    return similarity_score


def measure_robustness(candidate_answer_path, reference_answer_path, metric):
    candidate_answer_files = os.listdir(candidate_answer_path)
    reference_answer_files = os.listdir(reference_answer_path)
    candidate_answer_files.remove("answer_scores.pt")
    reference_answer_files.remove("answer_scores.pt")


    if len(candidate_answer_files) != len(reference_answer_files):
        raise ValueError("The numbers of answers are not same")

    candidate_answers_list, reference_answers_list = list(), list()
    for answer_index in range(len(candidate_answer_files)):
        with open("{}/answer_{}.pkl".format(candidate_answer_path, answer_index), "rb") as answer_file:
            candidate_answer = pickle.load(answer_file)
        answer_file.close()

        with open("{}/answer_{}.pkl".format(reference_answer_path, answer_index), "rb") as answer_file:
            reference_answer=pickle.load(answer_file)
        answer_file.close()

        candidate_answers_list.append(candidate_answer)
        reference_answers_list.append(reference_answer)
    
    answer_dict = {"candidate_answers": candidate_answers_list, "reference_answers": reference_answers_list}

    similarity_score = measure_sentences(answer_dict, metric)

    return similarity_score


def split_sentence(sentence):
    return re.findall(r'\b\w+\b', sentence)


def thresholding_tensor(tensor_path, threshold, specific_index=None):
    answer_tensor = torch.load(tensor_path)
    compared_tensor = (answer_tensor > threshold)

    larger_threshold_num = 0
    if specific_index == None:
        for i in range(compared_tensor.shape[0]):
            if compared_tensor[i] == True:
                larger_threshold_num += 1
    else:
        for i in specific_index:
            if compared_tensor[i] == True:
                larger_threshold_num += 1

    return larger_threshold_num


def compare_answers(model_answer_dir, dataset_local_name, args):
    bare_prefix = "../ref_llm_answers/bare"
    finetuned_prefix = "../ref_llm_answers/finetuned"

    bare_model_list = os.listdir(bare_prefix)
    finetuned_model_list = os.listdir(finetuned_prefix)
    verify_answer_list = os.listdir(model_answer_dir)
    if "answer_scores.pt" in verify_answer_list:
        verify_answer_list.remove("answer_scores.pt")
    if "BERT_scores.pt" in verify_answer_list:
        verify_answer_list.remove("BERT_scores.pt")
    
    reference_model_num = len(bare_model_list)
    verify_answer_num = int(len(verify_answer_list) / args["inference_times"])
    similarity_scores = torch.zeros(2*reference_model_num, verify_answer_num)
    if args["metric"] == "TF-IDF":
        tfidf_vectorizer = TfidfVectorizer()
        for answer_index in tqdm(range(verify_answer_num)):
            answers = []

            # load answers from reference models and suspicious models
            for time_index in range(args["inference_times"]):
                with open("{}/answer_{}_{}.pkl".format(model_answer_dir, answer_index, time_index), "rb") as answer_file:
                    answer = pickle.load(answer_file)
                    answers.append(answer)
                answer_file.close()

            for name in bare_model_list:
                with open("{}/{}/{}/answer_{}.pkl".format(bare_prefix, name, dataset_local_name, answer_index), "rb") as answer_file:
                    answer = pickle.load(answer_file)
                    answers.append(answer)
                answer_file.close()
            
            for name in finetuned_model_list:
                with open("{}/{}/{}/answer_{}.pkl".format(finetuned_prefix, name, dataset_local_name, answer_index), "rb") as answer_file:
                    answer = pickle.load(answer_file)
                    answers.append(answer)
                answer_file.close()
            
            # obtain TF-IDF vectors of all answers
            tfidf_matrix = tfidf_vectorizer.fit_transform(answers)
            # calculate the best similar scores between answers from reference models and suspicious models
            for i in range(reference_model_num):
                best_simi_with_bare, best_simi_with_finetuned = 0, 0
                for time_index in range(args["inference_times"]):
                    best_simi_with_bare = max(best_simi_with_bare, cosine_similarity(tfidf_matrix[time_index], tfidf_matrix[args["inference_times"]+1])[0][0].item())
                    best_simi_with_finetuned = max(best_simi_with_finetuned, cosine_similarity(tfidf_matrix[time_index], tfidf_matrix[args["inference_times"]+1+reference_model_num])[0][0].item())
                similarity_scores[i, answer_index] = best_simi_with_bare
                similarity_scores[i+reference_model_num, answer_index] = best_simi_with_finetuned
        torch.save(similarity_scores, "{}/TF-IDF_scores.pt".format(model_answer_dir))
    elif args["metric"] == "BERT":
        bertscore = evaluate.load("bertscore")
        test_answer_list = list(list() for _ in range(args["inference_times"]))
        bare_answer_list, finetuned_answer_list = list(list() for _ in range(reference_model_num)), list(list() for _ in range(reference_model_num))
        for answer_index in range(len(verify_answer_list)):

            # load answers from reference models and suspicious models
            for time_index in range(args["inference_times"]):
                with open("{}/answer_{}_{}.pkl".format(model_answer_dir, answer_index, time_index), "rb") as answer_file:
                    test_answer = pickle.load(answer_file)
                    test_answer_list[time_index].append(test_answer)
                answer_file.close()

            for i in range(reference_model_num):
                with open("{}/{}/{}/answer_{}.pkl".format(bare_prefix, bare_model_list[i], dataset_local_name, answer_index), "rb") as answer_file:
                    bare_answer = pickle.load(answer_file)
                    bare_answer_list[i].append(bare_answer)
                answer_file.close()
            
            for i in range(reference_model_num):
                with open("{}/{}/{}/answer_{}.pkl".format(finetuned_prefix, finetuned_model_list[i], dataset_local_name, answer_index), "rb") as answer_file:
                    finetuned_answer = pickle.load(answer_file)
                    finetuned_answer_list[i].append(finetuned_answer)
                answer_file.close()
        
        # calculate BERT scores
        bare_bert_results, finetuned_bert_results = list(list() for _ in range(args["inference_times"])), list(list() for _ in range(args["inference_times"]))
        for i in range(reference_model_num):
            for j in range(args["inference_times"]):
                results = bertscore.compute(predictions=test_answer_list[j], references=bare_answer_list[i], model_type="distilbert-base-uncased")
                bare_bert_results[j].append(results)
                results = bertscore.compute(predictions=test_answer_list[j], references=finetuned_answer_list[i], model_type="distilbert-base-uncased")
                finetuned_bert_results[j].append(results)
        
        # save the best scores
        for i in range(len(test_answer_list[0])):
            for j in range(reference_model_num):
                best_simi_with_bare, best_simi_with_finetuned = 0, 0
                for time_index in range(args["inference_times"]):
                    best_simi_with_bare = max(best_simi_with_bare, bare_bert_results[time_index][j]['f1'][i])
                    best_simi_with_finetuned = max(best_simi_with_finetuned, finetuned_bert_results[time_index][j]['f1'][i])
                similarity_scores[j, i] = best_simi_with_bare
                similarity_scores[j+reference_model_num, i] = best_simi_with_finetuned

        torch.save(similarity_scores, "{}/BERT_scores.pt".format(model_answer_dir))


def threshold_answers(model_answer_dir, args):
    if args["metric"] == "TF-IDF":
        similarity_scores = torch.load("{}/TF-IDF_scores.pt".format(model_answer_dir))
    elif args["metric"] == "BERT":
        similarity_scores = torch.load("{}/BERT_scores.pt".format(model_answer_dir))
    
    model_num, question_num = similarity_scores.shape
    reference_model_num = int(model_num/2)

    nonmem_answer_num, mem_answer_num = 0, 0
    for j in range(question_num):
        nonmem_ref_above, mem_ref_above = 0, 0
        for i in range(reference_model_num):
            if similarity_scores[i, j] >= args["filter_threshold"]:
                nonmem_ref_above += 1
            if similarity_scores[i+reference_model_num, j] >= args["filter_threshold"]:
                mem_ref_above += 1
            
        if nonmem_ref_above == mem_ref_above:
            continue

        nonmem_simi_list = similarity_scores[:reference_model_num, j].tolist()
        mem_simi_list = similarity_scores[reference_model_num:, j].tolist()

        _, p_value = stats.ttest_ind(mem_simi_list, nonmem_simi_list, alternative='greater')
        if p_value < 0.05:
            mem_answer_num += 1
        else:
            nonmem_answer_num += 1

    print(mem_answer_num, nonmem_answer_num)


def insight_eval(bare_answer_dir, finetuned_answer_dir):
    dataset = utils.get_dataset("databricks/databricks-dolly-15k", "../dataset/dd_15k.pkl")
    dataset = dataset["train"]
    
    category_list = list()
    for data in dataset:
        category_list.append(data["category"])
    category_set = set(category_list)

    category_index = dict()
    category_bare_simi = dict()
    category_finetuned_simi = dict()
    for category in category_set:
        category_index.update({category:list()})
        category_bare_simi.update({category:list()})
        category_finetuned_simi.update({category:list()})
    
    for i in range(len(dataset)):
        category_index[dataset[i]["category"]].append(i)
    
    for category_name in tqdm(category_set):
        tfidf_vectorizer = TfidfVectorizer()
        for answer_index in tqdm(category_index[category_name]):
            answers = []
            original_answer = dataset[answer_index]["response"]
            if len(original_answer) < 5:
                continue
            answers.append(original_answer)

            with open("{}/answer_{}.pkl".format(bare_answer_dir, answer_index), "rb") as answer_file:
                bare_answer = pickle.load(answer_file)
                answers.append(bare_answer)
            answer_file.close()

            with open("{}/answer_{}.pkl".format(finetuned_answer_dir, answer_index), "rb") as answer_file:
                finetuned_answer = pickle.load(answer_file)
                answers.append(finetuned_answer)
            answer_file.close()
            
            tfidf_matrix = tfidf_vectorizer.fit_transform(answers)
            bare_score = round(cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0].item(), 2)
            finetuned_score = round(cosine_similarity(tfidf_matrix[0], tfidf_matrix[2])[0][0].item(), 2)
            
            category_bare_simi[category_name].append(bare_score)
            category_finetuned_simi[category_name].append(finetuned_score)
    
    threshold_list = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    for category_name in tqdm(category_set):
        bare_sum_list, finetuned_sum_list = list(), list()
        for threshold in threshold_list:
            bare_sum_list.append(sum(simi_value > threshold for simi_value in category_bare_simi[category_name]))
            finetuned_sum_list.append(sum(simi_value > threshold for simi_value in category_finetuned_simi[category_name]))
        print("bare:", category_name, ":", bare_sum_list)
        print("finetuned:", category_name, ":", finetuned_sum_list)


if __name__ == '__main__':
    with open(os.path.join("../setting", "qa_config.yaml"), 'r') as file:
        global_cfg = yaml.safe_load(file)

    mem_list = []
    for name in mem_list:
        compare_answers("../answers/dd15k/{}".format(name), "dd15k", global_cfg)
    
    nonmem_list = []
    for name in nonmem_list:
        compare_answers("../answers/dd15k/{}".format(name), "dd15k", global_cfg)
    

    mem_list = []
    for name in mem_list:
       threshold_answers("../answers/dd15k/{}".format(name), global_cfg)
    
    print("----------------------------------")
    nonmem_list = []
    for name in nonmem_list:
        threshold_answers("../answers/dd15k/{}".format(name), global_cfg)

    # compare_answers("../com_llm_answers/chatgpt3.5/ultrafeedback", "ultrafeedback", "TF-IDF")
    # compare_answers("../com_llm_answers/chatgpt3.5_ft/dd_15k", "dd_15k", "TF-IDF")
    # threshold_answers("../com_llm_answers/chatgpt3.5/ultrafeedback")
    # threshold_answers("../com_llm_answers/chatgpt3.5_ft/dd_15k")

    # mem_list = ["zephyralpha", "zephyrbeta", "stablelm3B", "danube1.8B", "starchat15B", "juanako7B", "yi6B", "cybertron7B", "tulu7B", "tulu13B"]
    # print("Member Models:\n")
    # for name in tqdm(mem_list):
    #     BERT_simi = measure_robustness("../answers/ultrafeedback/{}".format(name), "../paraphrase_answers/ultrafeedback/{}".format(name), "BLEU")
    #     print(BERT_simi)
    # print("---------------------")

    # for name in tqdm(mem_list):
    #     BERT_simi = measure_robustness("../answers/ultrafeedback/{}".format(name), "../paraphrase_answers/ultrafeedback/{}".format(name), "BERTscore")
    #     print(BERT_simi)
    # print("---------------------")

    # nonmem_list = ["dolly3B", "dolly7B", "dolly12B", "flangpt4", "flanxl", "flanxxl", "bloom3B", "redpajama3B", "platypus7B", "meditron7B"]
    # print("Non-member Models:\n")
    # for name in tqdm(nonmem_list):
    #     BERT_simi = measure_robustness("../answers/ultrafeedback/{}".format(name), "../paraphrase_answers/ultrafeedback/{}".format(name), "BLEU")
    #     print(BERT_simi)
    # print("---------------------")

    # for name in tqdm(nonmem_list):
    #     BERT_simi = measure_robustness("../answers/ultrafeedback/{}".format(name), "../paraphrase_answers/ultrafeedback/{}".format(name), "BERTscore")
    #     print(BERT_simi)
    # print("---------------------")

    # insight_eval("../insight_answers/chatgpt3.5/dd_15k", "../insight_answers/chatgpt3.5_ft/dd_15k")
