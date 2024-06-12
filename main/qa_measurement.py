import torch
import pickle

from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import re
import numpy as np
import utils
from sentence_transformers import util

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def measure_sentences(dict):
    sen_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    sen_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


    saved_answer_list = [dict["benchmark_answer"], dict["member_answer"], dict["nonmember_answer"]]
    encoded_input = sen_tokenizer(saved_answer_list, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = sen_model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    similarity_score0 = util.pytorch_cos_sim(sentence_embeddings[0], sentence_embeddings[1]).cpu()
    similarity_score1 = util.pytorch_cos_sim(sentence_embeddings[0], sentence_embeddings[2]).cpu()

    return similarity_score0, similarity_score1
    
    


def split_sentence(sentence):
    return re.findall(r'\b\w+\b', sentence)


def calculate_match_score(dict):
    benchmark_answer_set = set(split_sentence(dict['benchmark_answer']))
    member_answer_set = set(split_sentence(dict['member_answer']))
    nonmember_answer_set = set(split_sentence(dict['nonmember_answer']))

    bm_common = benchmark_answer_set.intersection(member_answer_set)
    bn_common = benchmark_answer_set.intersection(nonmember_answer_set)

    bm_common_ratio = len(bm_common) / len(benchmark_answer_set)
    bn_common_ratio = len(bn_common) / len(benchmark_answer_set)

    if bm_common_ratio == 1.0 and bn_common_ratio == 1.0:
        mem_penalty = len(benchmark_answer_set) / len(member_answer_set) - 1.0
        nonmem_penalty = len(benchmark_answer_set) / len(nonmember_answer_set) - 1.0
        return mem_penalty, nonmem_penalty

    return bm_common_ratio, bn_common_ratio


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

if __name__ == '__main__':
    # print("member models:\n")
    # model_name = ["falcon1B"]
    # dataset_name = "einstein7B"
    # thres = 0.7
    # for name in model_name:
    #  thres_num = thresholding_tensor("../answers/{}/{}/answer_scores.pt".format(dataset_name, name), thres)
    #  print(thres_num)


    # print("nonmember models:\n")
    # model_name = [""]
    # dataset_name = ""
    # thres = 0.7
    # for name in model_name:
    #    thres_num = thresholding_tensor("../answers/{}/{}/answer_scores.pt".format(dataset_name, name), thres)
    #    print(thres_num)

    thres_list = [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
    for thres in thres_list:
        print(thresholding_tensor("../answers/slimorca/monarch7B/answer_scores.pt", thres))
