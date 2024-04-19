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


if __name__ == '__main__':
    dataset = utils.get_dataset("databricks/databricks-dolly-15k", "../dataset/dd_15k.pkl")

    cate_num = dict()
    for i in range(1500):
        current_cate = dataset['train'][i]['category']
        if current_cate in cate_num.keys():
            cate_num[current_cate] += 1
        else:
            cate_num.update({current_cate: 1})
    print(cate_num)
    print('-----------------------------')

    # temp = torch.load("../answers/dd_15k/nonmem_models/solar10.7B/answer_scores.pt")

    # answer_index = 3
    # with open("../answers/dd_15k/mem_models/dolly3B/answer_{}.pkl".format(answer_index), "rb") as file:
    #     temp = pickle.load(file)
    # print(temp)


    member_tensor = torch.load("../answers/dd_15k/mem_models/dolly3B/answer_scores.pt")
    nonmember_tensor = torch.load("../answers/dd_15k/nonmem_models/solar10.7B/answer_scores.pt")

    # print(member_tensor)
    # print(nonmember_tensor)

    mem_cate_num = dict()
    mem_threshold_num = 0
    mem_compare_threshold = (member_tensor > 0.95)
    for i in range(mem_compare_threshold.shape[0]):
        if mem_compare_threshold[i] == True:
            mem_threshold_num += 1
            current_cate = dataset['train'][i]['category']
            if current_cate in mem_cate_num.keys():
                mem_cate_num[current_cate] += 1
            else:
                mem_cate_num.update({current_cate: 1})
    print(mem_cate_num)
    print('-----------------------------')

    nonmem_cate_num = dict()
    nonmem_threshold_num = 0
    nonmem_compare_threshold = (nonmember_tensor > 0.95)
    for i in range(nonmem_compare_threshold.shape[0]):
        if nonmem_compare_threshold[i] == True:
            nonmem_threshold_num += 1
            current_cate = dataset['train'][i]['category']
            if current_cate in nonmem_cate_num.keys():
                nonmem_cate_num[current_cate] += 1
            else:
                nonmem_cate_num.update({current_cate: 1})
    print(nonmem_cate_num)
    print(mem_threshold_num, nonmem_threshold_num)
    
    # num = 0
    # for i in range(member_tensor.shape[0]):
    #     # diff = member_tensor[i] - nonmember_tensor[i]
    #     if member_tensor[i] >= 0.95 and nonmember_tensor[i] < 0.95:
    #         print(i, dataset['train'][i]['category'])
    #         print(dataset['train'][i]['instruction'], dataset['train'][i]['context'])
    #         num += 1
    # print(num)
