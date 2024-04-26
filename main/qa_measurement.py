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

    print(larger_threshold_num)

if __name__ == '__main__':
    dataset = utils.get_dataset("databricks/databricks-dolly-15k", "../dataset/dd_15k.pkl")

    cate_num = dict()
    for i in range(len(dataset['train'])):
        current_cate = dataset['train'][i]['category']
        if current_cate in cate_num.keys():
            cate_num[current_cate] += 1
        else:
            cate_num.update({current_cate: 1})
    print(cate_num)
    print('-----------------------------')

    thresholding_tensor("../answers/dd_15k/open_qa/DeciLM7B/answer_scores.pt", 0.95)