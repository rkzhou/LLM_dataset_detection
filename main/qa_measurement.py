import torch
import pickle
import torch.nn.functional as F
import re
import os

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
from tqdm import tqdm


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


def compare_answers(model_answer_dir, dataset_local_name):
    ### load answers of reference models before and after fine-tuning
    bare_prefix = "../ref_llm_answers/bare"
    finetuned_prefix = "../ref_llm_answers/finetuned"

    vocab_set = set()

    bare_model_list = os.listdir(bare_prefix)
    for name in bare_model_list:
        bare_reference_answer_path = "{}/{}/{}".format(bare_prefix, name, dataset_local_name)
        answer_list = os.listdir(bare_reference_answer_path)
        for answer_index in range(len(answer_list)):
            with open("{}/answer_{}.pkl".format(bare_reference_answer_path, answer_index), "rb") as answer_file:
                answer = pickle.load(answer_file)
                token_list = split_sentence(answer)
                for token in token_list:
                    vocab_set.add(token)
            answer_file.close()
    
    finetuned_model_list = os.listdir(finetuned_prefix)
    for name in finetuned_model_list:
        finetuned_reference_answer_path = "{}/{}/{}".format(finetuned_prefix, name, dataset_local_name)
        answer_list = os.listdir(finetuned_reference_answer_path)
        for answer_index in range(len(answer_list)):
            with open("{}/answer_{}.pkl".format(finetuned_reference_answer_path, answer_index), "rb") as answer_file:
                answer = pickle.load(answer_file)
                token_list = split_sentence(answer)
                for token in token_list:
                    vocab_set.add(token)
            answer_file.close()
    
    verify_answer_list = os.listdir(model_answer_dir)
    for answer_index in range(len(verify_answer_list)):
        # with open("{}/answer_{}.pkl".format(model_answer_dir, answer_index), "rb") as answer_file:
        #     answer = torch.load(answer_file, map_location="cpu")
        # with open("{}/answer_{}.pkl".format(model_answer_dir, answer_index), "wb") as answer_file:
        #     pickle.dump(answer, answer_file)
        with open("{}/answer_{}.pkl".format(model_answer_dir, answer_index), "rb") as answer_file:
            answer = pickle.load(answer_file)
            token_list = split_sentence(answer)
            for token in token_list:
                vocab_set.add(token)
        answer_file.close()
    
    vocab_list = list(vocab_set)
    vocab_size = len(vocab_list)
    reference_model_num = len(bare_model_list)
    similarity_scores = torch.zeros(2*reference_model_num, len(verify_answer_list))
    cos_simi = torch.nn.CosineSimilarity(dim=0)

    for answer_index in tqdm(range(len(verify_answer_list))):
        bare_answers_vec = torch.zeros(reference_model_num, vocab_size)
        finetuned_answers_vec = torch.zeros(reference_model_num, vocab_size)
        verify_answer_vec = torch.zeros(vocab_size)
        for name_index in range(len(bare_model_list)):
            bare_answer_path = "{}/{}/{}/answer_{}.pkl".format(bare_prefix, bare_model_list[name_index], dataset_local_name, answer_index)
            with open(bare_answer_path, "rb") as bare_answer_file:
                bare_answer = pickle.load(bare_answer_file)
                bare_token_list = split_sentence(bare_answer)
                for token in bare_token_list:
                    token_index = vocab_list.index(token)
                    bare_answers_vec[name_index][token_index] += 1
            bare_answer_file.close()

        for name_index in range(len(finetuned_model_list)):
            finetuned_answer_path = "{}/{}/{}/answer_{}.pkl".format(finetuned_prefix, finetuned_model_list[name_index], dataset_local_name, answer_index)
            with open(finetuned_answer_path, "rb") as finetuned_answer_file:
                finetuned_answer = pickle.load(finetuned_answer_file)
                finetuned_token_list = split_sentence(finetuned_answer)
                for token in finetuned_token_list:
                    token_index = vocab_list.index(token)
                    finetuned_answers_vec[name_index][token_index] += 1
            finetuned_answer_file.close()
        
        verify_answer_path = "{}/answer_{}.pkl".format(model_answer_dir, answer_index)
        with open(verify_answer_path, "rb") as verify_answer_file:
            verify_answer = pickle.load(verify_answer_file)
            verify_token_list = split_sentence(verify_answer)
            for token in verify_token_list:
                token_index = vocab_list.index(token)
                verify_answer_vec[token_index] += 1
        verify_answer_file.close()

        for i in range(reference_model_num):
            score = cos_simi(verify_answer_vec, bare_answers_vec[i, :])
            similarity_scores[i, answer_index] = score
        
        for i in range(reference_model_num):
            score = cos_simi(verify_answer_vec, finetuned_answers_vec[i, :])
            similarity_scores[i+reference_model_num, answer_index] = score
        
    torch.save(similarity_scores, "{}/answer_scores.pt".format(model_answer_dir))


if __name__ == '__main__':
    # thres_list = [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
    # for thres in thres_list:
    #     print(thresholding_tensor("../ref_llm_answers/bare/glm/dd_15k/answer_scores.pt", thres))

    compare_answers("../answers/dd_15k/bloom3B", "dd_15k")