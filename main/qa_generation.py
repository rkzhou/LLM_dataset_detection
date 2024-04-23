import utils
import os
import preprocess
import torch
import numpy
import pickle
import re
import math
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModel


def fetch_dataset(args, name, hf_path):
    strings = [args.dataset_path, name, ".pkl"]
    check_path = "".join(strings)
    if os.path.exists(check_path):
        dataset = utils.load_local_dataset(check_path)
    else:
        dataset = utils.save_hf_dataset(hf_path, check_path)

    return dataset


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                                                                              

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


def generate_answers(args):
    dataset = utils.get_dataset(args["dataset_name"], args["dataset_path"])
    select_category_list = ["open_qa"]
    dataset = dataset.filter(lambda x: x['category'] in select_category_list)

    data_group_num = math.ceil(len(dataset["train"]) / args["batch_size"])

    ### initialize model or pipeline
    if args["model_type"] == "pipeline":
        pipeline_tokenizer = AutoTokenizer.from_pretrained(args["model_name"], padding_side="left")
        pipe = pipeline(model=args["model_name"], torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", batch_size=args["batch_size"], tokenizer=pipeline_tokenizer)
    elif args["model_type"] == "kernel":
        valid_model_template = [index for index in range(6)]
        if args["model_template"] in valid_model_template:
            function_to_call = "Chatmodel_{}".format(args["model_template"])
            llm_model = getattr(preprocess, function_to_call)(args["model_name"])
        else:
            raise ValueError("Invalid Model Template")
    else:
        raise ValueError("Invalid Model Type")
    
    if not os.path.exists(args["answer_root"]):
        os.makedirs(args["answer_root"])

    model_scores = list()
    cos_simi = torch.nn.CosineSimilarity(dim=0)
    saved_answer_num = 0
    ### loop every batch of questions
    for group_index in tqdm(range(data_group_num)):
        current_group_complete = False
        current_group_saved_answer_num = 0
        ### check if answers have been already saved
        for i in range(args["batch_size"]):
            data_index = group_index * args["batch_size"] + i
            if os.path.exists("{}/answer_{}.pkl".format(args["answer_root"], data_index)):
                saved_answer_num += 1
                current_group_saved_answer_num += 1
                with open("{}/answer_{}.pkl".format(args["answer_root"], data_index), 'rb') as file:
                    saved_model_answer = pickle.load(file)
                
                benchmark_answer = dataset["train"][data_index]["response"]
                benchmark_split_tokens = split_sentence(benchmark_answer)
                model_split_tokens = split_sentence(saved_model_answer)

                vocab = list(set(benchmark_split_tokens + model_split_tokens))
                vocab_size = len(vocab)
                benchmark_tfidf = torch.zeros(vocab_size)
                model_tfidf = torch.zeros(vocab_size)

                for token in benchmark_split_tokens:
                    position = vocab.index(token)
                    benchmark_tfidf[position] += 1
                for token in model_split_tokens:
                    position = vocab.index(token)
                    model_tfidf[position] += 1

                similarity_score = cos_simi(benchmark_tfidf, model_tfidf).item()

                model_scores.append(similarity_score)
            if data_index >= len(dataset["train"]):
                print("Reach the end of dataset")
                model_scores = numpy.array(model_scores)
                scores_tensor = torch.flatten(torch.Tensor(model_scores))
                torch.save(scores_tensor, "{}/answer_scores.pt".format(args["answer_root"]))
                exit()
        if current_group_saved_answer_num == args["batch_size"]:
            current_group_complete = True
        
        if saved_answer_num >= args["early_stop_num"] and args["early_stop_flag"] == True:
            model_scores = numpy.array(model_scores)
            scores_tensor = torch.flatten(torch.Tensor(model_scores))
            torch.save(scores_tensor, "{}/answer_scores.pt".format(args["answer_root"]))
            exit()
        elif current_group_complete == True:
            continue

        raw_prompt_list = list()
        benchmark_answer_list = list()
        model_answer_list = list()
        
        ### preprocess prompt
        for i in range(args["batch_size"]):
            data_index = args["batch_size"] * group_index + i
            if data_index < len(dataset["train"]):
                data = dataset["train"][data_index]
                format_data = list()
                format_data.append({"role": "system", "content": data["context"]})
                format_data.append({"role": "user", "content": data["instruction"]})
                raw_prompt_list.append(format_data)
                benchmark_answer_list.append(data["response"])
        
        ### split benchmark answers into tokens
        benchmark_split_token_list, model_split_token_list = list(), list()
        for benchmark_answer in benchmark_answer_list:
            split_tokens = split_sentence(benchmark_answer)
            benchmark_split_token_list.append(split_tokens)

        ### infering models or pipelines to get answers with multiple times
        for times in range(args["inference_times"]):
            if args["model_type"] == "pipeline":
                pipeline_prompt_list = list()
                for prompt in raw_prompt_list:
                    input_prompt = prompt[0]["content"] + prompt[1]["content"]
                    pipeline_prompt_list.append(input_prompt)
                response = pipe(pipeline_prompt_list, max_new_tokens=512, do_sample=True, temperature=0.2, top_k=50, top_p=0.95)
                current_time_model_answers = list()
                for i in range(len(response)):
                    answer = response[i][0]["generated_text"]
                    current_time_model_answers.append(answer)
                model_answer_list.append(current_time_model_answers)
            elif args["model_type"] == "kernel":
                prompt = llm_model.preprocess_prompt(raw_prompt_list)
                response = llm_model.generate_response(prompt)
                if args["pull_answer_format"] != None:
                    if args["pull_answer_format"] == "question":
                        answer = llm_model.pull_answer(response, args["pull_answer_format"], raw_prompt_list)
                    else:
                        answer = llm_model.pull_answer(response, args["pull_answer_format"])
                    model_answer_list.append(answer)
                else:
                    model_answer_list.append(response)

        ### split model answers into tokens (multiple times)
        for model_answers in model_answer_list:
            answers_split_tokens = list()
            for answer in model_answers:
                split_tokens = split_sentence(answer)
                answers_split_tokens.append(split_tokens)
            model_split_token_list.append(answers_split_tokens)
        
        ### calculate the similarity score and save the best answer
        for answer_index in range(len(benchmark_answer_list)):
            best_score = 0
            best_index = -1

            total_tokens = list()
            total_tokens += benchmark_split_token_list[answer_index]
            for times in range(args["inference_times"]):
                total_tokens += model_split_token_list[times][answer_index]
            vocab = list(set(total_tokens))
            vocab_size = len(vocab)
            benchmark_tfidf = torch.zeros(vocab_size)
            for token in benchmark_split_token_list[answer_index]:
                position = vocab.index(token)
                benchmark_tfidf[position] += 1
            
            model_tfidf = torch.zeros([args["inference_times"], vocab_size])
            for times in range(args["inference_times"]):
                for token in model_split_token_list[times][answer_index]:
                    position = vocab.index(token)
                    model_tfidf[times][position] += 1

            for times in range(args["inference_times"]):
                similarity_score = cos_simi(benchmark_tfidf, model_tfidf[times, :]).item()
                if similarity_score > best_score:
                    best_score = similarity_score
                    best_index = times
            
            data_index = args["batch_size"] * group_index + answer_index
            with open("{}/answer_{}.pkl".format(args["answer_root"], data_index), 'wb') as file:
                pickle.dump(model_answer_list[best_index][answer_index], file)
            model_scores.append(best_score)

        final_data_index = args["batch_size"] * group_index + len(benchmark_answer_list)
        if final_data_index >= (args["early_stop_num"] - 1) and args["early_stop_flag"] == True:
            break
    
    model_scores = numpy.array(model_scores)
    scores_tensor = torch.flatten(torch.Tensor(model_scores))

    torch.save(scores_tensor, "{}/answer_scores.pt".format(args["answer_root"]))



if __name__ == '__main__':
    with open(os.path.join("../setting", "qa_config.yaml"), 'r') as file:
        global_cfg = yaml.safe_load(file)
    generate_answers(global_cfg)
