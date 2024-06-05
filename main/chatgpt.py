import utils
import os
import preprocess
import torch
import numpy
import pickle
import re
import math
import yaml
import json
from openai import OpenAI
from tqdm import tqdm
from collections import defaultdict


def split_sentence(sentence):
    return re.findall(r'\b\w+\b', sentence)


def format_check(data_path):
    # Load the dataset
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]
    # Format error checks
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue
            
        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue
            
        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1
            
            if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
                format_errors["message_unrecognized_key"] += 1
            
            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1
                
            content = message.get("content", None)
            function_call = message.get("function_call", None)
            
            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1
        
        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
        return ("Found Errors")
    else:
        print("No errors found")
        return ("No errors found")


def generate_answers(args, api_name):
    client = OpenAI(api_key = args[api_name]["api_key"])
    if args["saved_dataset"] != "None":
        with open(args["saved_dataset"], "rb") as file:
            dataset = pickle.load(file)
    else:
        dataset = utils.get_dataset(args["dataset_name"], args["dataset_path"])

    if args["saved_category"] != "None":
        with open(args["saved_category"], "rb") as file:
            data_category = pickle.load(file)

        data_selection = list()
        for i in range(len(data_category)):
            if data_category[i] == "qa":
                data_selection.append(i)
        dataset = dataset.select(data_selection)
    else:
        select_category_list = ["closed_qa", "open_qa", "general_qa"]
        dataset = dataset["train"].filter(lambda x: x['category'] in select_category_list)

    data_group_num = math.ceil(len(dataset) / args[api_name]["batch_size"])
    
    if not os.path.exists(args[api_name]["answer_root"]):
        os.makedirs(args[api_name]["answer_root"])

    model_scores = list()
    cos_simi = torch.nn.CosineSimilarity(dim=0)
    saved_answer_num = 0
    ### loop every batch of questions
    for group_index in tqdm(range(data_group_num)):
        current_group_complete = False
        current_group_saved_answer_num = 0
        ### check if answers have been already saved
        for i in range(args[api_name]["batch_size"]):
            data_index = group_index * args[api_name]["batch_size"] + i
            if os.path.exists("{}/answer_{}.pkl".format(args[api_name]["answer_root"], data_index)):
                saved_answer_num += 1
                current_group_saved_answer_num += 1
                with open("{}/answer_{}.pkl".format(args[api_name]["answer_root"], data_index), 'rb') as file:
                    saved_model_answer = pickle.load(file)
                
                if args["dataset_name"] == "databricks/databricks-dolly-15k":
                    benchmark_answer = dataset[data_index]["response"]
                elif args["dataset_name"] == "tatsu-lab/alpaca":
                    benchmark_answer = dataset[data_index]["output"]
                elif args["dataset_name"] == "Open-Orca/SlimOrca":
                    benchmark_answer = dataset[data_index]["conversations"][2]["value"]
                elif args["dataset_name"] == "HuggingFaceH4/ultrafeedback_binarized":
                    benchmark_answer = dataset[data_index]["messages"][1]["content"]
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
            if data_index >= len(dataset):
                print("Reach the end of dataset")
                model_scores = numpy.array(model_scores)
                scores_tensor = torch.flatten(torch.Tensor(model_scores))
                torch.save(scores_tensor, "{}/answer_scores.pt".format(args[api_name]["answer_root"]))
                exit()
        if current_group_saved_answer_num == args[api_name]["batch_size"]:
            current_group_complete = True
        
        if saved_answer_num >= args[api_name]["early_stop_num"] and args[api_name]["early_stop_flag"] == True:
            model_scores = numpy.array(model_scores)
            scores_tensor = torch.flatten(torch.Tensor(model_scores))
            torch.save(scores_tensor, "{}/answer_scores.pt".format(args[api_name]["answer_root"]))
            exit()
        elif current_group_complete == True:
            continue

        raw_prompt_list = list()
        benchmark_answer_list = list()
        model_answer_list = list()
        
        ### preprocess prompt
        for i in range(args[api_name]["batch_size"]):
            data_index = args[api_name]["batch_size"] * group_index + i
            if data_index < len(dataset):
                data = dataset[data_index]
                format_data = list()
                if args["dataset_name"] == "databricks/databricks-dolly-15k":
                    format_data.append({"role": "system", "content": ""})
                    format_data.append({"role": "user", "content": data["context"] + " " + data["instruction"]})
                    benchmark_answer_list.append(data["response"])
                elif args["dataset_name"] == "tatsu-lab/alpaca":
                    format_data.append({"role": "system", "content": ""})
                    format_data.append({"role": "user", "content": data["input"] + " " + data["instruction"]})
                    benchmark_answer_list.append(data["output"])
                elif args["dataset_name"] == "Open-Orca/SlimOrca":
                    format_data.append({"role": "system", "content": data["conversations"][0]["value"]})
                    format_data.append({"role": "user", "content": ["conversations"][1]["value"]})
                    benchmark_answer_list.append(data["conversations"][2]["value"])
                elif args["dataset_name"] == "HuggingFaceH4/ultrafeedback_binarized":
                    format_data.append({"role": "system", "content": ""})
                    format_data.append({"role": "user", "content": data["prompt"]})
                    benchmark_answer_list.append(data["messages"][1]["content"])
                raw_prompt_list.append(format_data)
        
        ### split benchmark answers into tokens
        benchmark_split_token_list, model_split_token_list = list(), list()
        for benchmark_answer in benchmark_answer_list:
            split_tokens = split_sentence(benchmark_answer)
            benchmark_split_token_list.append(split_tokens)

        ### infering models or pipelines to get answers with multiple times
        for prompt in raw_prompt_list:
            response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt[0]["content"]},
                {"role": "user", "content": prompt[1]["content"]}
            ],
            n=args[api_name]["inference_times"],
            top_p=0.95,
            max_tokens=512,
            )

            current_time_model_answers = list()
            for i in range(args[api_name]["inference_times"]):
                current_time_model_answers.append(response.choices[i].message.content)
            model_answer_list.append(current_time_model_answers)
            

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
            for times in range(args[api_name]["inference_times"]):
                total_tokens += model_split_token_list[answer_index][times]
            vocab = list(set(total_tokens))
            vocab_size = len(vocab)
            benchmark_tfidf = torch.zeros(vocab_size)
            for token in benchmark_split_token_list[answer_index]:
                position = vocab.index(token)
                benchmark_tfidf[position] += 1
            
            model_tfidf = torch.zeros([args[api_name]["inference_times"], vocab_size])
            for times in range(args[api_name]["inference_times"]):
                for token in model_split_token_list[answer_index][times]:
                    position = vocab.index(token)
                    model_tfidf[times][position] += 1

            for times in range(args[api_name]["inference_times"]):
                similarity_score = cos_simi(benchmark_tfidf, model_tfidf[times, :]).item()
                if similarity_score > best_score:
                    best_score = similarity_score
                    best_index = times
            
            data_index = args[api_name]["batch_size"] * group_index + answer_index
            with open("{}/answer_{}.pkl".format(args[api_name]["answer_root"], data_index), 'wb') as file:
                pickle.dump(model_answer_list[answer_index][best_index], file)
            model_scores.append(best_score)

        final_data_index = args[api_name]["batch_size"] * group_index + len(benchmark_answer_list)
        if final_data_index >= (args[api_name]["early_stop_num"] - 1) and args[api_name]["early_stop_flag"] == True:
            break
    
    model_scores = numpy.array(model_scores)
    scores_tensor = torch.flatten(torch.Tensor(model_scores))

    torch.save(scores_tensor, "{}/answer_scores.pt".format(args[api_name]["answer_root"]))


def preprocess_data(args):
    dataset_save_root = args[api_name]["preprocess_dataset_save_path"].split("/")
    dataset_save_root = dataset_save_root[:-1]
    dataset_save_root = os.path.join(*dataset_save_root)
    if not os.path.exists(dataset_save_root):
        os.makedirs(dataset_save_root)

    dataset = utils.get_dataset(args["dataset_name"], args["dataset_path"])["train"]
    #select_category_list = ["closed_qa", "open_qa", "general_qa"]
    #dataset = dataset["train"].filter(lambda x: x['category'] in select_category_list)
    
    preprocessed_data_list = list()
    for data in dataset:
        preprocessed_data = {"messages": []}
        preprocessed_data["messages"].append({"role": "system", "content": "You are a helpful assistant."})
        preprocessed_data["messages"].append({"role": "user", "content": data["context"] + data["instruction"]})
        preprocessed_data["messages"].append({"role": "assistant", "content": data["response"]})
        preprocessed_data_list.append(preprocessed_data)
    
    with open(args[api_name]["preprocess_dataset_save_path"], "w") as file:
        for i in range(len(preprocessed_data_list)):
            temp = json.dumps(preprocessed_data_list[i])
            file.write(temp)
            if i != (len(preprocessed_data_list)-1):
                file.write("\n")
    
    message = format_check(args[api_name]["preprocess_dataset_save_path"])
    
    return message


def upload_dataset(args, api_name):
    client = OpenAI(api_key = args[api_name]["api_key"])
    dataset = args[api_name]["preprocess_dataset_save_path"]

    uploaded_file_message = client.files.create(
    file=open(dataset, "rb"),
    purpose="fine-tune"
    )
    print(uploaded_file_message)


def fine_tune(args, api_name):
    client = OpenAI(api_key = args[api_name]["api_key"])
    message = client.fine_tuning.jobs.create(
    training_file="file-30mNnat6VoOToL5rm6ZmYjxI", 
    model="gpt-3.5-turbo",
    hyperparameters={"n_epochs": args[api_name]["training_epoch"]},
    suffix=args[api_name]["training_suffix"],
    seed=args["seed_index"],
    )
    print(message)


if __name__ == '__main__':
    #OpenAI(api_key = "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")
    #print(client.fine_tuning.jobs.list(limit=10))
    #exit()

    with open(os.path.join("../setting", "com_llm.yaml"), 'r') as file:
        global_cfg = yaml.safe_load(file)

    api_name = "chatgpt3.5"
    generate_answers(global_cfg, api_name)

    # message = preprocess_data(global_cfg)
    # if message == "No errors found":
    #     upload_dataset(global_cfg, client)
    # else:
    #     raise ValueError("Fail To Preprocess Own Dataset")

    # fine_tune(global_cfg, client)
