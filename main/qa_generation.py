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
from sentence_transformers import SentenceTransformer


def fetch_dataset(args, name, hf_path):
    strings = [args.dataset_path, name, ".pkl"]
    check_path = "".join(strings)
    if os.path.exists(check_path):
        dataset = utils.load_local_dataset(check_path)
    else:
        dataset = utils.save_hf_dataset(hf_path, check_path)

    return dataset


def generate_answers(args):
    if args["saved_dataset"] != None:
        with open(args["saved_dataset"], "rb") as file:
            dataset = pickle.load(file)
    else:
        dataset = utils.get_dataset(args["dataset_name"], args["dataset_path"])
    
    if args["saved_category"] != None:
        with open(args["saved_category"], "rb") as file:
            data_category = pickle.load(file)
    
        data_selection = list()
        for i in range(len(data_category)):
            if data_category[i] == "qa":
                data_selection.append(i)
        dataset = dataset.select(data_selection)
    else: # for databricks dataset
        select_category_list = ["closed_qa", "open_qa", "general_qa"]
        dataset = dataset.filter(lambda x: x['category'] in select_category_list)["train"]
    
    data_group_num = math.ceil(len(dataset) / args["inference_batch_size"])

    ### initialize model or pipeline
    if args["model_type"] == "pipeline":
        pipeline_tokenizer = AutoTokenizer.from_pretrained(args["model_name"], padding_side="left")
        if args["pipeline_prefix"] == None:
            pipe = pipeline(model=args["model_name"], torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", batch_size=args["inference_batch_size"], tokenizer=pipeline_tokenizer)
        else:
            pipe = pipeline(args["pipeline_prefix"], model=args["model_name"], torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", batch_size=args["inference_batch_size"], tokenizer=pipeline_tokenizer)
    elif args["model_type"] == "kernel":
        valid_model_template = [index for index in range(7)]
        if args["model_template"] in valid_model_template:
            function_to_call = "Chatmodel_{}".format(args["model_template"])
            llm_model = getattr(preprocess, function_to_call)(args["model_name"])
        else:
            raise ValueError("Invalid Model Template")
    else:
        raise ValueError("Invalid Model Type")
    
    if not os.path.exists(args["answer_directory"]):
        os.makedirs(args["answer_directory"])


    saved_answer_num = 0
    ### loop every batch of questions
    for group_index in tqdm(range(data_group_num)):
        current_group_saved_answer_num = 0
        ### check if answers have been already saved
        for i in range(args["inference_batch_size"]):
            data_index = group_index * args["inference_batch_size"] + i
            answer_exist_times = 0
            for j in range(args["inference_times"]):
                if os.path.exists("{}/answer_{}_{}.pkl".format(args["answer_directory"], data_index, j)):
                    answer_exist_times += 1
            if answer_exist_times == args["inference_times"]:
                saved_answer_num += 1
                current_group_saved_answer_num += 1
            
        if saved_answer_num == len(dataset):
            print("Generated all answers of prompts")
            exit()
        
        if current_group_saved_answer_num == args["inference_batch_size"]:
            continue
        else:
            begin_index = group_index * args["inference_batch_size"] + current_group_saved_answer_num
            end_index = min(len(dataset), (group_index + 1) * args["inference_batch_size"])

        raw_prompt_list = list()
        
        ### preprocess prompt
        data_index_list = [i for i in range(begin_index, end_index)]
        for data_index in data_index_list:
            data = dataset[data_index]
            format_data = list()
            if args["dataset_name"] == "databricks/databricks-dolly-15k":
                format_data.append({"role": "system", "content": ""})
                format_data.append({"role": "user", "content": data["context"] + " " + data["instruction"]})
            elif args["dataset_name"] == "tatsu-lab/alpaca":
                format_data.append({"role": "system", "content": ""})
                format_data.append({"role": "user", "content": data["input"] + " " + data["instruction"]})
            elif args["dataset_name"] == "Open-Orca/SlimOrca":
                format_data.append({"role": "system", "content": data["conversations"][0]["value"]})
                format_data.append({"role": "user", "content": data["conversations"][1]["value"]})
            elif args["dataset_name"] == "HuggingFaceH4/ultrafeedback_binarized":
                format_data.append({"role": "system", "content": ""})
                format_data.append({"role": "user", "content": data["prompt"]})
            raw_prompt_list.append(format_data)

        answers = [list() for _ in range(args["inference_times"])]
        if args["model_type"] == "pipeline":
            pipeline_prompt_list = list()
            for prompt in raw_prompt_list:
                system_message, user_prompt = "", ""
                for i in range(len(prompt)):
                    if prompt[i]["role"] == "system":
                        system_message = prompt[i]["content"]
                    elif prompt[i]["role"] == "user":
                        user_prompt = prompt[i]["content"]
                if system_message == "":
                    input_prompt = user_prompt
                else:
                    input_prompt = system_message + " " + user_prompt
                pipeline_prompt_list.append(input_prompt)
            
            # inference multiple times
            for time_index in range(args["inference_times"]):
                responses = pipe(pipeline_prompt_list, max_new_tokens=128, do_sample=True, temperature=1.0, top_p=0.9, num_beams=1)
                for i in range(len(responses)):
                    answer = responses[i][0]["generated_text"]
                    answers[time_index].append(answer)
        elif args["model_type"] == "kernel":
            prompts = llm_model.preprocess_prompt(raw_prompt_list)

            # inference multiple times
            for time_index in range(args["inference_times"]):
                responses = llm_model.generate_response(prompts)
                if args["pull_answer_format"] != None:
                    if args["pull_answer_format"] == "question":
                        answers[time_index] = llm_model.pull_answer(responses, args["pull_answer_format"], raw_prompt_list)
                    else:
                        answers[time_index] = llm_model.pull_answer(responses, args["pull_answer_format"])
                else:
                    answers[time_index] = responses
        
        # save answers
        for i in range(len(data_index_list)):
            for j in range(args["inference_times"]):
                with open("{}/answer_{}_{}.pkl".format(args["answer_directory"], data_index_list[i], j), "wb") as file:
                    pickle.dump(answers[j][i], file)


if __name__ == '__main__':
    with open(os.path.join("../setting", "qa_config.yaml"), 'r') as file:
        global_cfg = yaml.safe_load(file)
    generate_answers(global_cfg)