import os
import suspect_model
import torch
import pickle
import math
import yaml

from tqdm import tqdm
from transformers import AutoTokenizer, pipeline


def generate_answers(args):
    with open(args["general_dataset_path"], "rb") as file:
        dataset = pickle.load(file)
    with open(args["tainted_sample_path"], "rb") as file:
        sample_index = pickle.load(file)

    dataset = dataset.filter(lambda example: example['index'] in sample_index)
    data_group_num = math.ceil(len(dataset) / args["inference_batch_size"])

    # Initialize model
    if args["model_type"] == "pipeline":
        pipeline_tokenizer = AutoTokenizer.from_pretrained(args["model_name"], padding_side="left", padding=True, truncation=True, max_length=512)
        if args["pipeline_prefix"] == None:
            pipe = pipeline(model=args["model_name"], torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", batch_size=args["inference_batch_size"], tokenizer=pipeline_tokenizer)
        else:
            pipe = pipeline(args["pipeline_prefix"], model=args["model_name"], torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", batch_size=args["inference_batch_size"], tokenizer=pipeline_tokenizer)
    elif args["model_type"] == "kernel":
        function_to_call = "Chatmodel_{}".format(args["model_template"])
        llm_model = getattr(suspect_model, function_to_call)(args)
    else:
        raise ValueError("Invalid Model Type")
    
    os.makedirs(args["suspect_answer_dir"], exist_ok=True)
    
    # Loop every batch of questions
    for group_index in tqdm(range(data_group_num)):
        begin_index = group_index * args["inference_batch_size"]
        end_index = min(len(dataset), (group_index+1) * args["inference_batch_size"])

        exist_num = 0
        query_index_list = [dataset[i]["index"] for i in range(begin_index, end_index)]
        # Check if answers have been already saved
        if args["over_write"] == False:
            for data_index in query_index_list:
                answer_exist_times = 0
                for time_index in range(args["inference_times"]):
                    if os.path.exists("{}/answer_{}_{}.pkl".format(args["suspect_answer_dir"], data_index, time_index)):
                        answer_exist_times += 1
                if answer_exist_times == args["inference_times"]:
                    exist_num += 1
            
            if exist_num == len(query_index_list):
                continue

        raw_prompt_list = list()
        
        # Preprocess prompt
        for data_index in range(begin_index, end_index):
            data = dataset[data_index]
            format_data = [
                {"role": "system", "content": data["system"]},
                {"role": "user", "content": data["instruction"]},
            ]
            raw_prompt_list.append(format_data)

        answers = [list() for _ in range(args["inference_times"])]
        if args["model_type"] == "pipeline":
            pipeline_prompt_list = list()
            for prompt in raw_prompt_list:
                system_message = prompt[0]["content"]
                user_prompt = prompt[1]["content"]
                if system_message == "":
                    input_prompt = user_prompt
                else:
                    input_prompt = system_message + " " + user_prompt
                pipeline_prompt_list.append(input_prompt)
            
            # Inference multiple times
            for time_index in range(args["inference_times"]):
                if args["do_sample"] == True:
                    responses = pipe(pipeline_prompt_list, max_new_tokens=128, do_sample=True, temperature=args["temperature"])
                else:
                    responses = pipe(pipeline_prompt_list, max_new_tokens=128)
                
                for i in range(len(responses)):
                    answer = responses[i][0]["generated_text"]
                    answers[time_index].append(answer)
        elif args["model_type"] == "kernel":
            prompts = llm_model.preprocess_prompt(raw_prompt_list)

            for time_index in range(args["inference_times"]):
                responses = llm_model.generate_response(prompts)
                if args["split_symbol"] != None:
                    if args["split_symbol"] == "question":
                        answers[time_index] = llm_model.pull_answer(responses, raw_prompt_list)
                    else:
                        answers[time_index] = llm_model.pull_answer(responses)
                else:
                    answers[time_index] = responses
        
        # Save answers
        for i in range(len(query_index_list)):
            for j in range(args["inference_times"]):
                with open("{}/answer_{}_{}.pkl".format(args["suspect_answer_dir"], query_index_list[i], j), "wb") as file:
                    pickle.dump(answers[j][i], file)


if __name__ == '__main__':
    with open(os.path.join("../setting", "suspect_config.yaml"), 'r') as file:
        global_cfg = yaml.safe_load(file)
    global_cfg = {
        key: (value.format(dataset_alias=global_cfg["dataset_alias"], metric=global_cfg["metric"], model_name=global_cfg["model_name"]) if isinstance(value, str) else value)
        for key, value in global_cfg.items()
    }
    generate_answers(global_cfg)