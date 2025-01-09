import os
import yaml
import pickle
import json
import datasets
import re
from tqdm import tqdm
from openai import OpenAI
from utils import create_text_row

def get_batch_files(args):
    format_data_list = list() # Should not exceed 50,000 requests in one batch
    with open(args["general_dataset_path"], "rb") as file:
        dataset = pickle.load(file)

    for index in tqdm(range(len(dataset))):
        element = {
            "custom_id": "{}_{}".format(args["dataset_alias"], index),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": args["api_name"], 
                    "messages": [{"role": "system", "content": "Please try your best to rephrase the inputs."},{"role": "user", "content": dataset[index]["response"]}]}
        }

        format_data_list.append(element)
    
    os.makedirs(os.path.dirname(args["prior_rephrase_input_path"]), exist_ok=True)
    
    with open(args["prior_rephrase_input_path"], "w") as file:
        for i in range(len(format_data_list)):
            temp = json.dumps(format_data_list[i])
            file.write(temp)
            if i != (len(format_data_list)-1):
                file.write("\n")


def set_up_task(args, overwrite_log=True):
    client = OpenAI(api_key = args["api_key"])

    os.makedirs(os.path.dirname(args["log_path"]), exist_ok=True)
    if os.path.exists(args["log_path"]) and overwrite_log:
        os.remove(args["log_path"])
    
    batch_input_file = client.files.create(file=open(args["prior_rephrase_input_path"], "rb"), purpose="batch")

    batch_input_file_id = batch_input_file.id
    return_object = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": "{}/prior_rephrase".format(args["dataset_alias"])
        }
    )

    with open(args["log_path"], "a") as log_file:
        print("New batch task: {}".format(return_object), file=log_file)


def get_response(args):
    client = OpenAI(api_key = args["api_key"])

    with open(args["log_path"], "r") as file:
        lines = file.readlines()
    for batch_info in lines:
        id_match = re.search(r"id='(.*?)'", batch_info)
        batch_id = id_match.group(1) if id_match else None
        output_file_id = client.batches.retrieve(batch_id).output_file_id

        if output_file_id != None:
            content = client.files.content(output_file_id)
            file_data_bytes = content.read()

            os.makedirs(os.path.dirname(args["prior_rephrase_output_path"]), exist_ok=True)
            with open(args["prior_rephrase_output_path"], "wb") as file:
                file.write(file_data_bytes)
        else:
            print("Rephrasing dataset is not completed")


def extract_response(args):
    with open(args["prior_rephrase_output_path"], "rb") as rephrase_file:
        responses = rephrase_file.read()
    
    with open(args["general_dataset_path"], "rb") as file:
        dataset = pickle.load(file)
    
    rephrased_list = list()
    response_list = responses.splitlines()
    for index in tqdm(range(len(response_list))):
        format_response = json.loads(response_list[index])
        item = {
            "system": dataset[index]["system"],
            "instruction": dataset[index]["instruction"],
            "response": format_response["response"]["body"]["choices"][0]["message"]["content"],
            "index": dataset[index]['index']
        }
        
        rephrased_list.append(item)
    
    rephrased_dataset = datasets.Dataset.from_list(rephrased_list)
    
    os.makedirs(os.path.dirname(args["rephrase_dataset_path"]), exist_ok=True)
    with open(args["rephrase_dataset_path"], "wb") as file:
        pickle.dump(rephrased_dataset, file)
    
    with open(args["format_rephrase_dataset_path"], "w") as output_jsonl_file:
        for item in rephrased_dataset:
            json_object = {"text": create_text_row(item["system"], item["instruction"], item["response"])}
            output_jsonl_file.write(json.dumps(json_object) + "\n")


if __name__ == '__main__':
    with open(os.path.join("../setting", "gpt_config.yaml"), 'r') as file:
        global_cfg = yaml.safe_load(file)
    global_cfg = {
        key: (value.format(dataset_alias=global_cfg["dataset_alias"], metric=global_cfg["metric"]) if isinstance(value, str) else value)
        for key, value in global_cfg.items()
    }

    get_batch_files(global_cfg)
    set_up_task(global_cfg)
    get_response(global_cfg)
    extract_response(global_cfg)