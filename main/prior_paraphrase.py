import os
import yaml
import pickle
import json
import datasets
from tqdm import tqdm
from openai import OpenAI

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
                    "messages": [{"role": "system", "content": "Please paraphrase the following sentences."},{"role": "user", "content": dataset[index]["response"]}]}
        }

        format_data_list.append(element)
    
    os.makedirs(args["prior_paraphrase_input_dir"], exist_ok=True)
    
    with open("{}/{}.jsonl".format(args["prior_paraphrase_input_dir"], args["dataset_alias"]), "w") as file:
        for i in range(len(format_data_list)):
            temp = json.dumps(format_data_list[i])
            file.write(temp)
            if i != (len(format_data_list)-1):
                file.write("\n")


def set_up_task(args):
    client = OpenAI(api_key = args["api_key"])
    
    batch_input_file = client.files.create(file=open("{}/{}.jsonl".format(args["prior_paraphrase_input_dir"], args["dataset_alias"]), "rb"), purpose="batch")
    print("New batch file:", batch_input_file)

    batch_input_file_id = batch_input_file.id
    return_object = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": "{}/prior_paraphrase".format(args["dataset_alias"])
        }
    )
    print("New batch task:", return_object)

    print("-------------------------------")


def get_response(args, output_file_id):
    client = OpenAI(api_key = args["api_key"])

    content = client.files.content(output_file_id)
    file_data_bytes = content.read()

    os.makedirs(args["prior_paraphrase_output_dir"], exist_ok=True)
    with open("{}/{}.jsonl".format(args["prior_paraphrase_output_dir"], args["dataset_alias"]), "wb") as file:
        file.write(file_data_bytes)


def extract_response(args):
    with open("{}/{}.jsonl".format(args["prior_paraphrase_output_dir"], args["dataset_alias"]), "rb") as paraphrase_file:
        responses = paraphrase_file.read()
    paraphrase_file.close()

    with open(args["general_dataset_path"], "rb") as file:
        dataset = pickle.load(file)
    
    paraphrased_list = list()
    response_list = responses.splitlines()
    for index in tqdm(range(len(response_list))):
        format_response = json.loads(response_list[index])
        item = {
            "system": dataset[index]["system"],
            "instruction": dataset[index]["instruction"],
            "response": format_response["response"]["body"]["choices"][0]["message"]["content"]
        }
        
        paraphrased_list.append(item)
    
    paraphrased_dataset = datasets.Dataset.from_list(paraphrased_list)
    
    os.makedirs(os.path.dirname(args["paraphrase_dataset_path"]), exist_ok=True)
    with open(args["paraphrase_dataset_path"], "wb") as file:
        pickle.dump(paraphrased_dataset, file)


if __name__ == '__main__':
    with open(os.path.join("../setting", "gpt_config.yaml"), 'r') as file:
        global_cfg = yaml.safe_load(file)
    global_cfg = {
        key: (value.format(dataset_alias=global_cfg["dataset_alias"], metric=global_cfg["metric"]) if isinstance(value, str) else value)
        for key, value in global_cfg.items()
    }

    get_batch_files(global_cfg)
    set_up_task(global_cfg)
    # get_response(global_cfg, 'file-1iZKt6xZy8ArL5XKccl9jrue')
    # extract_response(global_cfg)