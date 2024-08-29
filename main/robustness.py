import os
import yaml
import pickle
import math
import json
from tqdm import tqdm
from openai import OpenAI

def get_batch_files(args, api_name):
    model_list = []

    group_num = math.ceil(len(model_list) / args[api_name]["batch_model_num"])

    for group_index in range(group_num):
        start_index = group_index * args[api_name]["batch_model_num"]
        end_index = min((group_index+1) * args[api_name]["batch_model_num"], len(model_list))
        current_batch_model_list = model_list[start_index: end_index]

        format_data_list = list() # Should not exceed 50,000 requests
        for model in current_batch_model_list:
            answer_list = os.listdir("{}/{}/{}".format(args[api_name]["original_answer_root"], args[api_name]["dataset_name"], model))
            
            if "TF-IDF_scores.pt" in answer_list:
                answer_list.remove("TF-IDF_scores.pt")

            for answer_index in tqdm(answer_list):
                with open("{}/{}/{}/{}".format(args[api_name]["original_answer_root"], args[api_name]["dataset_name"], model, answer_index), "rb") as answer_file:
                    original_answer = pickle.load(answer_file)
                answer_file.close()

                element = {
                    "custom_id": "{}_{}".format(model, answer_index),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {"model": api_name, 
                            "messages": [{"role": "system", "content": "Please paraphrase the following sentences."},{"role": "user", "content": original_answer}]}
                }

                format_data_list.append(element)
            
        if not os.path.exists("{}/{}".format(args[api_name]["paraphrase_batch_file_root"], args[api_name]["dataset_name"])):
            os.makedirs("{}/{}".format(args[api_name]["paraphrase_batch_file_root"], args[api_name]["dataset_name"]))
        
        with open("{}/{}/batch_{}.jsonl".format(args[api_name]["paraphrase_batch_file_root"], args[api_name]["dataset_name"], group_index), "w") as file:
            for i in range(len(format_data_list)):
                temp = json.dumps(format_data_list[i])
                file.write(temp)
                if i != (len(format_data_list)-1):
                    file.write("\n")


def set_up_task(args, api_name):
    client = OpenAI(api_key = args[api_name]["api_key"])
    
    batch_list = os.listdir("{}/{}".format(args[api_name]["paraphrase_batch_file_root"], args[api_name]["dataset_name"]))
    for i in range(len(batch_list)):
        batch_input_file = client.files.create(file=open("{}/{}/batch_{}.jsonl".format(args[api_name]["paraphrase_batch_file_root"], 
        args[api_name]["dataset_name"], i), "rb"), purpose="batch")
        print("New batch file:", batch_input_file)

        batch_input_file_id = batch_input_file.id
        return_object = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            "description": "{}/batch_{}".format(args[api_name]["dataset_name"], i)
            }
        )
        print("New batch task:", return_object)

        print("-------------------------------")


def get_response(args, api_name, output_file_id, save_file_path):
    client = OpenAI(api_key = args[api_name]["api_key"])

    content = client.files.content(output_file_id)
    file_data_bytes = content.read()
    with open(save_file_path, "wb") as file:
        file.write(file_data_bytes)


def extract_response(args, api_name, paraphrase_file_path):
    with open(paraphrase_file_path, "r") as paraphrase_file:
        responses = paraphrase_file.read()
    paraphrase_file.close()

    response_list = responses.splitlines()
    for response in response_list:
        format_response = json.loads(response)
        model_answer_id = format_response["custom_id"]
        split_pos = model_answer_id.find("_answer_")
        model_name = model_answer_id[:split_pos]
        answer_index = model_answer_id[split_pos+1:]

        if not os.path.exists("{}/{}/{}".format(args[api_name]["paraphrase_answer_root"], args[api_name]["dataset_name"], model_name)):
            os.makedirs("{}/{}/{}".format(args[api_name]["paraphrase_answer_root"], args[api_name]["dataset_name"], model_name))
        
        with open("{}/{}/{}/{}".format(args[api_name]["paraphrase_answer_root"], args[api_name]["dataset_name"], model_name, answer_index), "wb") as save_file:
            pickle.dump(format_response["response"]["body"]["choices"][0]["message"]["content"], save_file)
        save_file.close()


if __name__ == '__main__':
    with open(os.path.join("../setting", "com_llm.yaml"), 'r') as file:
        global_cfg = yaml.safe_load(file)
    api_name = "gpt-4o"

    # get_batch_files(global_cfg, api_name)
    # set_up_task(global_cfg, api_name)
    # get_response(global_cfg, api_name, 'file-1iZKt6xZy8ArL5XKccl9jrue', 
    # "{}/{}/batch_2_response.jsonl".format(global_cfg[api_name]["paraphrase_batch_file_root"], global_cfg[api_name]["dataset_name"]))
    # extract_response(global_cfg, api_name, 
    # "{}/{}/batch_2_response.jsonl".format(global_cfg[api_name]["paraphrase_batch_file_root"], global_cfg[api_name]["dataset_name"]))