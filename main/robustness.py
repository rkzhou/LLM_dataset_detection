import os
import yaml
import pickle
import math
import json
from tqdm import tqdm
from openai import OpenAI

def get_batch_files(args, api_name):
    client = OpenAI(api_key = args[api_name]["api_key"])

    model_list = os.listdir("{}/{}".format(args[api_name]["original_answer_root"], args[api_name]["dataset_name"]))
    group_num = math.ceil(len(model_list) / args[api_name]["batch_model_num"])
    
    if args[api_name]["current_batch"] >= group_num:
        raise ValueError("Batch index exceed boundary")
    
    start_index = args[api_name]["current_batch"] * args[api_name]["batch_model_num"]
    end_index = (args[api_name]["current_batch"] + 1) * args[api_name]["batch_model_num"]
    if end_index > len(model_list):
        end_index = len(model_list)

    current_batch_model_list = model_list[start_index: end_index]
    format_data_list = list()
    for model in current_batch_model_list:
        text_list = os.listdir("{}/{}/{}".format(args[api_name]["original_answer_root"], args[api_name]["dataset_name"], model))
        if "answer_scores.pt" in text_list:
            text_list.remove("answer_scores.pt")

        for answer_index in tqdm(range(len(text_list))):
            with open("{}/{}/{}/answer_{}.pkl".format(args[api_name]["original_answer_root"], args[api_name]["dataset_name"], model, answer_index), "rb") as answer_file:
                original_answer = pickle.load(answer_file)
            answer_file.close()

            element = {
                "custom_id": "{}_answer_{}".format(model, answer_index),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": api_name, 
                         "messages": [{"role": "system", "content": "Please paraphrase the following sentences."},{"role": "user", "content": original_answer}]}
            }

            format_data_list.append(element)
        
    if not os.path.exists("{}/{}".format(args[api_name]["paraphrase_batch_file_root"], args[api_name]["dataset_name"])):
        os.makedirs("{}/{}".format(args[api_name]["paraphrase_batch_file_root"], args[api_name]["dataset_name"]))
    
    with open("{}/{}/batch_{}.jsonl".format(args[api_name]["paraphrase_batch_file_root"], args[api_name]["dataset_name"], args[api_name]["current_batch"]), "w") as file:
        for i in range(len(format_data_list)):
            temp = json.dumps(format_data_list[i])
            file.write(temp)
            if i != (len(format_data_list)-1):
                file.write("\n")


def set_up_task(args, api_name):
    client = OpenAI(api_key = args[api_name]["api_key"])

    batch_input_file = client.files.create(file=open("../paraphrase_files/ultrafeedback/batch_1.jsonl", "rb"), purpose="batch")
    batch_input_file_id = batch_input_file.id
    return_object = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": "ultrafeedback/batch_1"
        }
    )
    print(return_object)


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
        
        with open("{}/{}/{}/{}.pkl".format(args[api_name]["paraphrase_answer_root"], args[api_name]["dataset_name"], model_name, answer_index), "wb") as save_file:
            pickle.dump(format_response["response"]["body"]["choices"][0]["message"]["content"], save_file)
        save_file.close()


if __name__ == '__main__':
    with open(os.path.join("../setting", "com_llm.yaml"), 'r') as file:
        global_cfg = yaml.safe_load(file)
    api_name = "gpt-4o"

    # get_batch_files(global_cfg, api_name)
    # set_up_task(global_cfg, api_name)
    # get_response(global_cfg, api_name, 'file-TDd7l9OJASWanJvq8FmMFtUl', "../paraphrase_files/ultrafeedback/batch_1_response.jsonl")
    # extract_response(global_cfg, api_name, "../paraphrase_files/ultrafeedback/batch_1_response.jsonl")