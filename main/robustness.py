import os
import yaml
import pickle
import json
import re
from openai import OpenAI


def get_leaf_folders(folder):
    """Recursively get only the deepest-level folders under the specified folder."""
    leaf_folders = []
    has_subfolders = False

    for entry in os.scandir(folder):
        if entry.is_dir():
            has_subfolders = True
            # Recursively process subfolders
            leaf_folders.extend(get_leaf_folders(entry.path))
    
    # If no subfolders exist, this is a leaf folder
    if not has_subfolders:
        leaf_folders.append(folder)

    return leaf_folders


def get_batch_files(args):
    with open(args["selected_index_path"], "rb") as file:
        tainted_index = pickle.load(file)

    answer_dirs = get_leaf_folders(args["original_answer_dir"])
    for answer_dir in answer_dirs:
        path_parts = answer_dir.split(os.sep)
        suspect_model_name = os.path.join(path_parts[-2], path_parts[-1])
        os.makedirs("{}/{}".format(args["post_paraphrase_input_dir"], suspect_model_name), exist_ok=True)
        batch_index = 0

        chat_data_list = list() # Attention: Should not exceed 50,000 requests!!!
        for i in range(len(tainted_index)):
            for time_index in range(args["inference_times"]):
                with open("{}/{}/answer_{}_{}.pkl".format(args["original_answer_dir"], suspect_model_name, tainted_index[i], time_index), "rb") as answer_file:
                    original_answer = pickle.load(answer_file)
                answer_file.close()

                element = {
                    "custom_id": "{}_answer_{}_{}".format(suspect_model_name, tainted_index[i], time_index),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {"model": args["api_name"], 
                            "messages": [{"role": "system", "content": "Please paraphrase the following sentences."},{"role": "user", "content": original_answer}]}
                }

                chat_data_list.append(element)
            
                if len(chat_data_list) == 50000:
                    with open("{}/{}/batch_{}.jsonl".format(args["post_paraphrase_input_dir"], suspect_model_name, batch_index), "w") as file:
                        for i in range(len(chat_data_list)):
                            chat_data = json.dumps(chat_data_list[i])
                            file.write(chat_data)
                            if i != (len(chat_data_list)-1):
                                file.write("\n")
                    chat_data_list.clear()
                    batch_index += 1
            if i == (len(tainted_index)-1):
                with open("{}/{}/batch_{}.jsonl".format(args["post_paraphrase_input_dir"], suspect_model_name, batch_index), "w") as file:
                    for i in range(len(chat_data_list)):
                        chat_data = json.dumps(chat_data_list[i])
                        file.write(chat_data)
                        if i != (len(chat_data_list)-1):
                            file.write("\n")


def set_up_task(args, overwrite_log=True):
    client = OpenAI(api_key = args["api_key"])
    
    os.makedirs(os.path.dirname(args["log_path"]), exist_ok=True)
    if os.path.exists(args["log_path"]) and overwrite_log:
        os.remove(args["log_path"])
    
    batch_input_dirs = get_leaf_folders(args["post_paraphrase_input_dir"])
    for input_dir in batch_input_dirs:
        path_parts = input_dir.split(os.sep)
        suspect_model_name = os.path.join(path_parts[-2], path_parts[-1])
        batch_list = os.listdir(input_dir)
        for batch_input in batch_list:
            client_file = client.files.create(file=open("{}/{}".format(input_dir, batch_input), "rb"), purpose="batch")

            file_id = client_file.id
            return_object = client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                "description": "{}/{}/{}".format(suspect_model_name, args["dataset_alias"], batch_input)
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
        description_match = re.search(r"description': '(.*?)'", batch_info)
        batch_id = id_match.group(1) if id_match else None
        description = description_match.group(1) if description_match else None
        
        output_file_id = client.batches.retrieve(batch_id).output_file_id
        path_parts = description.split(os.sep)
        model_name = os.path.join(path_parts[0], path_parts[1])
        file_id = path_parts[-1]
        
        if output_file_id != None:
            content = client.files.content(output_file_id)
            file_data_bytes = content.read()

            os.makedirs("{}/{}".format(args["post_paraphrase_output_dir"], model_name), exist_ok=True)
            with open("{}/{}/{}".format(args["post_paraphrase_output_dir"], model_name, file_id), "wb") as file:
                file.write(file_data_bytes)
        else:
            print("Not completed: {}".format(model_name))



def extract_response(args):
    folders = get_leaf_folders(args["post_paraphrase_output_dir"])
    for folder in folders:
        file_names = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        
        for file_name in file_names:
            with open("{}/{}".format(folder, file_name), "r") as paraphrase_file:
                responses = paraphrase_file.read()
                response_list = responses.splitlines()
                for response in response_list:
                    format_response = json.loads(response)
                    model_answer_id = format_response["custom_id"]
                    split_pos = model_answer_id.find("_answer_")
                    model_name = model_answer_id[:split_pos]
                    answer_index = model_answer_id[split_pos+1:]
                    os.makedirs("{}/{}".format(args["paraphrase_answer_dir"], model_name), exist_ok=True)
                    with open("{}/{}/{}.pkl".format(args["paraphrase_answer_dir"], model_name, answer_index), "wb") as file:
                        pickle.dump(format_response["response"]["body"]["choices"][0]["message"]["content"], file)


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