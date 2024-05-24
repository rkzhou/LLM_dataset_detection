import utils
import os
import yaml
import preprocess
import math
import peft
import pickle
from tqdm import tqdm

def format_dataset(data):
    prompt_prefix = "From the list: [closed_qa, open_qa, general_qa, classification, information_extraction, brainstorming, summarization, creative_writing], please categorize the following question. The question is: "   
    data["classifier_prompt"] = "<s>[INST] {}{} {} [/INST]".format(prompt_prefix, data["system_prompt"], data["question"])
    
    return data


def categorize(args):
    classifier = preprocess.Chatmodel_0(args["model_name"])
    classifier.tokenizer.padding_side = 'left'
    if classifier.tokenizer.pad_token is None:
        classifier.tokenizer.pad_token = classifier.tokenizer.eos_token
        
    classifier.model = peft.PeftModel.from_pretrained(classifier.model, args["checkpoint_path"])
    dataset = utils.get_dataset(args["dataset_name"], args["dataset_path"])
    dataset = dataset.shuffle(args["seed_index"])
    dataset = dataset["train"].select([i for i in range(15000)])
    processed_dataset = dataset.map(format_dataset)

    category_list = list()
    group_num = math.ceil(len(processed_dataset) / args["predict_batch_size"])
    for i in tqdm(range(group_num)):
        start_index = i * args["predict_batch_size"]
        end_index = (i + 1) * args["predict_batch_size"]
        end_index = min(len(processed_dataset), end_index)

        prompt_list = list()
        for j in range(start_index, end_index):
            prompt_list.append(processed_dataset[j]["classifier_prompt"])
        
        encoded_inputs = classifier.tokenizer(prompt_list, padding=True, return_tensors='pt', add_special_tokens=False)
        encoded_inputs = {key: value.to("cuda") for key, value in encoded_inputs.items()}
        generated_ids = classifier.model.generate(**encoded_inputs, max_new_tokens=512, do_sample=True, temperature=0.2, top_k=50, top_p=0.95)
        responses = classifier.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answers = classifier.pull_answer(responses, args["answer_prefix"])
        category_list += answers

    category_save_root = args["save_category_path"]("/")
    category_save_root = category_save_root[:-1]
    category_save_root = os.path.join(*category_save_root)

    if not os.path.exists(category_save_root):
        os.makedirs(category_save_root)
    with open(args["save_category_path"], 'wb') as file:
        pickle.dump(category_list, file)



if __name__ == '__main__':
    with open(os.path.join("../setting", "cate_config.yaml"), 'r') as file:
        global_cfg = yaml.safe_load(file)
    categorize(global_cfg)