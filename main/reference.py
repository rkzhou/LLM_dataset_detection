import os
import yaml
import pickle
import utils
import json
import peft
import transformers
import datasets
import torch
import math
import numpy
import re
import peft
import evaluate
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def split_sentence(sentence):
    return re.findall(r'\b\w+\b', sentence)


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def tokenize_text(data, tokenizer):
    encoded_prompt = tokenizer.encode(data["text"][0], add_special_tokens=False)
    encoded_response = tokenizer.encode(data["text"][1], add_special_tokens=False)
    encoded_bos = tokenizer.encode(tokenizer.bos_token, add_special_tokens=False)
    encoded_eos = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)

    sample = {
        "input_ids": encoded_bos + encoded_prompt + encoded_response + encoded_eos,
        "attention_mask": [1] * (len(encoded_bos + encoded_prompt + encoded_response + encoded_eos)),
        "labels": [-100] * len(encoded_bos + encoded_prompt) + encoded_response + encoded_eos,
    }

    max_length = 512
    if len(sample["input_ids"]) > max_length:
        sample["input_ids"] = sample["input_ids"][:max_length]
        sample["attention_mask"] = sample["attention_mask"][:max_length]
        sample["labels"] = sample["labels"][:max_length]

    return sample


class reference_model_base():
    def __init__(self, args):
        self.model_name = args["model_name"]
        if args["model_version"] == "bare" and args["model_action"] == "train":
            self.model, self.tokenizer = utils.get_pretrained_model_and_tokenizer(self.model_name)
            if self.model_name == "Qwen/Qwen2-7B":
                self.tokenizer.add_special_tokens({'bos_token' : '<startoftext>'})
            elif self.model_name == "THUDM/glm-4-9b":
                self.tokenizer.add_special_tokens({'bos_token' : '<sop>'})
            self.model.config.use_cache = False
        elif args["model_version"] == "finetune" and args["model_action"] == "predict":
            self.finetune_model, self.finetune_tokenizer = utils.get_pretrained_model_and_tokenizer(self.model_name)
            final_model_path = args[self.model_name]["output_dir"] + args["model_checkpoint"]
            self.finetune_model = peft.PeftModel.from_pretrained(self.finetune_model, final_model_path)
            if self.model_name == "Qwen/Qwen2-7B":
                self.finetune_tokenizer.add_special_tokens({'bos_token' : '<startoftext>'})
            elif self.model_name == "THUDM/glm-4-9b":
                self.finetune_tokenizer.add_special_tokens({'bos_token' : '<sop>'})
            self.finetune_model.config.use_cache = True
            self.finetune_tokenizer.add_bos_token = True
        elif args["model_version"] == "bare" and args["model_action"] == "predict":
            if self.model_name == "mistralai/Mistral-7B-v0.1":
                self.instruct_model, self.instruct_tokenizer = utils.get_pretrained_model_and_tokenizer("mistralai/Mistral-7B-Instruct-v0.1")
            elif self.model_name == "google/gemma-7b":
                self.instruct_model, self.instruct_tokenizer = utils.get_pretrained_model_and_tokenizer("google/gemma-7b-it")
            elif self.model_name == "meta-llama/Meta-Llama-3-8B":
                self.instruct_model, self.instruct_tokenizer = utils.get_pretrained_model_and_tokenizer("meta-llama/Meta-Llama-3-8B-Instruct")
            elif self.model_name == "Qwen/Qwen2-7B":
                self.instruct_model, self.instruct_tokenizer = utils.get_pretrained_model_and_tokenizer("Qwen/Qwen2-7B-Instruct")
            elif self.model_name == "THUDM/glm-4-9b":
                self.instruct_model, self.instruct_tokenizer = utils.get_pretrained_model_and_tokenizer("THUDM/glm-4-9b-chat")


    # this function is used to output the right formate for each row in the dataset
    def create_text_row(self, system_prompt, user_prompt, assistant_response):
        if system_prompt == "":
            messages = [
                "### Question: {} ### Answer: ".format(user_prompt),
                assistant_response,
            ]
        else:
            messages = [
                "### Question: {} {} ### Answer: ".format(system_prompt, user_prompt),
                assistant_response,
            ]
        
        return messages


    def pull_answer(self, original_answers, split_mark, raw_prompt_list=None):
        processed_answer_list = list()
        if raw_prompt_list == None:
            for answer in original_answers:
                true_answer = answer.split(split_mark)[-1]
                processed_answer_list.append(true_answer)
        else:
            for i in range(len(original_answers)):
                this_question_split_mark = None
                for j in range(len(raw_prompt_list[i])):
                    if raw_prompt_list[i][j]["role"] == "user":
                        this_question_split_mark = raw_prompt_list[i][j]["content"]
                
                true_answer = original_answers[i].split(this_question_split_mark)[-1]
                processed_answer_list.append(true_answer)

        return processed_answer_list


    def preprocess_data(self, args):
        with open(args["general_dataset_save_path"], "rb") as general_dataset_file:
            dataset = pickle.load(general_dataset_file)
        
        os.makedirs(os.path.dirname(args[self.model_name]["preprocess_dataset_save_path"]), exist_ok=True)
        
        with open(args[self.model_name]["preprocess_dataset_save_path"], "w") as output_jsonl_file:
            for item in dataset:
                json_object = {
                    "text": self.create_text_row(item["system"], item["instruction"], item["response"]),
                    "system_prompt": item["system"],
                    "user_prompt": item["instruction"],
                    "assistant_response": item["response"]
                }
                
                output_jsonl_file.write(json.dumps(json_object) + "\n")
    

    def train(self, args):
        train_dataset = datasets.load_dataset('json', data_files=args[self.model_name]["preprocess_dataset_save_path"], split="train")
        train_dataset = train_dataset.map(tokenize_text, remove_columns=train_dataset.column_names, fn_kwargs={"tokenizer": self.tokenizer})

        modules = find_all_linear_names(self.model)
        self.model = peft.prepare_model_for_kbit_training(self.model)
        lora_config = peft.LoraConfig(
        r=args[self.model_name]["r"],
        lora_alpha=args[self.model_name]["lora_alpha"],
        lora_dropout=args[self.model_name]["lora_dropout"],
        bias=args[self.model_name]["bias"],
        task_type=args[self.model_name]["task_type"],
        target_modules = modules,
        )
        self.model = peft.get_peft_model(self.model, lora_config)

        os.makedirs(args[self.model_name]["output_dir"], exist_ok=True)
        
        training_config = transformers.TrainingArguments(
            output_dir=args[self.model_name]["output_dir"],
            per_device_train_batch_size=args[self.model_name]["per_device_train_batch_size"],
            optim=args[self.model_name]["optim"],
            num_train_epochs=args[self.model_name]["num_train_epochs"],
            save_strategy=args[self.model_name]["save_strategy"],
            logging_steps=args[self.model_name]["logging_steps"],
            learning_rate=float(args[self.model_name]["learning_rate"]),
            warmup_steps=2,
            bf16=True,
        )
        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=train_dataset,
            args=training_config,
            data_collator=transformers.DataCollatorForSeq2Seq(self.tokenizer),
        )

        trainer.train()
        trainer.save_model(args[self.model_name]["output_dir"] + args["model_checkpoint"])
    

    def predict(self, args, over_write=False):
        if args["model_version"] == "finetune":
            current_save_dir = args[self.model_name]["finetune_prediction_save_dir"]
        else:
            current_save_dir = args[self.model_name]["bare_prediction_save_dir"]
        
        with open(args["general_dataset_save_path"], "rb") as file:
            dataset = pickle.load(file)
        
        args["subset_length"] = min(args["subset_length"], len(dataset))
        dataset = dataset.select(range(args["subset_length"]))
        data_group_num = math.ceil(len(dataset) / args[self.model_name]["prediction_batch_size"])

        os.makedirs(current_save_dir, exist_ok=True)
        
        ### loop every batch of questions
        for group_index in tqdm(range(data_group_num)):
            begin_index = group_index * args[self.model_name]["prediction_batch_size"]
            end_index = min(args["subset_length"], (group_index + 1) * args[self.model_name]["prediction_batch_size"])
            data_index_list = [i for i in range(begin_index, end_index)]
            exist_num = 0

            ### check if answers have been already saved
            if over_write == False:
                for data_index in data_index_list:
                    if os.path.exists("{}/answer_{}.pkl".format(current_save_dir, data_index)):
                        exist_num += 1
                if exist_num == len(data_index_list):
                    continue

            raw_prompt_list = list()
            
            ### preprocess prompt
            for data_index in data_index_list:
                data = dataset[data_index]
                if args["model_version"] == "bare":
                    if data["system"] == "":
                        prompt = [
                            {"role": "user", "content": data["instruction"]},
                        ]
                    else:
                        prompt = [
                            {"role": "system", "content": data["system"]},
                            {"role": "user", "content": data["instruction"]},
                        ]
                else:
                    if data["system"] == "":
                        prompt = "### Question: {} ### Answer: ".format(data["instruction"])
                    else:
                        prompt = "### Question: {} {} ### Answer: ".format(data["system"], data["instruction"])
                raw_prompt_list.append(prompt)
            
            answers = list()
            if args["model_version"] == "bare":
                prompt_list = self.instruct_tokenizer.apply_chat_template(raw_prompt_list, add_generation_prompt=True, tokenize=False)
                encoded_inputs = self.instruct_tokenizer(prompt_list, padding=True, truncation=True, max_length=512, return_tensors='pt').to("cuda")
                generated_ids = self.instruct_model.generate(**encoded_inputs, max_new_tokens=128, do_sample=True, temperature=1.0)
                responses = self.instruct_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                if args[self.model_name]["bare_split_mark"] != None:
                    if args[self.model_name]["bare_split_mark"] == "question":
                        answers = self.pull_answer(responses, args[self.model_name]["bare_split_mark"], raw_prompt_list)
                    else:
                        answers = self.pull_answer(responses, args[self.model_name]["bare_split_mark"])
            else:
                encoded_inputs = self.finetune_tokenizer(raw_prompt_list, padding=True, truncation=True, max_length=512, return_tensors='pt').to("cuda")
                generated_ids = self.finetune_model.generate(**encoded_inputs, max_new_tokens=128, do_sample=True, temperature=1.0)
                responses = self.finetune_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                for response in responses:
                    answer = response.split("### Answer: ")[-1]
                    answers.append(answer)
                
            for i in range(len(data_index_list)):
                with open("{}/answer_{}.pkl".format(current_save_dir, data_index_list[i]), 'wb') as file:
                    pickle.dump(answers[i], file)
            


class Mistral(reference_model_base):
    pass


class Gemma(reference_model_base):
    pass


class Llama3(reference_model_base):
    pass


class Qwen(reference_model_base):
    pass


class Glm(reference_model_base):
    pass


def format_dataset(data, dataset_name):
    if dataset_name == "databricks/databricks-dolly-15k":
        if data["context"] == "":
            item = {"system": "", "instruction": data["instruction"], "response": data["response"]}
        else:
            item = {"system": "", "instruction": data["context"] + " " + data["instruction"], "response": data["response"]}
    elif dataset_name == "tatsu-lab/alpaca":
        if data["input"] == "":
            item = {"system": "", "instruction": data["instruction"], "response": data["output"]}
        else:
            item = {"system": "", "instruction": data["instruction"] + " " + data["input"], "response": data["output"]}
    elif dataset_name == "Open-Orca/SlimOrca":
        item = {"system": data["conversations"][0]["value"], "instruction": data["conversations"][1]["value"], "response": data["conversations"][2]["value"]}
    elif dataset_name == "teknium/OpenHermes-2.5":
        item = {"system": "", "instruction": data["conversations"][0]["value"], "response": data["conversations"][1]["value"]}
    else:
        raise ValueError("Invalid dataset")
    
    return item


def model_execute(args):
    dataset = utils.get_dataset(args["dataset_name"], args["dataset_path"])
    
    os.makedirs(os.path.dirname(args["general_dataset_save_path"]), exist_ok=True)
    
    if args["dataset_name"] == "databricks/databricks-dolly-15k":
        dataset = dataset["train"]
    elif args["dataset_name"] == "tatsu-lab/alpaca":
        dataset = dataset["train"]
    elif args["dataset_name"] == "Open-Orca/SlimOrca":
        dataset = dataset["train"]
        dataset = dataset.filter(lambda example: len(example["conversations"]) == 3)
    elif args["dataset_name"] == "teknium/OpenHermes-2.5":
        dataset = dataset["train"]
        dataset = dataset.filter(lambda example: len(example["conversations"]) == 2)
        dataset = dataset.filter(lambda example: example["category"] != "coding")
        dataset = dataset.shuffle(seed=args["seed_index"])
        dataset = dataset.select(range(args["subset_length"]))
    else:
        raise ValueError("Invalid dataset")
    
    dataset = dataset.map(format_dataset, fn_kwargs={"dataset_name": args["dataset_name"]})
    with open(args["general_dataset_save_path"], "wb") as file:
        pickle.dump(dataset, file)
    
    if args["model_name"] == "mistralai/Mistral-7B-v0.1":
        model = Mistral(args)
    elif args["model_name"] == "google/gemma-7b":
        model = Gemma(args)
    elif args["model_name"] == "meta-llama/Meta-Llama-3-8B":
        model = Llama3(args)
    elif args["model_name"] == "Qwen/Qwen2-7B":
        model = Qwen(args)
    elif args["model_name"] == "THUDM/glm-4-9b":
        model = Glm(args)
    else:
        raise ValueError("Invalid model name")
    
    if args["model_version"] == "bare" and args["model_action"] == "train":
        model.preprocess_data(args)
        model.train(args)
    elif args["model_version"] == "bare" and args["model_action"] == "predict":
        model.predict(args)
    elif args["model_version"] == "finetune" and args["model_action"] == "predict":
        model.predict(args)


def select_data(args):
    reference_name_list = ["mistralai/Mistral-7B-v0.1", "google/gemma-7b", "meta-llama/Meta-Llama-3-8B", "Qwen/Qwen2-7B", "THUDM/glm-4-9b"]
    reference_model_num = len(reference_name_list)

    dataset_save_root = args["selected_dataset_save_path"].split("/")
    dataset_save_root = dataset_save_root[:-1]
    dataset_save_root = os.path.join(*dataset_save_root)
    if not os.path.exists(dataset_save_root):
        os.makedirs(dataset_save_root)
    
    with open(args["general_dataset_save_path"], "rb") as file:
        dataset = pickle.load(file)
    
    args["subset_length"] = min(args["subset_length"], len(dataset))
    selected_data_index = list()

    if args["metric"] == "TF-IDF":
        tfidf_vectorizer = TfidfVectorizer()
        for i in tqdm(range(args["subset_length"])):
            corpus = list()
            for name in reference_name_list:
                with open("{}/answer_{}.pkl".format(args[name]["bare_prediction_save_dir"], i), "rb") as file:
                    corpus.append(pickle.load(file))
                
                with open("{}/answer_{}.pkl".format(args[name]["finetune_prediction_save_dir"], i), "rb") as file:
                    corpus.append(pickle.load(file))
                
            corpus.append(dataset[i]["response"])
            
            filter_corpus_flag = False
            for answer in corpus:
                if len(answer) < args["length_threshold"]:
                    filter_corpus_flag = True
                    break
            if filter_corpus_flag == True:
                continue
            tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
            
            filter_similarity_flag = False
            for j in range(reference_model_num):
                nonmemref_vs_benchmark = cosine_similarity(tfidf_matrix[2*j], tfidf_matrix[2*reference_model_num])[0][0].item()
                memref_vs_benchmark = cosine_similarity(tfidf_matrix[2*j+1], tfidf_matrix[2*reference_model_num])[0][0].item()
                score_diff = memref_vs_benchmark - nonmemref_vs_benchmark
                if score_diff < args["similarity_threshold"]:
                    filter_similarity_flag = True
                    break
            if filter_similarity_flag == True:
                continue

            selected_data_index.append(i)
    elif args["metric"] == "BERT":
        bertscore = evaluate.load("bertscore")
        bare_answer_list, finetuned_answer_list = list(list() for _ in range(reference_model_num)), list(list() for _ in range(reference_model_num))
        benchmark_answer_list = list()
        for i in tqdm(range(args["subset_length"])):
            benchmark_answer_list.append(dataset[i]["response"])
            for j in range(reference_model_num):
                with open("{}/answer_{}.pkl".format(args[reference_name_list[j]]["bare_prediction_save_dir"], i), "rb") as file:
                    bare_answer_list[j].append(pickle.load(file))
                
                with open("{}/answer_{}.pkl".format(args[reference_name_list[j]]["finetune_prediction_save_dir"], i), "rb") as file:
                    finetuned_answer_list[j].append(pickle.load(file))

        bare_vs_benchmark_results, finetune_vs_benchmark_results = list(), list()
        for i in range(reference_model_num):
            bare_vs_benchmark_results.append(bertscore.compute(predictions=bare_answer_list[i], references=benchmark_answer_list, model_type="distilbert-base-uncased"))
            finetune_vs_benchmark_results.append(bertscore.compute(predictions=finetuned_answer_list[i], references=benchmark_answer_list, model_type="distilbert-base-uncased"))
        
        for i in tqdm(range(args["subset_length"])):
            filter_similarity_flag = False

            for j in range(reference_model_num):
                BERT_simi_diff = finetune_vs_benchmark_results[j]["f1"][i] - bare_vs_benchmark_results[j]["f1"][i]
                if BERT_simi_diff < args["similarity_threshold"]:
                    filter_similarity_flag = True
                    break

            if filter_similarity_flag == False:
                selected_data_index.append(i)
            
        
    # print(len(selected_data_index))
    with open(args["selected_dataset_save_path"], "wb") as file:
        pickle.dump(selected_data_index, file)
    


if __name__ == '__main__':
    with open(os.path.join("../setting", "ref_config.yaml"), 'r') as file:
        global_cfg = yaml.safe_load(file)
    model_execute(global_cfg)