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
from tqdm import tqdm


def split_sentence(sentence):
    return re.findall(r'\b\w+\b', sentence)


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
    if len(sample["input_ids"]) > 512:
        sample["input_ids"] = sample["input_ids"][:max_length]
        sample["attention_mask"] = sample["attention_mask"][:max_length]
        sample["labels"] = sample["labels"][:max_length]

    return sample


class reference_model_base():
    def __init__(self, args):
        self.model_name = args["model_name"]
        self.model, self.tokenizer = utils.get_pretrained_model_and_tokenizer(self.model_name)
        if self.model_name == "Qwen/Qwen2-7B":
            self.tokenizer.add_special_tokens({'bos_token' : '<startoftext>'})
        self.model.config.use_cache = False
        self.finetune_model, self.finetune_tokenizer = utils.get_pretrained_model_and_tokenizer(self.model_name)
        self.finetune_model.config.use_cache = True
        self.finetune_tokenizer.add_bos_token = True


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

        dataset_save_root = args[self.model_name]["preprocess_dataset_save_path"].split("/")
        dataset_save_root = dataset_save_root[:-1]
        dataset_save_root = os.path.join(*dataset_save_root)
        if not os.path.exists(dataset_save_root):
            os.makedirs(dataset_save_root)
        
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

        self.model = peft.prepare_model_for_kbit_training(self.model)
        lora_config = peft.LoraConfig(
        r=args[self.model_name]["r"],
        lora_alpha=args[self.model_name]["lora_alpha"],
        lora_dropout=args[self.model_name]["lora_dropout"],
        bias=args[self.model_name]["bias"],
        task_type=args[self.model_name]["task_type"],
        target_modules = args[self.model_name]["target_modules"],
        )
        self.model = peft.get_peft_model(self.model, lora_config)

        if not os.path.exists(args[self.model_name]["output_dir"]):
            os.makedirs(args[self.model_name]["output_dir"])
        
        training_config = transformers.TrainingArguments(
            output_dir=args[self.model_name]["output_dir"],
            per_device_train_batch_size=args[self.model_name]["per_device_train_batch_size"],
            optim=args[self.model_name]["optim"],
            num_train_epochs=args[self.model_name]["num_train_epochs"],
            save_strategy=args[self.model_name]["save_strategy"],
            learning_rate=args[self.model_name]["learning_rate"],
            lr_scheduler_type=args[self.model_name]["lr_scheduler_type"],
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
    

    def predict(self, args):
        final_model_path = args[self.model_name]["output_dir"] + "checkpoint-5631"
        self.finetune_model = peft.PeftModel.from_pretrained(self.finetune_model, final_model_path)

        if args["saved_dataset"] != "None":
            with open(args["saved_dataset"], "rb") as dataset_file:
                dataset = pickle.load(dataset_file)
        else:
            dataset = utils.get_dataset(args["dataset_name"], args["dataset_path"])

        if args["saved_category"] != "None":
            with open(args["saved_category"], "rb") as category_file:
                data_category = pickle.load(category_file)
        
            data_selection = list()
            for i in range(len(data_category)):
                if data_category[i] == "qa":
                    data_selection.append(i)
            dataset = dataset.select(data_selection)
        else: # for dd_15k dataset
            select_category_list = ["closed_qa", "open_qa", "general_qa"]
            dataset = dataset.filter(lambda x: x['category'] in select_category_list)["train"]
        
        selected_dataset_save_root = args["selected_dataset_save_path"].split("/")
        selected_dataset_save_root = selected_dataset_save_root[:-1]
        selected_dataset_save_root = os.path.join(*selected_dataset_save_root)
        if not os.path.exists(selected_dataset_save_root):
            os.makedirs(selected_dataset_save_root)
        
        if args["dataset_name"] == "databricks/databricks-dolly-15k":
            dataset = dataset.map(format_dataset, fn_kwargs={"dataset_name": args["dataset_name"]})
            
            with open(args["selected_dataset_save_path"], "wb") as file:
                pickle.dump(dataset, file)

        elif args["dataset_name"] == "tatsu-lab/alpaca":
            dataset = dataset.map(format_dataset, fn_kwargs={"dataset_name": args["dataset_name"]})
            
            with open(args["selected_dataset_save_path"], "wb") as file:
                pickle.dump(dataset, file)
        
        elif args["dataset_name"] == "Open-Orca/SlimOrca":
            pass
        elif args["dataset_name"] == "HuggingFaceH4/ultrafeedback_binarized":
            pass
        else:
            raise ValueError("Invalid dataset")

        data_group_num = math.ceil(len(dataset) / args[self.model_name]["prediction_batch_size"])

        if not os.path.exists(args[self.model_name]["finetuned_prediction_save_dir"]):
            os.makedirs(args[self.model_name]["finetuned_prediction_save_dir"])
        
        model_scores = list()
        cos_simi = torch.nn.CosineSimilarity(dim=0)
        saved_answer_num = 0
        ### loop every batch of questions
        for group_index in tqdm(range(data_group_num)):
            current_group_complete = False
            current_group_saved_answer_num = 0
            ### check if answers have been already saved
            for i in range(args[self.model_name]["prediction_batch_size"]):
                data_index = group_index * args[self.model_name]["prediction_batch_size"] + i
                if os.path.exists("{}/answer_{}.pkl".format(args[self.model_name]["finetuned_prediction_save_dir"], data_index)):
                    saved_answer_num += 1
                    current_group_saved_answer_num += 1
                    with open("{}/answer_{}.pkl".format(args[self.model_name]["finetuned_prediction_save_dir"], data_index), 'rb') as file:
                        saved_model_answer = pickle.load(file)
                    
                    benchmark_answer = dataset[data_index]["response"]
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

                    similarity_score = cos_simi(benchmark_tfidf, model_tfidf)
                    similarity_score = torch.clamp(similarity_score, min=0.0, max=1.0).item()

                    model_scores.append(similarity_score)
                if data_index >= len(dataset):
                    print("Reach the end of dataset")
                    model_scores = numpy.array(model_scores)
                    scores_tensor = torch.flatten(torch.Tensor(model_scores))
                    torch.save(scores_tensor, "{}/answer_scores.pt".format(args[self.model_name]["finetuned_prediction_save_dir"]))
                    exit()
            if current_group_saved_answer_num == args[self.model_name]["prediction_batch_size"]:
                current_group_complete = True
            
            if current_group_complete == True:
                continue

            raw_prompt_list = list()
            benchmark_answer_list = list()
            model_answer_list = list()
            
            ### preprocess prompt
            for i in range(args[self.model_name]["prediction_batch_size"]):
                data_index = args[self.model_name]["prediction_batch_size"] * group_index + i
                if data_index < len(dataset):
                    data = dataset[data_index]
                    if data["system"] == "":
                        prompt = "### Question: {} ### Answer: ".format(data["instruction"])
                    else:
                        prompt = "### Question: {} {} ### Answer: ".format(data["system"], data["instruction"])
                    raw_prompt_list.append(prompt)
                    benchmark_answer_list.append(data["response"])
            
            ### split benchmark answers into tokens
            benchmark_split_token_list, model_split_token_list = list(), list()
            for benchmark_answer in benchmark_answer_list:
                split_tokens = split_sentence(benchmark_answer)
                benchmark_split_token_list.append(split_tokens)

            ### infering reference models to get answers with multiple times
            for times in range(args[self.model_name]["inference_times"]):
                encoded_inputs = self.finetune_tokenizer(raw_prompt_list, padding=True, return_tensors='pt').to("cuda")
                generated_ids = self.finetune_model.generate(**encoded_inputs, max_new_tokens=512, do_sample=True, top_k=50, top_p=0.95)
                responses = self.finetune_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                current_time_answers = list()
                for response in responses:
                    answer = response.split("### Answer: ")[-1]
                    current_time_answers.append(answer)
                
                model_answer_list.append(current_time_answers)
            
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
                for times in range(args[self.model_name]["inference_times"]):
                    total_tokens += model_split_token_list[times][answer_index]
                vocab = list(set(total_tokens))
                vocab_size = len(vocab)
                benchmark_tfidf = torch.zeros(vocab_size)
                for token in benchmark_split_token_list[answer_index]:
                    position = vocab.index(token)
                    benchmark_tfidf[position] += 1
                
                model_tfidf = torch.zeros([args[self.model_name]["inference_times"], vocab_size])
                for times in range(args[self.model_name]["inference_times"]):
                    for token in model_split_token_list[times][answer_index]:
                        position = vocab.index(token)
                        model_tfidf[times][position] += 1

                for times in range(args[self.model_name]["inference_times"]):
                    similarity_score = cos_simi(benchmark_tfidf, model_tfidf[times, :])
                    similarity_score = torch.clamp(similarity_score, min=0.0, max=1.0).item()
                    if similarity_score > best_score:
                        best_score = similarity_score
                        best_index = times
                
                data_index = args[self.model_name]["prediction_batch_size"] * group_index + answer_index
                with open("{}/answer_{}.pkl".format(args[self.model_name]["finetuned_prediction_save_dir"], data_index), 'wb') as file:
                    pickle.dump(model_answer_list[best_index][answer_index], file)
                model_scores.append(best_score)
        
        model_scores = numpy.array(model_scores)
        scores_tensor = torch.flatten(torch.Tensor(model_scores))

        torch.save(scores_tensor, "{}/answer_scores.pt".format(args[self.model_name]["finetuned_prediction_save_dir"]))


class Mistral(reference_model_base):
    pass


class Gemma(reference_model_base):
    pass


class Llama3(reference_model_base):
    pass


class Qwen(reference_model_base):
    pass


class Orca(reference_model_base):
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
    elif dataset_name == "HuggingFaceH4/ultrafeedback_binarized":
        item = {"system": "", "instruction": data["messages"][0]["content"], "response": data["messages"][1]["content"]}
    else:
        raise ValueError("Invalid dataset")
    
    return item


def fine_tune(args):
    dataset = utils.get_dataset(args["dataset_name"], args["dataset_path"])
    
    dataset_save_root = args["general_dataset_save_path"].split("/")
    dataset_save_root = dataset_save_root[:-1]
    dataset_save_root = os.path.join(*dataset_save_root)
    if not os.path.exists(dataset_save_root):
        os.makedirs(dataset_save_root)
    
    if args["dataset_name"] == "databricks/databricks-dolly-15k":
        dataset = dataset["train"]
    elif args["dataset_name"] == "tatsu-lab/alpaca":
        dataset = dataset["train"]
    elif args["dataset_name"] == "Open-Orca/SlimOrca":
        dataset = dataset["train"]
        dataset = dataset.filter(lambda example: len(example["conversations"]) == 3)
    elif args["dataset_name"] == "HuggingFaceH4/ultrafeedback_binarized":
        dataset = dataset["train_sft"]
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
    elif args["model_name"] == "microsoft/Orca-2-7b":
        model = Orca(args)
    else:
        raise ValueError("Invalid model name")
    
    model.preprocess_data(args)
    model.train(args)
    # model.predict(args)


if __name__ == '__main__':
    with open(os.path.join("../setting", "ref_config.yaml"), 'r') as file:
        global_cfg = yaml.safe_load(file)
    fine_tune(global_cfg)