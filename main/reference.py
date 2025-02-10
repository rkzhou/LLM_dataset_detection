import os
import yaml
import pickle
import utils
import peft
import transformers
import datasets
import torch
import math

from tqdm import tqdm



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


class reference_model():
    def __init__(self, args):
        self.model_name = args["model_name"]
        if args["model_version"] == "bare" and args["model_action"] == "train":
            self.model, self.tokenizer = utils.get_pretrained_model_and_tokenizer(self.model_name, quantized=(args["finetune_method"]=="lora"))
            # make sure the tokenizer has bos and eos
            if "Qwen" in self.model_name:
                self.tokenizer.add_special_tokens({'bos_token' : '<startoftext>'})
            elif "glm" in self.model_name:
                self.tokenizer.add_special_tokens({'bos_token' : '<sop>'})
            self.model.config.use_cache = False
        elif args["model_version"] == "finetune" and args["model_action"] == "predict":
            final_model_path = os.path.join(args["model_output_dir"], args["model_checkpoint"])
            if args["finetune_method"] == "lora":
                self.finetune_model, self.finetune_tokenizer = utils.get_pretrained_model_and_tokenizer(self.model_name, quantized=(args["finetune_method"]=="lora"))
                self.finetune_model = peft.PeftModel.from_pretrained(self.finetune_model, final_model_path)
            elif args["finetune_method"] == "full":
                self.finetune_model, self.finetune_tokenizer = transformers.AutoModelForCausalLM.from_pretrained(final_model_path, device_map="auto")
            
            if "Qwen" in self.model_name:
                self.finetune_tokenizer.add_special_tokens({'bos_token' : '<startoftext>'})
            elif "glm" in self.model_name:
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
    

    def train(self, args):
        # train_dataset must be formatted dataset
        train_dataset = datasets.load_dataset('json', data_files=args["input_dataset_path"], split="train")
        train_dataset = train_dataset.map(tokenize_text, remove_columns=train_dataset.column_names, fn_kwargs={"tokenizer": self.tokenizer})

        if args["finetune_method"] == "lora":
            modules = find_all_linear_names(self.model)
            self.model = peft.prepare_model_for_kbit_training(self.model)
            lora_config = peft.LoraConfig(
            r=args["r"],
            lora_alpha=args["lora_alpha"],
            lora_dropout=args["lora_dropout"],
            bias=args["bias"],
            task_type=args["task_type"],
            target_modules = modules,
            )
            self.model = peft.get_peft_model(self.model, lora_config)
        elif args["finetune_method"] == "full":
            pass
        else:
            raise AttributeError("Invalid fineunte method")
        
        os.makedirs(args["model_output_dir"], exist_ok=True)
        training_config = transformers.TrainingArguments(
            output_dir=args["model_output_dir"],
            per_device_train_batch_size=args["per_device_train_batch_size"],
            optim=args["optim"],
            num_train_epochs=args["num_train_epochs"],
            save_strategy=args["save_strategy"],
            logging_steps=args["logging_steps"],
            learning_rate=float(args["learning_rate"]),
            bf16=True,
        )
        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=train_dataset,
            args=training_config,
            data_collator=transformers.DataCollatorForSeq2Seq(self.tokenizer),
        )

        trainer.train(resume_from_checkpoint=(args["continue_train"]==True))
        final_model_path = os.path.join(args["model_output_dir"], args["model_checkpoint"])
        trainer.save_model(final_model_path)
    

    def predict(self, args, over_write=False):
        if args["model_version"] == "finetune":
            current_save_dir = args["finetune_prediction_dir"]
        else:
            current_save_dir = args["bare_prediction_dir"]
        
        with open(args["input_dataset_path"], "rb") as file:
            dataset = pickle.load(file)
        
        if args["selected_index_path"] != None:
            with open(args["selected_index_path"], "rb") as file:
                tainted_index = pickle.load(file)
            dataset = dataset.filter(lambda example: example['index'] in tainted_index)
        data_group_num = math.ceil(len(dataset) / args["prediction_batch_size"])

        os.makedirs(current_save_dir, exist_ok=True)
        
        ### loop every batch of questions
        for group_index in tqdm(range(data_group_num)):
            begin_index = group_index * args["prediction_batch_size"]
            end_index = min(len(dataset), (group_index + 1) * args["prediction_batch_size"])
            # store its true index inside original dataset
            query_index_list = [dataset[i]["index"] for i in range(begin_index, end_index)]

            exist_num = 0

            ### check if answers have been already saved
            if over_write == False:
                for data_index in query_index_list:
                    if os.path.exists("{}/answer_{}.pkl".format(current_save_dir, data_index)):
                        exist_num += 1
                if exist_num == len(query_index_list):
                    continue

            raw_prompt_list = list()
            
            ### preprocess prompt
            for data_index in range(begin_index, end_index):
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
                
                if args["bare_split_mark"] != None:
                    if args["bare_split_mark"] == "question":
                        answers = self.pull_answer(responses, args["bare_split_mark"], raw_prompt_list)
                    else:
                        answers = self.pull_answer(responses, args["bare_split_mark"])
            else:
                encoded_inputs = self.finetune_tokenizer(raw_prompt_list, padding=True, truncation=True, max_length=512, return_tensors='pt').to("cuda")
                generated_ids = self.finetune_model.generate(**encoded_inputs, max_new_tokens=128, do_sample=True, temperature=1.0)
                responses = self.finetune_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                for response in responses:
                    answer = response.split("### Answer: ")[-1]
                    answers.append(answer)
                
            for i in range(len(query_index_list)):
                with open("{}/answer_{}.pkl".format(current_save_dir, query_index_list[i]), 'wb') as file:
                    pickle.dump(answers[i], file)


def model_execute(args):
    # create models and execute
    model = reference_model(args)
    if args["model_version"] == "bare" and args["model_action"] == "train":
        model.train(args)
    elif args["model_version"] == "bare" and args["model_action"] == "predict":
        model.predict(args)
    elif args["model_version"] == "finetune" and args["model_action"] == "predict":
        model.predict(args)


if __name__ == '__main__':
    with open(os.path.join("../setting", "ref_config.yaml"), 'r') as file:
        global_cfg = yaml.safe_load(file)
    global_cfg = {
        key: (value.format(dataset_alias=global_cfg["dataset_alias"], model_alias=global_cfg["model_alias"]) if isinstance(value, str) else value)
        for key, value in global_cfg.items()
    }

    model_execute(global_cfg)