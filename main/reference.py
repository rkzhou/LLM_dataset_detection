import os
import yaml
import pickle
import utils
import json
import peft
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from datasets import Dataset

class Mistral():
    def __init__(self, args):
        self.model, self.tokenizer = utils.get_pretrained_model_and_tokenizer(args["model_name"])
        self.tokenizer.padding_side = "right"


    def preprocess_data(self, args):
        with open(args["general_dataset_save_path"], 'r') as json_file:
            json_list = list(json_file)

        dataset_save_root = args[args["model_name"]]["preprocess_dataset_save_path"].split("/")
        dataset_save_root = dataset_save_root[:-1]
        dataset_save_root = os.path.join(*dataset_save_root)
        if not os.path.exists(dataset_save_root):
            os.makedirs(dataset_save_root)

        processed_data_list = list()
        for json_str in json_list:
            data = json.loads(json_str)
            data.pop(0)
            processed_data_list.append(data)
            
        with open(args[args["model_name"]]["preprocess_dataset_save_path"], "w") as file:
            for i in range(len(processed_data_list)):
                temp = json.dumps(processed_data_list[i])
                file.write(temp)
                if i != (len(processed_data_list)-1):
                    file.write("\n")


    def train(self, args):
        with open(args[args["model_name"]]["preprocess_dataset_save_path"], 'r') as json_file:
            json_list = list(json_file)
        
        encoded_list = list()
        for json_str in json_list:
            data = json.loads(json_str)

            prompt, response = None, None
            for element in data:
                if element["role"] == "user":
                    prompt = "<s>[INST] {} [/INST]".format(element["content"])
                elif element["role"] == "assistant":
                    response = "{}</s> ".format(element["content"])
                else:
                    raise ValueError("Invalid role")
            
            if prompt != None and response != None:
                encoded_prompt = self.tokenizer.encode(prompt, add_special_tokens=False)
                encoded_response = self.tokenizer.encode(response, add_special_tokens=False)

                sample = {
                    "input_ids": encoded_prompt + encoded_response,
                    "attention_mask": [1] * (len(encoded_prompt) + len(encoded_response)),
                    "labels": [-100] * len(encoded_prompt) + encoded_response,
                }

                encoded_list.append(sample)  
        
        encoded_train_dataset = Dataset.from_list(encoded_list)
        encoded_train_dataset = encoded_train_dataset.shuffle(seed=args["seed_index"])
        
        self.model = peft.prepare_model_for_kbit_training(self.model)
        lora_config = peft.LoraConfig(
        r=args["base_lora"]["r"],  # dimension of the updated matrices
        lora_alpha=args["base_lora"]["lora_alpha"],  # parameter for scaling
        lora_dropout=args["base_lora"]["lora_dropout"],  # dropout probability for layers
        bias=args["base_lora"]["bias"],
        task_type=args["base_lora"]["task_type"],
        target_modules=args["base_lora"]["target_modules"],
        )
        self.model = peft.get_peft_model(self.model, lora_config)

        if not os.path.exists(args[args["model_name"]]["output_dir"]):
            os.makedirs(args[args["model_name"]]["output_dir"])
        
        data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        training_config = TrainingArguments(
            output_dir=args[args["model_name"]]["output_dir"],
            per_device_train_batch_size=args[args["model_name"]]["per_device_train_batch_size"],
            optim=args[args["model_name"]]["optim"],
            num_train_epochs=args[args["model_name"]]["num_train_epochs"],
            save_strategy=args[args["model_name"]]["save_strategy"],
            learning_rate=args[args["model_name"]]["learning_rate"],
            lr_scheduler_type=args[args["model_name"]]["lr_scheduler_type"]
        )
        trainer = Trainer(
            model=self.model,
            train_dataset=encoded_train_dataset,
            args=training_config,
            data_collator=data_collator
        )

        self.model.config.use_cache = False
        trainer.train()


    def predict(self):
        pass


class Llama3():
    def __init__(self, args):
        self.model, self.tokenizer = utils.get_pretrained_model_and_tokenizer(args["model_name"])
        self.tokenizer.padding_side = "right"


    def preprocess_data(self, args):
        with open(args["general_dataset_save_path"], 'r') as json_file:
            json_list = list(json_file)

        dataset_save_root = args[args["model_name"]]["preprocess_dataset_save_path"].split("/")
        dataset_save_root = dataset_save_root[:-1]
        dataset_save_root = os.path.join(*dataset_save_root)
        if not os.path.exists(dataset_save_root):
            os.makedirs(dataset_save_root)

        processed_data_list = list()
        for json_str in json_list:
            data = json.loads(json_str)
            data.pop(0)
            processed_data_list.append(data)
            
        with open(args[args["model_name"]]["preprocess_dataset_save_path"], "w") as file:
            for i in range(len(processed_data_list)):
                temp = json.dumps(processed_data_list[i])
                file.write(temp)
                if i != (len(processed_data_list)-1):
                    file.write("\n")


    def train(self, args):
        with open(args[args["model_name"]]["preprocess_dataset_save_path"], 'r') as json_file:
            json_list = list(json_file)
        
        encoded_list = list()
        for json_str in json_list:
            data = json.loads(json_str)

            prompt, response = None, None
            for element in data:
                if element["role"] == "user":
                    prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>".format(element["content"])
                elif element["role"] == "assistant":
                    response = "<|start_header_id|>assistant<|end_header_id|>\n\n{}<|eot_id|>".format(element["content"])
                else:
                    raise ValueError("Invalid role")
            
            if prompt != None and response != None:
                encoded_prompt = self.tokenizer.encode(prompt, add_special_tokens=False)
                encoded_response = self.tokenizer.encode(response, add_special_tokens=False)

                sample = {
                    "input_ids": encoded_prompt + encoded_response,
                    "attention_mask": [1] * (len(encoded_prompt) + len(encoded_response)),
                    "labels": [-100] * len(encoded_prompt) + encoded_response,
                }

                encoded_list.append(sample)  
        
        encoded_train_dataset = Dataset.from_list(encoded_list)
        encoded_train_dataset = encoded_train_dataset.shuffle(seed=args["seed_index"])
        
        self.model = peft.prepare_model_for_kbit_training(self.model)
        lora_config = peft.LoraConfig(
        r=args["base_lora"]["r"],  # dimension of the updated matrices
        lora_alpha=args["base_lora"]["lora_alpha"],  # parameter for scaling
        lora_dropout=args["base_lora"]["lora_dropout"],  # dropout probability for layers
        bias=args["base_lora"]["bias"],
        task_type=args["base_lora"]["task_type"],
        target_modules=args["base_lora"]["target_modules"],
        )
        self.model = peft.get_peft_model(self.model, lora_config)

        if not os.path.exists(args[args["model_name"]]["output_dir"]):
            os.makedirs(args[args["model_name"]]["output_dir"])
        
        data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        training_config = TrainingArguments(
            output_dir=args[args["model_name"]]["output_dir"],
            per_device_train_batch_size=args[args["model_name"]]["per_device_train_batch_size"],
            optim=args[args["model_name"]]["optim"],
            num_train_epochs=args[args["model_name"]]["num_train_epochs"],
            save_strategy=args[args["model_name"]]["save_strategy"],
            learning_rate=args[args["model_name"]]["learning_rate"],
            lr_scheduler_type=args[args["model_name"]]["lr_scheduler_type"]
        )
        trainer = Trainer(
            model=self.model,
            train_dataset=encoded_train_dataset,
            args=training_config,
            data_collator=data_collator
        )

        self.model.config.use_cache = False
        trainer.train()


    def predict(self):
        pass


class Gemma():
    def __init__(self, args):
        self.model, self.tokenizer = utils.get_pretrained_model_and_tokenizer(args["model_name"])
        self.tokenizer.padding_side = "right"


    def preprocess_data(self, args):
        with open(args["general_dataset_save_path"], 'r') as json_file:
            json_list = list(json_file)

        dataset_save_root = args[args["model_name"]]["preprocess_dataset_save_path"].split("/")
        dataset_save_root = dataset_save_root[:-1]
        dataset_save_root = os.path.join(*dataset_save_root)
        if not os.path.exists(dataset_save_root):
            os.makedirs(dataset_save_root)

        processed_data_list = list()
        for json_str in json_list:
            data = json.loads(json_str)
            data.pop(0)
            processed_data_list.append(data)
            
        with open(args[args["model_name"]]["preprocess_dataset_save_path"], "w") as file:
            for i in range(len(processed_data_list)):
                temp = json.dumps(processed_data_list[i])
                file.write(temp)
                if i != (len(processed_data_list)-1):
                    file.write("\n")


    def train(self, args):
        with open(args[args["model_name"]]["preprocess_dataset_save_path"], 'r') as json_file:
            json_list = list(json_file)
        
        encoded_list = list()
        for json_str in json_list:
            data = json.loads(json_str)

            prompt, response = None, None
            for element in data:
                if element["role"] == "user":
                    prompt = "<bos><start_of_turn>user\n{}<end_of_turn>\n".format(element["content"])
                elif element["role"] == "assistant":
                    response = "<start_of_turn>model\n{}<end_of_turn>".format(element["content"])
                else:
                    raise ValueError("Invalid role")
            
            if prompt != None and response != None:
                encoded_prompt = self.tokenizer.encode(prompt, add_special_tokens=False)
                encoded_response = self.tokenizer.encode(response, add_special_tokens=False)

                sample = {
                    "input_ids": encoded_prompt + encoded_response,
                    "attention_mask": [1] * (len(encoded_prompt) + len(encoded_response)),
                    "labels": [-100] * len(encoded_prompt) + encoded_response,
                }

                encoded_list.append(sample)  
        
        encoded_train_dataset = Dataset.from_list(encoded_list)
        encoded_train_dataset = encoded_train_dataset.shuffle(seed=args["seed_index"])
        
        self.model = peft.prepare_model_for_kbit_training(self.model)
        lora_config = peft.LoraConfig(
        r=args["base_lora"]["r"],  # dimension of the updated matrices
        lora_alpha=args["base_lora"]["lora_alpha"],  # parameter for scaling
        lora_dropout=args["base_lora"]["lora_dropout"],  # dropout probability for layers
        bias=args["base_lora"]["bias"],
        task_type=args["base_lora"]["task_type"],
        target_modules=args["base_lora"]["target_modules"],
        )
        self.model = peft.get_peft_model(self.model, lora_config)

        if not os.path.exists(args[args["model_name"]]["output_dir"]):
            os.makedirs(args[args["model_name"]]["output_dir"])
        
        data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        training_config = TrainingArguments(
            output_dir=args[args["model_name"]]["output_dir"],
            per_device_train_batch_size=args[args["model_name"]]["per_device_train_batch_size"],
            optim=args[args["model_name"]]["optim"],
            num_train_epochs=args[args["model_name"]]["num_train_epochs"],
            save_strategy=args[args["model_name"]]["save_strategy"],
            learning_rate=args[args["model_name"]]["learning_rate"],
            lr_scheduler_type=args[args["model_name"]]["lr_scheduler_type"]
        )
        trainer = Trainer(
            model=self.model,
            train_dataset=encoded_train_dataset,
            args=training_config,
            data_collator=data_collator
        )

        self.model.config.use_cache = False
        trainer.train()


    def predict(self):
        pass


class Qwen():
    def __init__(self, args):
        self.model, self.tokenizer = utils.get_pretrained_model_and_tokenizer(args["model_name"])
        self.tokenizer.padding_side = "right"


    def preprocess_data(self, args):
        with open(args["general_dataset_save_path"], 'r') as json_file:
            json_list = list(json_file)

        dataset_save_root = args[args["model_name"]]["preprocess_dataset_save_path"].split("/")
        dataset_save_root = dataset_save_root[:-1]
        dataset_save_root = os.path.join(*dataset_save_root)
        if not os.path.exists(dataset_save_root):
            os.makedirs(dataset_save_root)

        processed_data_list = list()
        for json_str in json_list:
            data = json.loads(json_str)
            data.pop(0)
            processed_data_list.append(data)
            
        with open(args[args["model_name"]]["preprocess_dataset_save_path"], "w") as file:
            for i in range(len(processed_data_list)):
                temp = json.dumps(processed_data_list[i])
                file.write(temp)
                if i != (len(processed_data_list)-1):
                    file.write("\n")


    def train(self, args):
        with open(args[args["model_name"]]["preprocess_dataset_save_path"], 'r') as json_file:
            json_list = list(json_file)
        
        encoded_list = list()
        for json_str in json_list:
            data = json.loads(json_str)

            prompt, response = None, None
            for element in data:
                if element["role"] == "user":
                    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>User\n{}<|im_end|>\n".format(element["content"])
                elif element["role"] == "assistant":
                    response = "<|im_start|>assistant\n{}<|im_end|>".format(element["content"])
                else:
                    raise ValueError("Invalid role")
            
            if prompt != None and response != None:
                encoded_prompt = self.tokenizer.encode(prompt, add_special_tokens=False)
                encoded_response = self.tokenizer.encode(response, add_special_tokens=False)

                sample = {
                    "input_ids": encoded_prompt + encoded_response,
                    "attention_mask": [1] * (len(encoded_prompt) + len(encoded_response)),
                    "labels": [-100] * len(encoded_prompt) + encoded_response,
                }

                encoded_list.append(sample)  
        
        encoded_train_dataset = Dataset.from_list(encoded_list)
        encoded_train_dataset = encoded_train_dataset.shuffle(seed=args["seed_index"])
        
        self.model = peft.prepare_model_for_kbit_training(self.model)
        lora_config = peft.LoraConfig(
        r=args["base_lora"]["r"],  # dimension of the updated matrices
        lora_alpha=args["base_lora"]["lora_alpha"],  # parameter for scaling
        lora_dropout=args["base_lora"]["lora_dropout"],  # dropout probability for layers
        bias=args["base_lora"]["bias"],
        task_type=args["base_lora"]["task_type"],
        target_modules=args["base_lora"]["target_modules"],
        )
        self.model = peft.get_peft_model(self.model, lora_config)

        if not os.path.exists(args[args["model_name"]]["output_dir"]):
            os.makedirs(args[args["model_name"]]["output_dir"])
        
        data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        training_config = TrainingArguments(
            output_dir=args[args["model_name"]]["output_dir"],
            per_device_train_batch_size=args[args["model_name"]]["per_device_train_batch_size"],
            optim=args[args["model_name"]]["optim"],
            num_train_epochs=args[args["model_name"]]["num_train_epochs"],
            save_strategy=args[args["model_name"]]["save_strategy"],
            learning_rate=args[args["model_name"]]["learning_rate"],
            lr_scheduler_type=args[args["model_name"]]["lr_scheduler_type"]
        )
        trainer = Trainer(
            model=self.model,
            train_dataset=encoded_train_dataset,
            args=training_config,
            data_collator=data_collator
        )

        self.model.config.use_cache = False
        trainer.train()


    def predict(self):
        pass


class Falcon():
    def __init__(self, args):
        self.model, self.tokenizer = utils.get_pretrained_model_and_tokenizer(args["model_name"])
        self.tokenizer.padding_side = "right"


    def preprocess_data(self, args):
        with open(args["general_dataset_save_path"], 'r') as json_file:
            json_list = list(json_file)

        dataset_save_root = args[args["model_name"]]["preprocess_dataset_save_path"].split("/")
        dataset_save_root = dataset_save_root[:-1]
        dataset_save_root = os.path.join(*dataset_save_root)
        if not os.path.exists(dataset_save_root):
            os.makedirs(dataset_save_root)

        processed_data_list = list()
        for json_str in json_list:
            data = json.loads(json_str)
            data.pop(0)
            processed_data_list.append(data)
            
        with open(args[args["model_name"]]["preprocess_dataset_save_path"], "w") as file:
            for i in range(len(processed_data_list)):
                temp = json.dumps(processed_data_list[i])
                file.write(temp)
                if i != (len(processed_data_list)-1):
                    file.write("\n")


    def train(self, args):
        with open(args[args["model_name"]]["preprocess_dataset_save_path"], 'r') as json_file:
            json_list = list(json_file)
        
        encoded_list = list()
        for json_str in json_list:
            data = json.loads(json_str)

            prompt, response = None, None
            for element in data:
                if element["role"] == "user":
                    prompt = "<|im_start|>user\n{}<|im_end|>\n".format(element["content"])
                elif element["role"] == "assistant":
                    response = "<|im_start|>assistant\n{}<|im_end|>".format(element["content"])
                else:
                    raise ValueError("Invalid role")
            
            if prompt != None and response != None:
                encoded_prompt = self.tokenizer.encode(prompt, add_special_tokens=False)
                encoded_response = self.tokenizer.encode(response, add_special_tokens=False)

                sample = {
                    "input_ids": encoded_prompt + encoded_response,
                    "attention_mask": [1] * (len(encoded_prompt) + len(encoded_response)),
                    "labels": [-100] * len(encoded_prompt) + encoded_response,
                }

                encoded_list.append(sample)  
        
        encoded_train_dataset = Dataset.from_list(encoded_list)
        encoded_train_dataset = encoded_train_dataset.shuffle(seed=args["seed_index"])
        
        self.model = peft.prepare_model_for_kbit_training(self.model)
        lora_config = peft.LoraConfig(
        r=args["base_lora"]["r"],  # dimension of the updated matrices
        lora_alpha=args["base_lora"]["lora_alpha"],  # parameter for scaling
        lora_dropout=args["base_lora"]["lora_dropout"],  # dropout probability for layers
        bias=args["base_lora"]["bias"],
        task_type=args["base_lora"]["task_type"],
        )
        self.model = peft.get_peft_model(self.model, lora_config)

        if not os.path.exists(args[args["model_name"]]["output_dir"]):
            os.makedirs(args[args["model_name"]]["output_dir"])
        
        data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        training_config = TrainingArguments(
            output_dir=args[args["model_name"]]["output_dir"],
            per_device_train_batch_size=args[args["model_name"]]["per_device_train_batch_size"],
            optim=args[args["model_name"]]["optim"],
            num_train_epochs=args[args["model_name"]]["num_train_epochs"],
            save_strategy=args[args["model_name"]]["save_strategy"],
            learning_rate=args[args["model_name"]]["learning_rate"],
            lr_scheduler_type=args[args["model_name"]]["lr_scheduler_type"]
        )
        trainer = Trainer(
            model=self.model,
            train_dataset=encoded_train_dataset,
            args=training_config,
            data_collator=data_collator
        )

        self.model.config.use_cache = False
        trainer.train()


    def predict(self):
        pass


def fine_tune(args):
    dataset = utils.get_dataset(args["dataset_name"], args["dataset_path"])
    dataset = dataset["train"]
    
    dataset_save_root = args["general_dataset_save_path"].split("/")
    dataset_save_root = dataset_save_root[:-1]
    dataset_save_root = os.path.join(*dataset_save_root)
    if not os.path.exists(dataset_save_root):
        os.makedirs(dataset_save_root)
    
    general_data_list = list()
    if args["dataset_name"] == "databricks/databricks-dolly-15k":
        for data in dataset:
            format_data = list()
            format_data.append({"role": "system", "content": ""})
            format_data.append({"role": "user", "content": data["context"] + data["instruction"]})
            format_data.append({"role": "assistant", "content": data["response"]})
            general_data_list.append(format_data)
    
        with open(args["general_dataset_save_path"], "w") as file:
            for i in range(len(general_data_list)):
                temp = json.dumps(general_data_list[i])
                file.write(temp)
                if i != (len(general_data_list)-1):
                    file.write("\n")
    elif args["dataset_name"] == "tatsu-lab/alpaca":
        pass
    elif args["dataset_name"] == "Open-Orca/SlimOrca":
        pass
    elif args["dataset_name"] == "HuggingFaceH4/ultrafeedback_binarized":
        pass
    else:
        raise ValueError("Invalid dataset")
    
    # model = Mistral(args)
    # model = Gemma(args)
    # model = Llama3(args)
    # model = Qwen(args)
    model = Falcon(args)
    
    model.preprocess_data(args)
    model.train(args)


if __name__ == '__main__':
    with open(os.path.join("../setting", "ref_config.yaml"), 'r') as file:
        global_cfg = yaml.safe_load(file)
    fine_tune(global_cfg)