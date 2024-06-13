import os
import yaml
import pickle
import utils
import json
import peft
import transformers
import trl
import datasets


class reference_model_base():
    def __init__(self, args):
        self.model_name = args["model_name"]
        self.model, self.tokenizer = utils.get_pretrained_model_and_tokenizer(self.model_name)
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1


    # this function is used to output the right formate for each row in the dataset
    def create_text_row(self, system_prompt, user_prompt, assistant_response):
        pass


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

        lora_config = peft.LoraConfig(
        r=args[self.model_name]["r"],
        lora_alpha=args[self.model_name]["lora_alpha"],
        lora_dropout=args[self.model_name]["lora_dropout"],
        bias=args[self.model_name]["bias"],
        task_type=args[self.model_name]["task_type"],
        )
        if self.model_name != "tiiuae/falcon-7b-instruct":
            lora_config.target_modules=args[self.model_name]["target_modules"]

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
            weight_decay = args[self.model_name]["weight_decay"],
            warmup_ratio = args[self.model_name]["warmup_ratio"],
            group_by_length = args[self.model_name]["group_by_length"],
        )
        trainer = trl.SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            peft_config=lora_config,
            dataset_text_field="text",
            tokenizer=self.tokenizer,
            args=training_config,
            packing=args[self.model_name]["packing"],
        )


        trainer.train()
        final_model_save_path = args[self.model_name]["output_dir"] + "final_model"
        trainer.model.save_pretrained(final_model_save_path)
    

    def predict(self):
        pass


class Mistral(reference_model_base):
    # Mistral doesn't support system role
    def create_text_row(self, system, instruction, response):
        if system == "":
            text_row = "<s>[INST] {} [/INST]\n{}</s>".format(instruction, response)
        else:
            text_row = "<s>[INST] {} {} [/INST]\n{}</s>".format(system, instruction, response)
        
        return text_row


class Gemma(reference_model_base):
    # Gemma doesn't support system role
    def create_text_row(self, system, instruction, response):
        if system == "":
            text_row = "<bos><start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n{}<end_of_turn>".format(instruction, response)
        else:
            text_row = "<bos><start_of_turn>user\n{} {}<end_of_turn>\n<start_of_turn>model\n{}<end_of_turn>".format(system, instruction, response)
        
        return text_row


class Llama3(reference_model_base):
    def create_text_row(self, system, instruction, response):
        if system == "":
            text_row = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{}<|eot_id|>".format(instruction, response)
        else:
            text_row = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{}<|eot_id|>".format(system, instruction, response)
        
        return text_row


class Qwen(reference_model_base):
    def create_text_row(self, system, instruction, response):
        if system == "":
            text_row = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}<|im_end|>".format(instruction, response)
        else:
            text_row = "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}<|im_end|>".format(system, instruction, response)
        
        return text_row


class Falcon(reference_model_base):
    def create_text_row(self, system, instruction, response):
        if system == "":
            text_row = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}<|im_end|>".format(instruction, response)
        else:
            text_row = "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}<|im_end|>".format(system, instruction, response)
        
        return text_row


def format_dataset(data):
    if data["context"] == "":
        item = {"system": "", "instruction": data["instruction"], "response": data["response"]}
    else:
        item = {"system": "", "instruction": data["context"] + " " + data["instruction"], "response": data["response"]}
    
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
        dataset = dataset.map(format_dataset)
        
        with open(args["general_dataset_save_path"], "wb") as file:
            pickle.dump(dataset, file)

    elif args["dataset_name"] == "tatsu-lab/alpaca":
        pass
    elif args["dataset_name"] == "Open-Orca/SlimOrca":
        pass
    elif args["dataset_name"] == "HuggingFaceH4/ultrafeedback_binarized":
        pass
    else:
        raise ValueError("Invalid dataset")
    
    if args["model_name"] == "mistralai/Mistral-7B-Instruct-v0.2":
        model = Mistral(args)
    elif args["model_name"] == "google/gemma-7b-it":
        model = Gemma(args)
    elif args["model_name"] == "meta-llama/Meta-Llama-3-8B-Instruct":
        model = Llama3(args)
    elif args["model_name"] == "Qwen/Qwen2-7B-Instruct":
        model = Qwen(args)
    elif args["model_name"] == "tiiuae/falcon-7b-instruct":
        model = Falcon(args)
    else:
        raise ValueError("Invalid model name")
    
    model.preprocess_data(args)
    model.train(args)


if __name__ == '__main__':
    with open(os.path.join("../setting", "ref_config.yaml"), 'r') as file:
        global_cfg = yaml.safe_load(file)
    fine_tune(global_cfg)