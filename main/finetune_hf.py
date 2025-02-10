import os
import torch
import pickle
import yaml
import utils
import copy
import peft
import transformers
import math

from tqdm import tqdm
from datasets import Dataset


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


def model_train(args):
    with open(args["general_dataset_path"], "rb") as file:
        dataset = pickle.load(file)
    os.makedirs(os.path.dirname(args["model_output_dir"]), exist_ok=True)

    model, tokenizer = utils.get_pretrained_model_and_tokenizer(args["model_name"])

    raw_prompts = list()
    for element in dataset:
        raw_prompt = [
            {"role": "system", "content": element["system"]},
            {"role": "user", "content": element["instruction"]},
            {"role": "assistant", "content": element["response"]},
        ]
        raw_prompts.append(raw_prompt)
    
    encoded_inputs = preprocess_prompt(args, tokenizer, raw_prompts, stage="train")

    modules = find_all_linear_names(model)
    model = peft.prepare_model_for_kbit_training(model)
    lora_config = peft.LoraConfig(
    r=args["r"],
    lora_alpha=args["lora_alpha"],
    lora_dropout=args["lora_dropout"],
    bias=args["bias"],
    task_type=args["task_type"],
    target_modules = modules,
    )
    model = peft.get_peft_model(model, lora_config)
    model.config.use_cache = False
    
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
        model=model,
        train_dataset=encoded_inputs,
        args=training_config,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer),
    )

    trainer.train(resume_from_checkpoint=(args["continue_train"]==True))
    final_model_path = os.path.join(args["model_output_dir"], args["model_checkpoint"])
    trainer.save_model(final_model_path)


def model_predict(args, over_write=False):
    final_model_path = os.path.join(args["model_output_dir"], args["model_checkpoint"])
    model, tokenizer = utils.get_pretrained_model_and_tokenizer(args["model_name"])
    model = peft.PeftModel.from_pretrained(model, final_model_path)

    with open(args["general_dataset_path"], "rb") as file:
        dataset = pickle.load(file)
    
    if args["selected_index_path"] != None:
        with open(args["selected_index_path"], "rb") as file:
            tainted_index = pickle.load(file)
        dataset = dataset.filter(lambda example: example['index'] in tainted_index)

    raw_prompts = list()
    for element in dataset:
        raw_prompt = [
            {"role": "system", "content": element["system"]},
            {"role": "user", "content": element["instruction"]},
            {"role": "assistant", "content": element["response"]},
        ]
        raw_prompts.append(raw_prompt)
    encoded_inputs = preprocess_prompt(args, tokenizer, raw_prompts, stage="test")

    data_group_num = math.ceil(len(dataset) / args["inference_batch_size"])
    os.makedirs(args["prediction_dir"], exist_ok=True)

    for group_index in tqdm(range(data_group_num)):
        begin_index = group_index * args["inference_batch_size"]
        end_index = min(len(dataset), (group_index + 1) * args["inference_batch_size"])
        # store its true index inside original dataset
        query_index_list = list(i for i in range(begin_index, end_index))

        exist_num = 0
        if over_write == False:
            for data_index in query_index_list:
                answer_exist_times = 0
                for time_index in range(args["inference_times"]):
                    if os.path.exists("{}/answer_{}_{}.pkl".format(args["prediction_dir"], dataset[data_index]["index"], time_index)):
                        answer_exist_times += 1
                if answer_exist_times == args["inference_times"]:
                    exist_num += 1
            
            if exist_num == len(query_index_list):
                continue
        
        input_ids = torch.tensor(encoded_inputs[query_index_list]["input_ids"]).to(model.device)
        attention_mask = torch.tensor(encoded_inputs[query_index_list]["attention_mask"]).to(model.device)

        for i in range(args["inference_times"]):
            output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=512, 
                                    do_sample=(args["do_sample"] == True), temperature=args["temperature"], use_cache=True, pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.batch_decode(output[:, input_ids.shape[-1]:], skip_special_tokens=True)

            for j in range(len(query_index_list)):
                    with open("{}/answer_{}_{}.pkl".format(args["prediction_dir"], dataset[query_index_list[j]]["index"], i), "wb") as file:
                        pickle.dump(response[j], file)




def preprocess_prompt(args, tokenizer, raw_prompts, stage):
    input_ids_list = list()
    attention_mask_list = list()
    labels_list = list()

    if args["model_template"] == 0:
        for prompt in raw_prompts:
            whole_prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt=False, tokenize=False)
            prompt.pop()
            input_string = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)

            encoded_prompt = tokenizer(whole_prompt, truncation=True, max_length=512)
            encoded_string = tokenizer(input_string, truncation=True, max_length=512)

            if stage == "train":
                pad_length = 512 - len(encoded_prompt.input_ids)
                input_ids = encoded_prompt['input_ids'] + [tokenizer.eos_token_id] * pad_length
                attention_mask = encoded_prompt['attention_mask'] + [0] * pad_length
                
                if len(encoded_prompt["input_ids"]) == 512:
                    labels = encoded_prompt["input_ids"]
                else:
                    labels = encoded_prompt["input_ids"] + [tokenizer.eos_token_id] + [-100] * (pad_length - 1)
                for i in range(len(encoded_string["input_ids"])):
                    labels[i] = -100

                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                labels_list.append(labels)
            elif stage == "test":
                pad_length = 512 - len(encoded_string.input_ids)
                input_ids = [tokenizer.eos_token_id] * pad_length + encoded_string['input_ids']
                attention_mask = [0] * pad_length + encoded_string["attention_mask"]

                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
    else:
        for input in raw_prompts:
            system_message = input[0]["content"]
            user_prompt = input[1]["content"]
            benchmark_response = input[2]["content"]
            if args["model_template"] == 1:
                if system_message == "":
                    whole_prompt = "### Instruction:\n{}\n### Response:\n{}".format(user_prompt, benchmark_response)
                    input_string = "### Instruction:\n{}\n### Response:\n".format(user_prompt)
                else:
                    whole_prompt = "### Instruction:\n{} {}\n### Response:\n{}".format(system_message, user_prompt, benchmark_response)
                    input_string = "### Instruction:\n{} {}\n### Response:\n".format(system_message, user_prompt)
        
            elif args["model_template"] == 2:
                if system_message == "":
                    whole_prompt = "<|user|> {} <|model|> {}".format(user_prompt, benchmark_response)
                    input_string = "<|user|> {} <|model|>".format(user_prompt)
                else:
                    whole_prompt = "<|system|> {} <|user|> {} <|model|> {}".format(system_message, user_prompt, benchmark_response)
                    input_string = "<|system|> {} <|user|> {} <|model|>".format(system_message, user_prompt)
        
            elif args["model_template"] == 3:
                if system_message == "":
                    whole_prompt = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}<|im_end|>\n".format(user_prompt, benchmark_response)
                    input_string = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant".format(user_prompt)
                else:
                    whole_prompt = "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}<|im_end|>\n".format(
                        system_message, user_prompt, benchmark_response)
                    input_string = "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant".format(system_message, user_prompt)
        
            elif args["model_template"] == 4:
                if system_message == "":
                    whole_prompt = "<|prompter|>{}</s><|assistant|>{}</s>".format(user_prompt, benchmark_response)
                    input_string = "<|prompter|>{}</s><|assistant|>".format(user_prompt)
                else:
                    whole_prompt = "<|prompter|>{} {}</s><|assistant|>{}</s>".format(system_message, user_prompt, benchmark_response)
                    input_string = "<|prompter|>{} {}</s><|assistant|>".format(system_message, user_prompt)
        
            elif args["model_template"] == 5:
                if system_message == "":
                    whole_prompt = "<|user|>\n{}\n<|assistant|>\n{}\n".format(user_prompt, benchmark_response)
                    input_string = "<|user|>\n{}\n<|assistant|>\n".format(user_prompt)
                else:
                    whole_prompt = "<|user|>\n{} {}\n<|assistant|>\n{}\n".format(system_message, user_prompt, benchmark_response)
                    input_string = "<|user|>\n{} {}\n<|assistant|>\n".format(system_message, user_prompt)
        
            elif args["model_template"] == 6:
                if system_message == "":
                    whole_prompt = "<|prompter|>{}<|endoftext|><|assistant|>{}<|endoftext|>".format(user_prompt, benchmark_response)
                    input_string = "<|prompter|>{}<|endoftext|><|assistant|>".format(user_prompt)
                else:
                    whole_prompt = "<|prompter|>{} {}<|endoftext|><|assistant|>{}<|endoftext|>".format(system_message, user_prompt, benchmark_response)
                    input_string = "<|prompter|>{} {}<|endoftext|><|assistant|>".format(system_message, user_prompt)
        
            elif args["model_template"] == 7:
                if system_message == "":
                    whole_prompt = "### User:\n{}\n### Assistant:\n{}\n".format(user_prompt, benchmark_response)
                    input_string = "### User:\n{}\n### Assistant:\n".format(user_prompt)
                else:
                    whole_prompt = "### System:\n{}\n### User:\n{}\n### Assistant:\n{}\n".format(system_message, user_prompt, benchmark_response)
                    input_string = "### System:\n{}\n### User:\n{}\n### Assistant:\n".format(system_message, user_prompt)

            encoded_prompt = tokenizer(whole_prompt, truncation=True, max_length=512)
            encoded_string = tokenizer(input_string, truncation=True, max_length=512)

            if stage == "train":
                pad_length = 512 - len(encoded_prompt.input_ids)
                input_ids = encoded_prompt['input_ids'] + [tokenizer.eos_token_id] * pad_length
                attention_mask = encoded_prompt['attention_mask'] + [0] * pad_length

                if len(encoded_prompt["input_ids"]) == 512:
                    labels = encoded_prompt["input_ids"]
                else:
                    labels = encoded_prompt["input_ids"] + [tokenizer.eos_token_id] + [-100] * (pad_length - 1)
                for i in range(len(encoded_string["input_ids"])):
                    labels[i] = -100

                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                labels_list.append(labels)
            elif stage == "test":
                pad_length = 512 - len(encoded_string.input_ids)
                input_ids = [tokenizer.eos_token_id] * pad_length + encoded_string['input_ids']
                attention_mask = [0] * pad_length + encoded_string["attention_mask"]

                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
    
    if stage == "train":
        encoded_inputs = Dataset.from_dict({
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        })
    elif stage == "test":
        encoded_inputs = Dataset.from_dict({
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
        })

    return encoded_inputs


if __name__ == '__main__':
    with open(os.path.join("../setting", "hf_config.yaml"), 'r') as file:
        global_cfg = yaml.safe_load(file)
    global_cfg = {
        key: (value.format(dataset_alias=global_cfg["dataset_alias"], model_name=global_cfg["model_name"]) if isinstance(value, str) else value)
        for key, value in global_cfg.items()
    }

    if global_cfg["model_action"] == "train":
        model_train(global_cfg)
    elif global_cfg["model_action"] == "predict":
        model_predict(global_cfg)