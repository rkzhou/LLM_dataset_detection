import utils


class model_base():
    def __init__(self, args):
        self.args = args
        self.model, self.tokenizer = utils.get_pretrained_model_and_tokenizer(self.args["model_name"])


    def preprocess_prompt(self, raw_prompts):
        pass


    def generate_response(self, prompt_list):
        encoded_inputs = self.tokenizer(prompt_list, padding=True, truncation=True, max_length=512, return_tensors='pt')
        encoded_inputs = {key: value.to("cuda") for key, value in encoded_inputs.items()}
        if self.args["do_sample"] == True:
            generated_ids = self.model.generate(encoded_inputs, max_new_tokens=128, do_sample=True, temperature=self.args["temperature"])
        else:
            generated_ids = self.model.generate(encoded_inputs, max_new_tokens=128)
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return responses


    def pull_answer(self, original_answers, raw_prompt_list=None):
        final_answer_list = list()
        if raw_prompt_list == None:
            for answer in original_answers:
                real_answer = answer.split(self.args["split_symbol"])[-1]
                final_answer_list.append(real_answer)
        else:
            for i in range(len(original_answers)):
                current_question_split_mark = None
                for j in range(len(raw_prompt_list[i])):
                    if raw_prompt_list[i][j]["role"] == "user":
                        current_question_split_mark = raw_prompt_list[i][j]["content"]
                
                real_answer = original_answers[i].split(current_question_split_mark)[-1]
                final_answer_list.append(real_answer)

        return final_answer_list


class Chatmodel_0(model_base):
    '''
    apply chat template directly
    '''
    def preprocess_prompt(self, raw_prompts):
        prompt_list = list()
        for prompt in raw_prompts:
            format_prompt = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
            prompt_list.append(format_prompt)

        return prompt_list

    

class Chatmodel_1(model_base):
    '''
    ------Alpaca format-------
    ### Instruction:
    {instruction}
    ### Response:
    '''
    
    def preprocess_prompt(self, inputs):
        prompt_list = list()
        for input in inputs:
            system_message = input[0]["content"]
            user_prompt = input[1]["content"]
            
            if system_message == "":
                prompt = "### Instruction:\n{}\n### Response:\n".format(user_prompt)
            else:
                prompt = "### Instruction:\n{} {}\n### Response:\n".format(system_message, user_prompt)
            prompt_list.append(prompt)

        return prompt_list



class Chatmodel_2(model_base):
    '''
    <|system|> system message <|user|> user prompt <|model|>
    '''
    
    def preprocess_prompt(self, inputs):
        prompt_list = list()
        for input in inputs:
            system_message = input[0]["content"]
            user_prompt = input[1]["content"]
            
            if system_message == "":
                prompt = "<|user|> {} <|model|>".format(user_prompt)
            else:
                prompt = "<|system|> {} <|user|> {} <|model|>".format(system_message, user_prompt)
            prompt_list.append(prompt)

        
        return prompt_list

    

class Chatmodel_3(model_base):
    '''
    <|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant
    '''

    
    def preprocess_prompt(self, inputs):
        prompt_list = list()
        for input in inputs:
            system_message = input[0]["content"]
            user_prompt = input[1]["content"]
        
            if system_message == "":
                prompt = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant".format(user_prompt)
            else:
                prompt = "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant".format(system_message, user_prompt)
            prompt_list.append(prompt)

        return prompt_list



class Chatmodel_4(model_base):
    '''
    <|prompter|><input></s><|assistant|>

    '''
    
    def preprocess_prompt(self, inputs):    
        prompt_list = list()
        for input in inputs:
            system_message = input[0]["content"]
            user_prompt = input[1]["content"]
        
            if system_message == "":
                prompt = "<|prompter|>{}</s><|assistant|>".format(user_prompt)
            else:
                prompt = "<|prompter|>{} {}</s><|assistant|>".format(system_message, user_prompt)
            prompt_list.append(prompt)
        
        return prompt_list


class Chatmodel_5(model_base):
    '''
    <|user|>
    Your message here!
    <|assistant|>

    '''
    
    def preprocess_prompt(self, inputs):    
        prompt_list = list()
        for input in inputs:
            system_message = input[0]["content"]
            user_prompt = input[1]["content"]

            if system_message == "":
                prompt = "<|user|>\n{}\n<|assistant|>\n".format(user_prompt)
            else:
                prompt = "<|user|>\n{} {}\n<|assistant|>\n".format(system_message, user_prompt)
            prompt_list.append(prompt)
        
        return prompt_list


class Chatmodel_6(model_base):
    '''
    <|prompter|><input><|endoftext|><|assistant|>

    '''
    
    def preprocess_prompt(self, inputs):    
        prompt_list = list()
        for input in inputs:
            system_message = input[0]["content"]
            user_prompt = input[1]["content"]
        
            if system_message == "":
                prompt = "<|prompter|>{}<|endoftext|><|assistant|>".format(user_prompt)
            else:
                prompt = "<|prompter|>{} {}<|endoftext|><|assistant|>".format(system_message, user_prompt)
            prompt_list.append(prompt)

        return prompt_list


class Chatmodel_7(model_base):
    '''
    ### System:
    {system_prompt}
    ### User:
    {user_prompt}
    ### Assistant:
    '''
    
    def preprocess_prompt(self, inputs):
        prompt_list = list()
        for input in inputs:
            system_message = input[0]["content"]
            user_prompt = input[1]["content"]
            
            if system_message == "":
                prompt = "### User:\n{}\n### Assistant:\n".format(user_prompt)
            else:
                prompt = "### System:\n{}\n### User:\n{}\n### Assistant:\n".format(system_message, user_prompt)
            prompt_list.append(prompt)
        
        return prompt_list