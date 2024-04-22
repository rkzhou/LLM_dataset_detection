import utils


class model_base():
    def __init__(self, name=None):
        self.name = name
        if self.name == 'None':
            raise ValueError("Must state the name of model")
        self.model, self.tokenizer = utils.get_pretrained_model_and_tokenizer(self.name)

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


class Chatmodel_0(model_base):
    '''
    chat = [
    {"role": "system", "content": ""},
    {"role": "user", "content": ""},
    ]

    apply chat template directly
    '''

    def __init__(self, name=None):
        super().__init__(name)
        
    def preprocess_prompt(self, inputs):
        prompt_list = list()
        for input in inputs:
            prompt = self.tokenizer.apply_chat_template(input, add_generation_prompt=True, tokenize=False)
            prompt_list.append(prompt)

        encoded_inputs = self.tokenizer(prompt_list, padding=True, return_tensors='pt', add_special_tokens=False)
        encoded_inputs = {key: value.to("cuda") for key, value in encoded_inputs.items()}

        return encoded_inputs
    
    def generate_response(self, prompt):
        generated_ids = self.model.generate(**prompt, max_new_tokens=512, do_sample=True, temperature=0.2, top_k=50, top_p=0.95)
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return responses
    

class Chatmodel_1(model_base):
    '''
    chat = [
    {"role": "system", "content": ""},
    {"role": "user", "content": ""},
    ]

    ------Alpaca format-------
    ### Instruction:
    {instruction}

    ### Input:
    {input}

    ### Response:
    '''

    def __init__(self, name=None):
        super().__init__(name)
    
    def preprocess_prompt(self, inputs):
        prompt_prefix0 = "### Instruction:\n"
        prompt_prefix1 = "### Input:\n"
        prompt_prefix2 = "### Response:\n"

        prompt_list = list()
        for input in inputs:
            system_message, user_prompt = "", ""
            for i in range(len(input)):
                if input[i]["role"] == "system":
                    system_message = input[i]["content"]
                elif input[i]["role"] == "user":
                    user_prompt = input[i]["content"]
            
            prompt = "{}{}\n\n{}{}\n\n{}\n".format(prompt_prefix0, user_prompt, prompt_prefix1, system_message, prompt_prefix2)
            prompt_list.append(prompt)

        encoded_inputs = self.tokenizer(prompt_list, padding=True, return_tensors='pt', add_special_tokens=False)
        encoded_inputs = {key: value.to("cuda") for key, value in encoded_inputs.items()}
        
        return encoded_inputs
    
    def generate_response(self, prompt):
        generated_ids = self.model.generate(**prompt, max_new_tokens=512, do_sample=True, temperature=0.2, top_k=50, top_p=0.95)
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return responses


class Chatmodel_2(model_base):
    '''
    <|system|> system message <|user|> user prompt <|model|>
    '''

    def __init__(self, name=None):
        super().__init__(name)
    
    def preprocess_prompt(self, inputs):
        prompt_list = list()
        for input in inputs:
            system_message, user_prompt = "", ""
            for i in range(len(input)):
                if input[i]["role"] == "system":
                    system_message = input[i]["content"]
                elif input[i]["role"] == "user":
                    user_prompt = input[i]["content"]
            
            prompt = "<|system|> {} <|user|> {} <|model|>".format(system_message, user_prompt)
            prompt_list.append(prompt)

        
        encoded_inputs = self.tokenizer(prompt_list, padding=True, return_tensors='pt', add_special_tokens=False)
        encoded_inputs = {key: value.to("cuda") for key, value in encoded_inputs.items()}

        return encoded_inputs

    def generate_response(self, prompt):
        generated_ids = self.model.generate(**prompt, max_new_tokens=512, do_sample=True, temperature=0.2, top_k=50, top_p=0.95)
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return responses
    

class Chatmodel_3(model_base):
    '''
    <|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant
    '''

    def __init__(self, name=None):
        super().__init__(name)
    
    def preprocess_prompt(self, input):
        system_message, user_prompt = "", ""
        for i in range(len(input)):
            if input[i]["role"] == "system":
                system_message = input[i]["content"]
            elif input[i]["role"] == "user":
                user_prompt = input[i]["content"]
        
        prompt = f"""<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant"""
        prompt = self.tokenizer(prompt, return_tensors="pt").input_ids
        prompt = prompt.to("cuda")

        return prompt

    def generate_response(self, prompt):
        generated_ids = self.model.generate(prompt, max_new_tokens=512, do_sample=True, temperature=0.2, top_k=50, top_p=0.95)
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        best_response = responses[0]

        return best_response


class Chatmodel_4(model_base):
    '''
    <|prompter|><input></s><|assistant|>

    '''

    def __init__(self, name=None):
        super().__init__(name)
    
    def preprocess_prompt(self, inputs):    
        prompt_list = list()
        for input in inputs:
            system_message, user_prompt = "", ""
            for i in range(len(input)):
                if input[i]["role"] == "system":
                    system_message = input[i]["content"]
                elif input[i]["role"] == "user":
                    user_prompt = input[i]["content"]
        
            prompt = f"""<|prompter|>{system_message} {user_prompt}</s><|assistant|>"""
            prompt_list.append(prompt)
        
        encoded_inputs = self.tokenizer(prompt_list, padding=True, return_tensors='pt', add_special_tokens=False)
        encoded_inputs = {key: value.to("cuda") for key, value in encoded_inputs.items()}

        return encoded_inputs

    def generate_response(self, prompt):
        generated_ids = self.model.generate(**prompt, max_new_tokens=512, do_sample=True, temperature=0.2, top_k=50, top_p=0.95)
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return responses