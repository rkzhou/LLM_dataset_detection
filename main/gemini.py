import os
import yaml
import google.generativeai as genai

def generate_answers(args):
    genai.configure(api_key=args["gemini-1.5-flash"]["api_key"],)

    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Write a story about a magic backpack.",
        generation_config=genai.types.GenerationConfig(
        # Only one candidate for now.
        candidate_count=1,
        max_output_tokens=10,
        temperature=1.0))
    print(response.text)



if __name__ == '__main__':
    with open(os.path.join("../setting", "com_llm.yaml"), 'r') as file:
        global_cfg = yaml.safe_load(file)

    generate_answers(global_cfg)