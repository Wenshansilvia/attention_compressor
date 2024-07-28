from fastchat.model.model_adapter import load_model, get_conversation_template
import torch
import json
import os
from tqdm import tqdm


class longchat_generator():
    def __init__(self, model):
        self.model_path = model
        model, tokenizer = load_model(model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side='left'
        self.model, self.tokenizer = model, tokenizer

    def chat(self, query):
        conv = get_conversation_template(self.model_path)
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = self.tokenizer([prompt]).input_ids

        output_ids = self.model.module.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            max_new_tokens=512,
        )
        if self.model.module.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(input_ids[0]) :]
        outputs = self.tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        ) 
        return outputs

    def batch_chat(self, querys):
        prompts = []
        for query in querys:
            conv = get_conversation_template(self.model_path)
            conv.append_message(conv.roles[0], query)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            prompts.append(prompt)
            
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                do_sample=True,
                max_new_tokens=512,
            )
        responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
        for i, response in enumerate(responses):
            input_length = len(self.tokenizer.decode(inputs.input_ids[i], skip_special_tokens=True))
            responses[i] = response[input_length:].strip()

        return responses
        
        
    
if __name__ == '__main__':
    longchat_model_path = ''
    generator = longchat_generator(longchat_model_path)
    print(torch.cuda.device_count())
    prompts = ['hello', 'hello world']
    print([generator.chat(prompt) for prompt in prompts])