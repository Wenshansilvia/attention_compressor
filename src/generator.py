from fastchat.model.model_adapter import load_model, get_conversation_template
import torch
import json
import os
from tqdm import tqdm


class longchat_generator():
    def __init__(self, model):
        # 将模型放入DataParallel中
        model, tokenizer = load_model(model)
        model = torch.nn.DataParallel(model)
        model = model.to('cuda')
        self.model, self.tokenizer = model, tokenizer

    def chat(self, query):
        conv = get_conversation_template('../models/longchat-13b-16k')
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
    
if __name__ == '__main__':
    longchat_model_path = '/home/gomall/models/longchat-13b-16k'
    generator = longchat_generator(longchat_model_path)
    print(torch.cuda.device_count())
    prompts = ['你是一个中文新闻编辑', 'hi', 'hi', 'hi']
    print([generator.chat(prompt) for prompt in prompts])