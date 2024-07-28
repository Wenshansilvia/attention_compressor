from llmlingua import PromptCompressor
from src.load_data import load_data
from src.prompts import (nq_prompt)
from src.generator import longchat_generator
from tqdm import tqdm
import torch
import os



def batch_generate(nq, prompts, res_file_name):
    batch_size = 4  # Adjust this based on your GPU memory
    
    longchat_model_path = '/home/gomall/models/longchat-13b-16k'
    generator = longchat_generator(longchat_model_path)
    
    # generate answers
    results = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        try:
            batch = prompts[i:i+batch_size]
        except:
            print('last batch')
            batch = prompts[i:]
        responses = generator.batch_chat(batch)
        results+=responses
        
    nq = nq.add_column('outputs', results)
    #save to json
    nq.to_json(res_file_name)


if __name__ == '__main__':
    for i in [4,9,14,19]:
        print('\n\ni is:', i)
        if not os.path.exists('/home/gomall/work/nq_0.25_prompt_{num}_longllmlingua_2gen.json'.format(num = str(i))):

            model_path = '/home/gomall/models/llama2-7B-chat/'
            llm_lingua = PromptCompressor(model_name=model_path)
            data_path = '/home/gomall/work/NQ/{num}.json'.format(num = str(i))
            nq = load_data(data_path)
        
            def compress(row):
                question = row['question']
                contexts = row['contexts']
                compressed_prompt = llm_lingua.compress_prompt(
                                        contexts,
                                        question=question,
                                        rate=0.25,
                                        # Set the special parameter for LongLLMLingua
                                        condition_in_question="after_condition",
                                        use_context_level_filter=False,
                                        reorder_context="original",
                                        #dynamic_context_compression_ratio=0.3, # or 0.4
                                        condition_compare=True,
                                        context_budget="+100",
                                        rank_method="longllmlingua",
                                        concate_question=False,
                                    )
                return compressed_prompt
            nq = nq.map(lambda x: {"compressed_contexts": compress(x)})  
    
            data_path = '/home/gomall/work/nq_0.25_prompt_{num}_longllmlingua_2gen.json'.format(num = str(i))
            nq.to_json(data_path)
            
            nq = nq.select_columns(['question', 'compressed_contexts', 'answer'])
            nq.to_json(data_path)
    
            import gc
            del nq
            del llm_lingua
            torch.cuda.empty_cache()
            gc.collect()
        else:
            data_path = '/home/gomall/work/nq_0.25_prompt_{num}_longllmlingua_2gen.json'.format(num = str(i))
            nq = load_data(data_path)
            prompts = [
                nq_prompt.format(contexts=data['compressed_contexts'],
                                question=data['question'])
                for data in nq
            ]
            batch_generate(nq, prompts, res_file_name = '/home/gomall/work/test_results/nq_0.25_prompt_{num}_longllmlingua.json'.format(num = str(i)))
    


