from llmlingua import PromptCompressor
from experiments.load_data import load_data
from experiments.prompts import (nq_prompt)
from experiments.generator import longchat_generator
from tqdm import tqdm
import torch
import os


def batch_test_on_nq():
    batch_size = 4  # Adjust this based on your GPU memory
    longchat_model_path = ''
    res_file_name = ''
    compress_model_path = ''
    data_path = ''
    ratio = 0.5

    # compress contexts
    llm_lingua = PromptCompressor(model_name=compress_model_path)
    nq = load_data(data_path)
    def compress(row):
        question = row['question']
        contexts = row['contexts']
        compressed_prompt = llm_lingua.compress_prompt(
                                contexts,
                                question=question,
                                rate=ratio,
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
    nq = nq.select_columns(['question', 'compressed_contexts', 'answer'])
    nq.to_json(res_file_name)
    import gc
    del nq
    del llm_lingua
    torch.cuda.empty_cache()
    gc.collect()
    
    # generate answers
    nq = load_data(res_file_name)
    prompts = [
        nq_prompt.format(contexts=data['compressed_contexts'],
                        question=data['question'])
        for data in nq
    ]
    generator = longchat_generator(longchat_model_path)
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


def batch_test_on_nq():
    batch_size = 4  # Adjust this based on your GPU memory
    longchat_model_path = ''
    res_file_name = ''
    compress_model_path = ''
    data_path = ''
    ratio = 0.5

    # compress contexts
    llm_lingua = PromptCompressor(model_name=compress_model_path)
    asqa = load_data(data_path)
    asqa = asqa.map(lambda example: {"contexts": [s["text"] for s in example["docs"]]})

    def compress(row):
        question = row['question']
        contexts = row['contexts']
        compressed_prompt = llm_lingua.compress_prompt(
                                contexts,
                                question=question,
                                rate=ratio,
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
    asqa = asqa.map(lambda x: {"compressed_contexts": compress(x)})  
    asqa.to_json(res_file_name)
    import gc
    del asqa
    del llm_lingua
    torch.cuda.empty_cache()
    gc.collect()
    
    # generate answers
    asqa = load_data(res_file_name)
    prompts = [
        nasqa_prompt.format(contexts=data['compressed_contexts'],
                        question=data['question'])
        for data in asqa
    ]
    generator = longchat_generator(longchat_model_path)
    results = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        try:
            batch = prompts[i:i+batch_size]
        except:
            print('last batch')
            batch = prompts[i:]
        responses = generator.batch_chat(batch)
        results+=responses
        
    asqa = asqa.add_column('outputs', results)
    #save to json
    asqa.to_json(res_file_name)



if __name__ == '__main__':
    print('Run test')
    #batch_test_on_nq()
    #batch_test_on_asqa() 



