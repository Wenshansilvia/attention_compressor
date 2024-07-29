from datasets import load_dataset
from experiments.prompts import (nq_prompt, asqa_prompt)
from quito.compressor import Compressor


def load_data(path):
    print('start load dataset:', path)
    data = load_dataset('json', data_files=path)['train'] 
    print('dataset loaded')
    return data

def get_prompts(data_path, ratio, filter, model_path = 'Qwen/Qwen2-0.5B-Instruct'):
    nq = load_data(data_path)

    if ratio == 1:
        prompts = [
            nq_prompt.format(contexts='\n\n'.join(data['contexts']),
                            question=data['question'])
            for data in nq
        ]
    else:
        compressor = Compressor(model_path)
        def compress(row):
            query = row['question']
            docs = row['contexts']
            if not filter:
                ctxs = '\n\n'.join([compressor.compress(doc, query, ratio) for doc in docs])
            elif filter == 'sentence':
                ctxs = '\n\n'.join([compressor.compress_sentence(doc, query, ratio) for doc in docs])
            elif filter == 'sentence_token':
                ctxs = '\n\n'.join([compressor.compress_sentence_token(doc, query, ratio) for doc in docs])
            return ctxs
        nq = nq.map(lambda x: {"compressed_contexts": compress(x)})  
        prompts = [
            nq_prompt.format(contexts=data['compressed_contexts'],
                            question=data['question'])
            for data in nq
        ]
    return nq, prompts


def get_prompts_asqa(data_path, ratio, filter, model_path = 'Qwen/Qwen2-0.5B-Instruct'):
    asqa = load_data(data_path)
    asqa = asqa.map(lambda example: {"contexts": [s["text"] for s in example["docs"]]})
    
    if ratio == 1:
        prompts = [
            asqa_prompt.format(contexts='\n\n'.join(data['contexts']),
                            question=data['question'])
            for data in asqa
        ]
    else:
        compressor = Compressor(model_path)
        def compress(row):
            query = row['question']
            docs = row['contexts']
            if not filter:
                ctxs = '\n\n'.join([compressor.compress(doc, query, ratio) for doc in docs])
            elif filter == 'sentence':
                ctxs = '\n\n'.join([compressor.compress_sentence(doc, query, ratio) for doc in docs])
            elif filter == 'sentence_token':
                ctxs = '\n\n'.join([compressor.compress_sentence_token(doc, query, ratio) for doc in docs])
            return ctxs
        asqa = asqa.map(lambda x: {"compressed_contexts": compress(x)})  
        prompts = [
            asqa_prompt.format(contexts=data['compressed_contexts'],
                            question=data['question'])
            for data in asqa
        ]
    return asqa, prompts
