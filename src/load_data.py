from datasets import load_dataset
from src.prompts import (nq_prompt)
from src.compressor import Compressor

def load_data(path):
    print('start load dataset:', path)
    data = load_dataset('json', data_files=path)['train'] 
    print('dataset loaded')
    return data

def get_prompts(data_path, ratio = 1):
    nq = load_data(data_path)
    if ratio == 1:
        prompts = [
            nq_prompt.format(contexts=''.join(data['contexts']),
                            question=data['question'])
            for data in nq
        ]
    else:
        model_path = '/home/gomall/models/Qwen2-0.5B-Instruct'
        compressor = Compressor(model_path)
        def compress(row):
            query = row['question']
            docs = row['contexts']
            ctxs = '\n'.join([compressor.compress(doc, query, ratio=0.5) for doc in docs])
            return ctxs
        nq = nq.map(lambda x: {"compressed_contexts": compress(x)})  
        prompts = [
            nq_prompt.format(contexts=data['compressed_contexts'],
                            question=data['question'])
            for data in nq
        ]
        print(prompts)
    return nq, prompts


if __name__ == '__main__':
    
    path = '/Users/wenshan/Desktop/ccir/注意力压缩/NQ/0.json'
    nq, prompts = get_prompts(path)
