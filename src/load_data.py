from datasets import load_dataset
from src.prompts import (nq_prompt)

def load_data(path):
    print('start load dataset:', path)
    data = load_dataset('json', data_files=path)['train'] 
    print('dataset loaded')
    return data

def get_prompts(data_path):
    nq = load_data(data_path)
    prompts = [
        nq_prompt.format(contexts=''.join(data['contexts']),
                           question=data['question'])
        for data in nq
    ]
    return nq, prompts


if __name__ == '__main__':
    
    path = '/Users/wenshan/Desktop/ccir/注意力压缩/NQ/0.json'
    nq, prompts = get_prompts(path)
