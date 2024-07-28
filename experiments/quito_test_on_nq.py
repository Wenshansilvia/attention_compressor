from experiments.load_data import get_prompts
from experiments.generator import longchat_generator
from tqdm import tqdm
from datasets import load_dataset


def batch_test(data_path, res_file_name):
    batch_size = 4  # Adjust this based on your GPU memory
    model_path = ''
    data_path = ''
    res_file_name = ''
    longchat_model_path = ''

    nq, prompts  = get_prompts(data_path, ratio = 0.25, filter = 'sentence_token', model_path = model_path)
    nq = nq.add_column('prompts', prompts)
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
    for i in [0,4,9,14,19]:
        data_path = '/home/gomall/work/NQ/{num}.json'.format(num = str(i))
        res_file_name = '/home/gomall/work/test_results/nq_0.25_prompt_{num}_sentence_token.json'.format(num=str(i+1))        
        #res_file_name = 'test_results/nq_0.5_prompt_{num}.json'.format(num=str(i+1))
        #batch_generate('/home/gomall/work/tmp.json', res_file_name)
        batch_test(data_path, res_file_name)




