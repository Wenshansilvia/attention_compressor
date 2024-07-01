from src.load_data import get_prompts
from src.generator import longchat_generator
from tqdm import tqdm

def test(data_path, res_file_name):
    nq, prompts  = get_prompts(data_path, ratio = 0.5)
    longchat_model_path = '/home/gomall/models/longchat-13b-16k'
    generator = longchat_generator(longchat_model_path)
    # generate answers
    results = []
    for prompt in tqdm(prompts):
        results.append(generator.chat(prompt))
    nq = nq.add_column('outputs', results)
    #save to json
    nq.to_json(res_file_name)

if __name__ == '__main__':
    for i in [14,19]:
        data_path = '/home/gomall/work/NQ/{num}.json'.format(num=str(i))
        res_file_name = 'test_results/nq_0.5_prompt_{num}.json'.format(num=str(i+1))
        test(data_path, res_file_name)