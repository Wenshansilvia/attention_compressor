from experiments.load_data import get_prompts, get_prompts_asqa
from experiments.generator import longchat_generator
from tqdm import tqdm
from datasets import load_dataset


def batch_test_on_nq(batch_size, model_path, data_path, res_file_name, longchat_model_path, ratio, filter):

    nq, prompts  = get_prompts(data_path, ratio = ratio, filter = filter, model_path = model_path)
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


def batch_test_on_asqa(batch_size, model_path, data_path, res_file_name, longchat_model_path, ratio, filter):

    asqa, prompts  = get_prompts_asqa(data_path, ratio = ratio, filter = filter, model_path = model_path)
    asqa = asqa.add_column('prompts', prompts)
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
        
    asqa = asqa.add_column('outputs', results)
    #save to json
    asqa.to_json(res_file_name)



if __name__ == '__main__':
    print('Run test')
    #batch_test_on_nq()
    #batch_test_on_asqa()

