import json
from tqdm import tqdm
import unicodedata
import re
import os


def normalize_text(text):
    # 用问号替换所有非ASCII字符
    text = re.sub(r'[^\x00-\x7F]', '?', text)
    # 进行Unicode规范化并将文本转为小写
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8').lower()
    return text

def calculate_sample_score(sample):
    answer_list = sample['answer']
    response = sample['outputs']
    
    # 归一化响应文本
    normalized_response = normalize_text(response)
    
    for answer in answer_list:
        # 归一化答案文本
        normalized_answer = normalize_text(answer)
        if normalized_answer in normalized_response:
            return 1
    return 0


if __name__ == '__main__':
    file_name = ''
    data = []
    with open(file_name) as f:
        for line in tqdm(f.readlines()):
            try:
                data.append(json.loads(line))
            except:
                pass

    i = 0
    for d in data:
        i += calculate_sample_score(d)

    print(i / len(data))



    