import matplotlib.pyplot as plt
import numpy as np
import torch
from nltk.tokenize import sent_tokenize
import re

def reconstruct(tokens, attention):
    reconstructed_words = []
    reconstructed_attn = []
    reconstructed_cnt = []

    # 初始化一个变量来保存当前正在构建的单词
    current_word = ""
    current_attn = 0
    current_cnt = 1

    for token, attn in zip(tokens, attention):
        token = token.replace('Ċ', '\n')
        if token.startswith("Ġ") or token.startswith("▁"):
            if current_word:
                reconstructed_words.append(current_word)
                reconstructed_attn.append(current_attn)
                reconstructed_cnt.append(current_cnt)
            current_word = token[1:]
            current_attn = attn
            current_cnt = 1
        else:
            current_word += token
            current_attn = max(current_attn, attn)
            current_cnt += 1
    if current_word:
        reconstructed_words.append(current_word)
        reconstructed_attn.append(current_attn)
        reconstructed_cnt.append(current_cnt)
    return reconstructed_words, reconstructed_attn, reconstructed_cnt

def reconstruct_sentence(tokens, attention):
    reconstructed_sentences = []
    reconstructed_attn = []
    words, attention, _ = reconstruct(tokens, attention)
    current_sentence = ""
    current_attn = 0
    for word, attn in zip(words, attention):
        if current_sentence:
            current_sentence += (' ' + word)
        else:
            current_sentence += word
        current_attn = max(current_attn, attn)
        if word.endswith("."):
            reconstructed_sentences.append(current_sentence)
            reconstructed_attn.append(current_attn)
            current_sentence = ""
            current_attn = 0
    if current_sentence:
        reconstructed_sentences.append(current_sentence)
        reconstructed_attn.append(current_attn)
    return reconstructed_sentences, reconstructed_attn

def reconstruct_paragraph(tokens, attention):
    reconstructed_paragraph = []
    reconstructed_attn = []
    words, attention, _ = reconstruct(tokens, attention)
    current_paragraph = ""
    current_attn = 0
    for word, attn in zip(words, attention):
        if word == 'ĊĊ':
            reconstructed_paragraph.append(current_paragraph)
            reconstructed_attn.append(current_attn)
            current_paragraph = ""
            current_attn = 0
            continue
        if current_paragraph:
            current_paragraph += (' ' + word)
        else:
            current_paragraph += word
        current_attn = max(current_attn, attn)
    if current_paragraph:
        reconstructed_paragraph.append(current_paragraph)
        reconstructed_attn.append(current_attn)
    return reconstructed_paragraph, reconstructed_attn

def plot_line_chart(tokens, scores, title="Line Chart", xlabel="X Axis", ylabel="Y Axis"):
    """
    Plot a line chart with string labels on the x-axis.
    
    Parameters:
        x_labels (list of str): List of strings for the x-axis labels.
        y_values (list of float): List of values for the y-axis.
        title (str): Title of the chart.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    fig, ax = plt.subplots(figsize=(48, 6))
    ax.plot(range(len(tokens)), scores, marker='o')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    
    plt.tight_layout()
    plt.show()

def attn(input_text, model, tokenizer):
    try: 
        inputs = tokenizer(input_text, return_tensors="pt").to('cuda')
    except:
        device = torch.device("mps")
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Forward pass to get attention weights
    outputs = model(**inputs)
    attention = outputs.attentions  # list of (num_layers, batch_size, num_heads, seq_length, seq_length)

    # Extract attention weights from the last layer and convert tokens to their string representation
    attention_weights = attention[-1].squeeze().cpu().detach().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze().tolist())

    # Plot mean attention weights
    mean_attention_weights = np.mean(attention_weights, axis=0)
    del inputs, outputs, attention
    torch.cuda.empty_cache()
    return mean_attention_weights, tokens

def softmax(x, axis=0):
    # 归一化输入，防止溢出
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    # 计算Softmax值
    return e_x / e_x.sum(axis=axis, keepdims=True)

def attention_score(doc, query, model, tokenizer, prefix=14):
    messages = [
        {"role": "user", "content": doc + ' ' + query},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    mean_attention_weights, tokens = attn(text, model, tokenizer)
    doc_len = len(tokenizer(doc, return_tensors="pt")['input_ids'][0])
    return softmax(mean_attention_weights[-1][prefix: prefix+doc_len], axis=0), tokens[prefix: prefix+doc_len]

def full_attention_score(doc, query, model, tokenizer, prefix=14):
    messages = [
        {"role": "user", "content": doc + '\n' + query},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    mean_attention_weights, tokens = attn(text, model, tokenizer)
    len_dict = {'prefix_end': 14}
    len_dict['doc_end'] = len_dict['prefix_end'] + \
        len(tokenizer(doc, return_tensors="pt")['input_ids'][0])
    len_dict['query_end'] = len_dict['doc_end'] + \
        len(tokenizer(query, return_tensors="pt")['input_ids'][0])
    return mean_attention_weights[-1], tokens, len_dict

def get_rank_list(score):
    sorted_index_list = [index for index, _ in sorted(enumerate(score), key=lambda x: x[1])]
    index_mapping = {index: sorted_index_list.index(index) for index in range(len(sorted_index_list))}
    sorted_index_list = [index_mapping[index] for index in range(len(score))]
    return sorted_index_list

def text_filter(text, score, cnt, ratio=0.5):
    rank = get_sorted_ids(score)
    word_filter = [False] * len(text)
    threshold = sum(cnt) * ratio
    current_cnt = 0
    for i in rank:
        current_cnt += cnt[i]
        if current_cnt < threshold:
            word_filter[i] = True
        else:
            break
    res_text = []
    res_score = []
    for w, s, j in zip(text, score, word_filter):
        if j:
            res_text.append(w)
            res_score.append(s)
    return res_text, res_score

def sentence_filter(words, attention, ratio):
    ori_words = words
    budget_length = len(words) * ratio
    text = ' '.join(words)
    sentences = sent_tokenize(text)
    sentence_scores = []
    for sentence in sentences:
        sentence_score = 0
        words_in_sentence = []
        pattern = r'[^\w\s]'
        for i, w in enumerate(words):
            _w = w.split('.')[0] if ('.' in w) else w
            _w = _w.split('!')[0] if ('!' in _w) else _w
            _w = _w.split('?')[0] if ('?' in _w) else _w
            _w = re.sub(pattern, '', _w)
            _w = _w[:-1] if _w.endswith('s') else _w
            if _w in re.sub(pattern, '', sentence):
                sentence_score = max(attention[i], sentence_score)
                sentence.replace(w, '')
                words_in_sentence.append((w, attention[i], 1))
            else:
                words = words[len(words_in_sentence):]
                attention = attention[len(words_in_sentence):]
                break
        sentence_scores.append((sentence, sentence_score, words_in_sentence, len(words_in_sentence)))
    assert(len(sentence_scores) == len(sentences))

    # 选择前k个句子，使得总长度不超过整个段落长度*ratio
    selected_index = []
    current_length = 0
    sorted_indices = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i][1], reverse=True)
    for index in sorted_indices:
        if current_length + sentence_scores[index][-1] > budget_length:
            breakpoint_sentence = sentence_scores[index]
            budget_left = budget_length - current_length
            break
        selected_index.append(index)
        current_length += sentence_scores[index][-1]
    
    selected_index.sort()
    selected_sentences = [sentence_scores[index][0] for index in selected_index]
    try:
        return selected_sentences, breakpoint_sentence, budget_left
    except:
        print(sentence_scores)
        print(words)
        print(ori_words)
        print(sentences)


def _select_tuples(breakpoint_sentence, m):
    tuples_list = breakpoint_sentence[-2] #拿到这个句子中的word list
    # 找到score最大的index
    max_index = max(range(len(tuples_list)), key=lambda i: tuples_list[i][1])
    
    selected_list = [tuples_list[max_index]]
    left, right = max_index - 1, max_index + 1
    
    # 以max_index为中心，扩展selected_list直到其个数达到m
    while sum([s[-1] for s in selected_list]) < m:
        if left >= 0 and right < len(tuples_list):
            selected_list.insert(0, tuples_list[left])
            left -= 1
            selected_list.append(tuples_list[right])
            right += 1
        elif left >= 0:
            selected_list.insert(0, tuples_list[left])
            left -= 1
        elif right < len(tuples_list):
            selected_list.append(tuples_list[right])
            right += 1
        else:
            break  # 防止m大于tuples_list的长度，避免无限循环
    
    # 确保结果中的元素顺序与原列表中的顺序一致
    selected_list.sort(key=lambda x: tuples_list.index(x))
    
    return selected_list


def sentence_token_filter(words, attention, ratio):
    selected_sentences, breakpoint_sentence, budget_left = sentence_filter(words, attention, ratio)
    selected = _select_tuples(breakpoint_sentence, budget_left)
    last_sentence = ' '.join([w[0] for w in selected])
    selected_sentences.append(last_sentence)
    return selected_sentences

