import matplotlib.pyplot as plt
import numpy as np
import torch


def reconstruct(tokens, attention):
    reconstructed_words = []
    reconstructed_attn = []

    # 初始化一个变量来保存当前正在构建的单词
    current_word = ""
    current_attn = 0

    for token, attn in zip(tokens, attention):
        token = token.replace('Ċ', '\n')
        if token.startswith("Ġ"):
            if current_word:
                reconstructed_words.append(current_word)
                reconstructed_attn.append(current_attn)
            current_word = token[1:]
            current_attn = attn
        else:
            current_word += token
            current_attn = max(current_attn, attn)
    if current_word:
        reconstructed_words.append(current_word)
        reconstructed_attn.append(current_attn)
    return reconstructed_words, reconstructed_attn

def reconstruct_sentence(tokens, attention):
    reconstructed_sentences = []
    reconstructed_attn = []
    words, attention = reconstruct(tokens, attention)
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
    words, attention = reconstruct(tokens, attention)
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
    inputs = tokenizer(input_text, return_tensors="pt").to('cuda')

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

def get_rank_list(score):
    sorted_index_list = [index for index, _ in sorted(enumerate(score), key=lambda x: x[1])]
    index_mapping = {index: sorted_index_list.index(index) for index in range(len(sorted_index_list))}
    sorted_index_list = [index_mapping[index] for index in range(len(score))]
    return sorted_index_list

def text_filter(text, score, ratio=0.5):
    rank = get_rank_list(score)
    threshold = len(text) * ratio
    res_text = []
    res_score = []
    for i in range(len(text)):
        if rank[i] > threshold:
            res_text.append(text[i])
            res_score.append(score[i])
    return res_text, res_score