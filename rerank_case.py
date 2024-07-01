from src.load_data import load_data
from src.compressor import Compressor
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def filter_case(data, num_of_case):
    data = data.select(range(num_of_case))
    return data['question'], [c[0] for c in data['contexts']], [c[3] for c in data['contexts']]

def plot_line_chart(tokens, scores, line_loc, title="Line Chart", xlabel="X Axis", ylabel="Y Axis"):
    fig, ax = plt.subplots(figsize=(60, 6))
    ax.plot(range(len(tokens)), scores, marker='o')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=4)

    # 在 x 轴位置添加竖线
    plt.axvline(x=line_loc['prefix_end'], color='r', linestyle='--', linewidth=1)
    plt.axvline(x=line_loc['doc_end'], color='g', linestyle='--', linewidth=1)
    plt.axvline(x=line_loc['query_end'], color='b', linestyle='--', linewidth=1)

    # 制定y轴统一长度，并添加每段的平均值横线
    avg1 = np.mean(scores[:line_loc['prefix_end']])
    avg2 = np.mean(scores[line_loc['prefix_end']+1:line_loc['doc_end']])
    avg3 = np.mean(scores[line_loc['doc_end']+1:line_loc['query_end']])
    avg4 = np.mean(scores[line_loc['query_end']+1:])
    plt.ylim(0, 0.07)
    # Add horizontal lines for the mean values
    ax.hlines(y=avg1, xmin=0, xmax=line_loc['prefix_end'], colors='r', linestyles='-')
    ax.hlines(y=avg2, xmin=line_loc['prefix_end']+1, xmax=line_loc['doc_end'], colors='g', linestyles='-')
    ax.hlines(y=avg3, xmin=line_loc['doc_end']+1, xmax=line_loc['query_end'], colors='b', linestyles='-')
    ax.hlines(y=avg4, xmin=line_loc['query_end']+1, xmax=len(tokens)-1, colors='purple', linestyles='-')
    
    plt.tight_layout()
    return plt

def concat_png(png_name_list, res_path):
    # 打开所有图片
    images = [Image.open(img) for img in png_name_list]

    # 获取每张图片的宽度和高度
    widths, heights = zip(*(i.size for i in images))

    # 总宽度是所有图片的最大宽度，总高度是所有图片高度的总和
    total_width = max(widths)
    total_height = sum(heights)

    # 创建一个新的空白图像（白色背景）
    combined_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    # 将每张图片粘贴到新图像上
    y_offset = 0
    for img in images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height

    # 保存组合后的图片
    combined_image.save(res_path)
    return

if __name__ == '__main__':
    path = '/Users/wenshan/Desktop/ccir/注意力压缩/NQ/0.json'
    nq = load_data(path)
    num_of_case = 10
    qsts, re_ctxs, irr_ctxs = filter_case(nq, num_of_case)
    print(len(qsts), len(re_ctxs), len(re_ctxs))

    model_path = '/Users/wenshan/Desktop/ccir/注意力压缩/Qwen2-0.5B-Instruct'
    compressor = Compressor(model_path, device_map = None)
 
    # plot relevant contexts
    i = 1
    for doc, query in zip(re_ctxs, qsts):
        attention, tokens, len_dict = compressor.get_attention(doc, query)
        fig = plot_line_chart(tokens[:-2], attention[:-2], len_dict)
        fig_name = '/Users/wenshan/Desktop/ccir/注意力压缩/figs/re_{num}.png'.format(num = i)
        fig.savefig(fig_name, bbox_inches='tight')  # 保存为 PNG 格式
        i+=1
    
    # plot irrelevant contexts
    i = 1
    for doc, query in zip(irr_ctxs, qsts):
        attention, tokens, len_dict = compressor.get_attention(doc, query)
        fig = plot_line_chart(tokens[:-2], attention[:-2], len_dict)
        fig_name = '/Users/wenshan/Desktop/ccir/注意力压缩/figs/irr_{num}.png'.format(num = i)
        fig.savefig(fig_name, bbox_inches='tight')  # 保存为 PNG 格式
        i+=1

    png_name_list = ['/Users/wenshan/Desktop/ccir/注意力压缩/figs/re_{num}.png'.format(num = i)\
                     for i in range(1,num_of_case+1)]
    res_path = '/Users/wenshan/Desktop/ccir/注意力压缩/figs/combined_re_fig.png'
    concat_png(png_name_list, res_path)

    png_name_list = ['/Users/wenshan/Desktop/ccir/注意力压缩/figs/irr_{num}.png'.format(num = i)\
                     for i in range(1,num_of_case+1)]
    res_path = '/Users/wenshan/Desktop/ccir/注意力压缩/figs/combined_irr_fig.png'
    concat_png(png_name_list, res_path)

