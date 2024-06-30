from src.load_data import load_data
from src.compressor import Compressor


if __name__ == '__main__':
    path = '/Users/wenshan/Desktop/ccir/注意力压缩/NQ/0.json'
    nq = load_data(path)
    model_path = '/Users/wenshan/Desktop/ccir/注意力压缩/Qwen2-0.5B-Instruct'
    compressor = Compressor(model_path)