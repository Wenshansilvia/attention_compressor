import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    T5Tokenizer, 
    T5ForConditionalGeneration
)
from scipy.ndimage import gaussian_filter1d
from .utils import *


def get_device():
    """自动检测设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class Compressor:
    def __init__(
        self,
        model_path: str,
        sigma: float = 1.0,
        device_map: str = 'auto'
    ):
        """
        通用压缩器基类
        :param model_path: 模型路径或名称
        :param sigma: 平滑系数，用于 attention 的高斯滤波
        :param device_map: 模型加载策略（'auto'、'cuda'、'cpu'）
        """
        self.model_path = model_path
        self.device = get_device()
        self.device_map = device_map
        self.sigma = sigma
        self.load_model()

    def __repr__(self):
        return f"<Compressor model={self.model_path}, device={self.device}, sigma={self.sigma}>"

    def load_model(self):
        """加载语言模型"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            output_attentions=True
        ).to(self.device)

    def compress(self, doc, query, ratio: float = 0.5, word: bool = True):
        """压缩文本（支持 word 模式和 token 模式）"""
        attention, text = attention_score(doc, query, self.model, self.tokenizer)
    
        # 对齐 T5Compressor 的逻辑
        if word:
            words, attention, cnt = reconstruct(text, attention)
        else:
            words = text
            cnt = [1] * len(words)
    
        # 可选高斯平滑
        if self.sigma:
            attention = gaussian_filter1d(attention, self.sigma)
    
        # 过滤文本
        words, attention = text_filter(words, attention, cnt, ratio)
    
        # 如果是 token 模式，重新聚合回词
        if not word:
            words, _, _ = reconstruct(words, attention)
    
        return ' '.join(words)
    
    def compress_sentence(self, doc, query, ratio: float = 0.5):
        attention, text = attention_score(doc, query, self.model, self.tokenizer)
        words, attention, _ = reconstruct(text, attention)
        if self.sigma:
            attention = gaussian_filter1d(attention, self.sigma)
        sentences, _, _ = sentence_filter(words, attention, ratio)
        return ' '.join(sentences)
    
    def compress_sentence_token(self, doc, query, ratio: float = 0.5):
        attention, text = attention_score(doc, query, self.model, self.tokenizer)
        words, attention, _ = reconstruct(text, attention)
        if self.sigma:
            attention = gaussian_filter1d(attention, self.sigma)
        sentences = sentence_token_filter(words, attention, ratio)
        return ' '.join(sentences)


class T5Compressor(Compressor):
    def __init__(
        self,
        model_path: str,
        sigma: float = 1.0,
        device_map: str = 'auto'
    ):
        """
        基于 T5 的压缩器，支持 cross-attention 提取
        """
        self.model_path = model_path
        self.device = get_device()
        self.device_map = device_map
        self.sigma = sigma
        self.load_model()

    def __repr__(self):
        return f"<T5Compressor model={self.model_path}, device={self.device}, sigma={self.sigma}>"

    def load_model(self):
        """加载 T5 模型"""
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            output_attentions=True
        ).to(self.device)

    def t5_attn(self, doc, query):
        """计算 cross-attention"""
        input_text = doc + " " + query
        input_ids = self.tokenizer(input_text, return_tensors="pt").to(self.device).input_ids
        context_len = len(self.tokenizer(doc, return_tensors="pt").input_ids[0]) - 1
        
        decoder_start_token_id = self.model.config.decoder_start_token_id
        decoder_input_ids = torch.tensor([[decoder_start_token_id]], device=self.device)

        outputs = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        attention = outputs.cross_attentions  # list: layers × (batch, heads, tgt_len, src_len)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        first_token_attention = attention[-1][0, :, 0, :]  
        mean_attention_weights = first_token_attention.mean(dim=0).cpu().detach().numpy()

        return mean_attention_weights[:context_len], tokens[:context_len]
    
    def t5_attention_score(self, doc, query):
        mean_attention_weights, tokens = self.t5_attn(doc, query)
        return softmax(mean_attention_weights), tokens
    
    def compress(self, doc, query, ratio: float = 0.5, word: bool = True):
        """压缩文本"""
        attention, text = self.t5_attention_score(doc, query)
        if word:
            words, attention, cnt = reconstruct(text, attention)
        else:
            words = text
            cnt = [1] * len(words)

        if self.sigma:
            attention = gaussian_filter1d(attention, self.sigma)

        words, attention = text_filter(words, attention, cnt, ratio)
        if not word:
            words, _, _ = reconstruct(words, attention)
        return ' '.join(words)

