from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.ndimage import gaussian_filter1d
from .utils import *


class Compressor:
    def __init__(
            self,
            model_path: str,
            device_map: str ='cuda',
    ):
        self.model_path = model_path
        self.device_map = device_map
        self.load_model()
    
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device_map,
            trust_remote_code=True,
            output_attentions=True
        )
    
    def compress(self, doc, query, ratio=0.5):
        attention, text = attention_score(doc, query, self.model, self.tokenizer)
        words, attention = reconstruct(text, attention)
        attention = gaussian_filter1d(attention, 1)
        words, _ = text_filter(words, attention, ratio)
        return ' '.join(words)