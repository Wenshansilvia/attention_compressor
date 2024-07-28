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
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                output_attentions=True
            ).to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=self.device_map,
                trust_remote_code=True,
                output_attentions=True
            )

    
    def compress(self, doc, query, ratio):
        attention, text = attention_score(doc, query, self.model, self.tokenizer)
        words, attention = reconstruct(text, attention)
        attention = gaussian_filter1d(attention, 1)
        words, _ = text_filter(words, attention, ratio)
        return ' '.join(words)
    
    def get_attention(self, doc, query):
        attention, tokens, len_dict = full_attention_score(doc, query, self.model, self.tokenizer)
        return attention, tokens, len_dict
    
    def compress_sentence(self, doc, query, ratio):
        attention, text = attention_score(doc, query, self.model, self.tokenizer)
        words, attention = reconstruct(text, attention)
        sentences, _, _ = sentence_filter(words, attention, ratio)
        return ' '.join(sentences)
    
    def compress_sentence_token(self, doc, query, ratio):
        attention, text = attention_score(doc, query, self.model, self.tokenizer)
        words, attention = reconstruct(text, attention)
        sentences = sentence_token_filter(words, attention, ratio)
        return ' '.join(sentences)