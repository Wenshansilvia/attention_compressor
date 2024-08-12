<h1 align="center"> QUITO: Accelerating Long-Context Reasoning through Query-Guided Context Compression </h1>

<p align="center">
<a href="https://arxiv.org/abs/2408.00274">📃 Paper</a>

# 🔍 Overview
We release QUITO, a powerful **context compressor** that leverages attention of the question over the contexts to filter useless information. 

![quito framework](assets/method.png)

# 🎯 Quick Start
## 1. Installation

```bash
git clone https://github.com/Wenshansilvia/attention_compressor
cd attention_compressor/
pip install -r requirements.txt
```

## 2. Usage

```python
from quito.compressor import Compressor

compressor = Compressor('Qwen/Qwen2-0.5B-Instruct')
# Use Phrase Level Filtering 
compressed_context = compressor.compress(doc="", query="", ratio=0.5)

# Or use Sentence Level Filtering
compressed_context = compressor.compress_sentence(doc="", query="", ratio=0.5)

# Or use Dynamic Sentence Level Filtering
compressed_context = compressor.compress_sentence_token(doc="", query="", ratio=0.5)
```


# 📌 Citation

If you find the repository or paper helpful, please cite our work:

```
@article{
    @misc{wang2024quitoacceleratinglongcontextreasoning,
      title={QUITO: Accelerating Long-Context Reasoning through Query-Guided Context Compression}, 
      author={Wenshan Wang and Yihang Wang and Yixing Fan and Huaming Liao and Jiafeng Guo},
      year={2024},
      eprint={2408.00274},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.00274}, 
}
}
```



