# Transformers

导入transformers包

```python
import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
```

在NER任务中，对句子进行token to id的转化（官网样例）

```python
sentence = "Hello, my dog is cuting"
input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0)
```



