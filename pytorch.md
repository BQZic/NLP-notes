[toc]

# Pytorch 笔记

## BERT for MLM

```python
import torch
from transformers import BertForMaskedLM, BertTokenizer

if torch.cuda.is_available():
    device = torch.device('cuda', 0)
else:
    device = torch.device('cpu')
tokenizer = BertTokenizer.from_pretrained('chinese-bert-wwm-ext')
model = BertForMaskedLM.from_pretrained('chinese-bert-wwm-ext')
model.to(device)
```
在上述代码中加载了pretrained BertTokenizer和BertForMaskedLM的model。

```python
text = "我今天收到一个我不喜欢的礼物，我好[MASK][MASK]。"
tokenzied_text =  tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
```

- `tokenizer.tokenize`: 分词，在上面的代码中结果为
  
    `['我', '今', '天', '收', '到', '一', '个', '我', '不', '喜', '欢', '的', '礼', '物', '，', '我', '好', '[MASK]', '[MASK]', '。']`

- `tokenizer.convert_tokens_to_ids`: 在`vocab.txt`中寻找分词出来的list里面每个character的index，结果为

    `[2769, 791, 1921, 3119, 1168, 671, 702, 2769, 679, 1599, 3614, 4638, 4851, 4289, 8024, 2769, 1962, 103, 103, 511]`

另外一种进行tokenize的方法为直接使用`tokenizer.encode()`。
```python
indexed_tokens = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)
```
结果为
    
`tensor([[ 101, 2769,  791, 1921, 3119, 1168,  671,  702, 2769,  679, 1599, 3614,
         4638, 4851, 4289, 8024, 2769, 1962,  103,  103,  511,  102]])`

在这边可以发现前后多了`101`和`102`，分别为`[CLS]`和`[SEP]`。

接下来需要把刚刚得到的`indexed_tokens`转化为`torch.tensor`放入model。
```python
# 如果有GPU的话可以.to('cuda')
tokens_tensor = torch.tensor([indexed_tokens]).to(device) # --- 1
```
在[Bert pytorch](https://pypi.org/project/pytorch-pretrained-bert/)里还有提到`segement_tensors`，这是为了区分不同sentences A 和 B。比如：
```python
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 8
tokenized_text[masked_index] = '[MASK]'
assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']

# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

segments_tensors = torch.tensor([segments_ids]) # --- 1
segments_tensors = segments_tensors.to('cuda')
```
注意在这里将list转成tensor的时候多加了`[]`（`1`的部分）。也可以通过`unsqueeze(0)`来达到同样效果。

```python
# Predict all tokens
with torch.no_grad():
    predictions = model(tokens_tensor, segments_tensors)

# confirm we were able to predict 'henson'
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
assert predicted_token == 'henson'
```

也可以使用`tokenizer.encode_plus()`，`encode_plus`返回的是字典dict，使用字典访问的方式取出结果数据。示例如下：
```python
token2id = tokenizer.encode_plus(
    sequence,                       # 输入文本
    add_special_tokens = True,      # 添加'[CLS]'和'[SEP]'token
    max_length = 20,                # 填充和截断长度
    pad_to_max_length - True,       
    return_tensors = 'pt'           # 返回pytorch格式的数据
)

print(token2id)
```
`token2id`的结果为
```python
{'input_ids': tensor([[  101,   138, 18696,   155,  1942,  3190,  1144,  1572, 13745,  1104,
           159,  9664,  2107,   102,     0,     0,     0,     0,     0,     0]]), 
'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]])}
```
其中
- `input_ids`: token id（包括padding补零）
- `token_type_ids`: 如果是next sentence任务，用1标识第一个句子，0标识第二个句子
- `attention_mask`: `attention_mask`是可选的，对于self-attention句子后面补零的部分需要mask不参与之后的softmax计算，保证计算结果正确

`label`是个可选参数，用于模型的预测值pred之间进行计算loss。

- 分类任务 如 **BertForSequenceClassification**: label的维度为batch_size大小的tensor
- token classification model 如 **BertForTokenClassification**：label维度为`[batch_size, seq_length]`大小的tensor
- MLM 如 **BertForMaskedLM**：label的维度为`[batch_size, seq_length]`大小的tensor
- seq2seq tasks 如 **BertForConditionalGeneration**, **MBartForConditionalGeneration**：label的维度为`[batch_size, tgt_seq_length]`大小的tensor

基本的模型如BertModel不接受label，因为它仅仅是transformer模型，只能输出特征值。

示例
```python
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes).to(device)
for _ in range(2):#训练2个epoch
    print(train_dataloader)
    for i, batch in enumerate(train_dataloader):#迭代器遍历数据，四个tensor对应
        batch = tuple(t.to(device) for t in batch)#转换成tupe并且指定cuda运行
        loss = model(input_ids=batch[0], token_type_ids=batch[1], attention_mask=batch[2], labels=batch[3])[0]
        print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

[Back to Top](#pytorch-笔记)

### BERTModel Transformer 源码

[huggingface source code ref](https://huggingface.co/transformers/model_doc/bert.html#bertformaskedlm)

[ref](http://xtf615.com/2020/07/05/transformers/)

```python
# BertModel的构造函数
def __init__(self, config):
    super().__init__(config)
    self.config = config
    self.embeddings = BertEmbeddings(config)
    self.encoder = BertEncoder(config)
    self.pooler = BertPooler(config)
    self.init_weights()
    
def forward(self, input_ids=None, attention_mask=None,token_type_ids=None,    
            position_ids=None, head_mask=None, inputs_embeds=None,
            encoder_hidden_states=None, encoder_attention_mask=None,
            output_attentions=None, output_hidden_states=None,):
    # ignore some code here...
    
    # step 1: obtain sequence embedding, BertEmbeddings 
    embedding_output = self.embeddings(
        input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, 
        inputs_embeds=inputs_embeds)
    
    # step 2: transformer encoder, BertEncoder
    encoder_outputs = self.encoder(
        embedding_output,
        attention_mask=extended_attention_mask,
        head_mask=head_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_extended_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )
    sequence_output = encoder_outputs[0]
    
    # step 3: pooling to obtain sequence-level encoding, BertPooler
    pooled_output = self.pooler(sequence_output)

    outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
    return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

```

#### `tokenizer.encode()`参数介绍

```python
    def encode(
        self, 
        text: str, 
        text_pair: Optional[str] = None,
        add_special_tokens: bool = True, 
        max_length: Optional[int] = None, 
        stride: int = 0,
        truncation_strategy: str = "longest_first", 
        pad_to_max_length: bool = False, 
        return_tensors: Optional[str] = None, 
        **kwargs
    )
```

- `add_special_tokens: bool = True`将句子转化成对应模型的输入形式，默认开启
- `max_length`设置最大长度，如果不设置的话默认长度为512，如果句子长度超过512会报错
- `pad_to_max_length: bool = False`是否按照最长长度补齐，默认关闭，此处可以通过`tokenizer.padding_side='left'`
- `truncation_strategy: str = "longest_first"`截断机制，有四种方式来读取句子内容：
    - `longest_first`: 一直迭代，读到不能再读，直到读满为止
    - `only_first`: 只读第一个序列
    - `only_second`: 只读第二个序列
    - `do_not_truncate`: 不做截取，长了就报错
- `return_tensors: Optional[str] = None`返回的数据类型，默认是`None`，可以选择tensorflow版本`tf`或pytorch版本`pt`

#### `forward()`返回值
`Return Type`为`MaskedLMOutput`或者是`tuple(torch.FloatTensor)`。
- `loss` (optional): MLM loss，如果`torch.no_grad()`, loss则为None；shape为 `(1,)`
- `logits`: prediction score of the language modeling head (scores for each vocabulary token **before** SoftMax)；shape为`(batch_size, sequence_length, config.vocab_size)`
- `hidden_states` (optional): hidden-states of the model at the output of each layer plus the initial embedding outputs
- `atetentions` (optional): 

[Back to Top](#pytorch-笔记)


## Sidenotes


[padding](https://zhuanlan.zhihu.com/p/161972223)

[mask](https://zhuanlan.zhihu.com/p/139595546)