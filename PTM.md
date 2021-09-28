# Pre-trained Models

## Pre-trained Models for Natural Language Processing: A survey [[paper](https://arxiv.org/abs/2003.08271)]

#### Language Representation Learning

- Non-contextual Embeddings
- Contextual Embeddings
  - Neural Contextual Encoders
    - Sequence Models
      - CNN
      - RNN
    - Non-sequence Models
      - Fully-connected self-attention model

#### Pretrained Models

- Why?
  - Learn universal language representations
  - better model initialization -> better generalization
  - regularization to avoid overfitting on small data
- 1st Generation - Pretrained word embeddings
  - Neural Network LM
    - Word2vec, GloVe (Based on CBOW, Skip-Gram)
- 2nd Generation - Pretrained __*Contextual encoders*__
  - ELMo
  - ULMFit
  - GPT, BERT

#### Overview of Pretrained Models

- Pretraining Tasks
  - Language Modeling (LM)
  - Masked Language Model (MLM)
  - Permuted Language Model
  - Denoising Autoencoder (DAE)
  - Contrastive Learning (CTL)



### Pre-training with Whole Word Masking for Chinese BERT

- 在训练BERT任务时，不随机mask单词，而是mask连续的词

- 比如：

  ```
  [Original Sentence]
  
  使用语言模型来预测下一个词的probability。
  使用语言[MASK]型来[MASK]测下一次的pro[MASK]##lity。 -随机mask
  使用语言[MASK][MASK]来[MASK][MASK]下一个词的[MASK][MASK][MASK]。 -本工作中使用的mask方法
  ```

### SpanBERT: Improving Pre-training by Representing and Predicting Spans

**Contributions**

1. 提出了**更好的Span Mask方案**，随即遮盖连续一段字比随机遮盖分散字更好。
2. 通过加入**Span Boundary Objective（SBO）训练目标**，增强了BERT的性能，特别在一些与span相关的任务，如抽取式问答。
3. 用实验获得了和XLNet类似的结果，发现不加入Next Sentence Prediction任务，直接用连续一长句训练效果更好。

**Method**

1. 



## Sidenotes

1. [知乎](https://zhuanlan.zhihu.com/p/406512290) 常见预训练语言模型总结

