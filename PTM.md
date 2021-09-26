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



## Sidenotes

1. [知乎](https://zhuanlan.zhihu.com/p/406512290) 常见预训练语言模型总结

