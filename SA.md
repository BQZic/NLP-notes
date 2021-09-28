# Sentiment Analysis & ERC

### Table of Contents

- [Overview](#overview)
  - Emotion Recognition in Conversation
- [Dataset](#dataset)
- Paper
  - ERC
    - Infusing Multi-Source Knowledge with Heterogeneous Graph Neural Network for Emotional Conversation Generation [AAAI '21]
    - Graph Based Network with Contextualized Representations of Turns in Dialogue [EMNLP '21]
    - Directed Acyclic Graph Network for Conversational Emotion Recognition [ACL '21]
  - Sentiment Analysis
    - Does syntax matter? A strong baseline for Aspect-based Sentiment Analysis with RoBERTa [NAACL '21]
    - A Unified Generative Framework for Aspect-based Sentiment Analysis [ACL '21]
- [Sidenotes](#sidenotes)

****

### Overview

在Emotion Recognition in Conversation (ERC) task中，一般来说有两种model conversation context的方式：

1. Graph-based Method

   - DialogGCN [2019]：每个dialog作为一张图
   - RGAT [2020]：在DialogGCN的基础上加入positional encoding
   - ConGCN [2019]：整个ERC dataset作为一张图，speaker和utterances都是graph nodes
   - KET [2019]：hierarchical Transformers with external knowledge
   - DialogXL [2020]：基于XLNet，提高了memory和dialog-aware self-attention

   将KET和DialogXL看作Graph-based method是因为他们都基于transformer，transformer中的self-attention可以被当作fully-connected graph。

2. Recurrence-based Method

   - ICON and CMN [2018]：使用了GRU和memory network
   - HiGRU [2019]：2个GRU，一个utterance encoder一个conversation encoder
   - DialogRNN [2019]：model dialogue dynamics with several RNNs
   - COSMIC [2020]：和DialogRNN很像，但是加了commonsense knowledge

3. Directed Acyclic Graph Neural Network

   - DAGNN [2021]：有multiple layers

****

### Dataset

1. MELD

   - data from _Friends_
   - 七分类：Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise
   - [Dataset Github Link](https://affective-meld.github.io)

2. DailyDialog - huggingface

   - Default
     - `dialog`: a `list` of `string` features.
     - `act`: a `list` of classification labels, with possible values including `__dummy__`(0), `inform` (1), `question` (2), `directive` (3), `commissive` (4).
     - `emotion`: a `list` of classification labels, with possible values including `no emotion` (0), `anger` (1), `disgust` (2), `fear` (3), `happiness` (4).
   - Data splits

   | name    | train | validation | test |
   | ------- | ----- | ---------- | ---- |
   | default | 11118 | 1000       | 1000 |

   ```
   {
       "act": [2, 1, 1, 1, 1, 2, 3, 2, 3, 4],
       "dialog": "[\"Good afternoon . This is Michelle Li speaking , calling on behalf of IBA . Is Mr Meng available at all ? \", \" This is Mr Meng ...",
       "emotion": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
   }
   ```

| Dataset     | Total Dialogues (train, test, dev) | Average Speaker Turns Per Dialogue | Average Tokens Per Dialogue | Average Tokens Per Utterance |
| ----------- | ---------------------------------- | ---------------------------------- | --------------------------- | ---------------------------- |
| DailyDialog | 13,118 (11,118, 1000, 1000)        | 7.9                                | 114.7                       | 14.6                         |
| MELD        | 1,433 (1039, 114, 280)             |                                    |                             |                              |
| EmoryNLP    | (713, 99, 85)                      |                                    |                             |                              |
| IEMOCAP     | (120, 31)                          |                                    |                             |                              |

3. EmoryNLP

****

### Infusing Multi-Source Knowledge with Heterogeneous Graph Neural Network for Emotional Conversation Generation [[paper](https://arxiv.org/pdf/2012.04882.pdf)] [[code](https://github.com/XL2248/HGNN)] [AAAI '21]

**Motivation**

Recent emotional conversation systems fall into two categories:

1. Inject the specific emotion vector into the response generation (ignores facial expression and personality)

2. Automatically track the emotional state in dialogue history to generate emotional responses (ignores facial expression and personality)

**Contribution**

1. _Heterogeneous Graph-Based Encoder_
   - build a graph on the conversation content with four-source knowledge
   - conduct _representation learning_ over the constructed graph with the encoder
2. _Emotion Personality Aware Decoder_
   - take in graph-enhanced representation, the predicted emotions, and the current speaker's personality

**Method**

1. Task Definition

   - _Input_: 5-tuples <U, F, E, S, s_{N+1}> (Dialogue history till round N, Facial Expression Sequence, Emotion Sequence, Speaker Sequence, the speaker for the next response)
   - _Output_: Y = {y_1, ..., y_J} target response

2. Encoder

   - Graph Contruction: construct the heterogeneous graph with multi-source knowledge
   - Graph Initialization: initializing different kinds of nodes
   - Heterogeneous Graph Encoding: perceive emotions and representing the conversation context based on the constructed graph
   - Emotion Predictor: predict suitable emotions with the graph-enhanced representations for feedback

3. Heterogeneous Graph-based Encoder Graph Construction

   ![Screen Shot 2021-09-16 at 5.23.02 PM](/Users/beiqizou/Library/Application Support/typora-user-images/Screen Shot 2021-09-16 at 5.23.02 PM.png)

**Experiment**

word embedding dimension 128

hidden size 256

number of encoder layers 2

number of attention heads (ReCoSa & model itself) 4

batch size 16

max turn of dialogue 35, max sentence length 50, number of speakers 13 (remove speakers just said several utterances -> 12)

PPL - fluency evaluation

BLEU - relevance of responses 

D-1, D-2: degree of diversity

semantic-level similarity between the generated responses and the ground truth

fine-tune a BERT-based emotion classifier (W-avg 62.55 > DialogGCN) on the text utterances of MELD as one evaluation tool for the accuracy of emotion expression for generated response

**Conclusion**

[Back to Top](#table-of-contents)

****

### Graph Based Network with Contextualized Representations of Turns in Dialogue [[paper](https://arxiv.org/pdf/2109.04008.pdf)] [[code](https://github.com/BlackNoodle/TUCORE-GCN)] [EMNLP '21]

**Motivation**

Effective relation extraction中有几个问题：

1. Speaker和utterance之间的关系
2. Surrounding turns和当前utterance的关系
3. dialogue有several turns

为了捕捉到以上信息，使用GCN来做。Encoding部分使用了$BERT_s$ 和 SA-BERT的speaker embedding。Extract each turn的representation使用了Masked Multi-head self-attention。然后再construct heterogeneous dialogue graph to capture **relation information between arguments in the dialogue**。最后对turn nodes使用BiLSTM，对heterogeneous graph使用GCN。

**Contribution**

1. Proposed **TUCORE-GCN** (Turn Context aware graph convolutional network) modeled by paying attention to the way people understand dialogues
2. Introduced **Surrounding Turn Mask** to better capture the representations of the turns
3. Proposed a novel approach which treats the task of emotion recognition in conversations (ERC) as a **dialogue-based RE**

**Method**

1. Encoding Module
2. Turn Attention Module
3. Dialogue graph with sequential nodes module
   - Type of edges
     - Dialogue edge
     - Argument edge
     - Speaker edge
4. Classification module

**Experiment**

| Method             | MELD  | EmoryNLP | DailyDialog |
| ------------------ | ----- | -------- | ----------- |
| TUCORE-GCN Roberta | 65.36 | 39.24    | 61.961      |

- Weighted-F1
- 在MELD和EmoryNLP上用RoBERTa的效果更好，CESTa在DailyDialog上比该方法好，为63.12

**Conclusion**

[Back to Top](#table-of-contents)

****

### Directed Acyclic Graph Neural Networks [[paper](https://arxiv.org/abs/2101.07965)] [ICLR '21]

****

### Directed Acyclic Graph Network for Conversational Emotion Recognition [[paper](https://arxiv.org/abs/2105.12907)] [[code](https://github.com/shenwzh3/DAG-ERC)] [ACL '21]

**Motivation**

- Conversational context一般被建模成两种形式
  - Graph-based method：可以收集附近文字的信息，但是忽视了distant utterances和sequential information
  - Recurrence-based method：考虑到了distant utterance和sequential information但是很难get到临近的information
- 因此，为了可以更好的solve ERC，可以利用graph-based method的优势。在一段conversation中，每一句话只从前面的某几句话提取信息，并且不能propagate information backward

**Contribution**

- 把对话看作是DAG
- 将positional relation和speaker information纳入DAG的考量范围
- 使用DAGNN作为基准

**Method**

- Build a DAG from conversation

  - DAG：$(V, E, R)$
    - $V = \{u_1, ..., u_N\}$ (utterances - node)
    - $(i, j, r_{ij}) \in E$ (edge)
    - $r_{ij} \in R = \{0, 1\}$ (relation): $r_{ij} = 1$ if two utterances are spoken by the same speaker; $r_{ij} = 0$ otherwise 
  - 以下三个方面决定了当前信息是否应该传递
    - Direction: $\forall j > i, (j, i, r_{ji}) \notin E$. 前一句utterance可以pass information给后一句，但是后一句utterance不能把information传给前面一句。（保证构建的Graph是一个单项图，具体算法paper中有pseudocode）
    - Remote information: $\exists \tau < i, p(u_{\tau}) = p(u_i), (\tau, i, r_{\tau i} \in E)$ and $\forall j < \tau, (j, i, r_{ji} \notin E)$. 除去speaker说的第一句话以外，后面当前speaker说的utterance $u_i$ 前面会有同一speaker说的$u_{\tau}$, 这句话被当作是相关信息。再前面的话不重要，因此不纳入考虑范围。（这边的$u_{\tau}$是可控的，$\omega$ 是hyperparameter，用来控制一共几句被纳入“remote information”的范畴）
    - Local information: $\forall l, \tau < l < i, (l, i, r_{li}) \in E$. 在$\tau$和$i$之间的utterance都有local information

- Directed Acyclic Graph Neural Network (DAG-ERC)

  - Utterance Feature Extraction
    - RoBERTa-large as feature extractor, 在每一个数据集上fine-tune，最后freeze parameter
    - 用最后一层[CLS] token的pooled embedding作为当前utterance的feature representation

  作者在3.3.2中粗略概括了GNN，RNN和DAGNN的区别。DAGNN在ERC task上的优势是相对显著的，因为能够 1). get access to distant utterances; 2). model the information flow through whole conversation; 3). gathers information from several neighboring utterances 

  - DAG-ERC layers (详见paper推导)
  - Training and Prediction
    - ReLU, Softmax, argmax
    - CrossEntropy loss

**Experiment**

$\omega = 1$ as default, size of all hidden vectors 300, feature size of RobERTa is 1024

each training process 60 epochs, 50s per epoch

result is avg of 5 random runs on the test set

使用micro-avg F1 excluding the majority class (neutral) for DailyDialog 和 weighted-avg F1 for the other datasets

datasets使用了IEMOCAP, MELD, DailyDialog, EmoryNLP。DAG-ERC在IEMOCAP、DailyDialog、EmoryNLP中分别为68.03、59.33、39.02，为SOTA。在MELD dataset上，COSMIC有最好的成绩（64.28），DAG-ERC为63.65。

最后结果发现$\omega = 1$时F1最高，但跟其余两个其实差别不大。

**Conclusion**

- 通过ablation studies，DAG structure的确对结果有影响
- 在多人对话中，两句utterances有相同说话人的信息对最终结果没有太大作用
- 当layer增加时，DAG network不会像GNN一样那么容易受over-smoothing的影响
- 很多DAG-ERC的错误都是发生在misjudge neutral samples，emotional shift的时候

[Back to Top](#table-of-contents)

****

### DialogXL: All-in-One XLNet for Multi-Party Conversation Emotion Recognition [[paper](https://arxiv.org/pdf/2012.08695.pdf)] [[code](https://github.com/shenwzh3/DialogXL)] [AAAI '21]

****

### Topic Driven and Knowledge-Aware Transformer for Dialogue Emotion Detection [[paper](https://arxiv.org/abs/2106.01071)] 

****

### Does syntax matter? A strong baseline for Aspect-based Sentiment Analysis with RoBERTa [[paper](https://arxiv.org/abs/2104.04986)] [[code](https://github.com/ROGERDJQ/RoBERTaABSA)] [ACL '21]

**Motivation**

主要在这篇文章中回答两个问题：

1. Tree induced from PTM会不会比dependency parser给出的tree有更好的performance？
2. fine-tuning时，PTM会不会adapt entailed tree structure implicitly？

缩写：ALSC - aspect-level sentiment classification

**Contribution**

- FT-RoBERTa SOTA or near SOTA on datasets
- 回答前两个问题

**Method**

- Perturbed Masking Method
  - Detect syntactic information from pre-trained models
  - Impact value：impact a token $x_j$ has on another token $x_i$
    - Apply "[MASK]" on $x_i$, and get the representation $H_{\theta}(x\setminus\{x_i\})_i$
    - Apply "[MASK]" on $x_j$, and get the representation $H_{\theta}(x\setminus\{x_i, x_j\})_i$
    - Impact value is the Euclidean distance between the above two representations
      - $f(x_i, x_j) = \|H_{\theta}(x\setminus\{x_i\})_i - H_{\theta}(x\setminus\{x_i, x_j\})_i\|_2$
- ALSC Models based on Trees
  - Aspect-specific Graph Convolutional Networks (ASGCN)
    - Build dependency as a graph
    - Use GCN to model dependencies between each word
  - Proximity-weighted Convolution network (PWCN)
    - Use PWCN to get the dependency tree
    - Assign proximity value to each word based on the tree
    - Proximity value for each word: the shortest path in the dependency tree between this word and the aspects
  - Relational Graph Attention Network (RGAT)
    - Aspect-oriented dependency tree
      - Use aspects as the root node, all other words depend on the aspect directly
      - Relation between the aspect and other words: syntactic tag or tree-based distance

**Experiment**

- Dataset
  - English datasets: Rest14, Laptop14, Twitter
- Tree structures
  - Off-the-shelf dependency tree parser: spaCy and allenNLP
  - pre-trained BERT and RoBERTa by perturbed masking method
  - Fine-tuned BERT and RoBERTa with perturbed masking (Fine-tuning in the corresponding datasets - ALSC datasets)
- Implementation
  - FT-PTM: batch size 32, dropout 0.1, lr 2e-4, AdamW optimizer
  - Tree-decoding: Chu-Liu/Edmond's Algorithm
- Result
  - Models with dependecy trees usually achieve better performance than PTMs induced trees
    - PTMs induced trees tend to learn from neighbors
  - FT-RoBERTa leads to the best results on all the datasets
  - FT models on ALSC could adapt the induced tree implicitly
    - Less proportion of neighboring connections
    - Less aspect-sentiment distance
  - RoBERTa with MLP layer achieve SOTA or near SOTA performance (?)
- Analysis
  - Proportions of Neighboring Connections
  - Aspects-sentiment Distance - Avg distance between aspect and sentiment words

**Conclusion**

- PTMs-suitable tree-based models
- Tree-inducing methods from PTMs

[tx分析](https://cloud.tencent.com/developer/article/1817052)

[Back to Top](#table-of-contents)

****

### A Unified Generative Framework for Aspect-based Sentiment Analysis [[paper](https://arxiv.org/abs/2106.04300)] [[code](https://github.com/yhcc/BARTABSA)] [ACL '21]

**Motivation**



**Contribution**

**Method**

**Experiment**

**Conclusion**

[Back to Top](#table-of-contents)

****

### Thoughts on using MDP/HMM to transfer emotion state

Input: `robot current emotion state`, `user utterance`, `user utterance emotion state`, `情感强度`

Output: `robot next emotion state`

Fixed number of states: 

```
sentiment_map = {
    "恐惧": 0,
    "快乐": 1,
    "喜欢": 2,
    "惊讶": 3,
    "疑惑": 4,
    "悲伤": 5,
    "沮丧": 6,
    "厌恶": 7,
    "愤怒": 8,
    "其他": 9
}
label_map = {
    "恐惧": "负面",
    "快乐": "正面",
    "喜欢": "正面",
    "惊讶": "其他",
    "疑惑": "其他",
    "悲伤": "负面",
    "沮丧": "负面",
    "厌恶": "负面",
    "愤怒": "负面",
    "其他": "其他"
}
```



对于目前conversation中的情感识别一般需要使用graph的structure，因为需要多方的信息去判断当前的情感。对于目前的情感模块来说，应该只是一个pretrain model在游戏数据上fine-tune，并没有考虑到对话的特殊性。

****

### Sidenotes

1. [medium post](https://medium.com/neuronio/from-sentiment-analysis-to-emotion-recognition-a-nlp-story-bcc9d6ff61ae) Sentiment Analysis to Emotion Recognition
   - Use LSTM + CNN to predict next state emotion
2. [Hidden Markov Model](https://towardsdatascience.com/hidden-markov-model-implemented-from-scratch-72865bda430e)
3. [HMM with Pomegranate](https://medium.com/analytics-vidhya/how-to-build-a-simple-hidden-markov-models-with-pomegranate-dfa885b337fb)
4. [HMM的python实现](https://www.cnblogs.com/d-roger/articles/5719979.html)
5. 知乎  [文本情感分析](https://zhuanlan.zhihu.com/p/63852350)
6. [MDP](https://amy12xx.github.io/rl/2020/08/23/solving-markov-descision-processes.html) 图解
7. https://zhuanlan.zhihu.com/p/104337841
8. [github](https://github.com/yirui-wang-0212/NLP-SentimentAnalysisForChineseText) Chinese Sentiment Analysis



[Back to Top](#table-of-contents)



