[toc]

## 1. Sentiment Analysis

### 1.1 Aspect-based Sentiment Analysis

#### 1.1.1 A unified generative framework for aspect-based sentiment analysis [ACL '21]

[paper](https://arxiv.org/pdf/2106.04300.pdf); [code](https://github.com/yhcc/BARTABSA)

##### Motivation

absa的问题定义为识别aspect，opinion term和sentiment polarity。因此，作者根据input和output将absa分为七种类型：

1. aspect term extraction
2. opinion term extraction
3. aspect-level sentiment classification - 给定sentence + aspect，返回sentiment classification结果
4. aspect-oriented opinion extraction
5. aspect term extraction and sentiment classification
6. pair extraction
7. triplet extraction - 给定sentence S，返回aspect，opinion和sentiment classification结果

大部分的研究都集中于这些subtask的subset，没有一个可以将所有subtasks整合的unified generative formulation，因此本篇工作针对该问题进行解决。

##### Method: bart based, end-to-end

**思路**：
1. 根据不同task定义target sequence
2. 给定text sequence使用Bart encoder进行编码
3. 编码结果和sentiment class拼接
4. Bart decoder解码得到last hidden state
5. 3和4的结果进行dot-product并最终得出probability

**问题定义**

我们主要关注triplet extraction的问题定义：

`$Y = [a_1^s, a_1^e, o_1^s, o_1^e, s_1^p, ...]$`

target sequence中每5个元素为一组，上标s和e分辨代表start和end的index，p代表polarity，在本文中共三种neutral, positive和negative。该target sequence是一个index sequence。

例如，`<s> the battery life is good </s>`的target sequence为`[2,3,5,5,8,6]`，6代表下一个generation index (placeholder)，8代表polarity为positive。

[图 Figure 2]

**Model**

1. Bart Encoder：`$H^e = BARTEncoder([x_1, ..., x_n]), H^e \in \mathbb{R}^{n\times d}$`
2. 使用Index2Token进行转换 【TODO：check code】
3. BART decoder：`$h_t^d = BARTDecoder(H^e; \hat Y_{<t})$`
4. 预测probability distribution `$P_t$`: 【插入图】

**Training**

1. Training：teacher forcing，negative loglikelihood
2. Inference：beam search，autoregressive
3. Decoding algorithm：将sequence转化为term spans和sentiment polarity

##### Experiment

**Dataset**

triplet使用 [D20b](https://aclanthology.org/2020.emnlp-main.183.pdf) 【训练数据图 Table 1】

**Baseline**

【Backbone】

【Table 6图】

**Parameters**

3090，24G mem，每个dataset训练~15分钟

    1). BART-base: 12 layers, 768 hidden dimensions, 16 heads, total # of parameters 139M
    2). BERT-base: 12 layers, 768 hidden dimensions, 12 heads, total # of parameters 110M


#### 1.1.2 Does syntax matter? A strong baseline for aspect-based sentiment analysis with RoBERTa [NAACL '21]

[paper](https://arxiv.org/pdf/2104.04986.pdf); [code](https://github.com/ROGERDJQ/RoBERTaABSA)

##### Motivation

ALSC task - 给定句子和aspect，输出sentiment classification。主要探索预训练语言模型是否可以生成更为准确的句法树(induced tree)，并对absa task有所帮助。本文主要想要解决两个问题：

1. Induced tree from PTM是否会比dependency parser给出的tree在ALSC task上有更好的performance？
2. Fine-tuning时，PTM会不会隐性地学习到tree structure？

最终发现使用一个RoBERTa+MLP就可以实现比较好的效果（主要是指生成的tree比较好，ALSC task好像没有太大改善）。

##### Method - BERT, RoBERTa

**Perturbed Masking - 抽取Induced tree的方法**

主要做法为给定一个sequence `$x = [x_1, ..., x_T]$`，计算词于词之间的impact value。

`$x_j$`对`$x_i$`的impact value具体算法为：

1. 用[MASK] token替代`$x_i$`
2. 将masked sequence输入BERT或者RoBERTa，得到句子的表示`$H_{\theta}(x \setminus \{x_i\})_i$`
3. 在已经mask `$x_i$`的sequence基础上继续mask `$x_j$`
4. 并使用BERT或者RoBERTa，得到句子表示`$H_{\theta}(x \setminus \{x_i,x_j\})_i$`
5. impact value使用euclidean distance进行计算 `$f(x_i, x_j) = \|H_{\theta}(x \setminus \{x_i\})_i - H_{\theta}(x \setminus \{x_i,x_j\})_i \|_2$`

一个sequence中每个单词进行impact value的计算可以得到一个大小为 `$T \times T$`的matrix。使用tree decoding algorithm（如Eisner、Chu-Liu/Edmounds' algorithm）抽取dependency tree。

**ALSC Models Based on Trees**

1. Aspect-Specific Graph Convolutional Networks (ASGCN)
    - 将dependency tree看作一个graph，每个单词时一个node，dependencies看作edges。将tree转化为graph之后使用GCN model dependencies。
    - topological
2. Proximity-Weighted Convolution Network (PWCN)
    - 根据proximity value找到aspect对应的contextual words。proximity value通过tree中的shortest path计算。
    - tree-based distance
3. Relational Graph Attention Network
    - 将aspect作为root node，aspect和其他单词的relation通过syntactic tag或者tree-based distance表示。
    - topological + tree-based distance

##### Experiment

**Dataset**

SemEval 2014 task 4 (restaurant, laptop); Twitter; 3 non-English datasets - Dutch, French, Spanish

**Tree Structures**

1. off-the-shelf dependency tree parser (spaCy, allenNLP)
2. pre-trained BERT and RoBERTa + perturbed masking
3. fine-tune BERT and RoBERTa + perturbed masking 【图 Figure 1】
4. Left-chain & right-chain

**dependency tree的评判标准**

1. **neighboring connections的比例**。发现BERT/RoBERTa的比例有70%，但是FT-RoBERTa在50%左右。可能的原因是上游预训练时使用的是MLM任务，邻居提供的信息比例更大。
2. **aspect-sentiment distance**。作者predefine了一个sentiment words set（Amazon-2），然后再sentence里面找sentiment words，并计算distance。【公式】其中，FT-RoBERTa的distance最短。

**与ALSC models的比较**

对于ALSC task来说，使用RoBERTa的方法会比BERT好，但是提升没有太多。但对于非英语的dataset结果有提高。(Acc, F1)【结果】

#### 1.1.3 PASTE: A Tagging-Free Decoding Framework Using Pointer Networks for Aspect Sentiment Triplet Extraction [EMNLP '21]
[paper](https://arxiv.org/pdf/2110.04794.pdf)

##### Motivation

其他关于absa的triplet extraction不能理解sentiments背后的意义。
- 例: The film was good, but could have been better. 这句话修饰film的主题其实在"could have been better"，是一个偏负面的词汇。

贡献主要在于：
1. 提出了一个end-to-end tagging-free的解决方法。这个解决方法不仅可以挖掘aspect-opinion之间的连接关系，也可以对span-level的sentiment prediction进行建模，从而真正理解opinion triplet包含的相关性。
2. 提出了一个基于位置的scheme，可以统一地表示opinion三元组（这块儿不是和第一篇一样了？unified framework？）
3. 在ASTE-Data-V2上达到了SOTA

##### Method - End to end ？？？怎么感觉和第一篇一模一样？？？除了用的模型不一样

使用了Pointer Network-based decoding framework。

**问题定义**

输入一句utterance，输出[0, 0, 2, 2, POS] (0, 0)表示aspect，(2, 2)表示opinion，POS表示polarity。在这个framework中一共分三类，pos，neg，neutral。

**Framework**

？？？？？而且怎么结果也是这么低

#### 1.1.4 Recommend for a Reason: Unlocking the Power of Unsupervised Aspect-Sentiment Co-Extraction [EMNLP '21]

##### Motivation


## 2. Emotion Related

### 2.1 Emotional Conversation Generation

#### 2.1.1 Emotional Chatting Machine: Emotional Conversation Generation with Internal and External Memory [AAAI '18]

[paper](https://arxiv.org/abs/1704.01074)

第一个使用在大规模对话生成中加入emotion factor的工作。提出了emotion category，internal emotion memory，和external memeory。使用了encoder-decoder structure。emotion category的embedding是dynamically changing的(需要训练)，internal emotion memory主要还是用了GRU的特性。external memory的作用有点类似于copy，主要控制copy emotion words还是generic words(词表里的单词)。evaluation分为automatic和manual。Automatic evaluation采用了perplexity和accuracy。

#### 2.1.2 Perspective-taking and Pragmatics for Generating Empathetic Responses Focused on Emotion Causes [EMNLP '21] 



[paper](https://arxiv.org/abs/2109.08828)

##### Motivation and Task

empathy分为weak empathy和stronger empathy，weak empathy是没有共情主体的，如普通的问候"Are you okay?"，stronger empathy是有具体的situation，如"how is your headache"。本篇工作主要关注strong empathy中的两个问题，
1. 哪些单词会影响对方的情绪
2. 在生成回复时考虑到这些单词

之前工作的解决方式都需要sub-utterance level的annotations，因此在本篇工作中作者使用generative estimator去推测emotion cause words，可以使用无词义级别标注的utterance。

##### Method

1. 提取emotion cause words
    1. 训练一个generative emotion estimator (GEE，基于BART)，对句子里的每个单词计算"emotion cause weight"。主要model `P(C, E) = P(E)P(C|E)`。首先计算给定E生成C的概率，然后用用Bayes计算`P(E|C)`。C是描述当前情况的一句话。
        - 用于训练GEE的dataset时EmpaheticDialogues：多轮英语对话，专门针对emotional situation，且听者会表达empathy。一共有32中emotion labels，均匀分布。
        - Training的主要过程即给定emotion label E，GEE生成相应的emotional situation，这样GEE可以学习到joint probability (?这样的话是否会太针对于这个dataset了)
            - 如：给定emotion joyful，GEE生成 I got accepted into a masters program in neuroscience
        - 计算P(W|E=e),取top-k作为emotion cause words
2. 生成empathetic responses
    - 使用RSA framework 
        - 这里reference到了作者的去年的一篇工作，关于使用RSA framework防止生成和定义的persona冲突的结果
            - Will i sound like me? improving persona consistancy in dialogues through pragmatic self-consciousness [EMNLP '20] [paper](https://arxiv.org/pdf/2004.05816.pdf)
    - 基本思想是通过随机替换(从GEE生成的别的单词里面抽取)识别出的emotion cause words来构建distractor，这样可以使基于RSA的模型更加focus在targeted words上。(构建负样本的思想？？)

##### Experiment

基于EmpatheticDialogues dataset，找人标注了EmoCause dataset。

**评价指标**

Emotion cause word recognition rate: Top-1,3,5 recall

Exploration and Interpretation: RoBERTa

**结果**


#### 2.1.3 Towards Persona-Based Empathetic Conversational Models [EMNLP '20]

[code](https://github.com/zhongpeixiang/PEC)

阿里的工作，主要贡献是

1. 针对persona-based empathetic conversations的数据集，PEC
2. CoBERT, 基于BERT的回复选择模型，SOTA on PEC
3. CoBERT在empathetic conversation dataset上进行训练

PEC主要是reddit里面的data，persona的抽取方式是根据规则来抽取，感觉是一个历史的post。

CoBERT主要是将context，response，persona分别放入三个BERT，算两两之间的co-attention，softmax之后concatenate。Loss是


### 2.2 Emotion Recognition/Detection

在Emotion Recognition in Conversation (ERC) task中，一般有以下几种model conversation text的方式：

1. Graph-based Method
    - DialogGCN [2019]: 每个dialogue作为一张图
    - RGAT [2020]: 在DialogGCN的基础上加入positional encoding
    - ConGCN [2019]: 整个ERC dataset作为一张图，speaker和utterances都是graph nodes
    - KET [2019]: hierarchical transformers with external knowledge
    - DialogXL [2020]: 基于XLNet，提高了memory和dialog-aware self-attention
2. Recurrence-based Method
    - ICON and CMN [2018]：使用了GRU和memory network
    - HiGRU [2019]：2个GRU，一个utterance encoder一个conversation encoder
    - DialogRNN [2019]：model dialogue dynamics with several RNNs
    - COSMIC [2020]：和DialogRNN很像，但是加了commonsense knowledge
3. Directed Acyclic Graph Neural Network
   - DAGNN [2021]：有multiple layers

#### 2.2.1 Topic-Driven and Knowledge-Aware Transformer for Dialogue Emotion Detection [ACL '21]

[paper](https://arxiv.org/pdf/2106.01071.pdf); [code](https://github.com/something678/TodKat)

##### Motivation and Task

对话中的情感检测，识别每句话对应的情感。

由于情感的表达方式、话语的意义因讨论的主题而变化、对话者之间共享的隐形知识，检测对话的情感仍然有挑战性，因为
- 现有的对话情感检测方法没有将重点放在对话的整体属性上
- 情感和话题检测依赖于对话者之间共享的基本常识知识，虽然COSMIC加入了常识，但是现有的方法没有基于涉及的主题和情感相关信息进行细粒度提取
- 主题发现的低资源性

##### Method - Pretrained LM + VAE

1. Topic Representation Learning
    - 在预训练语言模型中加入topic layer并fine tune。topic之间的关系使用transformer的multi-head attention。
        - Input -> Encoder(LM) -> Latent Vector -> Decoder(LM) -> Output 
2. 融合commonsence knowledge，使用transformer encoder-decoder structure当作classifier预测emotion labels
    - commonsence knowledge retrieval
        - 使用ATOMIC作为commensence knowledge retrieval。将utterance和knowledge graph里的每一个node做匹配，用SBERT计算utterance和event之间的相似度，取top-k。
    - knowledge generation model
        - 使用COMET(一个在ATOMIC上train的模型)。将utterance作为输入input到COMET中生成K个最可能的事件
    - knowledge selection
        - 使用pointer network从SBERT或者COMET中选择commonsense knowledge (计算候选知识源概率)。pointer network通过gumbel softmax生成one-hot distribution，然后结合SBERT和COMET的commonsense knowledge。
        - 使用在步骤一中训练好的语言模型计算knowledge的[CLS]和topic representations。attention就是utterance和之前得到的knowledge representation的点乘。
    - Transformer encoder-decoder：将utterance映射成emotion label sequence
        - 每个utterance用[CLS] token representation + topic representation + knowledge representation表示。

##### Experiment

**Datasets**

| Dataset Name | # of emotions | # of Dial. | # of Utt. | Train, dev, test  |
| ------------ | ------------- | ---------- | --------- | ----------------- |
| DailyDialog  | 6 + neutral   | 13,118     | 102,979   | 11118, 1000, 1000 |
| MELD         | 6 + neutral   | 1,432      | 13,708    | 1038, 114, 280    |
| IEMOCAP      | 5 + neutral   | 151        | 7,333     | 100, 20, 31       |
| EmoryNLP     | 6 + neutral   | 827        | 9,489     | 659, 89, 79       |

MELD和EmoryNLP都是从'Friends'里面抽取的对话。DD里面有太多neutral label，所以在evaluation的时候去掉了。

**Baseline**

| Model       | Structure                                                    |
| ----------- | ------------------------------------------------------------ |
| HiGRU       | encoder GRU + attention layer + decoder GRU                  |
| DialogueGCN | speaker aware GCN, global context + 说话人status -> emotion labels |
| KET         | transformer encoder + common-sense knowledge from ConceptNet |
| COSMIC      | 本文之前的SOTA，GRU + ATOMIC，以事件为中心的常识             |

【结果图 Table 2】

- 不算本工作COSMIC性能最优；TODKAT在个别数据集上效果不好（作者认为可能是数据量和划分的问题）
- 消融实验：
    - 小数据集阻碍了模型发现主题的能力，所以IEMOCAP数据集上删掉topic性能会好
    - 删除常识知识下降很厉害
    - pointer调和了SBERT和COMET

【Table 4】

上图表示注意力机制可以引导模型关注更相关的事件，从而预测正确的情绪标签。


#### 2.2.2 Few-Shot Emotion Recognition in Conversation with Sequential Prototypical Networks [EMNLP '21]

[paper](https://arxiv.org/abs/2109.09366)

智能客服。

### 2.3 Emotion相关的评价指标

1. Relevant emotion ranking 相关情感排名
    - 基于多标签分类器
        - Subset Accuracy
        - Hamming Loss
        - Example F1
        - Micro F1
        - Macro F1
    - 基于实值函数
        - One Error
        - Coverage
        - Ranking loss
        - average precision
    - ProLoss弥补上面两类的不足
    https://zhangxin.liumengyang.xyz/rer-relevant-emotion-ranking-ren-wu-de-shi-xiang-zhi-biao-jie-du/

## 3. 落地思考

1. 文章里focus on emotion cause words生成的response看起来结果不错，但是这篇文章的方法好像不太适用于现在的框架，但是可以借鉴找emotion cause words的想法。如果npc的人设是一个比较有同理心的人的话，可以focus more attention on those words，通过一个threshold去控制focus的attention。
2. 在对于对话来说的emotion detection中，可以融合dialogue history做emotion prediction
3. 第二篇absa的方法可以借鉴用来生成dependency parse tree，后续可以看一下别的关于使用dependency parse tree来做emotion prediction

## Sidenotes

1. [pdf](https://arxiv.org/ftp/arxiv/papers/1801/1801.07883.pdf) DL for SA: A survey
2. https://github.com/l294265421/ACSA
3. [pdf](https://aclanthology.org/D16-1169.pdf) Context-Sensitive Lexicon Features for Neural Sentiment Analysis [EMNLP '16] - 情感词典，加权和，双向LSTM，学习情感强度(强化或弱化)
4. https://zhuanlan.zhihu.com/p/108168121
5. EMNLP 2021 情感分析 [整理](https://zhuanlan.zhihu.com/p/415880221)
    - 没出的
        -  Improving Empathetic Response Generation by Recognizing Emotion Cause in Conversations
6. Neural Temporal Opinion Modelling for Opinion Prediction on Twitter [ACL '20] [paper](https://arxiv.org/pdf/2005.13486.pdf)
    - tweet representation，topic extraction和neighborhood context attention
    - 主体模型是LSTM和GRU。LSTM捕捉临近context的影响，使用GRU建模temporal point process（下一条tweet在下一个时间间隔发布的可能性）。
