#### Code

Gitlab: https://gitlab.fuxi.netease.com:8081/zoubeiqi/dialoguesum（基于plm annotator）

#### Chat Summarization

1. Language Model as an Annotator: Exploring DialoGPT for Dialogue Summarization

    [ACL '21] [

   Code

   ]

   - Motivation

     - 一篇好的abstract需要有以下特性：信息丰富、无冗余、内容相关性高
     - 因此，本文利用DialogGPT的一些特性为原始文本注释了三类信息：**关键词、冗余句、主题切换**

   - Method

     - 利用

       DialoGPT

       将以上三类信息做标注（利用了DialoGPT所有的特性），具体实现为：

       - 关键词
         - Keyword empirically是word level loss比较大的（模型难以预测接下来的单词），因此挑选每句话前r%的单词作为keywords，加在最后keywords list里
         - r根据句子长度做变换
       - 冗余句
         - 通过计算<EOS> token的cosine similarity，如果后面一句和前一句的similarity超过threshold t1，那么在后面一句前加[RD] token
       - 主题切换
         - 根据之前算出来的word level loss，对每一句话做加权平均，算出每一句话的loss
         - 对于一个dialogue来说，如果这句话的loss超过threshold t2，那么就认为topic switch了，在前面加[TS] token
         - Intuition和关键词的标注一样，如果loss较大说明模型本来预测接下来的话不是“容易想到的”，那么认为只有换了一个话题才会有这种效果

     - 将dialogue标注成以上形式之后，放入BART生成summarization

   - Experiment

     - 在SAMSum Dataset上做了实验 

2. Low-Resource Dialogue Summarization with Domain-Agnostic Multi-Source Pretraining

    [EMNLP '21] [

   Code (Not Yet Released)

   ]

   - Motivation
     - 现有low-resource dialogue summarization直接使用了在其他领域（如新闻）的pre-train model，但新闻和dialogue差距很大
     - 因此提出了multi-source pretraining paradigm，基于large-scale in-domain non-summary data和summary decoder。encoder-decoder model 用adversarial critics在out-of-domain pretrain。
   - Method
   - Experiment 
     - 用了external news dataset的DAMS的R1，R2，RL和TGDGA相差不大
     - 20% training data R1在40左右

3. Unsupervised Summarization for Chat Logs with Topic-Oriented Ranking and Context-Aware Auto-Encoders

    [AAAI '21] [

   Code

   ]

   - Dataset - Large-scale chat log collected from e-commerce platform （淘宝对话数据）

4. Enhancing Semantic Understanding with Self-Supervised Methods for Abstractive Dialogue Summarization

    [Interspeech '21]

   - 结果并不理想，和TGDGA差不多（SAMSum dataset）

5. RepSum: Unsupervised Dialogue Summarization based on Replacement Strategy

    [ACL '21]

   - 只在AMI和Justice Dataset上测了

6. Controllable Abstractive Dialogue Summarization with Sketch Supervision

    [ACL-Findings '21]

   - 和PLM annotator在SAMSum dataset上有差不多的效果

7. [Structure-Aware Abstractive Conversation Summarization via Discourse and Action Graphs](https://arxiv.org/abs/2104.08400) [NAACL '21]

8. Multi-View Sequence-to-Sequence Models with Conversational Structure for Abstractive Dialogue Summarization

    [EMNLP '20]

   - Motivation

     - 每一段对话可以从不同角度view，从而导致有多种不同的patterns

   - Method

     - ![网易伏羲 > [邹北琪] 对话摘要 > Screen Shot 2021-09-10 at 4.51.33 PM.png](https://confluence.leihuo.netease.com/download/attachments/108630482/Screen%20Shot%202021-09-10%20at%204.51.33%20PM.png?version=1&modificationDate=1631263898061&api=v2) 

     - Conversation View Extraction

       - Topic View：先用Sentence-BERT将conversation中的每一句utterance encode，然后使用C99将对话分成几个block，每一个block里面有连续的几句对话（）
       - Stage View：使用HMM对进行stage view的分割，设置了hidden stage view的数量为4
       - Global View and Discrete View
         - Global：将所有utterances当作一个整体
         - Discrete：将每一句话分为一个block
       - 先将utterance分为不同block，每个view里面有几个block，以上几个view就是分block的方式不同

     - Multi-view Seq2Seq Model 

       - Conversation Encoder

         - 对于每一个view来说，先将每一个block中的每句话通过BART进行encoding，并在每一个block最开始加入特殊token，用token representation作为每个block的representation

         - describe view的方式为使用LSTM，将每一个block的information aggregate，得到的最后一个state

           

           作为当前view的representation

           - ![img](https://confluence.leihuo.netease.com/plugins/servlet/confluence/placeholder/macro?definition=e21hdGhqYXgtaW5saW5lLW1hY3JvOmVxdWF0aW9uPVNfal5rID0gTFNUTShoXzBee2osa30sIFNfe2otMX1eayksIGogXGluIFsxLG5dfQ&locale=en_US&version=2), 下标表示第几个block，上标表示当前的view k。

       - Multi-view Decoder （Figure 1b）

         - Motivation：不同view可以提供不同的conversational aspects可以让模型进行学习，并且可以让模型决定对哪些utterances pay more attention 

         - 在每一个transformer block中，加入multi-view attention layer

           - 

             , 

             

             - v是随机的context vector，W和b是parameters

           - 为了防止attention weights太过相像，apply sharpening function over ![img](https://confluence.leihuo.netease.com/plugins/servlet/confluence/placeholder/macro?definition=e21hdGhqYXgtaW5saW5lLW1hY3JvOmVxdWF0aW9uPVxhbHBoYV9rfQ&locale=en_US&version=2)：![img](https://confluence.leihuo.netease.com/plugins/servlet/confluence/placeholder/macro?definition=e21hdGhqYXgtaW5saW5lLW1hY3JvOmVxdWF0aW9uPVQ6IFx0aWxkZSBcYWxwaGFfayA9IFxmcmFje1xhbHBoYV9rXnsxL1R9fXtcc3VtX2kgXGFscGhhX2leezEvVH19fQ&locale=en_US&version=2)

           - ![img](https://confluence.leihuo.netease.com/plugins/servlet/confluence/placeholder/macro?definition=e21hdGhqYXgtaW5saW5lLW1hY3JvOmVxdWF0aW9uPVx0aWxkZSBBID0gXHN1bV9rIFx0aWxkZSBcYWxwaGFfayBBXmt9&locale=en_US&version=2)

       - Training

         - teacher forcing strategy, CE loss

   - Experiment

     - SAMSum dataset，14732 dialogues with human-written summaries
     - topic view比stage view来说略有更多贡献
     - multi-view BART(Topic + stage)主要是precision score比较高，因此model preserve了necessary information

9. [Improving Abstractive Dialogue Summarization with Graph Structures and Topic Words](https://aclanthology.org/2020.coling-main.39.pdf) [COLING '20]

#### Domain Adaptation

1. [AdaptSum: Towards Low-Resource Domain Adaptation for Abstractive Summarization](https://arxiv.org/abs/2103.11332) [NAACL '21]

#### Factual Consistency

1. [The Factual Inconsistency Problem in Abstractive Text Summarization: A Survey](https://arxiv.org/abs/2104.14839)

#### Theory

1. [RefSum: Refactoring Neural Summarization](https://arxiv.org/abs/2104.07210) [NAACL '21]

#### Extractive Dialogue Summarization

1. PreSumm

   :

    

   Text Summarization with Pretrained Encoders

    [EMNLP '19]

   - 

#### 数据集

1. DialogSum

   :

    

   A Real-Life Scenario Dialogue Summarization Dataset

    [ACL Findings '21]

   - 13k multi-turn dialogues (original purpose is to help EN learners to practice English speaking)
   - 作者认为online-chat messages有更多unique tokens和emojis，但是daily conversation更formal
   - Collect from online English listening exam materials and crawl from English learning website
   - 
   - Abstractive summarization result
     - 

2. SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization

   - 16k chat dialogues with manually annotated summaries
   - Train: 14,732 Validation: 818 Test: 819
   - linguists create dialogue - chitchats, gossiping about friends, arranging meetings, discussing politics, consulting university assignments with colleagues, ...; summary - short, extract important pieces of information, including names, written in the third person
   - 
   - 模仿messenger app上可能出现的对话

问题：

1. 需要大量人力标注数据
2. dialogue content规整，大部分有目标而非chitchat
3. 只有英文数据集

#### **Chat Summarization Leaderboard on SAMSum Dataset**