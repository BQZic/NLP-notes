## Evaluation Metrics

[知乎](https://zhuanlan.zhihu.com/p/108630305)

### **_BLEU_**: Bilingual Evaluaton Understudy 双语评估辅助工具

主要应用于machine translation。[BLEU scores](https://towardsdatascience.com/foundations-of-nlp-explained-bleu-score-and-wer-metrics-1a5ba06d812b)

**核心思想**

比较候选译文和参考译文里的n-gram的重合程度，重合程度越高就认为译文质量越高。unigram用于衡量单词翻译的准确性，高阶n-gram用于衡量句子翻译的流畅性。实践中，通常是N = 1～4，然后取加权平均。

**计算公式**

$$
BLEU = BP \cdot \exp(\sum_{n = 1}^N w_n \log p_n)
$$
其中$n$表示n-gram, $w_n$表示n-gram的权重。

BP(brevity penalty)为短句子惩罚因子，$r$表示最短的参考翻译长度，$c$表示候选翻译长度。当$c > r$时，$BP = 1$, 即不做惩罚；当$c \leq r$时，$BP = \exp(1-r/c)$. 

$p_n$表示n-gram的覆盖率，具体计算方式为：

$$
p_n = \frac{\sum_{C \in \{Candidates\}}\sum_{n-gram \in C} Count_{clip}(n-gram)}{\sum_{C' \in \{Candidates\} }\sum_{n-gram \in C'} Count(n-gram)}
$$
$Count_{clip}$是截断计数，其计数方式为：将一个n-gram在候选翻译中出现的次数，与在各个参考翻译中出现的最大值进行比较，取最小的那一个。


**主要特点**

1. n-gram共现统计
2. Based on precision

**缺点**

1. Only based on precision, not including recall
2. 存在常用词干扰（可用截断的方法解决）
3. 短句得分较高，即使有了BP

****

### **_ROUGE_**: Recall-Oriented Understudy for Gisting Evaluation，面向召回率的摘要评估辅助工具

主要应用于text summarization。

**核心思想**

大致分为四种：ROUGE-N，ROUGE-L，ROUGE-W，ROUGE-S。常用的是前两种（-N与-L）。

- ROUGE-N中的“N”指的是N-gram，其计算方式与BLEU类似，只是BLEU基于精确率，而ROUGE基于召回率。
- ROUGE-L中的“L”指的是Longest Common Subsequence，计算的是候选摘要与参考摘要的最长公共子序列长度，长度越长，得分越高，基于F值。

**计算公式**

$$
ROUGE-N=\frac{\sum_{S \in \{reference summaries\}}\sum_{n-gram \in S}Count_{match} (n-gram)}{\sum_{S \in \{reference summaries\}}\sum_{n-gram \in S}Count (n-gram)}
$$
其中， $n$ 表示n-gram， $Count (n-gram)$ 表示一个n-gram的出现次数， $Count_{match} (n-gram)$ 表示一个n-gram的共现次数。

****

### Normalization

Normalization在DL中就是“通过把一部分不重要的复杂信息损失掉”，以此来降低拟合难度以及过拟合的风险，从而加速了模型的收敛。降低了各维度数据的方差，让分布更稳定。

不同Normalization的区别是操作的信息维度不同，即选择损失信息的维度不同。目前在NLP领域中使用LN主要是因为BN效果很差。2020年的一篇工作中有提到“[powernorm](https://arxiv.org/abs/2003.07845)”，即对BN作一定改动，效果可以超越LN。

```
Sample 1: x1, x2, x3, ...
Sample 2: y1, y2, y3, ...
Sample 3: z1, z2, z3, ...
```

Batch Normalization: x1, y1, z1

Layer Normalization: x1, x2, x3

****

### Cross Entropy Loss vs KL divergence

****

### Optimizer

[Medium Post](https://medium.com/mlearning-ai/optimizers-in-deep-learning-7bf81fed78a0)



