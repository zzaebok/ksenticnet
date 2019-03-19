# KSenticNet 한국어 감성 사전

### Overview
- There are several Korean sentiment analysis resources such as [KNU SentiLex](https://github.com/park1200656/KnuSentiLex), [KOSAC](http://word.snu.ac.kr/kosac/).
- However, sentiment lexicons like them require a lot of time and human resources.
- So I decided to make it easier and automatically from [SenticNet](https://sentic.net/) and KAIST Korean wordnet([KWN](http://wordnet.kaist.ac.kr/)).

### Building Process
![KSenticNet Structure](/KsenticNetStructure.png)
> #### Features
- It follows major process of [CSenticNet](https://sentic.net/csenticnet.pdf)
- But it resolved duplicated sentic value problems
> #### Process
1. Make english word - synsets dictionary through KWN.
2. Direct mapping ( Compare each synset's hypernyms to semantics in SenticNet )
3. Apply lesk algorithm to the non-matched words in SenticNet
4. During 2, 3 there are synsets which get several different sentic values. Average sentic values according to AffectNet frequencies.
5. For Korean words in synset, assign sentic value which was given toward the synset.
6. During 5, there are synsets which have only one same Korean word. For those, use weighted average sentic value as 4.
7. During 5, several Korean words are assigned different sentic values but we cannot use weighted average because each synset contains multiple Korean words. So compute average cosine similarity * of synsets for that Korean word and use only the most adequate synset to give sentic value.

\* Cosine similarity is computed from Korean tuned-embedding vectors. The vectors of Korean words are tuned by [Context2Vec structure](https://github.com/SenticNet/context2vec). In this structure, I scraped example sentences for target words from several dictionaries. While applying Bi-LSTM, Self-Attention, Neural Tensor Network, pre-trained Fasttext vectors are modified and adjusted. By using these tuned vectors we can compute cosine similarities among other Korean words in a synset and use average similarity as an index of 'adequacy'.

### Resources
- SenticNet5
- CSenticNet
- KAIST Korean Wordnet
- [연세 한국어 사전](https://ilis.yonsei.ac.kr/ysdic/), 표준 국어 대사전, 고려대 한국어 대사전, 우리말샘

### Results and validation
- We can assign sentic value to 5465 Korean words.
- Validate it through [NAVER movie review corpus](https://github.com/e9t/nsmc) ( simple count after tokenizing by [Kkma](https://github.com/konlpy/kkma) )
<사진>
