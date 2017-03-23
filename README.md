# 1. Language Processing and Python

* text.concordance("word")
 word가 등장한 문장을 검색한다.
* text.similar("word")
 word와 같은 구조에 끼어있는 단어들을 리터한다.
 예를들어 "monstrous"는 the _ pictures, a _ size 등의 사이에서 등장했는데,
 text.similar("monstrous")를 하면, 위와 같은 구조에서 등장하는 단어들을 리턴한다.
* text.common_contexts(["word1", "word2"])
 word1과 word2의 주변의 공통 구조를 리턴한다.
* text.dispersion_plot(['word1', ...])
 word들이 text내에서 등장하는 위치를 plot
* len(set(text3)) / len(text3)
 text3을 집합으로 바꾸고 갯수를 세고 그것을 text3의 총 단어수로 나눈다.
 즉, text3에서 평균적으로 단어들이 몇 % 등장하는지 알 수 있다.
* lexical diversity
 unique 총 단어 수 / 총 단어수
* fdist = FreqDist(text)
 text내의 단어들에 대해 빈도 분포표(frequency distribution)를 만든다.
* fdist.most_common(n)
 가장 빈번한 단어들을 n등까지 출력한다.
* fdist.plot(50, cumulative=True)
 누적 분포 그래프를 그린다. 50위까지.
* hapaxes
 텍스트에서 한번만 등장하는 단어.
 
* collocation
 함께 자주 쓰이는 단어들의 결합 ex. red wine
 => 의미가 비슷한 다른 단어와 대체되지 않는다는 특징이 있다. ex. marron(고동색) wine은 잘 쓰이지 않는다.
* list(bigrams(['word1', 'word2', ...]))
 []안의 인접한 word들을 쌍으로 묶는다. list로 리턴.

 => 즉, collocation은 빈번히 발생하는 bigram들이라고 볼 수 있다.
 우리는 거의 발생하지 않는 단어들을 포함하는 bigram들에 집중하고 싶다.
 특히, 우리가 각각의 단어들의 빈도수를 토대로 하여 예측하는 것보다 더 자주 발생하는 bigram을 찾기 원한다.
* text.collocations()
 위의 내용을 처리해주는 함수.


# 2. Accessing Text Corpora and Lexical Resources

## 1 Accessing Text Corpora

```python
from nltk.corpus import brown
news_text = brown.words(categories='news')
fdist = nltk.FreqDist(w.lower() for w in news_text)
for m in modals:
	print(m + ":", fdist[m], end=' ')
```

Next, we need to obtain counts for each genre of interest.

```python
>>> cfd = nltk.ConditionalFreqDist(
...           (genre, word)
...           for genre in brown.categories()
...           for word in brown.words(categories=genre))
>>> genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
>>> modals = ['can', 'could', 'may', 'might', 'must', 'will']
>>> cfd.tabulate(conditions=genres, samples=modals)
```

## 2 Conditional Frequency Distributions

When the texts of a corpus are divided into several categories, by genre, topic, author, etc, we can maintain separate frequency distributions for each category.

A conditional frequency distribution is a collection of frequency distributions, each one for a different "condition". The condition will often be the category of the text.

### 2.1 Conditions and Events
A conditional frequency distribution needs to pair each event with a condition.

Each pair has the form *(condition, event)*.

### 2.2 Coundting Words by Genre

Whereas FreqDist() takes a simple list as input, ConditionalFreqDist() takes a list of pairs.

```python
>>> from nltk.corpus import brown
>>> cfd = nltk.ConditionalFreqDist(
...           (genre, word)
...           for genre in brown.categories()
...           for word in brown.words(categories=genre))
```

_ConditionalFreqDist()_ 는 해당 condition에 따른 event들의 발생 빈도를 저장하고 있는다.
즉, `condition => {event1: n1, event2: n2, ...}`

```python
>>> cfd
<ConditionalFreqDist with 2 conditions>
>>> cfd.conditions()
['news', 'romance'] # [_conditions-cfd]
```

```python
>>> print(cfd['news'])
<FreqDist with 14394 samples and 100554 outcomes>
>>> print(cfd['romance'])
<FreqDist with 8452 samples and 70022 outcomes>
>>> cfd['romance'].most_common(20)
[(',', 3899), ('.', 3736), ('the', 2758), ('and', 1776), ('to', 1502), ...]
>>> cfd['romance']['could']
193
```

### 2.3 Plotting and Tabulating Distributions

a ConditionalFreqDist provides some useful methods for tabulation and plotting.

```python
>>> from nltk.corpus import inaugural
>>> cfd = nltk.ConditionalFreqDist(
...           (target, fileid[:4]) [1]
...           for fileid in inaugural.fileids()
...           for w in inaugural.words(fileid)
...           for target in ['america', 'citizen'] [2]
...           if w.lower().startswith(target))
```

```python
>>> from nltk.corpus import inaugural
>>> cfd = nltk.ConditionalFreqDist(
...           (target, fileid[:4]) [1]
...           for fileid in inaugural.fileids()
...           for w in inaugural.words(fileid)
...           for target in ['america', 'citizen'] [2]
...           if w.lower().startswith(target))
```

위의 코드는 inaugural의 각 파일들이 'america', 'citizen'으로 시작하는 단어를 몇 개나 포함하고 있는지를 계산한다.


In the plot() and tabulate() methods, we can optionally specify which conditions to display with a conditions= parameter. When we omit it, we get all the conditions.

```python
>>> from nltk.corpus import udhr
>>> languages = ['Chickasaw', 'English', 'German_Deutsch',
...     'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
>>> cfd = nltk.ConditionalFreqDist(
...           (lang, len(word)) [1]
...           for lang in languages
...           for word in udhr.words(lang + '-Latin1'))
```

```python
>>> cfd.tabulate(conditions=['English', 'German_Deutsch'],
...              samples=range(10), cumulative=True)
                  0    1    2    3    4    5    6    7    8    9
       English    0  185  525  883  997 1166 1283 1440 1558 1638
German_Deutsch    0  171  263  614  717  894 1013 1110 1213 1275
```

각 언어로서 쓰여진 Human right ... 문서를 단어수 별로 tabulate해서 보여준다.


### 2.4 Generating Random Text with Bigrams

We can use a conditional frequency distribution to create a table of bigrams (word pairs). The bigrams() function takes a list of words and builds a list of consecutive word pairs.

```python
def generate_model(cfdist, word, num=15):
    for i in range(num):
        print(word, end=' ')
        word = cfdist[word].max()

text = nltk.corpus.genesis.words('english-kjv.txt')
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams)
```

```python
>>> cfd['living']
FreqDist({'creature': 7, 'thing': 4, 'substance': 2, ',': 1, '.': 1, 'soul': 1})
>>> generate_model(cfd, 'living')
living creature that he said , and the land of the land of the land
```

Conditional frequency distributions are a useful data structure for many NLP tasks. Their commonly-used methods are summarized in 2.1.


## 3 More Python: Reusing Code

### 3.1 Creating Programs with a Text Editor

### 3.2 Functions

### 3.3 Modules

## 4 Lexical Resources

A lexicon, or lexical resource, is a collection of words and/or phrases along with associated information such as part of speech and sense definitions.

Lexical resources are secondary to texts, and are usually created and enriched with the help of texts.

For example, if we have defined a text `my_text`, then `vocab = sorted(set(my_text))` builds the vocabulary of `my_text`, while `word_freq = FreqDist(my_text)` counts the frequency of each word in the text. Both of `vocab` and `word_freq` are simple lexical resources.

A lexical entry consists of a headword (also known as a lemma) along with additional information such as the part of speech and the sense definition. Two distinct words having the same spelling are called homonyms.


### 4.1 Wordlist Corpora

NLTK includes some corpora that are nothing more than wordlists.

We can use it to find unusual or mis-spelt words in a text corpus:

```python
def unusual_words(text):
    text_vocab = set(w.lower() for w in text if w.isalpha())
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    unusual = text_vocab - english_vocab
    return sorted(unusual)

>>> unusual_words(nltk.corpus.gutenberg.words('austen-sense.txt'))
['abbeyland', 'abhorred', 'abilities', 'abounded', 'abridgement', 'abused', 'abuses', 'accents', ...]
>>> unusual_words(nltk.corpus.nps_chat.words())
['aaaaaaaaaaaaaaaaa', 'aaahhhh', 'abortions', 'abou', 'abourted', 'abs', ...]
```

Stopwords usually have little lexical content, and their presence in a text fails to distinguish it from other texts.

stopword: 색인 시에 도움이 되지 않는 단어들. (the, a, also 등)

Let's define a function to compute what fraction of words in a text are not in the stopwords list:

```python
>>> def content_fraction(text):
...     stopwords = nltk.corpus.stopwords.words('english')
...     content = [w for w in text if w.lower() not in stopwords]
...     return len(content) / len(text)
...
>>> content_fraction(nltk.corpus.reuters.words())
0.7364374824583169
```

Thus, with the help of stopwords we filter out over a quarter of the words of the text.

Word Puzzle: How many words of four letters or more can you make from those shown her? Each letter may be used once per word. Each word must contain the center letter and there must be at least one nine-letter word. No plurals ending in "s"; no foreign words; no proper names.

A wordlist is useful for solving word puzzles.

It is trickier to check that candidate solutions only use combinations of the supplied letters, especially since some of the supplied letters appear twice (here, the letter v). The FreqDist comparison method [3] permits us to check that the frequency of each letter in the candidate word is less than or equal to the frequency of the corresponding letter in the puzzle.

```python
>>> puzzle_letters = nltk.FreqDist('egivrvonl')
>>> obligatory = 'r'
>>> wordlist = nltk.corpus.words.words()
>>> [w for w in wordlist if len(w) >= 6
... 	and obligatory in w
...		and nltk.FreqDist(w) <= puzzle_letters]
['glover', 'gorlin', 'govern', ...]
```

It is well known that names ending in the letter **a** are almost always female. We can see this and some other patterns in the graph in 4.4, produced by the following code. Remember that name[-1] is the last letter of name.

```python
>>> cfd = nltk.ConditionalFreqDist(
...           (fileid, name[-1])
...           for fileid in names.fileids()
...           for name in names.words(fileid))
>>> cfd.plot()
```

### 4.2 A pronouncing Dictionary

A slightly richer kind of lexical resource is a table (or spreadsheet), containing a word plus some properties in each row. NLTK includes the CMU Pronouncing Dictionary for US English, which was designed for use by speech synthesizers.

```python
>>> entries = nltk.corpus.cmudict.entries()
>>> len(entries)
133737
>>> for entry in entries[42371:42379]:
...     print(entry)
...
('fir', ['F', 'ER1'])
('fire', ['F', 'AY1', 'ER0'])
('fire', ['F', 'AY1', 'R'])
('firearm', ['F', 'AY1', 'ER0', 'AA2', 'R', 'M'])
```

Observe that fire has two pronunciations (in US English): the one-syllable F AY1 R, and the two-syllable F AY1 ER0. The symbols in the CMU Pronouncing Dictionary are from the Arpabet, described in more detail at  http://en.wikipedia.org/wiki/Arpabet

```python
>>> for word, pron in entries: [1]
...     if len(pron) == 3: [2]
...         ph1, ph2, ph3 = pron [3]
...         if ph1 == 'P' and ph3 == 'T':
...             print(word, ph2, end=' ')
...
pait EY1 pat AE1 pate EY1 patt AE1 peart ER1 peat IY1 peet IY1 peete IY1 pert ER1
pet EH1 pete IY1 pett EH1 piet IY1 piette IY1 pit IH1 pitt IH1 pot AA1 pote OW1
pott AA1 pout AW1 puett UW1 purt ER1 put UH1 putt AH1
```

Can you summarize the purpose of the following examples and explain how they work?

```python
>>> [w for w, pron in entries if pron[-1] == 'M' and w[-1] == 'n']
['autumn', 'column', 'condemn', 'damn', 'goddamn', 'hymn', 'solemn']
>>> sorted(set(w[:2] for w, pron in entries if pron[0] == 'N' and w[0] != 'n'))
['gn', 'kn', 'mn', 'pn']
```

묵음을 체크하는 듯 하다.

The phones contain digits to represent primary stress (1), secondary stress (2) and no stress (0).

As our final example, we define a function to extract the stress digits and then scan our lexicon to find words having a particular stress pattern.

```python
>>> def stress(pron):
...     return [char for phone in pron for char in phone if char.isdigit()]
>>> [w for w, pron in entries if stress(pron) == ['0', '1', '0', '2', '0']]
['abbreviated', 'abbreviated', 'abbreviating', 'accelerated', 'accelerating', ...]
>>> [w for w, pron in entries if stress(pron) == ['0', '2', '0', '1', '0']]
['abbreviation', 'abbreviations', 'abomination', 'abortifacient', 'abortifacients', ...]
```

We can use a conditional frequency distribution to help us find minimally-contrasting sets of words.

```python
>>> p3 = [(pron[0]+'-'+pron[2], word) [1]
...       for (word, pron) in entries
...       if pron[0] == 'P' and len(pron) == 3] [2]
>>> cfd = nltk.ConditionalFreqDist(p3)
>>> for template in sorted(cfd.conditions()):
...     if len(cfd[template]) > 10:
...         words = sorted(cfd[template])
...         wordstring = ' '.join(words)
...         print(template, wordstring[:70] + "...")
...
P-CH patch pautsch peach perch petsch...
P-K pac pack paek paik pak pake paque...
P-L pahl pail paille pal pale pall paul...
```

Rather than iterating over the whole dictionary, we can also access it by looking up particular words.

```python
>>> prondict = nltk.corpus.cmudict.dict()
>>> prondict['fire']
[['F', 'AY1', 'ER0'], ['F', 'AY1', 'R']]
>>> prondict['blog']
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: 'blog'
>>> prondict['blog'] = [['B', 'L', 'AA1', 'G']]
>>> prondict['blog']
[['B', 'L', 'AA1', 'G']]
```


### 4.3 Comparative Wordlists

NLTK includes so-called Swadesh wordlists, lists of about 200 common words in several languages. The languages are identified using an ISO 639 two-letter code.

We can access cognate words from multiple languages using the entries() method, specifying a list of languages. With one further step we can convert this into a simple dictionary.

```python
>>> from nltk.corpus import swadesh
>>> fr2en = swadesh.entries(['fr', 'en'])
>>> fr2en
[('je', 'I'), ('tu, vous', 'you (singular), thou'), ('il', 'he'), ...]
>>> translate = dict(fr2en) # convert to dict type
>>> translate['chien']
'dog'
>>> translate['jeter']
'throw'
```


### 4.4 Shoebox and Toolbox Lexicons

## 5 WordNet

WordNet is a semantically-oriented dictionary of English, similar to a traditional thesaurus but with a richer structure.

### 5.1 Senses and Synonyms(의미와 유의어)

Consider the sentence in (1a). If we replace the word *motorcar* in (1a) by *automobile*, to get (1b), the meaning of the sentence stays pretty much the same:

(1a) Benz is credited with the invention of the *motorcar*.

(1b) Benz is credited with the invention of the *automobile*.

Since everything else in the sentence has remained unchanged, we can conclude that the words motorcar and automobile have the same meaning, i.e. they are synonyms. We can explore these words with the help of WordNet:

```python
>>> from nltk.corpus import wordnet as wn
>>> wn.synsets('motorcar')
[Synset('car.n.01')]
```

Thus, motorcar has just one possible meaning and it is identified as car.n.01, the first noun sense of car. The entity car.n.01 is called **a synset, or "synonym set", a collection of synonymous words (or "lemmas")**:

```python
>>> wn.synset('car.n.01').lemma_names()
['car', 'auto', 'automobile', 'machine', 'motorcar']
```

Each word of a synset can have several meanings. However, we are only interested in the single meaning that is common to all words of the above synset. Synsets also come with a prose definition and some example sentences:

```python
>>> wn.synset('car.n.01').definition()
'a motor vehicle with four wheels; usually propelled by an internal combustion engine'
>>> wn.synset('car.n.01').examples()
['he needs a car to get to work']
```

The words of the synset are often more useful for our programs. To eliminate ambiguity, we will identify these words as car.n.01.automobile, car.n.01.motorcar, and so on. This pairing of a synset with a word is called a lemma.

```python
>>> wn.synset('car.n.01').lemmas()
[Lemma('car.n.01.car'), Lemma('car.n.01.auto'), Lemma('car.n.01.automobile'),
Lemma('car.n.01.machine'), Lemma('car.n.01.motorcar')]
>>> wn.lemma('car.n.01.automobile')
Lemma('car.n.01.automobile')
>>> wn.lemma('car.n.01.automobile').synset()
Synset('car.n.01')
>>> wn.lemma('car.n.01.automobile').name()
'automobile'
```

Unlike the word motorcar, which is unambiguous and has one synset, the word car is ambiguous, having five synsets:

```python
>>> wn.synsets('car')
[Synset('car.n.01'), Synset('car.n.02'), Synset('car.n.03'), Synset('car.n.04'),
Synset('cable_car.n.01')]
>>> for synset in wn.synsets('car'):
...     print(synset.lemma_names())
...
['car', 'auto', 'automobile', 'machine', 'motorcar']
['car', 'railcar', 'railway_car', 'railroad_car']
['car', 'gondola']
['car', 'elevator_car']
['cable_car', 'car']
```

For convenience, we can access all the lemmas involving the word car as follows.

```python
>>> wn.lemmas('car')
[Lemma('car.n.01.car'), Lemma('car.n.02.car'), Lemma('car.n.03.car'), Lemma('car.n.04.car'), Lemma('cable_car.n.01.car')]
```

**`synsets(word)`는 word가 속한(이름이 word인 lemma를 포함하는) synset들을 리턴하고, `synset(name)`은 'name'이라는 synset을 접근하도록 하며, `lemma(word)`는 word를 이름으로 가지는 lemma들을 리턴한다.**


### 5.2 The WordNet Hierarchy

WordNet synsets correspond to abstract concepts, and they don't always have corresponding words in English. These concepts are linked together in a hierarchy.

WordNet makes it easy to navigate between concepts. For example, given a concept like motorcar, we can look at the concepts that are more specific; the (immediate) hyponyms.

```python
>>> motorcar = wn.synset('car.n.01')
>>> types_of_motorcar = motorcar.hyponyms()
>>> types_of_motorcar[0]
Synset('ambulance.n.01')
>>> sorted(lemma.name() for synset in types_of_motorcar for lemma in synset.lemmas())
['Model_T', 'S.U.V.', 'SUV', 'Stanley_Steamer', 'ambulance', 'beach_waggon', 'beach_wagon', 'bus', 'cab', 'compact', 'compact_car', 'convertible', ...]
```

We can also navigate up the hierarchy by visiting hypernyms. Some words have multiple paths, because they can be classified in more than one way.

```python
>>> motorcar.hypernyms()
[Synset('motor_vehicle.n.01')]
>>> paths = motorcar.hypernym_paths()
>>> len(paths)
2
>>> [synset.name() for synset in paths[0]]
['entity.n.01', 'physical_entity.n.01', 'object.n.01', 'whole.n.02', 'artifact.n.01', ...]
>>> [synset.name() for synset in paths[1]]
['entity.n.01', 'physical_entity.n.01', 'object.n.01', 'whole.n.02', 'artifact.n.01', ...]
```

We can get the most general hypernyms (or root hypernyms) of a synset as follows:

```python
>>> motorcar.root_hypernyms()
[Synset('entity.n.01')]
```

### 5.3 More Lexical Relations

Hypernyms and hyponyms are called **lexical relations** because they relate one synset to another.
Another important way to navigate the WordNet network is from items to their components (**meronyms**) or to the things they are contained in (**holonyms**).

나무는 뿌리의 *holonym*이고 뿌리는 나무의 *meronym*이다.

For example, the parts of a tree are its trunk, crown, and so on; the `part_meronyms()`. The substance a tree is made of includes heartwood and sapwood; the `substance_meronyms()`. A collection of trees forms a forest; the `member_holonyms()`:

```python
>>> for synset in wn.synsets('mint', wn.NOUN):
...     print(synset.name() + ':', synset.definition())
...
batch.n.02: (often followed by 'of') a large number or amount or extent
mint.n.02: any north temperate plant of the genus Mentha with aromatic leaves and small mauve flowers
mint.n.03: any member of the mint family of plants
mint.n.04: the leaves of a mint plant used fresh or candied
mint.n.05: a candy that is flavored with a mint oil
mint.n.06: a plant where money is coined by authority of the government
>>> wn.synset('mint.n.04').part_holonyms()
[Synset('mint.n.02')]
>>> wn.synset('mint.n.04').substance_holonyms()
[Synset('mint.n.05')]
```

There are also relationships between verbs. For example, the act of walking involves the act of stepping, so walking **entails** stepping. Some verbs have multiple entailments:

```python
>>> wn.synset('walk.v.01').entailments()
[Synset('step.v.01')]
```

Some lexical relationships hold between lemmas, e.g., **antonymy**(반의어):

```python
>>> wn.lemma('horizontal.a.01.horizontal').antonyms()
[Lemma('inclined.a.02.inclined'), Lemma('vertical.a.01.vertical')]
```

You can see the lexical relations, and the other methods defined on a synset, using dir(), for example: dir(wn.synset('harmony.n.02')).


### 5.4 Semantic Similarity

Given a particular synset, we can traverse the WordNet network to find synsets with related meanings.

Knowing which words are semantically related is useful for indexing a collection of texts, so that a search for a general term like vehicle will match documents containing specific terms like limousine.

If two synsets share a very specific hypernym — one that is low down in the hypernym hierarchy — they must be closely related.

```python
>>> right = wn.synset('right_whale.n.01') # 수염 고랫과 고래
>>> orca = wn.synset('orca.n.01') # killer whale
>>> minke = wn.synset('minke_whale.n.01') # 긴수염고래속의 소형 고래
>>> tortoise = wn.synset('tortoise.n.01') # 거북
>>> novel = wn.synset('novel.n.01') # 소설
>>> right.lowest_common_hypernyms(minke)
[Synset('baleen_whale.n.01')] # 수염고래
>>> right.lowest_common_hypernyms(orca)
[Synset('whale.n.02')] # 고래
>>> right.lowest_common_hypernyms(tortoise)
[Synset('vertebrate.n.01')] # 척추 동물
>>> right.lowest_common_hypernyms(novel)
[Synset('entity.n.01')] # 실재하는 것
```

We can quantify this concept of generality by looking up the depth of each synset:

```python
>>> wn.synset('baleen_whale.n.01').min_depth()
14
>>> wn.synset('whale.n.02').min_depth()
13
>>> wn.synset('vertebrate.n.01').min_depth()
8
>>> wn.synset('entity.n.01').min_depth()
0
```

For example, path_similarity assigns a score in the range  0–1 based on the shortest path that connects the concepts in the hypernym hierarchy (-1 is returned in those cases where a path cannot be found).

```python
>>> right.path_similarity(minke)
0.25
>>> right.path_similarity(orca)
0.16666666666666666
>>> right.path_similarity(tortoise)
0.07692307692307693
>>> right.path_similarity(novel)
0.043478260869565216
```

Several other similarity measures are available; you can type help(wn) for more information.


## 6 Summary

## 7 Further Reading

Extra materials for this chapter are posted at http://nltk.org/, including links to freely available resources on the web. The corpus methods are summarized in the Corpus HOWTO, at http://nltk.org/howto, and documented extensively in the online API documentation.

## 8 Exercises


# 3. Processing Raw Test