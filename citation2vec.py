import pandas as pd # provide sql-like data manipulation tools. very handy.
pd.options.mode.chained_assignment = None
import numpy as np # high dimensional vector computing library.
from copy import deepcopy
from string import punctuation
from random import shuffle

import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
LabeledSentence = gensim.models.doc2vec.LabeledSentence # we'll talk about this down below

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def ingest():
    data = pd.read_csv('~/citations.csv')
    data.drop(["year", "title", "event_type", "pdf_name", "abstract"], axis=1, inplace=True)
    data = data[data.id.isnull() == False]
    data['id'] = data['id'].map(int)
    data = data[data['citations'].isnull() == False]
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    print 'dataset loaded with shape', data.shape
    return data

data = ingest()
print data.head(5)


def tokenize(citation):
    try:
        citation = unicode(citation.decode('utf-8').lower())
        citation = ''.join([i for i in citation if not i.isdigit()])
        citation = ''.join([i for i in citation if i.isalpha() or i.isspace()])
        tokens = tokenizer.tokenize(citation)
        return tokens
    except:
        return 'NC'


def postprocess(data, n=67483):
    data = data.head(n)
    data['tokens'] = data['citations'].progress_map(tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

data = postprocess(data)

x_train, x_test, y_train, y_test = train_test_split(np.array(data.head(67483).tokens),np.array(data.head(67483).id), test_size=0.2)

def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in tqdm(enumerate(tweets)):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized
print x_train[0]
x_train = labelizeTweets(x_train, 'TRAIN')
x_test = labelizeTweets(x_test, 'TEST')


tweet_w2v = Word2Vec(size=200, min_count=10)
tweet_w2v.build_vocab([x.words for x in tqdm(x_train)])

# train model
tweet_w2v.train([x.words for x in tqdm(x_train)], total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)


#print tweet_w2v.most_similar(positive=['better', 'smaller'], negative=['best'], topn=1)
#print tweet_w2v.similarity('neural', 'network')



#print tweet_w2v['good']
#print tweet_w2v.most_similar('neural')



print 'building tf-idf matrix ...'
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x.words for x in x_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print 'vocab size :', len(tfidf)



def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec


from sklearn.preprocessing import scale
train_vecs_w2v = np.concatenate([buildWordVector(z, 200) for z in tqdm(map(lambda x: x.words, x_train))])
train_vecs_w2v = scale(train_vecs_w2v)

print train_vecs_w2v[0]
from scipy import spatial
result = 1 - spatial.distance.cosine(train_vecs_w2v[0], train_vecs_w2v[1])

print result
