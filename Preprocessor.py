from razdel import tokenize
import pymorphy2
morph = pymorphy2.MorphAnalyzer()

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
russian_stopwords = stopwords.words('russian')

from string import punctuation

import os
import numpy as np


## uncomment lines below to load navec
# cwd = os.getcwd()
navec_path = 'navec_pretrained.tar'
# os.system(f'wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar -O {navec_path}')

from navec import Navec

navec = Navec.load(navec_path)

class Preprocessor:
    '''
    takes as input raw sentence and does tokenization + lemmatization
    '''

    def __init__(self,tokenizer = tokenize, lemmatizer = morph,stopwords = russian_stopwords):
        # other tokenizer and lemmatizer might be used, code should be modified in that case, cause their returns differ
        self.tokenizer = tokenizer
        self.lemmatizer = lemmatizer
        self.stopwords = stopwords

    def prep(self,sentence):
        # tokenize:
        # # tokenize returns tuple of [start of seq pos, end of seq pos, token],

        print(f"sentence : {sentence}")
        tokens = [_.text for _ in self.tokenizer(sentence)]

        # lemmatize ()
        lemmas = [self.lemmatizer.parse(word)[0].normal_form for word in tokens]

        # remove unnecessary tokens
        prep_tokens = [word for word in lemmas if word not in self.stopwords 
                                                and word.strip() not in punctuation
                                                and word != ' ']

        return prep_tokens

    def vectorize(self,tokens):
        dim = navec.pq.dim
        embedding = np.zeros([1,dim])

        for word in tokens:
            if word in navec:
                embedding += navec[word]

        return embedding

    def encode(self,token):
        '''
        encodes word with it's index from navec vocab
        '''
        if token in navec:
            return navec.vocab[token]
        return navec.vocab['<unk>']