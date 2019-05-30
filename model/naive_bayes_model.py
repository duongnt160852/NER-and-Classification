#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
sys.path.append('D:\\20182\\ROJECT2\\RABILOO\\hehe\\text_classification_example\\kimi_nlp\\src')
# from transformer.feature_transformer import FeatureTransformer


class NaiveBayesModel(object):
    def __init__(self):
        self.clf = self._init_pipeline()

    @staticmethod
    def _init_pipeline():
        pipe_line = Pipeline([
            ("transformer", FeatureTransformer()),#sử dụng pyvi tiến hành word segmentation
            ("vect", CountVectorizer()),#bag-of-words
            ("tfidf", TfidfTransformer()),#tf-idf
            ("clf", MultinomialNB())#model naive bayes
        ])

        return pipe_line