#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import os, sys, re, json, pickle

sys.path.append('C:\\Users\\admin\\Downloads\\New folder\\New folder')
from model.svm_model import SVMModel
from pyvi import ViTokenizer, ViPosTagger
from flask import Flask, request, render_template
import ner_train
from underthesea import sent_tokenize, word_tokenize, pos_tag

app = Flask(__name__)
a = sys.path
crf = pickle.load(open("ner_model.sav", 'rb'))


@app.route("/")
def index():
    return render_template('hello.html')


@app.route("/ajax/ner", methods=['POST'])
def ner():
    a = request.form['content']
    corpus = sent_tokenize(a)
    b = []
    for sen in corpus:
        x_test = []
        for word in word_tokenize(sen, format="text").split(' '):
            x_test.extend(pos_tag(word))
        b.append(x_test)
    b1 = [ner_train.get_features(s) for s in b]
    c = crf.predict(b1)
    return json.dumps([b, c])


@app.route("/ajax/cls", methods=['POST'])
def cls():
    if request.method == 'POST':
        text = request.form['content']
        tcp = TextClassificationPredict()
        result = tcp.predict(text)[0]
        predict = result[0]
        predict_convert = convert(predict)
        d = tcp.predict(text)[1]
        giaitri = d[0]
        giaoduc = d[1]
        kinhdoanh = d[2]
        phapluatintuc = d[3]
        thegioi = d[4]
        thethao = d[5]
        thoisu = d[6]
        tuvan = d[7]
        a = []
        a.append(predict_convert)
        a.append(giaitri)
        a.append(giaoduc)
        a.append(kinhdoanh)
        a.append(phapluatintuc)
        a.append(thegioi)
        a.append(thethao)
        a.append(thoisu)
        a.append(tuvan)
        return json.dumps(a)


def convert(text):
    if text == 'giai_tri':
        a = 'Giải trí'
    elif text == 'giao_duc':
        a = 'Giáo dục'
    elif text == 'kinh_doanh':
        a = 'Kinh doanh'
    elif text == 'phap_luat_tin_tuc':
        a = 'Pháp luật tin tức'
    elif text == 'the_gioi':
        a = 'Thế giới'
    elif text == 'the_thao':
        a = 'Thể thao'
    elif text == 'thoi_su':
        a = 'Thời sự'
    else:
        a = 'Tư vấn'
    return a


class TextClassificationPredict:
    def __init__(self):
        self.test = None

    def train_data(self):
        train = open('data/giaitri.json', 'r', encoding='utf-8-sig')
        train_data = []
        for line in train:
            data = json.loads(line)
            train_data.append({"feature": (str(data['title']).strip() + ' ' + str(data['sapo']).strip() + ' ' + str(
                data['title']).strip() + ' ' + str(data['sapo']).strip() + ' ' + str(data['text']).strip()),
                               "target": "giai_tri"})

        train = open('data/giaoduc.json', 'r', encoding='utf-8-sig')
        for line in train:
            data = json.loads(line)
            train_data.append({"feature": (str(data['title']).strip() + ' ' + str(data['sapo']).strip() + ' ' + str(
                data['title']).strip() + ' ' + str(data['sapo']).strip() + ' ' + str(data['text']).strip()),
                               "target": "giao_duc"})

        train = open('data/kinhdoanh.json', 'r', encoding='utf-8-sig')
        for line in train:
            data = json.loads(line)
            train_data.append({"feature": (str(data['title']).strip() + ' ' + str(data['sapo']).strip() + ' ' + str(
                data['title']).strip() + ' ' + str(data['sapo']).strip() + ' ' + str(data['text']).strip()),
                               "target": "kinh_doanh"})

        train = open('data/phapluat-tintuc.json', 'r', encoding='utf-8-sig')
        for line in train:
            data = json.loads(line)
            train_data.append({"feature": (str(data['title']).strip() + ' ' + str(data['sapo']).strip() + ' ' + str(
                data['title']).strip() + ' ' + str(data['sapo']).strip() + ' ' + str(data['text']).strip()),
                               "target": "phap_luat_tin_tuc"})

        train = open('data/thegioi.json', 'r', encoding='utf-8-sig')
        for line in train:
            data = json.loads(line)
            train_data.append({"feature": (str(data['title']).strip() + ' ' + str(data['sapo']).strip() + ' ' + str(
                data['title']).strip() + ' ' + str(data['sapo']).strip() + ' ' + str(data['text']).strip()),
                               "target": "the_gioi"})

        train = open('data/thethao.json', 'r', encoding='utf-8-sig')
        for line in train:
            data = json.loads(line)
            train_data.append({"feature": (str(data['title']).strip() + ' ' + str(data['sapo']).strip() + ' ' + str(
                data['title']).strip() + ' ' + str(data['sapo']).strip() + ' ' + str(data['text']).strip()),
                               "target": "the_thao"})

        train = open('data/thoisu.json', 'r', encoding='utf-8-sig')
        for line in train:
            data = json.loads(line)
            train_data.append({"feature": (str(data['title']).strip() + ' ' + str(data['sapo']).strip() + ' ' + str(
                data['title']).strip() + ' ' + str(data['sapo']).strip() + ' ' + str(data['text']).strip()),
                               "target": "thoi_su"})

        train = open('data/tuvan.json', 'r', encoding='utf-8-sig')
        for line in train:
            data = json.loads(line)
            train_data.append({"feature": (str(data['title']).strip() + ' ' + str(data['sapo']).strip() + ' ' + str(
                data['title']).strip() + ' ' + str(data['sapo']).strip() + ' ' + str(data['text']).strip()),
                               "target": "tu_van"})

        df_train = pd.DataFrame(train_data)
        model = SVMModel()
        clf = model.clf.fit(df_train["feature"], df_train.target)
        filename = 'svm_model.sav'
        pickle.dump(clf, open(filename, 'wb'))

    def predict(self, input):
        test_data = []
        test_data.append({"feature": input, "target": "hoi_thoi_tiet"})
        df_test = pd.DataFrame(test_data)

        # init model naive bayes
        model = SVMModel()
        filename = 'svm_model.sav'
        clf = pickle.load(open(filename, 'rb'))
        predicted = clf.predict(df_test["feature"])
        Probability = clf.predict_proba(df_test["feature"])[0]

        # Print predicted result
        # print(predicted)
        # print(clf.predict_proba(df_test["feature"]))
        # print(clf.predict_proba(df_test["feature"]))

        return predicted, Probability


if __name__ == '__main__':
    tcp = TextClassificationPredict()
    result = tcp.predict('xin chào')
    print(result[0])
    print(result[1])
    # tcp.train_data()
