import sklearn_crfsuite
from sklearn_crfsuite import scorers, metrics
import re
import os
import pickle
import string
from underthesea import word_tokenize, sent_tokenize, chunk, pos_tag, ner


def read_dict(path):
    f = open(path, 'r', encoding='utf-8-sig')
    dic = []
    for line in f:
        dic.append(line.strip().lower())
    f.close()
    return dic


name_dict = read_dict('name_dict.txt')
dict_loc = read_dict('dic_location.txt')
vnese_dict = read_dict('vnese_dict.txt')


def cv_data(path):
    x = []
    files = os.listdir(path)
    for file in files:
        f = open(path + '/' + file, 'r', encoding='utf-8')
        corpus = f.readlines()
        f.close()
        lsen = []
        sen = 4
        while sen < len(corpus):
            tmp = [x for x in re.split("[\s]", corpus[sen].strip())]
            sen = sen + 1
            if len(tmp) == 5:
                tmp[1] = tmp[1].replace('NNP', 'Np')
                if tmp[3] == "B-PER":
                    while sen < len(corpus) and len([x for x in re.split("[\s]", corpus[sen].strip())]) == 5 and \
                            [x for x in re.split("[\s]", corpus[sen].strip())][3] == "I-PER":
                        tmp[0] = tmp[0] + "_" + [x for x in re.split("[\s]", corpus[sen].strip())][0]
                        sen = sen + 1
                lsen.append(tmp)
        x.append(lsen)
    return x


def is_location(word):
    location = ' '.join(word.lower().split('_'))
    if location in dict_loc:
        return True
    else:
        return False


# def is_vietnamese_word(word):
#    w = ' '.join(word.lower().split('_'))
#    if w in vnese_dict:
#        return True
#    else: return False

def is_lastname(word):
    name = re.split('[ \_]', word.lower())
    if name[-1] in name_dict:
        return True
    else:
        return False


def is_name(word):
    if '_' in word:
        broken = word.split('_')
        for i in range(len(broken)):
            if broken[i].islower(): return False
        return True
    else:
        return False


def is_mixcase(word):
    if len(word) > 2:
        if word[0].islower() and word[1].istitle(): return True
        return False
    return False


def word_shape(word):
    shape = ''
    for character in word:
        if character.istitle():
            shape += 'U'
        elif character.islower():
            shape += 'L'
        elif character.isdigit():
            shape += 'D'
        else:
            shape += character
    return shape


def word2feature(sent, i):
    word = sent[i][0]
    pos = sent[i][1]
    if word in string.punctuation:
        word = '<punct>'
    if word[0].isdigit():
        word = '<number>'
    features = {
        'w(0)': word,
        'w(0)[:1]': word[:1],
        'w(0)[:2]': word[:2],
        'w(0)[:3]': word[:3],
        'w(0)[:4]': word[:4],
        'w(0)[-1:]': word[-1:],
        'w(0)[-2:]': word[-2:],
        'w(0)[-3:]': word[-3:],
        'w(0)[-4:]': word[-4:],
        'word.islower': word.islower(),
        'word.lower': word.lower(),
        'isTitle': word[0].istitle(),
        'isNumber': word.isdigit(),
        'isUpper': word.isupper(),
        'isCapWithPeriod': word[0].istitle() and word[-1] == '.',
        'endsInDigit': word[-1].isdigit(),
        'containHyphen': '-' in word,
        'isDate': word[0].isdigit() and word[-1].isdigit() and '/' in word,
        'isCode': word[0].isdigit() and word[-1].istitle(),
        'isLastName': is_lastname(word),
        'isLocation': is_location(word),
               # 'isVietnameseWord' : is_vietnamese_word(word),
        'isName': is_name(word),
        'isMixCase': is_mixcase(word),
        'd&comma': word[0].isdigit() and word[-1].isdigit() and ',' in word,
        'd&period': word[0].isdigit() and word[-1].isdigit() and '.' in word,
        'wordShape': word_shape(word),
        'pos(0)': pos,
        'pos(0)[:2]': pos[:2],
    }

    if '_' in word:
        for index, _ in enumerate(word.split('_')):
            features.update({
                '{}thWord'.format(index): word.split('_')[index]
            })
    if (i > 0):
        prev_word = sent[i - 1][0]
        prev_pos = sent[i - 1][1]
        if prev_word in string.punctuation:
            prev_word = '<punct>'
        if prev_word[0].isdigit():
            prev_word = '<number>'
        features.update({
            'w(-1)': prev_word,
            'w(-1).isLower': prev_word.islower(),
            'w(-1).isUpper': prev_word.isupper(),
            'w(-1).lower': prev_word.lower(),
            'isTitle(-1)': prev_word[0].istitle(),
            'isNumber(-1)': prev_word.isdigit(),
            'isCapWithPeriod(-1)': prev_word[0].istitle() and prev_word[-1] == '.',

            'wordShape(-1)': word_shape(prev_word),
            'w(-1)+w(0)': prev_word + ' ' + word,
            'pos(-1)': prev_pos,
            'pos(-1)[:2]': prev_pos[:2],
        })
    else:
        features['BOS'] = True

    if i > 1:
        prev_2_word = sent[i - 2][0]
        prev_2_pos = sent[i - 2][1]
        if prev_2_word in string.punctuation:
            prev_2_word = '<punct>'
        if prev_2_word[0].isdigit():
            prev_2_word = '<number>'
            prev_2_pos = sent[i-2][1]
        #        prev_2_chunk = sent[i-2][2]
        features.update({
            'w(-2)': prev_2_word,
            'w(-2)+w(-1)': prev_2_word + ' ' + prev_word,
            'w(-2).isTitle()': prev_2_word[0].istitle(),
            'w(-2).isdigit': prev_2_word[0].isdigit(),
            'pos(-2)': prev_2_pos,
            'pos(-2)[:2]': prev_2_pos[:2],

        })

    if i < (len(sent) - 1):
        next_word = sent[i + 1][0]
        if next_word in string.punctuation:
            next_word = '<punct>'
        if next_word[0].isdigit():
            next_word = '<number>'
        next_pos = sent[i + 1][1]
        #        next_chunk = sent[i+1][2]
        features.update({
            'w(1)': next_word,
            'pos(1)': next_pos,
            #        'chunk(1)': next_chunk,
            #        'pos(0)+pos(1)': pos + ' ' + next_pos,
            #        'chunk(0)+chunk(1)': chunk +' '+ next_chunk,
            'w(1).lower': next_word.lower(),
            'w(-1).isLower': next_word.islower(),
            'w(-1).isUpper': next_word.isupper(),
            'isTitle(1)': next_word[0].istitle(),
            'isNumber(1)': next_word.isdigit(),
            'isCapWithPeriod(1)': next_word[0].istitle() and next_word[-1] == '.',
            'wordShape(1)': word_shape(next_word),
            'w(0)+w(1)': word + ' ' + next_word,
            'pos(+1)': next_pos,
            'pos(+1)[:2]': next_pos[:2],
        })
    else:
        features['EOS'] = True
    if i < (len(sent) - 2):
        next_2_word = sent[i + 2][0]
        if next_2_word in string.punctuation:
            next_2_word = '<punct>'
        if next_2_word[0].isdigit():
            next_2_word = '<number>'
        next_2_pos = sent[i + 2][1]
        #        next_2_chunk = sent[i+2][2]
        features.update({
            'w(2)': next_2_word,
            #        'pos(2)': next_2_pos,
            #        'chunk(2)':next_2_chunk,
            'w(1)+w(2)': next_word + ' ' + next_2_word,
            #        'pos(1)+pos(2)': next_pos +' '+ next_2_pos,
            #        'chunk(1)+chunk(2)': next_chunk+' '+next_2_chunk,
            'w(2).isTitle()': next_2_word[0].istitle(),
            'w(2).isdigit': next_2_word[0].isdigit(),
            'pos(+2)': next_2_pos,
            'pos(+2)[:2]': next_2_pos[:2],
        })
    return features


def get_features(sent):
    return [word2feature(sent, i) for i in range(len(sent))]


def get_labels(sent):
    return [label for token, _, _, label, _ in sent]


def get_tokens(sent):
    return [token for token, _, _, label, _ in sent]


def get_data():
    train_set = cv_data('train')
    test_set = cv_data('test')
    X_train = [get_features(s) for s in train_set]
    y_train = [get_labels(s) for s in train_set]
    X_test = [get_features(s) for s in test_set]
    y_test = [get_labels(s) for s in test_set]
    return X_train, y_train, X_test, y_test


def train_model():
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.05,
        c2=0.1,
        max_iterations=100,
    )
    X_train, y_train, X_test, y_test = get_data()
    crf.fit(X_train, y_train)
    labels = list(crf.classes_)
    labels.remove('O')
    filename = 'finalized_model.sav'
    pickle.dump(crf, open(filename, 'wb'))
    y_pred = crf.predict(X_test)
    print(metrics.flat_f1_score(y_test, y_pred,
                                 average='weighted', labels=labels))


if __name__ == '__main__':
    train_model()
    print('done')
