# -*- coding: utf-8 -*-
'''
论文《Online Transfer Learning by Leveraging Multiple Source Domain》算法
'''

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import metrics
import gensim.models.word2vec
from sklearn.metrics import hinge_loss
from scipy.sparse import csr_matrix
import numpy as np
import random
from Tools import sign


w_s = 0.5
w_t = 0.5
num_source = 3
C = 5
predicted = 0

domain_cate = []
domain_train_len = []
domain_test_len = []
domain_all_cate = ['comp.os.ms-windows.misc', 'sci.crypt', 'comp.sys.ibm.pc.hardware', 'sci.electronics',
                   'comp.sys.mac.hardware', 'sci.med', 'comp.windows.x', 'sci.space']
domain_cate = []
domain1_cate = ['comp.os.ms-windows.misc', 'sci.crypt']
domain2_cate = ['comp.sys.ibm.pc.hardware', 'sci.electronics']
domain3_cate = ['comp.sys.mac.hardware', 'sci.med']
domain4_cate = ['comp.windows.x', 'sci.space']
domain_cate.append(domain1_cate)
domain_cate.append(domain2_cate)
domain_cate.append(domain3_cate)
domain_cate.append(domain4_cate)

vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

domain_all_train = fetch_20newsgroups(data_home='Z:\\Study\\小论文\\paper3\\somedata\\20Newsgroup',
                                      categories=domain_all_cate,
                                      shuffle=True, random_state=0,
                                      remove=('headers', 'footers', 'quotes'))
# domain_all_test = fetch_20newsgroups(data_home='Z:\\Study\\小论文\\paper3\\somedata\\20Newsgroup',
#                                      subset='test', categories=domain_all_cate,
#                                      shuffle=True, random_state=0,
#                                      remove=('headers', 'footers', 'quotes'))
domain_all_train_vector = vectorizer.fit_transform(domain_all_train.data)
print("domain_all_train_vector shape : " + str(domain_all_train_vector.shape))
# domain_all_test_vector = vectorizer.transform(domain_all_test.data)
# print("domain_all_test_vector shape : " + str(domain_all_test_vector.shape))
vocabulary_list = vectorizer.vocabulary_
print("vocabulary_list len : " + str(len(vocabulary_list)))

domain_train = []
domain_test = []
domain_train_vector = []
domain_test_vector = []
f = []
index_train = 0
index_test = 0
data_train = []
data_test = []
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, vocabulary=vocabulary_list)
for i in range(num_source):
    domain_train_tmp = fetch_20newsgroups(data_home='Z:\\Study\\小论文\\paper3\\somedata\\20Newsgroup',
                                          subset='train',
                                          categories=domain_cate[i],
                                          shuffle=True, random_state=0,
                                          remove=('headers', 'footers', 'quotes'))
    domain_test_tmp = fetch_20newsgroups(data_home='Z:\\Study\\小论文\\paper3\\somedata\\20Newsgroup',
                                         subset='test',
                                         categories=domain_cate[i],
                                         shuffle=True, random_state=0,
                                         remove=('headers', 'footers', 'quotes'))
    data_train.append(domain_train_tmp)
    data_test.append(domain_test_tmp)

    domain_train_vector_tmp = vectorizer.fit_transform(domain_train_tmp.data)
    print("domain_train_vector_tmp shape : " + str(domain_train_vector_tmp.shape))
    domain_test_vector_tmp = vectorizer.transform(domain_test_tmp.data)
    print("domain_test_vector_tmp shape : " + str(domain_test_vector_tmp.shape))

    # 建立分类器
    lg = LogisticRegression(C=4, dual=True)
    lg.fit(domain_train_vector_tmp, domain_train_tmp.target)
    y_test = lg.predict(domain_test_vector_tmp)

    print("F1 Score : " + str(metrics.f1_score(domain_test_tmp.target, y_test)))
    f.append(lg)

print("======================================建立源域分类器结束======================================")

print("======================================for目标域======================================")
target_cate = ['comp.windows.x', 'sci.space']
target_train = fetch_20newsgroups(
    categories=target_cate, shuffle=True,
    random_state=0, remove=('headers', 'footers', 'quotes'))
# target_test = fetch_20newsgroups(
#     subset='test', categories=target_cate, shuffle=True,
#     random_state=0, remove=('headers', 'footers', 'quotes'))

target_train_vector = vectorizer.fit_transform(target_train.data)
# target_test_vector = vectorizer.transform(target_test.data)
print("target_train_vector shape : " + str(target_train_vector.shape))
# print("target_test_vector shape : " + str(target_test_vector.shape))

train_size = target_train_vector.shape[0]

pa = PassiveAggressiveClassifier(random_state=0, loss='hinge', C=5)
classes = set(target_train.target)
batch_size = 1
w_u = []
p_u = []
beta1 = 0.9
beta2 = 0.95

# num_errors用于统计组合多个源域后的预测错误数量
# num_errors_pa用于统计单纯在目标域增量建立分类器时的预测错误数量
num_errors = 0
num_errors_pa = 0

for s in range(num_source):
    tmp = 1/(2*num_source)
    w_u.append(tmp)
    p_u.append(tmp)

pa.partial_fit(target_train_vector[0:100], target_train.target[0:100], list(classes))

for i in range(100, train_size-batch_size, batch_size):
    target_inst_vector = vectorizer.transform(target_train)
    print("===================round " + str(i) + "错误率=====================")
    if i+batch_size >= train_size:
        break
    train_inst_tmp = []
    train_inst_vector = []
    class_tmp = []
    # 归一化
    sum_us = sum(w_u)
    p_v = w_t/(sum_us + w_t)
    for k in range(num_source):
        p_u[k] = w_u[k]/(sum_us + w_t)

    print("p_s : " + str(p_u) + "    p_v : " + str(p_v))

    for j in range(batch_size):
        class_tmp.append(target_train.target[i+j])

    pTrue = 1
    if target_train.target[i] == 0:
        pTrue = -1

    # 源域分类器分别预测并加权求和
    pred_s = 0
    pSource_res = []
    for s in range(num_source):
        pSource_tmp = f[s].predict(target_train_vector[i])
        if pSource_tmp == 0:
            pSource_tmp = -1
        else:
            pSource_tmp = 1
        # print("pSource_tmp : " + str(pSource_tmp))
        print("Source " + str(s) + " ==> " + "预测："+str(pSource_tmp) +
              " === " + "真实：" + str(pTrue))
        pSource_res.append(pSource_tmp)
        pred_s = pred_s + w_u[s]*pSource_tmp

    # 目标域分类器预测
    pLabelInTarget = pa.predict(target_train_vector[i])
    # predTest = pa.predict(target_test_vector)
    # print("====================evaluation====================")
    # print("target_test.target shape : " + str(target_test.target.shape))
    # print("pLabelInTarget shape : " + str(pLabelInTarget.shape))
    #
    # print("accuracy = " + str(metrics.accuracy_score(target_test.target, predTest)))
    # print("recall = " + str(metrics.recall_score(target_test.target, predTest)))

    # print("pLabelInTarget" + str(pLabelInTarget))

    # 将{0,1}类标空间转化为{-1,1}
    pLabel = pLabelInTarget
    if pLabel == 0:
        pLabel = -1
    else:
        pLabel = 1

    print("PA pLabel : " + str(pLabel) + "   True : " + str(pTrue))
    if pLabel != pTrue:
        num_errors_pa = num_errors_pa + 1

    # 线性组合
    f_final = 0
    for s in range(num_source):
        f_final = f_final + p_u[s]*pSource_res[s]
    predicted = sign(f_final + p_v*pLabel)

    print("OTLMS pLabel : " + str(predicted) + "   True : " + str(pTrue))

    # 判断f_t预测是否正确
    if predicted == 0:
        predicted = random.randint(0, 1)
        predicted = 2*(predicted - 1/2)
    if predicted != pTrue:
        num_errors = num_errors + 1

    sign_s = pred_s * pTrue
    if pred_s < 0:
        w_s = w_s * beta1

    sign_t = pTrue * pLabel
    if sign_t < 0:
        w_t = w_t * beta1
    # w_s = w_s * (beta1 ** sign_s)
    # w_t = w_t * (beta1 ** sign_t)

    # sum_u = 0
    for s in range(num_source):
        tmp = pTrue * pSource_res[s]
        if tmp < 0:
            w_u[s] = w_u[s] * beta2
        # w_u[s] = w_u[s] * (beta2 ** sign_tmp)
        # sum_u = sum_u + w_u[s]

    # for s in range(num_source):
    #     w_u[s] = w_u[s] * w_s / sum_u

    # loss = max(0, 1 - yf(x))
    loss = 0
    if pLabel*pTrue < 0:
        pa.partial_fit(target_train_vector[i], class_tmp, list(classes))

    print("只使用目标域分类器时错误率：" + str(num_errors_pa))
    print("结合源域分类器后错误率   ：" + str(num_errors))
    print()
