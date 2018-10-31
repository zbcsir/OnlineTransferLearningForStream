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
from sklearn.metrics import hinge_loss
from scipy.sparse import csr_matrix
import numpy as np

w_s = 0.5
w_t = 0.5
num_source = 7
C = 5

topics = []
# sample_cate = ['comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
#                'comp.windows.x', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space']
sample_cate = ['comp.os.ms-windows.misc', 'sci.crypt',
               'comp.sys.ibm.pc.hardware', 'sci.electronics',
               'comp.sys.mac.hardware', 'sci.med',
               'comp.windows.x', 'sci.space']
# num_source = len(sample_cate)-1
domain_cate = []
domain_train_len = []
domain_test_len = []
domain_cate = []
domain1_cate = ['comp.os.ms-windows.misc', 'sci.crypt']
domain2_cate = ['comp.sys.ibm.pc.hardware', 'sci.electronics']
domain3_cate = ['comp.sys.mac.hardware', 'sci.med']
domain4_cate = ['comp.windows.x', 'sci.space']
domain_cate.append(domain1_cate)
domain_cate.append(domain2_cate)
domain_cate.append(domain3_cate)
domain_cate.append(domain4_cate)

train_samples = fetch_20newsgroups(data_home='Z:\\Study\\小论文\\paper3\\somedata\\20Newsgroup',
                                   subset='train',
                                   categories=sample_cate,
                                   # shuffle=False,
                                   remove=('headers', 'footers', 'quotes'))
test_samples = fetch_20newsgroups(data_home='Z:\\Study\\小论文\\paper3\\somedata\\20Newsgroup',
                                  subset='test',
                                  categories=sample_cate,
                                  # shuffle=False,
                                  remove=('headers', 'footers', 'quotes'))

print("classes unique : " + str(set(train_samples.target)))
print("classes nounique :" + str(train_samples.target[0:10]))
len_train = len(train_samples.target)
len_test = len(test_samples.target)
print("len_train" + str(len_train))


vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
# vectorizer = TfidfTransformer()
train_samples_vector = vectorizer.fit_transform(train_samples.data)
print(train_samples_vector.shape)
print("type : " + str(type(train_samples_vector)))
# tdTrain_vector = vectorizer.fit_transform(targetDomain_train.data)

test_samples_vector = vectorizer.transform(test_samples.data)
print(test_samples_vector.shape)

print("vector sample 00 : " + str(train_samples_vector.toarray().shape))
# print("vector sample 01 : " + str(train_samples_vector[1][0]))

label_domain1 = []
label_domain2 = []
label_domain3 = []
vector_domain1 = []
vector_domain2 = []
vector_domain3 = []

# for i in range(num_source):
#     t_l = []
#     t_v = []
#     label_by_domain.append(t_l)
#     vector_by_domain.append(t_v)

# for i in range(train_samples_vector.shape[0]):
#     domain_id = int(train_samples.target[i]/2)
#     if domain_id == 0:
#         label_domain1.append(2*(train_samples.target[i] % 2 - 1/2))
#         vector_domain1.append(np.array(train_samples_vector[i]))
#         print(np.asarray(train_samples_vector[i]).shape)
#     if domain_id == 1:
#         label_domain2.append(2 * (train_samples.target[i] % 2 - 1/2))
#         vector_domain2.append(train_samples_vector[i])
#     if domain_id == 2:
#         label_domain3.append(2 * (train_samples.target[i] % 2 - 1/2))
#         vector_domain3.append(train_samples_vector[i])
#
# print("vector_domain1")
# print(np.asmatrix(label_domain1[1]).shape)
# print(csr_matrix(np.array(vector_domain1)).shape)
# print("label_domain1")
# print(np.asmatrix(label_domain1).shape)
# print("vector_domain2")
# print(np.asmatrix(vector_domain2).shape)
# print("label_domain2")
# print(np.asmatrix(label_domain2).shape)
# print("vector_domain3")
# print(np.asmatrix(vector_domain3).shape)
# print("label_domain3")
# print(np.asmatrix(label_domain3).shape)

# for i in range(num_source):
#     domain_train_tmp = fetch_20newsgroups(data_home='Z:\\Study\\小论文\\paper3\\somedata\\20Newsgroup',
#                                           subset='train',
#                                           categories=domain_cate[i],
#                                           shuffle=False,
#                                           remove=('headers', 'footers', 'quotes'))
#     domain_test_tmp = fetch_20newsgroups(data_home='Z:\\Study\\小论文\\paper3\\somedata\\20Newsgroup',
#                                          subset='test',
#                                          categories=domain_cate[i],
#                                          shuffle=False,
#                                          remove=('headers', 'footers', 'quotes'))
#     len_train = len(domain_train_tmp.data)
#     len_test = len(domain_test_tmp.data)
#     domain_train_len.append(len_train)
#     domain_test_len.append(len_test)
#     print("train length : " + str(len_train))
#     print("test length : " + str(len_test))

domain_train = []
domain_test = []
domain_train_vector = []
domain_test_vector = []
f = []
index_train = 0
index_test = 0
for i in range(num_source):
    domain_train_tmp = fetch_20newsgroups(subset='train',
                                          categories=sample_cate,
                                          shuffle=False,
                                          remove=('headers', 'footers', 'quotes'))
    domain_test_tmp = fetch_20newsgroups(subset='test',
                                         categories=sample_cate,
                                         shuffle=False,
                                         remove=('headers', 'footers', 'quotes'))
    # domain_train.append(domain_train_tmp)
    # domain_test.append(domain_train_tmp)
    # 如果实例等于当前子话题的类别，将其target设为1，否则为-1
    num_train_pos = 0
    num_train_neg = 0
    num_test_pos = 0
    num_test_neg = 0
    for j in range(len_train):
        if domain_train_tmp.target[j] == i:
            domain_train_tmp.target[j] = 1
            num_train_pos = num_train_pos + 1
        else:
            domain_train_tmp.target[j] = 0
            num_train_neg = num_train_neg + 1
    for k in range(len_test):
        if domain_test_tmp.target[k] == i:
            domain_train_tmp.target[k] = 1
            num_test_pos = num_test_pos + 1
        else:
            domain_train_tmp.target[k] = 0
            num_test_neg = num_test_neg + 1

    print("num_train_pos : " + str(num_train_pos))
    print("num_train_neg : " + str(num_train_neg))
    print("num_test_pos  : " + str(num_test_pos))
    print("num_test_neg  : " + str(num_test_neg))

    domain_train_vector_tmp = vectorizer.fit_transform(domain_train_tmp.data)
    domain_test_vector_tmp = vectorizer.transform(domain_test_tmp.data)

    # len_tmp_train = domain_train_len[i]
    # len_tmp_test = domain_test_len[i]
    # domain_train_vector_tmp = train_samples_vector[index_train:index_train+len_tmp_train]
    print(np.asmatrix(domain_train_vector_tmp).shape)
    # domain_test_vector_tmp = vecto    rizer.transform(domain_test_tmp.data)
    # domain_test_vector_tmp = test_samples_vector[index_test:index_test+len_tmp_test]
    print(domain_test_vector_tmp.shape)
    # domain_train_vector.append(domain_train_vector_tmp)
    # domain_test_vector.append(domain_test_vector_tmp)
    print("classes : " + str(set(domain_train_tmp.target)))
    # 建立分类器
    # lg = LogisticRegression(C=4, dual=True)
    # lg.fit(domain_train_vector_tmp, domain_train_tmp.target)
    # y_test = lg.predict(domain_test_vector_tmp)

    svm = SVC(kernel='linear', C=1.5)
    # svm.fit(domain_train_vector_tmp, domain_train_tmp.target)
    svm.fit(domain_train_vector_tmp, domain_train_tmp.target)
    y_test = svm.predict(domain_test_vector_tmp)
    print("classes predict : " + str(set(y_test)))
    print("Accuracy : " + str(metrics.accuracy_score(domain_test_tmp.target, y_test)))
    print("F1 Score : " + str(metrics.f1_score(domain_test_tmp.target, y_test)))
    f.append(svm)
    # index_train = index_train + len_tmp_train
    # index_test = index_test + len_tmp_test

print("======================================建立源域分类器结束======================================")

print("======================================for目标域======================================")
target_cate = ['comp.windows.x', 'sci.space']
target_train = fetch_20newsgroups(
    subset='train', categories=target_cate, shuffle=True,
    random_state=0, remove=('headers', 'footers', 'quotes'))
target_test = fetch_20newsgroups(
    subset='test', categories=target_cate, shuffle=True,
    random_state=0, remove=('headers', 'footers', 'quotes'))
# vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
# target_train_vector = vectorizer.fit_transform(target_train.data)
# target_test_vector = vectorizer.transform(target_test.data)
len_target_train = domain_train_len[num_source]
len_target_test = domain_test_len[num_source]
target_train_vector = train_samples_vector[index_train:index_train+len_target_train]
target_test_vector = train_samples_vector[index_test:index_test+len_target_test]

train_size = target_train_vector.shape[0]

pa = PassiveAggressiveClassifier(random_state=0, loss='hinge', C=5)
classes = set(target_train.target)
batch_size = 1
w_u = []
p_u = []
beta1 = 0.9
beta2 = 0.9

for s in range(num_source):
    tmp = 1/(2*num_source)
    w_u.append(tmp)
    p_u.append(tmp)

for i in range(0, train_size-batch_size, batch_size):
    print("round " + str(i))
    train_inst_tmp = []
    train_inst_vector = []
    class_tmp = []
    # 归一化
    p_v = w_t/(w_s + w_t)
    for k in range(num_source):
        p_u[k] = w_u[k]/(w_s + w_t)
    for j in range(batch_size):
        class_tmp.append(target_train.target[i+j])
    # pa.partial_fit(target_train_vector[i:i+batch_size], class_tmp, list(classes))
    pLabelInTarget = pa.predict(target_train_vector[i:i+batch_size])
    print("====================evaluation====================")
    print("accuracy = " + str(metrics.accuracy_score(target_test.target, pLabelInTarget)))
    print("recall = " + str(metrics.recall_score(target_test.target, pLabelInTarget)))

    print("pLabelInTarget" + str(pLabelInTarget))
    # 将{0,1}类标空间转化为{-1,1}
    pLabel = pLabelInTarget
    if pLabel == 0:
        pLabel = -1
    pTrue = 1
    if target_test.target[i] == 0:
        pTrue = -1

    pred_s = 0
    pSource_res = []
    for s in range(num_source):
        pSource_tmp = f[s].predict(target_test_vector[i])
        if pSource_tmp == 0:
            pSource_tmp = -1
        else:
            pSource_tmp = 1
        print("pSource_tmp : " + str(pSource_tmp))
        pSource_res.append(pSource_tmp)
        pred_s = pred_s + w_u[s]*pSource_tmp
    pred_s = pred_s * pTrue
    sign_s = 0
    if pred_s < 0:
        sign_s = 1
    sign_t = 0
    pred_t = pTrue * pLabel
    if pred_t < 0:
        sign_t = 1
    w_s = w_s * beta1 ** sign_s
    w_t = w_t * beta1 ** sign_t

    sum_u = 0
    for s in range(num_source):
        tmp = pTrue * pSource_res[s]
        sign_tmp = 0
        if tmp < 0:
            sign_tmp = 1
        w_u[s] = w_u[s] * beta2 ** sign_tmp
        sum_u = sum_u + w_u[s]

    for s in range(num_source):
        w_u[s] = w_u[s] * w_s / sum_u

    # loss = max(0, 1 - yf(x))
    loss = 0
    if pLabel*pTrue < 0:
        loss = 2
    # loss = hinge_loss(pTrue, pLabel)
    if loss > 0:
        pa.partial_fit(target_train_vector[i:i + batch_size],
                       class_tmp, list(classes))

    # 线性组合
    f_final = 0
    for s in range(num_source):
        f_final = f_final + p_u[s]*pSource_res[s]
    f_final = f_final + p_v*pLabel

    # i = i+batch_size
    if i+batch_size >= train_size:
        break

