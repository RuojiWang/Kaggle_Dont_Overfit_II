#coding=utf-8
#这个比赛的做法参照Titanic吧，简单的说就是先补充缺失数据然后
#转换数据之间的类型，然后选择特征或者进行特征的组合，以对模型进行超参选择
#最后进行多个模型之间的融合并进行stacking咯
#这个版本目前就只是实现了对于单模型的特征选择，考虑到之后的stacking现在不能用lr吧。。
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectFromModel, RFE, VarianceThreshold

import hyperopt
from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK

#读取数据并合并咯
data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")
temp = data_train["target"]
data_all = pd.concat([data_train, data_test], axis=0)
X_all = data_all.drop(["id","target"], axis=1)

#查看数据是否存在缺失咯
#下面的代码可以确定所有的数据非缺失，所以不用进行补充了
"""
feature_list = [str(i) for i in range(0, 300)]
#print(feature_list)
for feature in feature_list:
    if (not data_all[data_all[feature].isnull().values==True].empty):
        print(data_all[data_all[feature].isnull().values==True])
"""

#接下来是将特征进行缩放到一定的区间
X_all_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X_all), columns = X_all.columns)
X_all_scaled = pd.DataFrame(data = X_all_scaled, index = X_all.index, columns = X_all_scaled.columns.values)
X_train_scaled = X_all_scaled[:len(data_train)]
X_test_scaled = X_all_scaled[len(data_train):]
Y_train = temp

#现在开始准备选择或者创建特征了吧
#看了下面这两个文档，感觉对于机器学习的特征选择有了一定的了解
#https://blog.csdn.net/xmuecor/article/details/53432157
#https://blog.csdn.net/u013710265/article/details/71796758
#目前为止我还是打算采用SelectFromModel并将逻辑回归作为该函数的分类器
#这样做的好处主要在于：将模型调参和特征选择作为一个整体进行贝叶斯优化
#我个人感觉这做法比较简洁又优雅，应该能够取得比较靠谱的结果吧
#但是好像遇到了另外一个问题呀，就是这样选择的特征参与stacking是否合适？
#或者是说多对几个模型进行类似的操作，进行stacking应该效果会不错的吧
#我在想这里是不是可以不用逻辑回归，等到stacking的时候再使用逻辑回归咯
#我查了一下贝叶斯优化并没有什么技术使用的前提
#下面就开始对每个模型进行超惨搜索了吧
def save_inter_params(trials, space_nodes, best_nodes, title):
 
    files = open(str(title+"_intermediate_parameters.pickle"), "wb")
    pickle.dump([trials, space_nodes, best_nodes], files)
    files.close()

def load_inter_params(title):
  
    files = open(str(title+"_intermediate_parameters.pickle"), "rb")
    trials, space_nodes, best_nodes = pickle.load(files)
    files.close()
    
    return trials, space_nodes ,best_nodes

def save_stacked_dataset(stacked_train, stacked_test, title):
    
    files = open(str(title+"_stacked_dataset.pickle"), "wb")
    pickle.dump([stacked_train, stacked_test], files)
    files.close()

def load_stacked_dataset(title):
    
    files = open(str(title+"_stacked_dataset.pickle"), "rb")
    stacked_train, stacked_test = pickle.load(files)
    files.close()
    
    return stacked_train, stacked_test

def save_best_model(best_model, title):
    
    files = open(str(title+"_best_model.pickle"), "wb")
    pickle.dump(best_model, files)
    files.close()

def load_best_model(title_and_nodes):
    
    files = open(str(title_and_nodes+"_best_model.pickle"), "rb")
    best_model = pickle.load(files)
    files.close()
    
    return best_model

"""
def lr_f(params):
    
    clf = LogisticRegression(penalty=params["penalty"], C=params["C"],
                             fit_intercept=params["fit_intercept"],
                             class_weight=params["class_weight"])
    clf.fit(X_train_scaled, Y_train)
    model = SelectFromModel(clf, threshold=params["feature_num"]/len(X_train_scaled), prefit=True)
    X_train_scaled_new = model.transform(X_train_scaled)
    print(X_train_scaled.columns[model.get_support()])
    #X_train_scaled_new = X_train_scaled_new
    #现在遇到的问题是用什么作为评价准则
    #是采用交叉验证还是采用单模型多次训练呢
    #神经网络之所以采用后者是因为只能评价超参
    #而同一超参训练出的神经网络有一定差异
    #且K折超参搜索K取小了会影响网络计算结果，取大了计算量过大
    model = clf
    skf = StratifiedKFold(Y_train, n_folds=10, shuffle=True, random_state=None)
    metric = cross_val_score(model, X_train_scaled_new, Y_train.values, cv=skf, scoring="roc_auc").mean()
    return metric
"""

def lr_f(params):
    
    clf = LogisticRegression(penalty=params["penalty"], C=params["C"],
                             fit_intercept=params["fit_intercept"],
                             class_weight=params["class_weight"])
    selector = RFE(clf, n_features_to_select=params["feature_num"])
    X_train_scaled_new = selector.fit_transform(X_train_scaled, Y_train)
    skf = StratifiedKFold(Y_train, n_folds=10, shuffle=True, random_state=None)
    metric = cross_val_score(clf, X_train_scaled_new, Y_train, cv=skf, scoring="roc_auc").mean()
    return -metric
    
def parse_lr_nodes(trials, space_nodes):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    trials_list.sort(key=lambda item: item['result']['loss'])
    
    best_nodes = {}
    best_nodes["title"] = space_nodes["title"][trials_list[0]["misc"]["vals"]["title"][0]]
    best_nodes["path"] = space_nodes["path"][trials_list[0]["misc"]["vals"]["path"][0]]
    best_nodes["mean"] = space_nodes["mean"][trials_list[0]["misc"]["vals"]["mean"][0]]
    best_nodes["std"] = space_nodes["std"][trials_list[0]["misc"]["vals"]["std"][0]]
    
    best_nodes["feature_num"] = space_nodes["feature_num"][trials_list[0]["misc"]["vals"]["feature_num"][0]]
    best_nodes["penalty"] = space_nodes["penalty"][trials_list[0]["misc"]["vals"]["penalty"][0]]
    best_nodes["C"] = space_nodes["C"][trials_list[0]["misc"]["vals"]["C"][0]]
    best_nodes["fit_intercept"] = space_nodes["fit_intercept"][trials_list[0]["misc"]["vals"]["fit_intercept"][0]]
    best_nodes["class_weight"] = space_nodes["class_weight"][trials_list[0]["misc"]["vals"]["class_weight"][0]]
    
    return best_nodes

def train_lr_model(nodes, X_train_scaled, Y_train):
    
    return 

lr_space = {"title":hp.choice("title", ["stacked_don't_overfit!_II"]),
            "path":hp.choice("path", ["Don't_Overfit!_II_Prediction.csv"]),
            "mean":hp.choice("mean", [0]),
            "std":hp.choice("std", [0]),
            #这个linspace返回的类型是ndarray类型的数据
            "feature_num":hp.choice("feature_num", np.linspace(1,300,300)),
            "penalty":hp.choice("penalty", ["l1", "l2"]),
            "C":hp.choice("C", np.logspace(-3, 5, 100)),
            "fit_intercept":hp.choice("fit_intercept", ["True", "False"]),
            "class_weight":hp.choice("class_weight", ["balanced", None])
            }

lr_space_nodes = {"title":["stacked_don't_overfit!_II"],
                  "path":["Don't_Overfit!_II_Prediction.csv"],
                  "mean":[0],
                  "std":[0],
                  "feature_num":np.linspace(1,300,300),
                  "penalty":["l1", "l2"],
                  "C":np.logspace(-3, 5, 100),
                  "fit_intercept":["True", "False"],
                  "class_weight":["balanced", None]
                 }

trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
best = fmin(lr_f, lr_space, algo=tpe.suggest, max_evals=8000)
best_nodes = parse_lr_nodes(trials, lr_space_nodes)
save_inter_params(trials, lr_space_nodes, best_nodes, "lr_don't_overfit!_II")

