#coding=utf-8
#这个版本准备实现多个模型版本的超参搜索咯
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

from xgboost import XGBClassifier

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

#lr相关的部分咯
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

#xgboost相关的部分咯
xgb_space = {"title":hp.choice("title", ["stacked_don't_overfit!_II"]),
             "path":hp.choice("path", ["Don't_Overfit!_II_Prediction.csv"]),
             "mean":hp.choice("mean", [0]),
             "std":hp.choice("std", [0]),
             "feature_num":hp.choice("feature_num", np.linspace(1,300,300)),
             "gamma":hp.choice("gamma", [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.3, 0.5, 0.7, 0.9]),
             "max_depth":hp.choice("max_depth", [3, 5, 7, 9, 12, 15, 17, 25]),
             "learning_rate":hp.choice("learning_rate", np.linspace(0.01, 1.00, 100)),
             "min_child_weight":hp.choice("min_child_weight", [1, 3, 5, 7, 9]),
             "subsample":hp.choice("subsample", [0.6, 0.7, 0.8, 0.9, 1.0]),
             "colsample_bytree":hp.choice("colsample_bytree", [0.6, 0.7, 0.8, 0.9, 1.0]),
             "reg_alpha":hp.choice("reg_alpha", [0.0, 0.1, 0.5, 1.0]),
             "reg_lambda":hp.choice("reg_lambda", [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 1.0]),
             "n_estimators":hp.choice("n_estimators", np.linspace(50, 500, 45)),
             }

xgb_space_nodes = {"title":["stacked_don't_overfit!_II"],
                   "path":["Don't_Overfit!_II_Prediction.csv"],
                   "mean":[0],
                   "std":[0],
                  }

"""
trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
best = fmin(lr_f, lr_space, algo=tpe.suggest, max_evals=8000)
best_nodes = parse_lr_nodes(trials, lr_space_nodes)
save_inter_params(trials, lr_space_nodes, best_nodes, "lr_don't_overfit!_II")
"""

gxb = XGBClassifier()
