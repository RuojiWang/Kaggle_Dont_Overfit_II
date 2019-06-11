#coding=utf-8
#这个版本主要是复盘一下这两周参加比赛的一个新的，感觉还是难以释怀这个成绩
#这么有意思的比赛，然后我自己还挺有信心的，最后居然是这个成绩，主要还是时间不足吧
#最后test5提交的结果炸了呀，我个人感觉出现问题的地方应该是特征的选择和stacking模型的选择
#先实验一下选择出的特征和网络上的kernel选择出来的特征是否一样
#然我不淡定的就是有一个人用逻辑回归进行了简单的网格搜索然后就提交结果，居然也有0.83的成绩
#所以以后每次都先做个单模型，典型的就是逻辑回归和catboost试试水在说其他的吧。。
#https://www.kaggle.com/praxitelisk/don-t-overfit-ii-eda-ml/notebook
#对比了一下这个kernel我发现自己没对数据的分布做统计，采用相关性做的特征选择，因为他没有缺失值，所以我做的比较马虎
#这个人的特征选择也是通过rf选择的，我觉得很奇怪的是他选出的特征都和我不一样。
#https://www.kaggle.com/mitjasha/don-t-overfit-2-linear-models-with-hyperopt
#这个kernel为了防止过拟合故意增加了噪声，用的lr eli5和相关系数RFCV（采用lr模型）选择的特征，用的贝叶斯优化选择超参。。
#https://www.kaggle.com/iavinas/simple-short-solution-don-t-overfit-0-848
#这个kernel真的让我觉得耻辱，这么简单的lr单模型居然可以取得这么好的效果，属实带秀熬，我觉得很奇怪只用了lr连特征选择都没有熬，可能只是sample code吧
#还有我发现了一个非常神奇的现象就是别人提交的都是predict_proba(test)，原来在上面还可以采用提交概率的方式么，我还以为必须是严格的0或者1呢
#最后我个人倾向于认为不同的分类器应该自己选择自己的特征，而不是应该在一开始统一由某个分类器全部选择，至于为什么是12可能就是试出来的吧
#以后先单模型做到最佳就不会遇到类似的问题了，我之后将会继续使用现在的代码如果不是上述原因，那么我之后的比赛可能还会遇到这种情况。
import pickle
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, ElasticNet, Lasso
from sklearn.feature_selection import SelectFromModel, RFE, VarianceThreshold
from sklearn.cross_validation import cross_val_score, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest, AdaBoostClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, KFold

import hyperopt
from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK

from sklearn import svm
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from mlxtend.classifier import StackingClassifier, StackingCVClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import eli5
from eli5.sklearn import PermutationImportance

from catboost import CatBoostClassifier

from lightgbm.sklearn import LGBMClassifier
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
    
    print("title", params["title"])
    print("path", params["path"])
    print("mean", params["mean"])
    print("std", params["std"])
    print("feature_num", params["feature_num"])
    print("penalty", params["penalty"]) 
    print("C", params["C"])
    print("fit_intercept", params["fit_intercept"])
    print("class_weight", params["class_weight"])

    #设置一个random_state值以防分类器初始参数有些差异
    #但是好像据说sklearn的逻辑回归模型这个设置是无效的
    #怪不得我看别人的代码都会设置这个东西，原来是这个意思呀
    clf = LogisticRegression(penalty=params["penalty"], 
                             C=params["C"],
                             fit_intercept=params["fit_intercept"],
                             class_weight=params["class_weight"],
                             random_state=42)
    
    selector = RFE(clf, n_features_to_select=params["feature_num"])
    X_train_scaled_new = selector.fit_transform(X_train_scaled, Y_train)
    skf = StratifiedKFold(Y_train, n_folds=25, shuffle=True, random_state=42)
    metric = cross_val_score(clf, X_train_scaled_new, Y_train, cv=skf, scoring="roc_auc").mean()
    
    print(metric)
    print()
    return -metric
    
def parse_lr_nodes(trials, space_nodes):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    trials_list.sort(key=lambda item:(item['result']['loss'], item['misc']['vals']['feature_num']))
    
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

lr_space = {"title":hp.choice("title", ["stacked_don't_overfit!_II"]),
            "path":hp.choice("path", ["Don't_Overfit!_II_Prediction.csv"]),
            "mean":hp.choice("mean", [0]),
            "std":hp.choice("std", [0]),
            #这个linspace返回的类型是ndarray类型的数据
            "feature_num":hp.choice("feature_num", np.linspace(1,300,300)),
            #"penalty":hp.choice("penalty", ["l1", "l2"]),
            "penalty":hp.choice("penalty", ["l1"]),
            "C":hp.choice("C", np.logspace(-3, 5, 40)),
            #"C":hp.choice("C", np.logspace(0, 5, 24)),
            "fit_intercept":hp.choice("fit_intercept", ["True", "False"]),
            "class_weight":hp.choice("class_weight", ["balanced", None])
            }

lr_space_nodes = {"title":["stacked_don't_overfit!_II"],
                  "path":["Don't_Overfit!_II_Prediction.csv"],
                  "mean":[0],
                  "std":[0],
                  "feature_num":np.linspace(1,300,300),
                  #"penalty":["l1", "l2"],
                  "penalty":["l1"],
                  "C":np.logspace(-3, 5, 40),
                  #"C":np.logspace(0, 5, 24),
                  "fit_intercept":["True", "False"],
                  "class_weight":["balanced", None]
                 }

#几乎所有传统模型的random_state设置了具体的值，且超参一样那么结果完全一样
#如果设置的是None则每次的训练结果有所差异，这就是为什么那些代码都会设置random_state

#RFE的稳定性很大程度上取决于迭代时，底层用的哪种模型。
#比如RFE采用的是普通的回归（LR），没有经过正则化的回归是不稳定的，那么RFE就是不稳定的。
#假如采用的是Lasso/Ridge，正则化的回归是稳定的，那么RFE就是稳定的。
#所以底层是不是应该换一个分类器，然后增加Adaboost进行超参搜索咯？但是先按照这个做一个版本的代码吧
#sklearn的逻辑回归是带有正则化的，所以是稳定的，只是在这个问题最好使用l1正则化选择特征咯
def train_lr_model(best_nodes, X_train_scaled, Y_train):
    
    clf = LogisticRegression(penalty=best_nodes["penalty"], 
                             C=best_nodes["C"],
                             fit_intercept=best_nodes["fit_intercept"],
                             class_weight=best_nodes["class_weight"],
                             random_state=42)
    
    selector = RFE(clf, n_features_to_select=best_nodes["feature_num"])
    X_train_scaled_new = selector.fit_transform(X_train_scaled, Y_train)
    feature_names = selector.get_support(1)
    X_test_scaled_new = X_test_scaled[X_test_scaled.columns[feature_names]]
    
    clf.fit(X_train_scaled_new, Y_train)
    print(clf.score(X_train_scaled_new, Y_train))
    
    return clf, X_train_scaled_new, X_test_scaled_new.values

def create_lr_model(best_nodes):
    
    clf = LogisticRegression(penalty=best_nodes["penalty"], 
                             C=best_nodes["C"],
                             fit_intercept=best_nodes["fit_intercept"],
                             class_weight=best_nodes["class_weight"],
                             random_state=42)
    return clf

#这个主要是为了判断训练多次出来的逻辑回归模型是否完全一致
#判断逻辑回归模型完全一致的办法就是他们的截距和系数完全一致吧
#还记得模型训练过程其实是梯度下降算法，所以即便是所有参数以及超参相同
#训练数据的顺序改变也会导致最终训练出来的模型有所差异吧（这就是梯度下降算法）。
#可能这个真的是玄学很难判断同样的超参但是shuffle之后的数据训练出的模型谁更好吧？
#最后的实验结果证明train_lr_model中clf的random_state为常数时候训练出的所有模型一样
#设置为None的时候每次都不一样，这是因为训练数据的顺序被改变了，梯度下降算法就是这样的
def compare_lr_model(clf1, clf2):

    #print(type(clf1.coef_))
    #print(type(clf1.intercept_))
    #print(type(clf1.n_iter_))
    #print(type(clf1.get_params()))
    if((clf1.coef_==clf2.coef_).all() and (clf1.intercept_==clf2.intercept_).all() 
       and (clf1.n_iter_==clf2.n_iter_).all()):
        return True
    else:
        return False

"""
trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
best_params = fmin(lr_f, lr_space, algo=tpe.suggest, max_evals=10, trials=trials)
best_nodes = parse_lr_nodes(trials, lr_space_nodes)
save_inter_params(trials, lr_space_nodes, best_nodes, "lr_don't_overfit!_II")
train_lr_model(best_nodes, X_train_scaled, Y_train)
"""

"""
trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
best_params = fmin(lr_f, lr_space, algo=tpe.suggest, max_evals=1, trials=trials)
best_nodes = parse_lr_nodes(trials, lr_space_nodes)
save_inter_params(trials, lr_space_nodes, best_nodes, "lr_don't_overfit!_II")
clf1, X_train_scaled1 = train_lr_model(best_nodes, X_train_scaled, Y_train)
clf2, X_train_scaled2 = train_lr_model(best_nodes, X_train_scaled, Y_train)
if(compare_lr_model(clf1, clf2)):
    print("1")
else:
    print("2")
"""

#RandomForest
def rf_f(params):
    
    print("title",params["title"])
    print("path",params["path"]) 
    print("mean",params["mean"])
    print("std",params["std"])
    #print("feature_num",params["feature_num"])
    print("n_estimators",params["n_estimators"])
    print("criterion",params["criterion"])
    print("max_depth",params["max_depth"])
    print("min_samples_split",params["min_samples_split"])
    print("min_samples_leaf",params["min_samples_leaf"])
    print("max_features",params["max_features"])

    clf = RandomForestClassifier(n_estimators=int(params["n_estimators"]),
                                 criterion=params["criterion"],
                                 max_depth=params["max_depth"],
                                 min_samples_split=params["min_samples_split"],
                                 min_samples_leaf=params["min_samples_leaf"],
                                 max_features=params["max_features"],
                                 random_state=42)

    #这里不能够fit，不然交叉验证不就没意义了么
    #clf.fit(X_train_scaled, Y_train)
    """
    #现在不需要再这个模型内进行超参搜索咯
    selector = RFE(clf, n_features_to_select=params["feature_num"])
    X_train_scaled_new = selector.fit_transform(X_train_scaled, Y_train)
    skf = StratifiedKFold(Y_train, n_folds=25, shuffle=True, random_state=42)
    metric = cross_val_score(clf, X_train_scaled_new, Y_train, cv=skf, scoring="roc_auc").mean()
    """
    skf = StratifiedKFold(Y_train, n_folds=25, shuffle=True, random_state=42)
    metric = cross_val_score(clf, X_train_scaled, Y_train, cv=skf, scoring="roc_auc").mean()
    print(metric)
    print()
    return -metric

def parse_rf_nodes(trials, space_nodes):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    trials_list.sort(key=lambda item: item['result']['loss'])
    
    best_nodes = {}
    best_nodes["title"] = space_nodes["title"][trials_list[0]["misc"]["vals"]["title"][0]]
    best_nodes["path"] = space_nodes["path"][trials_list[0]["misc"]["vals"]["path"][0]]
    best_nodes["mean"] = space_nodes["mean"][trials_list[0]["misc"]["vals"]["mean"][0]]
    best_nodes["std"] = space_nodes["std"][trials_list[0]["misc"]["vals"]["std"][0]]
    
    #best_nodes["feature_num"] = space_nodes["feature_num"][trials_list[0]["misc"]["vals"]["feature_num"][0]]
    best_nodes["n_estimators"] = space_nodes["n_estimators"][trials_list[0]["misc"]["vals"]["n_estimators"][0]]
    best_nodes["criterion"] = space_nodes["criterion"][trials_list[0]["misc"]["vals"]["criterion"][0]]
    best_nodes["max_depth"] = space_nodes["max_depth"][trials_list[0]["misc"]["vals"]["max_depth"][0]]
    best_nodes["min_samples_split"] = space_nodes["min_samples_split"][trials_list[0]["misc"]["vals"]["min_samples_split"][0]]
    best_nodes["min_samples_leaf"] = space_nodes["min_samples_leaf"][trials_list[0]["misc"]["vals"]["min_samples_leaf"][0]]
    best_nodes["max_features"] = space_nodes["max_features"][trials_list[0]["misc"]["vals"]["max_features"][0]]
    
    return best_nodes

rf_space =  {"title":hp.choice("title", ["stacked_don't_overfit!_II"]),
             "path":hp.choice("path", ["Don't_Overfit!_II_Prediction.csv"]),
             "mean":hp.choice("mean", [0]),
             "std":hp.choice("std", [0]),
             #"feature_num":hp.choice("feature_num", np.linspace(1,300,300)),
             "n_estimators":hp.choice("n_estimators", [120, 150, 200, 250, 300, 500, 800, 1000, 1200]),
             "criterion":hp.choice("criterion", ["gini", "entropy"]),
             "max_depth":hp.choice("max_depth", [3, 5, 8, 10, 15, 20, 25]),
             "min_samples_split":hp.choice("min_samples_split", [2, 5, 10, 15, 20, 50, 100]),
             "min_samples_leaf":hp.choice("min_samples_leaf", [1, 2, 5, 10]),
             "max_features":hp.choice("max_features", [None, "log2", "sqrt"])
             }

rf_space_nodes = { "title":["stacked_don't_overfit!_II"],
                   "path":["Don't_Overfit!_II_Prediction.csv"],
                   "mean":[0],
                   "std":[0],
                   #"feature_num":np.linspace(1,300,300),
                   "n_estimators":[120, 150, 200, 250, 300, 500, 800, 1000, 1200],
                   "criterion":["gini", "entropy"],
                   "max_depth":[3, 5, 8, 10, 15, 20, 25],
                   "min_samples_split":[2, 5, 10, 15, 20, 50, 100],
                   "min_samples_leaf":[1, 2, 5, 10],
                   "max_features":[None, "log2", "sqrt"]
                   }

def train_rf_model(best_nodes, X_train_scaled, Y_train):
    
    clf = RandomForestClassifier(n_estimators=best_nodes["n_estimators"],
                                 criterion=best_nodes["criterion"],
                                 max_depth=best_nodes["max_depth"],
                                 min_samples_split=best_nodes["min_samples_split"],
                                 min_samples_leaf=best_nodes["min_samples_leaf"],
                                 max_features=best_nodes["max_features"],
                                 random_state=42)
    """
    #现在不需要在这个模型内做超参搜索咯
    selector = RFE(clf, n_features_to_select=best_nodes["feature_num"])
    X_train_scaled_new = selector.fit_transform(X_train_scaled, Y_train)
    
    clf.fit(X_train_scaled_new, Y_train)
    print(clf.score(X_train_scaled_new, Y_train))
    
    return clf, X_train_scaled_new
    """
    clf.fit(X_train_scaled, Y_train)
    print(clf.score(X_train_scaled, Y_train))
    return clf

def create_rf_model(best_nodes):
    
    clf = RandomForestClassifier(n_estimators=best_nodes["n_estimators"],
                                 criterion=best_nodes["criterion"],
                                 max_depth=best_nodes["max_depth"],
                                 min_samples_split=best_nodes["min_samples_split"],
                                 min_samples_leaf=best_nodes["min_samples_leaf"],
                                 max_features=best_nodes["max_features"],
                                 random_state=42)
    return clf
    
"""
trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
best_params = fmin(rf_f, rf_space, algo=tpe.suggest, max_evals=10, trials=trials)
best_nodes = parse_rf_nodes(trials, rf_space_nodes)
save_inter_params(trials, rf_space_nodes, best_nodes, "rf_don't_overfit!_II")
train_rf_model(best_nodes, X_train_scaled, Y_train)
"""

#一般存在下面的情况XGBOOST>=GBDT>=SVM>=RF>=Adaboost>=Other
#adaboost一般而言效果还不如决策树，所以感觉还是不用了吧
#所以可以考虑使用一个决策树的模型咯，下面是决策树的部分咯
#算了决策树还是不用了吧，因为模型数目不是当前最重要的问题，而且模型之间的差异也不是那么大

#素朴贝叶斯
def parse_gnb_nodes(trials, space_nodes):
    
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
    
    return best_nodes

gnb_space =  {"title":hp.choice("title", ["stacked_don't_overfit!_II"]),
              "path":hp.choice("path", ["Don't_Overfit!_II_Prediction.csv"]),
              "mean":hp.choice("mean", [0]),
              "std":hp.choice("std", [0]),
              #"feature_num":hp.choice("feature_num", np.linspace(1,300,300))
              }

gnb_space_nodes = {"title":["stacked_don't_overfit!_II"],
                   "path":["Don't_Overfit!_II_Prediction.csv"],
                   "mean":[0],
                   "std":[0],
                   #"feature_num":np.linspace(1,300,300)
                   }

def train_gnb_model(X_train_scaled, Y_train):
    
    """
    #The classifier does not expose "coef_" or "feature_importances_" attributes
    clf = GaussianNB()
    
    selector = RFE(clf, n_features_to_select=best_nodes["feature_num"])
    X_train_scaled_new = selector.fit_transform(X_train_scaled, Y_train)
    
    clf.fit(X_train_scaled_new, Y_train)
    print(clf.score(X_train_scaled_new, Y_train))
    
    return clf, X_train_scaled_new
    """
    
    clf = GaussianNB()
    clf.fit(X_train_scaled, Y_train)
    print(clf.score(X_train_scaled, Y_train))
    return clf

def create_gnb_model():
    
    clf = GaussianNB()
    return clf

"""
train_gnb_model(X_train_scaled, Y_train)
"""

#k近邻算法
def knn_f(params):
    
    print("title",params["title"])
    print("path",params["path"])
    print("mean",params["mean"])
    print("std",params["std"])
    #print("feature_num",params["feature_num"])
    print("n_neighbors",params["n_neighbors"])
    print("weights",params["weights"])
    print("algorithm",params["algorithm"])
    print("p",params["p"])

    clf = KNeighborsClassifier(n_neighbors=params["n_neighbors"],
                               weights=params["weights"],
                               algorithm=params["algorithm"],
                               p=params["p"])

    #这里不能够fit，不然交叉验证不就没意义了么
    #clf.fit(X_train_scaled, Y_train)
    """
    selector = RFE(clf, n_features_to_select=params["feature_num"])
    X_train_scaled_new = selector.fit_transform(X_train_scaled, Y_train)
    skf = StratifiedKFold(Y_train, n_folds=25, shuffle=True, random_state=42)
    metric = cross_val_score(clf, X_train_scaled_new, Y_train, cv=skf, scoring="roc_auc").mean()
    """
    
    skf = StratifiedKFold(Y_train, n_folds=25, shuffle=True, random_state=42)
    metric = cross_val_score(clf, X_train_scaled, Y_train, cv=skf, scoring="roc_auc").mean()    
    
    print(metric)
    print()
    return -metric

def parse_knn_nodes(trials, space_nodes):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    trials_list.sort(key=lambda item: item['result']['loss'])
    
    best_nodes = {}
    best_nodes["title"] = space_nodes["title"][trials_list[0]["misc"]["vals"]["title"][0]]
    best_nodes["path"] = space_nodes["path"][trials_list[0]["misc"]["vals"]["path"][0]]
    best_nodes["mean"] = space_nodes["mean"][trials_list[0]["misc"]["vals"]["mean"][0]]
    best_nodes["std"] = space_nodes["std"][trials_list[0]["misc"]["vals"]["std"][0]]
    
    #best_nodes["feature_num"] = space_nodes["feature_num"][trials_list[0]["misc"]["vals"]["feature_num"][0]]
    best_nodes["n_neighbors"] = space_nodes["n_neighbors"][trials_list[0]["misc"]["vals"]["n_neighbors"][0]]
    best_nodes["weights"] = space_nodes["weights"][trials_list[0]["misc"]["vals"]["weights"][0]]
    best_nodes["algorithm"] = space_nodes["algorithm"][trials_list[0]["misc"]["vals"]["algorithm"][0]]
    best_nodes["p"] = space_nodes["p"][trials_list[0]["misc"]["vals"]["p"][0]]
    
    return best_nodes

knn_space =  {"title":hp.choice("title", ["stacked_don't_overfit!_II"]),
              "path":hp.choice("path", ["Don't_Overfit!_II_Prediction.csv"]),
              "mean":hp.choice("mean", [0]),
              "std":hp.choice("std", [0]),
              #"feature_num":hp.choice("feature_num", np.linspace(1,300,300)),
              "n_neighbors":hp.choice("n_neighbors", [2, 4, 8, 16, 32, 64]),
              "weights":hp.choice("weights", ["uniform", "distance"]),
              "algorithm":hp.choice("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
              "p":hp.choice("p", [2, 3])
              }

knn_space_nodes = {"title":["stacked_don't_overfit!_II"],
                   "path":["Don't_Overfit!_II_Prediction.csv"],
                   "mean":[0],
                   "std":[0],
                   #"feature_num":np.linspace(1,300,300),
                   "n_neighbors":[2, 4, 8, 16, 32, 64],
                   "weights":["uniform", "distance"],
                   "algorithm":["auto", "ball_tree", "kd_tree", "brute"],
                   "p":[2, 3]
                   }

def train_knn_model(best_nodes, X_train_scaled, Y_train):
    
    """
    clf = KNeighborsClassifier(n_neighbors=best_nodes["n_neighbors"],
                               weights=best_nodes["weights"],
                               algorithm=best_nodes["algorithm"],
                               p=best_nodes["p"])
    selector = RFE(clf, n_features_to_select=best_nodes["feature_num"])
    X_train_scaled_new = selector.fit_transform(X_train_scaled, Y_train)
    
    clf.fit(X_train_scaled_new, Y_train)
    print(clf.score(X_train_scaled_new, Y_train))
    
    return clf, X_train_scaled_new
    """
    clf = KNeighborsClassifier(n_neighbors=best_nodes["n_neighbors"],
                               weights=best_nodes["weights"],
                               algorithm=best_nodes["algorithm"],
                               p=best_nodes["p"])
    clf.fit(X_train_scaled, Y_train)
    print(clf.score(X_train_scaled, Y_train))
    return clf

def create_knn_model(best_nodes):
    
    clf = KNeighborsClassifier(n_neighbors=best_nodes["n_neighbors"],
                               weights=best_nodes["weights"],
                               algorithm=best_nodes["algorithm"],
                               p=best_nodes["p"])
    return clf

"""
trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
best_params = fmin(knn_f, knn_space, algo=tpe.suggest, max_evals=10, trials=trials)
best_nodes = parse_knn_nodes(trials, knn_space_nodes)
save_inter_params(trials, knn_space_nodes, best_nodes, "knn_don't_overfit!_II")
train_knn_model(best_nodes, X_train_scaled, Y_train)
"""

#svm
def svm_f(params):
    
    print("title",params["title"])
    print("path",params["path"])
    print("mean",params["mean"])
    print("std",params["std"])
    #print("feature_num",params["feature_num"])
    print("gamma",params["gamma"])
    print("C",params["C"])
    print("class_weight",params["class_weight"])

    clf = svm.SVC(gamma=params["gamma"],
                  C=params["C"],
                  class_weight=params["class_weight"],
                  random_state=42)

    #这里不能够fit，不然交叉验证不就没意义了么
    #clf.fit(X_train_scaled, Y_train)
    """
    selector = RFE(clf, n_features_to_select=params["feature_num"])
    X_train_scaled_new = selector.fit_transform(X_train_scaled, Y_train)
    skf = StratifiedKFold(Y_train, n_folds=25, shuffle=True, random_state=42)
    metric = cross_val_score(clf, X_train_scaled_new, Y_train, cv=skf, scoring="roc_auc").mean()
    """
    
    skf = StratifiedKFold(Y_train, n_folds=25, shuffle=True, random_state=42)
    metric = cross_val_score(clf, X_train_scaled, Y_train, cv=skf, scoring="roc_auc").mean()
    
    print(metric)
    print()
    return -metric

def parse_svm_nodes(trials, space_nodes):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    trials_list.sort(key=lambda item: item['result']['loss'])
    
    best_nodes = {}
    best_nodes["title"] = space_nodes["title"][trials_list[0]["misc"]["vals"]["title"][0]]
    best_nodes["path"] = space_nodes["path"][trials_list[0]["misc"]["vals"]["path"][0]]
    best_nodes["mean"] = space_nodes["mean"][trials_list[0]["misc"]["vals"]["mean"][0]]
    best_nodes["std"] = space_nodes["std"][trials_list[0]["misc"]["vals"]["std"][0]]
    
    #best_nodes["feature_num"] = space_nodes["feature_num"][trials_list[0]["misc"]["vals"]["feature_num"][0]]
    best_nodes["gamma"] = space_nodes["gamma"][trials_list[0]["misc"]["vals"]["gamma"][0]]
    best_nodes["C"] = space_nodes["C"][trials_list[0]["misc"]["vals"]["C"][0]]
    best_nodes["class_weight"] = space_nodes["class_weight"][trials_list[0]["misc"]["vals"]["class_weight"][0]]

    return best_nodes

svm_space =  {"title":hp.choice("title", ["stacked_don't_overfit!_II"]),
              "path":hp.choice("path", ["Don't_Overfit!_II_Prediction.csv"]),
              "mean":hp.choice("mean", [0]),
              "std":hp.choice("std", [0]),
              #"feature_num":hp.choice("feature_num", np.linspace(1,300,300)),
              "gamma":hp.choice("gamma", ["auto"]),
              "C":hp.choice("C", np.logspace(-3, 5, 9)),
              "class_weight":hp.choice("class_weight", [None, "balanced"])
              }

svm_space_nodes = {"title":["stacked_don't_overfit!_II"],
                   "path":["Don't_Overfit!_II_Prediction.csv"],
                   "mean":[0],
                   "std":[0],
                   #"feature_num":np.linspace(1,300,300),
                   "gamma":["auto"],
                   "C":np.logspace(-3, 5, 9),
                   "class_weight":[None, "balanced"]
                   }

def train_svm_model(best_nodes, X_train_scaled, Y_train):
    
    """
    clf = svm.SVC(gamma=best_nodes["gamma"],
                  C=best_nodes["C"],
                  class_weight=best_nodes["class_weight"],
                  random_state=42)
    
    selector = RFE(clf, n_features_to_select=best_nodes["feature_num"])
    X_train_scaled_new = selector.fit_transform(X_train_scaled, Y_train)
    
    clf.fit(X_train_scaled_new, Y_train)
    print(clf.score(X_train_scaled_new, Y_train))
    
    return clf, X_train_scaled_new
    """
    
    clf = svm.SVC(gamma=best_nodes["gamma"],
                  C=best_nodes["C"],
                  class_weight=best_nodes["class_weight"],
                  random_state=42)
    clf.fit(X_train_scaled, Y_train)
    print(clf.score(X_train_scaled, Y_train))
    return clf   

def create_svm_model(best_nodes):

    clf = svm.SVC(gamma=best_nodes["gamma"],
                  C=best_nodes["C"],
                  class_weight=best_nodes["class_weight"],
                  random_state=42)
    return clf  

"""
trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
best_params = fmin(svm_f, svm_space, algo=tpe.suggest, max_evals=10, trials=trials)
best_nodes = parse_svm_nodes(trials, svm_space_nodes)
save_inter_params(trials, svm_space_nodes, best_nodes, "svm_don't_overfit!_II")
train_svm_model(best_nodes, X_train_scaled, Y_train)
"""

"""
感觉不能够用这个模型，因为我查了一下，这个模型是线性模型哦太简陋了
那么接下来换一个mlp的模型咯，但是这个模型会涉及到超参选择熬= =！
#Elastic Net咯
enet = ElasticNet(l1_ratio=0.7)
"""

#mlp
def mlp_f(params):
    
    print("title",params["title"])
    print("path",params["path"])
    print("mean",params["mean"])
    print("std",params["std"])
    #print("feature_num",params["feature_num"])    
    print("hidden_layer_sizes",params["hidden_layer_sizes"])
    print("activation",params["activation"])
    print("solver",params["solver"])
    print("batch_size",params["batch_size"])
    print("learning_rate_init",params["learning_rate_init"])
    print("max_iter",params["max_iter"])
    print("shuffle",params["shuffle"])
    print("early_stopping",params["early_stopping"])

    clf = MLPClassifier(hidden_layer_sizes=params["hidden_layer_sizes"],
                        activation=params["activation"],
                        solver=params["solver"],
                        batch_size=params["batch_size"],
                        learning_rate_init=params["learning_rate_init"],
                        max_iter=params["max_iter"],
                        shuffle=params["shuffle"],
                        early_stopping=params["early_stopping"],
                        random_state=42)
    #这里不能够fit，不然交叉验证不就没意义了么
    #clf.fit(X_train_scaled, Y_train)
    """
    selector = RFE(clf, n_features_to_select=params["feature_num"])
    X_train_scaled_new = selector.fit_transform(X_train_scaled, Y_train)
    skf = StratifiedKFold(Y_train, n_folds=25, shuffle=True, random_state=42)
    metric = cross_val_score(clf, X_train_scaled_new, Y_train, cv=skf, scoring="roc_auc").mean()
    """
    skf = StratifiedKFold(Y_train, n_folds=25, shuffle=True, random_state=42)
    metric = cross_val_score(clf, X_train_scaled, Y_train, cv=skf, scoring="roc_auc").mean()
    
    print(metric)
    print()
    return -metric

def parse_mlp_nodes(trials, space_nodes):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    trials_list.sort(key=lambda item: item['result']['loss'])
    
    best_nodes = {}
    best_nodes["title"] = space_nodes["title"][trials_list[0]["misc"]["vals"]["title"][0]]
    best_nodes["path"] = space_nodes["path"][trials_list[0]["misc"]["vals"]["path"][0]]
    best_nodes["mean"] = space_nodes["mean"][trials_list[0]["misc"]["vals"]["mean"][0]]
    best_nodes["std"] = space_nodes["std"][trials_list[0]["misc"]["vals"]["std"][0]]
    
    #best_nodes["feature_num"] = space_nodes["feature_num"][trials_list[0]["misc"]["vals"]["feature_num"][0]]
    best_nodes["hidden_layer_sizes"] = space_nodes["hidden_layer_sizes"][trials_list[0]["misc"]["vals"]["hidden_layer_sizes"][0]]
    best_nodes["activation"] = space_nodes["activation"][trials_list[0]["misc"]["vals"]["activation"][0]]
    best_nodes["solver"] = space_nodes["solver"][trials_list[0]["misc"]["vals"]["solver"][0]]
    best_nodes["batch_size"] = space_nodes["batch_size"][trials_list[0]["misc"]["vals"]["batch_size"][0]]
    best_nodes["learning_rate_init"] = space_nodes["learning_rate_init"][trials_list[0]["misc"]["vals"]["learning_rate_init"][0]]
    best_nodes["max_iter"] = space_nodes["max_iter"][trials_list[0]["misc"]["vals"]["max_iter"][0]]
    best_nodes["shuffle"] = space_nodes["shuffle"][trials_list[0]["misc"]["vals"]["shuffle"][0]]
    best_nodes["early_stopping"] = space_nodes["early_stopping"][trials_list[0]["misc"]["vals"]["early_stopping"][0]]

    return best_nodes

mlp_space =  {"title":hp.choice("title", ["stacked_don't_overfit!_II"]),
              "path":hp.choice("path", ["Don't_Overfit!_II_Prediction.csv"]),
              "mean":hp.choice("mean", [0]),
              "std":hp.choice("std", [0]),
              #"feature_num":hp.choice("feature_num", np.linspace(1,300,300)),
              "hidden_layer_sizes":hp.choice("hidden_layer_sizes", [(50,), (80,), (100,), (150,), (200,),
                                                                    (50,50), (80, 80), (120, 120), (150, 150),
                                                                    (50, 50, 50), (80, 80, 80), (120, 120, 120)]),
              "activation":hp.choice("activation", ["relu"]),
              "solver":hp.choice("solver", ["adam"]),
              "batch_size":hp.choice("batch_size", [64]),
              "learning_rate_init":hp.choice("learning_rate_init", np.linspace(0.0001, 0.0300, 300)),
              "max_iter":hp.choice("max_iter", [400]),
              "shuffle":hp.choice("shuffle", [True]),
              "early_stopping":hp.choice("early_stopping", [True])
              }

mlp_space_nodes = {"title":["stacked_don't_overfit!_II"],
                   "path":["Don't_Overfit!_II_Prediction.csv"],
                   "mean":[0],
                   "std":[0],
                   #"feature_num":np.linspace(1,300,300),
                   "hidden_layer_sizes":[(50,), (80,), (100,), (150,), (200,),
                                         (50,50), (80, 80), (120, 120), (150, 150),
                                         (50, 50, 50), (80, 80, 80), (120, 120, 120)],
                   "activation":["relu"],
                   "solver":["adam"],
                   "batch_size":[64],
                   "learning_rate_init":np.linspace(0.0001, 0.0300, 300),
                   "max_iter":[400],
                   "shuffle":[True],
                   "early_stopping":[True]
                   }

def train_mlp_model(best_nodes, X_train_scaled, Y_train):
    
    clf = MLPClassifier(hidden_layer_sizes=best_nodes["hidden_layer_sizes"],
                        activation=best_nodes["activation"],
                        solver=best_nodes["solver"],
                        batch_size=best_nodes["batch_size"],
                        learning_rate_init=best_nodes["learning_rate_init"],
                        max_iter=best_nodes["max_iter"],
                        shuffle=best_nodes["shuffle"],
                        early_stopping=best_nodes["early_stopping"],
                        random_state=42)
    clf.fit(X_train_scaled, Y_train)
    print(clf.score(X_train_scaled, Y_train))
    return clf   

def create_mlp_model(best_nodes):
    
    clf = MLPClassifier(hidden_layer_sizes=best_nodes["hidden_layer_sizes"],
                        activation=best_nodes["activation"],
                        solver=best_nodes["solver"],
                        batch_size=best_nodes["batch_size"],
                        learning_rate_init=best_nodes["learning_rate_init"],
                        max_iter=best_nodes["max_iter"],
                        shuffle=best_nodes["shuffle"],
                        early_stopping=best_nodes["early_stopping"],
                        random_state=42)
    return clf

#lgb相关的部分
def lgb_f(params):
    
    global cnt
    cnt +=1
    print(cnt)
    print("title", params["title"])
    print("path", params["path"])
    print("mean", params["mean"])
    print("std", params["std"])
    print("learning_rate", params["learning_rate"])
    print("n_estimators", params["n_estimators"])
    print("max_depth", params["max_depth"])
    #print("eval_metric", params["eval_metric"])
    print("num_leaves", params["num_leaves"])
    print("subsample", params["subsample"])
    print("colsample_bytree", params["colsample_bytree"])
    print("min_child_samples", params["min_child_samples"])
    print("min_child_weight", params["min_child_weight"])   

    clf = LGBMClassifier(learning_rate=params["learning_rate"],
                         n_estimators=int(params["n_estimators"]),
                         max_depth=params["max_depth"],
                         #eval_metric=params["eval_metric"],
                         num_leaves=params["num_leaves"],
                         subsample=params["subsample"],
                         colsample_bytree=params["colsample_bytree"],
                         min_child_samples=params["min_child_samples"],
                         min_child_weight=params["min_child_weight"],
                         random_state=42)

    #skf = StratifiedKFold(Y_train, n_folds=25, shuffle=True, random_state=42)
    metric = cross_val_score(clf, X_train_scaled, Y_train, cv=10, scoring="accuracy").mean()
    
    print(-metric)
    print() 
    return -metric
    
def parse_lgb_nodes(trials, space_nodes):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    trials_list.sort(key=lambda item: item['result']['loss'])
    
    best_nodes = {}
    best_nodes["title"] = space_nodes["title"][trials_list[0]["misc"]["vals"]["title"][0]]
    best_nodes["path"] = space_nodes["path"][trials_list[0]["misc"]["vals"]["path"][0]]
    best_nodes["mean"] = space_nodes["mean"][trials_list[0]["misc"]["vals"]["mean"][0]]
    best_nodes["std"] = space_nodes["std"][trials_list[0]["misc"]["vals"]["std"][0]]
        
    best_nodes["learning_rate"] = space_nodes["learning_rate"][trials_list[0]["misc"]["vals"]["learning_rate"][0]]
    best_nodes["n_estimators"] = space_nodes["n_estimators"][trials_list[0]["misc"]["vals"]["n_estimators"][0]]
    best_nodes["max_depth"] = space_nodes["max_depth"][trials_list[0]["misc"]["vals"]["max_depth"][0]]
    #best_nodes["eval_metric"]= space_nodes["eval_metric"][trials_list[0]["misc"]["vals"]["eval_metric"][0]]
    best_nodes["num_leaves"] = space_nodes["num_leaves"][trials_list[0]["misc"]["vals"]["num_leaves"][0]]
    best_nodes["subsample"] = space_nodes["subsample"][trials_list[0]["misc"]["vals"]["subsample"][0]]
    best_nodes["colsample_bytree"] = space_nodes["colsample_bytree"][trials_list[0]["misc"]["vals"]["colsample_bytree"][0]]
    best_nodes["min_child_samples"] = space_nodes["min_child_samples"][trials_list[0]["misc"]["vals"]["min_child_samples"][0]]
    best_nodes["min_child_weight"]= space_nodes["min_child_weight"][trials_list[0]["misc"]["vals"]["min_child_weight"][0]]
    
    return best_nodes

def train_lgb_model(best_nodes, X_train_scaled, Y_train):
    
    clf = LGBMClassifier(learning_rate=best_nodes["learning_rate"],
                         n_estimators=int(best_nodes["n_estimators"]),
                         max_depth=best_nodes["max_depth"],
                         #eval_metric=best_nodes["eval_metric"],
                         num_leaves=best_nodes["num_leaves"],
                         subsample=best_nodes["subsample"],
                         colsample_bytree=best_nodes["colsample_bytree"],
                         min_child_samples=best_nodes["min_child_samples"],
                         min_child_weight=best_nodes["min_child_weight"],
                         random_state=42)

    clf.fit(X_train_scaled, Y_train)
    print(clf.score(X_train_scaled, Y_train))
    return clf

def create_lgb_model(best_nodes, X_train_scaled, Y_train):
    
    clf = LGBMClassifier(learning_rate=best_nodes["learning_rate"],
                         n_estimators=int(best_nodes["n_estimators"]),
                         max_depth=best_nodes["max_depth"],
                         #eval_metric=best_nodes["eval_metric"],
                         num_leaves=best_nodes["num_leaves"],
                         subsample=best_nodes["subsample"],
                         colsample_bytree=best_nodes["colsample_bytree"],
                         min_child_samples=best_nodes["min_child_samples"],
                         min_child_weight=best_nodes["min_child_weight"],
                         random_state=42)
    return clf

#其他的参数暂时不知道如何设置，暂时就用这些超参吧，我觉得主要还是特征的创建咯
lgb_space = {"title":hp.choice("title", ["stacked_don't_overfit!_II"]),
             "path":hp.choice("path", ["Don't_Overfit!_II_Prediction.csv"]),
             "mean":hp.choice("mean", [0]),
             "std":hp.choice("std", [0]),
             "learning_rate":hp.choice("learning_rate", np.linspace(0.01, 0.40, 40)),
             "n_estimators":hp.choice("n_estimators", np.linspace(60, 200, 8)),
             "max_depth":hp.choice("max_depth", [-1, 3, 4, 6, 8]),
             #"eval_metric":hp.choice("eval_metric", ["l2_root"]),
             "num_leaves":hp.choice("num_leaves", [21, 31, 54]),
             "subsample":hp.choice("subsample", [0.8, 0.9, 1.0]),
             "colsample_bytree":hp.choice("colsample_bytree", [0.8, 0.9, 1.0]),
             "min_child_samples":hp.choice("min_child_samples", [16, 18, 20, 22, 24]),
             "min_child_weight":hp.choice("min_child_weight", [16, 18, 20, 22, 24])
             #"reg_alpha":hp.choice("reg_alpha", [0, 0.001, 0.01, 0.03, 0.1, 0.3]),
             #"reg_lambda":hp.choice("reg_lambda", [0, 0.001, 0.01, 0.03, 0.1, 0.3]),
            }

lgb_space_nodes = {"title":["stacked_don't_overfit!_II"],
                   "path":["Don't_Overfit!_II_Prediction.csv"],
                   "mean":[0],
                   "std":[0],
                   "learning_rate":np.linspace(0.01, 0.40, 40),
                   "n_estimators":np.linspace(60, 200, 8),
                   "max_depth":[-1, 3, 4, 6, 8],
                   #"eval_metric":["l2_root"],
                   "num_leaves":[21, 31, 54],
                   "subsample":[0.8, 0.9, 1.0],
                   "colsample_bytree":[0.8, 0.9, 1.0],
                   "min_child_samples":[16, 18, 20, 22, 24],
                   "min_child_weight":[16, 18, 20, 22, 24]
                   }

"""
trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
cnt = 0
best_params = fmin(lgb_f, lgb_space, algo=tpe.suggest, max_evals=10, trials=trials)
best_nodes = parse_lgb_nodes(trials, lgb_space_nodes)
save_inter_params(trials, lgb_space_nodes, best_nodes, "lgb_don't_overfit!_II")
train_lgb_model(best_nodes, X_train_scaled, Y_train)
"""

#xgboost相关的部分咯
#我还是有点担心这里面的函数输出的是错误的值，所以函数print一下吧
def xgb_f(params):
    
    global cnt
    cnt +=1
    print(cnt)
    print("title", params["title"])
    print("path", params["path"])
    print("mean", params["mean"])
    print("std", params["std"])
    #print("feature_num", params["feature_num"])
    print("gamma", params["gamma"]) 
    print("max_depth", params["max_depth"])
    print("learning_rate", params["learning_rate"])
    print("min_child_weight", params["min_child_weight"])
    print("subsample", params["subsample"])
    print("colsample_bytree", params["colsample_bytree"])
    print("reg_alpha", params["reg_alpha"])
    print("reg_lambda", params["reg_lambda"])
    print("n_estimators", params["n_estimators"])

    clf = XGBClassifier(gamma=params["gamma"],
                        max_depth=params["max_depth"],
                        learning_rate=params["learning_rate"],
                        min_child_weight=params["min_child_weight"],
                        subsample=params["subsample"],
                        colsample_bytree=params["colsample_bytree"],
                        reg_alpha=params["reg_alpha"],
                        reg_lambda=params["reg_lambda"],
                        n_estimators=int(params["n_estimators"]),
                        random_state=42)

    #这里不能够fit，不然交叉验证不就没意义了么
    #clf.fit(X_train_scaled, Y_train)
    """
    #不需要再这个模型中进行特征选择了
    selector = RFE(clf, n_features_to_select=params["feature_num"])
    X_train_scaled_new = selector.fit_transform(X_train_scaled, Y_train)
    skf = StratifiedKFold(Y_train, n_folds=25, shuffle=True, random_state=42)
    metric = cross_val_score(clf, X_train_scaled_new, Y_train, cv=skf, scoring="roc_auc").mean()
    """
    skf = StratifiedKFold(Y_train, n_folds=25, shuffle=True, random_state=42)
    metric = cross_val_score(clf, X_train_scaled, Y_train, cv=skf, scoring="roc_auc").mean()
    
    print(metric)
    print()
    return -metric
    
def parse_xgb_nodes(trials, space_nodes):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    trials_list.sort(key=lambda item: item['result']['loss'])
    
    best_nodes = {}
    best_nodes["title"] = space_nodes["title"][trials_list[0]["misc"]["vals"]["title"][0]]
    best_nodes["path"] = space_nodes["path"][trials_list[0]["misc"]["vals"]["path"][0]]
    best_nodes["mean"] = space_nodes["mean"][trials_list[0]["misc"]["vals"]["mean"][0]]
    best_nodes["std"] = space_nodes["std"][trials_list[0]["misc"]["vals"]["std"][0]]
    
    #best_nodes["feature_num"] = space_nodes["feature_num"][trials_list[0]["misc"]["vals"]["feature_num"][0]]
    best_nodes["gamma"] = space_nodes["gamma"][trials_list[0]["misc"]["vals"]["gamma"][0]]
    best_nodes["max_depth"] = space_nodes["max_depth"][trials_list[0]["misc"]["vals"]["max_depth"][0]]
    best_nodes["learning_rate"] = space_nodes["learning_rate"][trials_list[0]["misc"]["vals"]["learning_rate"][0]]
    best_nodes["min_child_weight"] = space_nodes["min_child_weight"][trials_list[0]["misc"]["vals"]["min_child_weight"][0]]
    best_nodes["subsample"] = space_nodes["subsample"][trials_list[0]["misc"]["vals"]["subsample"][0]]
    best_nodes["colsample_bytree"] = space_nodes["colsample_bytree"][trials_list[0]["misc"]["vals"]["colsample_bytree"][0]]
    best_nodes["reg_alpha"] = space_nodes["reg_alpha"][trials_list[0]["misc"]["vals"]["reg_alpha"][0]]
    best_nodes["reg_lambda"] = space_nodes["reg_lambda"][trials_list[0]["misc"]["vals"]["reg_lambda"][0]]
    best_nodes["n_estimators"] = space_nodes["n_estimators"][trials_list[0]["misc"]["vals"]["n_estimators"][0]]

    return best_nodes

xgb_space = {"title":hp.choice("title", ["stacked_don't_overfit!_II"]),
             "path":hp.choice("path", ["Don't_Overfit!_II_Prediction.csv"]),
             "mean":hp.choice("mean", [0]),
             "std":hp.choice("std", [0]),
             #"feature_num":hp.choice("feature_num", np.linspace(1,300,300)),
             "gamma":hp.choice("gamma", [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.3, 0.5, 0.7, 0.9]),
             "max_depth":hp.choice("max_depth", [3, 5, 7, 9, 12, 15, 17, 25]),
             "learning_rate":hp.choice("learning_rate", np.linspace(0.01, 0.50, 50)),
             "min_child_weight":hp.choice("min_child_weight", [1, 3, 5, 7, 9]),
             "subsample":hp.choice("subsample", [0.6, 0.7, 0.8, 0.9, 1.0]),
             "colsample_bytree":hp.choice("colsample_bytree", [0.6, 0.7, 0.8, 0.9, 1.0]),
             "reg_alpha":hp.choice("reg_alpha", [0.0, 0.1, 0.5, 1.0]),
             "reg_lambda":hp.choice("reg_lambda", [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 1.0]),
             "n_estimators":hp.choice("n_estimators", np.linspace(50, 500, 46))
             }

xgb_space_nodes = {"title":["stacked_don't_overfit!_II"],
                   "path":["Don't_Overfit!_II_Prediction.csv"],
                   "mean":[0],
                   "std":[0],
                   #"feature_num":np.linspace(1,300,300),
                   "gamma":[0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.3, 0.5, 0.7, 0.9],
                   "max_depth":[3, 5, 7, 9, 12, 15, 17, 25],
                   "learning_rate":np.linspace(0.01, 0.50, 50),
                   "min_child_weight":[1, 3, 5, 7, 9],
                   "subsample":[0.6, 0.7, 0.8, 0.9, 1.0],
                   "colsample_bytree":[0.6, 0.7, 0.8, 0.9, 1.0],
                   "reg_alpha":[0.0, 0.1, 0.5, 1.0],
                   "reg_lambda":[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 1.0],
                   "n_estimators":np.linspace(50, 500, 46)
                   }

def train_xgb_model(best_nodes, X_train_scaled, Y_train):
    
    clf = XGBClassifier(gamma=best_nodes["gamma"],
                        max_depth=best_nodes["max_depth"],
                        learning_rate=best_nodes["learning_rate"],
                        min_child_weight=best_nodes["min_child_weight"],
                        subsample=best_nodes["subsample"],
                        colsample_bytree=best_nodes["colsample_bytree"],
                        reg_alpha=best_nodes["reg_alpha"],
                        reg_lambda=best_nodes["reg_lambda"],
                        n_estimators=int(best_nodes["n_estimators"]),
                        random_state=42)
    
    """
    selector = RFE(clf, n_features_to_select=best_nodes["feature_num"])
    X_train_scaled_new = selector.fit_transform(X_train_scaled, Y_train)
    
    clf.fit(X_train_scaled_new, Y_train)
    print(clf.score(X_train_scaled_new, Y_train))
    
    return clf, X_train_scaled_new
    """
    clf.fit(X_train_scaled, Y_train)
    print(clf.score(X_train_scaled, Y_train))
    return clf 
    
def create_xgb_model(best_nodes, X_train_scaled, Y_train):
    
    clf = XGBClassifier(gamma=best_nodes["gamma"],
                        max_depth=best_nodes["max_depth"],
                        learning_rate=best_nodes["learning_rate"],
                        min_child_weight=best_nodes["min_child_weight"],
                        subsample=best_nodes["subsample"],
                        colsample_bytree=best_nodes["colsample_bytree"],
                        reg_alpha=best_nodes["reg_alpha"],
                        reg_lambda=best_nodes["reg_lambda"],
                        n_estimators=int(best_nodes["n_estimators"]),
                        random_state=42)
    return clf 

"""
trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
cnt =0
best_params = fmin(xgb_f, xgb_space, algo=tpe.suggest, max_evals=10, trials=trials)
best_nodes = parse_xgb_nodes(trials, xgb_space_nodes)
save_inter_params(trials, xgb_space_nodes, best_nodes, "xgb_don't_overfit!_II")
train_xgb_model(best_nodes, X_train_scaled, Y_train)
"""

#cat相关的部分
def cat_f(params):
    
    global cnt
    cnt+=1
    print(cnt)
    print("title", params["title"])
    print("path", params["path"])
    print("mean", params["mean"])
    print("std", params["std"])
    print("iterations", params["iterations"])
    print("learning_rate", params["learning_rate"])
    print("depth", params["depth"])
    print("l2_leaf_reg", params["l2_leaf_reg"])
    print("custom_metric", params["custom_metric"])

    clf = CatBoostClassifier(iterations=params["iterations"],
                             learning_rate=params["learning_rate"],
                             depth=params["depth"],
                             l2_leaf_reg=params["l2_leaf_reg"],
                             custom_metric=params["custom_metric"],
                             task_type='GPU',
                             random_state=42)

    #skf = StratifiedKFold(Y_train, n_folds=25, shuffle=True, random_state=42)
    metric = cross_val_score(clf, X_train_scaled, Y_train, cv=8, scoring="accuracy").mean()
    
    print(-metric)
    print() 
    return -metric

def parse_cat_nodes(trials, space_nodes):
    
    trials_list =[]
    for item in trials.trials:
        trials_list.append(item)
    trials_list.sort(key=lambda item: item['result']['loss'])
    
    best_nodes = {}
    best_nodes["title"] = space_nodes["title"][trials_list[0]["misc"]["vals"]["title"][0]]
    best_nodes["path"] = space_nodes["path"][trials_list[0]["misc"]["vals"]["path"][0]]
    best_nodes["mean"] = space_nodes["mean"][trials_list[0]["misc"]["vals"]["mean"][0]]
    best_nodes["std"] = space_nodes["std"][trials_list[0]["misc"]["vals"]["std"][0]]
    
    best_nodes["iterations"] = space_nodes["iterations"][trials_list[0]["misc"]["vals"]["iterations"][0]]
    best_nodes["learning_rate"] = space_nodes["learning_rate"][trials_list[0]["misc"]["vals"]["learning_rate"][0]]
    best_nodes["depth"] = space_nodes["depth"][trials_list[0]["misc"]["vals"]["depth"][0]]
    best_nodes["l2_leaf_reg"]= space_nodes["l2_leaf_reg"][trials_list[0]["misc"]["vals"]["l2_leaf_reg"][0]]
    best_nodes["custom_metric"] = space_nodes["custom_metric"][trials_list[0]["misc"]["vals"]["custom_metric"][0]]
    
    return best_nodes

def train_cat_model(best_nodes, X_train_scaled, Y_train):
    
    clf = CatBoostClassifier(iterations=best_nodes["iterations"],
                             learning_rate=best_nodes["learning_rate"],
                             depth=best_nodes["depth"],
                             l2_leaf_reg=best_nodes["l2_leaf_reg"],
                             custom_metric=best_nodes["custom_metric"],
                             task_type='GPU',
                             random_state=42)

    clf.fit(X_train_scaled, Y_train)
    print(clf.score(X_train_scaled, Y_train))
    return clf

def create_cat_model(best_nodes, X_train_scaled, Y_train):
    
    clf = CatBoostClassifier(iterations=best_nodes["iterations"],
                             learning_rate=best_nodes["learning_rate"],
                             depth=best_nodes["depth"],
                             l2_leaf_reg=best_nodes["l2_leaf_reg"],
                             custom_metric=best_nodes["custom_metric"],
                             task_type='GPU',
                             random_state=42)
    return clf

#其他的参数暂时不知道如何设置，暂时就用这些超参吧，我觉得主要还是特征的创建咯
cat_space = {"title":hp.choice("title", ["stacked_don't_overfit!_II"]),
             "path":hp.choice("path", ["Don't_Overfit!_II_Prediction.csv"]),
             "mean":hp.choice("mean", [0]),
             "std":hp.choice("std", [0]),
             "iterations":hp.choice("iterations", [800, 1000, 1200]),
             "learning_rate":hp.choice("learning_rate", np.linspace(0.01, 0.30, 30)),
             "depth":hp.choice("depth", [3, 4, 6, 9, 11]),
             "l2_leaf_reg":hp.choice("l2_leaf_reg", [2, 3, 5, 7, 9]),
             #"n_estimators":hp.choice("n_estimators", [3, 4, 5, 6, 8, 9, 11]),
             #"loss_function":hp.choice("loss_function", ["RMSE"]),#这个就是默认的
             "custom_metric":hp.choice("custom_metric", ["Accuracy"]),
             #"partition_random_seed":hp.choice("partition_random_seed", [42]),#这个是cv里面才用的参数
             #"n_estimators":hp.choice("n_estimators", np.linspace(50, 500, 46)),
             #"border_count":hp.choice("border_count", [16, 32, 48, 64, 96, 128]),
             #"ctr_border_count":hp.choice("ctr_border_count", [32, 48, 50, 64, 96, 128])
             }

cat_space_nodes = {"title":["stacked_don't_overfit!_II"],
                   "path":["Don't_Overfit!_II_Prediction.csv"],
                   "mean":[0],
                   "std":[0],
                   #"feature_num":np.linspace(1,300,300),
                   "iterations":[800, 1000, 1200],
                   "learning_rate":np.linspace(0.01, 0.30, 30),
                   "depth":[3, 4, 6, 9, 11],
                   "l2_leaf_reg":[2, 3, 5, 7, 9],
                   #"n_estimators":[3, 4, 5, 6, 8, 9, 11],
                   #"loss_function":["RMSE"],#这个就是默认的
                   "custom_metric":["Accuracy"],
                   #"partition_random_seed":[42],
                   #"n_estimators":np.linspace(50, 500, 46),
                   #"border_count":[16, 32, 48, 64, 96, 128],
                   #"ctr_border_count":[32, 48, 50, 64, 96, 128]
                   }

"""
trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
cnt =0
best_params = fmin(cat_f, cat_space, algo=tpe.suggest, max_evals=1, trials=trials)
best_nodes = parse_cat_nodes(trials, cat_space_nodes)
save_inter_params(trials, cat_space_nodes, best_nodes, "cat_don't_overfit!_II")
train_cat_model(best_nodes, X_train_scaled, Y_train)
"""

#非神经网络模型也就是传统机器学习模型的get_oof方法咯
def get_oof_nonnn(best_model, X_train_scaled, Y_train, X_test_scaled, n_folds = 5):
    
    """K-fold stacking"""
    num_train, num_test = X_train_scaled.shape[0], X_test_scaled.shape[0]
    oof_train = np.zeros((num_train,)) 
    oof_test = np.zeros((num_test,))
    oof_test_all_fold = np.zeros((num_test, n_folds))
    train_rmse = []
    valida_rmse = []

    KF = KFold(n_splits =n_folds, shuffle=True)
    for i, (train_index, valida_index) in enumerate(KF.split(X_train_scaled)):
        #划分数据集
        X_split_train, Y_split_train = X_train_scaled[train_index], Y_train[train_index]
        X_split_valida, Y_split_valida = X_train_scaled[valida_index], Y_train[valida_index]
        
        best_model.fit(X_split_train, Y_split_train)
        
        score1 = best_model.score(X_split_train, Y_split_train)
        print("当前模型：",type(best_model))
        print("训练集上的准确率", score1)
        train_rmse.append(score1)
        score2 = best_model.socre(X_split_valida, Y_split_valida)
        print("验证集上的准确率", score2)
        valida_rmse.append(score2)
        
        oof_train[valida_index] = (best_model.predict(X_split_valida.astype(np.float32))).reshape(1,-1)
        oof_test_all_fold[:, i] = (best_model.predict(X_test_scaled.astype(np.float32))).reshape(1,-1)
        
    oof_test = np.mean(oof_test_all_fold, axis=1)
    
    return oof_train, oof_test, best_model

#非神经网络模型也就是传统机器学习模型的stacked_features方法咯
def stacked_features_nonnn(model_list, X_train_scaled, Y_train, X_test_scaled, folds):
    
    input_train = [] 
    input_test = []
    nodes_num = len(model_list)
    
    for i in range(0, nodes_num):
        oof_train, oof_test, best_model= get_oof_nonnn(model_list[i], X_train_scaled.values, Y_train.values, X_test_scaled.values, folds)
        input_train.append(oof_train)
        input_test.append(oof_test)
    
    stacked_train = np.concatenate([f.reshape(-1, 1) for f in input_train], axis=1)
    stacked_test = np.concatenate([f.reshape(-1, 1) for f in input_test], axis=1)
    
    stacked_train = pd.DataFrame(stacked_train)
    stacked_test = pd.DataFrame(stacked_test)
    return stacked_train, stacked_test

#实现正常分类版本的stacking
def logistic_stacking_rscv_predict(nodes_list, data_test, stacked_train, Y_train, stacked_test, max_evals=50):
    
    clf = LogisticRegression(random_state=42)
    #这边的参数我觉得还是有必要进行这样的设置的，才能够比较好的表示
    param_dist = {"penalty": ["l1", "l2"],
                  "C": np.logspace(-3, 5, 100),
                  "fit_intercept": [True, False],
                  "class_weight": ["balance", None],
                  }
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=max_evals, cv=10, scoring="accuracy")#,scoring ="mother_fucker" #,scoring ="mean_squared_error" #, scoring="neg_mean_squared_error")
    random_search.fit(stacked_train, Y_train)
    best_score = random_search.best_estimator_.score(stacked_train, Y_train)

    save_best_model(random_search.best_estimator_, nodes_list[0]["title"]+"_"+str(len(nodes_list)))
    Y_pred = random_search.best_estimator_.predict(stacked_test.values.astype(np.float32))

    data = {"id":data_test["id"], "target":Y_pred}
    output = pd.DataFrame(data = data)            
    output.to_csv(nodes_list[0]["path"], index=False)
    print("prediction file has been written.")
     
    print("the best coefficient R^2 of the model on the whole train dataset is:", best_score)
    print()
    return random_search.best_estimator_, Y_pred

#进行一次stacking的测试咯
#这个是lgb进行超参搜索的版本
start_time = datetime.datetime.now()
trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
cnt = 0
best = fmin(lgb_f, lgb_space, algo=tpe.suggest, max_evals=1, trials=trials)
best_lgb_nodes = parse_lgb_nodes(trials, lgb_space_nodes)
save_inter_params(trials, lgb_space_nodes, best_lgb_nodes, "lgb_don't_overfit!_II")
clf = train_lgb_model(best_lgb_nodes, X_train_scaled, Y_train)
end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))
print()
#这个是xgb进行超参搜索的版本
start_time = datetime.datetime.now()
trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
cnt = 0
best = fmin(xgb_f, xgb_space, algo=tpe.suggest, max_evals=1, trials=trials)
best_xgb_nodes = parse_xgb_nodes(trials, xgb_space_nodes)
save_inter_params(trials, xgb_space_nodes, best_xgb_nodes, "xgb_don't_overfit!_II")
clf = train_xgb_model(best_xgb_nodes, X_train_scaled, Y_train)
end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))
print()
"""
#这个是cat进行超参搜索的版本
start_time = datetime.datetime.now()
trials = Trials()
algo = partial(tpe.suggest, n_startup_jobs=10)
cnt = 0
best = fmin(cat_f, cat_space, algo=tpe.suggest, max_evals=1, trials=trials)
best_cat_nodes = parse_cat_nodes(trials, cat_space_nodes)
save_inter_params(trials, cat_space_nodes, best_cat_nodes, "cat_don't_overfit!_II")
rsg = train_cat_model(best_cat_nodes, X_train_scaled, Y_train)
end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))
print()
"""
start_time = datetime.datetime.now()
#进行stacking的部分，我之前看这个比赛别人都是blending还以为不适合stacking，其实回归问题以及这个比赛还是有stacking的
trials, space_nodes, best_lgb_nodes = load_inter_params("lgb_don't_overfit!_II")
best_lgb_model = create_lgb_model(best_lgb_nodes, X_train_scaled, Y_train)
trials, space_nodes, best_xgb_nodes = load_inter_params("xgb_don't_overfit!_II")
best_xgb_model = create_xgb_model(best_xgb_nodes, X_train_scaled, Y_train)
#trials, space_nodes, best_cat_nodes = load_inter_params("cgb_don't_overfit!_II")
#best_cat_model = create_cat_model(best_cat_nodes, X_train_scaled, Y_train)

nodes_list = [best_lgb_nodes, best_xgb_nodes] 
models_list = [best_lgb_model, best_xgb_model] 
#nodes_list = [best_lgb_nodes, best_xgb_nodes, best_cat_nodes] 
#models_list = [best_lgb_model, best_xgb_model, best_cat_model]
stacked_train, stacked_test = stacked_features_nonnn(models_list, X_train_scaled, Y_train, X_test_scaled, 2)
save_stacked_dataset(stacked_train, stacked_test, "tmdb_box_office_prediction")
logistic_stacking_rscv_predict(nodes_list, data_test, stacked_train, Y_train, stacked_test, 600)
end_time = datetime.datetime.now()
print("time cost", (end_time - start_time))
