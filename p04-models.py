import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neural_network import MLPClassifier

# new helpers:
from shared import dataset_local_path, bootstrap_accuracy, simple_boxplot, TODO

# stdlib:
from dataclasses import dataclass
import json
from typing import Dict, Any, List

# warnings:
import warnings

#%% load up the data
examples = []
ys = []

with open(dataset_local_path("poetry_id.jsonl")) as fp:
    for line in fp:
        info = json.loads(line)
        # Note: the data contains a whole bunch of extra stuff; we just want numeric features for now.
        keep = info["features"]
        # whether or not it's poetry is our label.
        ys.append(info["poetry"])
        # hold onto this single dictionary.
        examples.append(keep)

## CONVERT TO MATRIX:

feature_numbering = DictVectorizer(sort=True)

# Ensure Feauture is within the range from 0 to 1 (very rough now)
# I did this so it doesn't complain...
X = feature_numbering.fit_transform(examples) / 1000

print("Features as {} matrix.".format(X.shape))


## SPLIT DATA:

RANDOM_SEED = 12345678

# Numpy-arrays are more useful than python's lists.
y = np.array(ys)
# split off train/validate (tv) pieces.
X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, train_size=0.75, shuffle=True, random_state=RANDOM_SEED
)
# split off train, validate from (tv) pieces.
X_train, X_vali, y_train, y_vali = train_test_split(
    X_tv, y_tv, train_size=0.66, shuffle=True, random_state=RANDOM_SEED
)

print(X_train.shape, X_vali.shape, X_test.shape)

#%% Define & Run Experiments
@dataclass
class ExperimentResult:
    vali_acc: float
    params: Dict[str, Any]
    model: ClassifierMixin


# Added "gini"
def consider_decision_trees():
    print("Consider Decision Tree.")
    performances: List[ExperimentResult] = []

    for rnd in range(3):
        for crit in ["entropy", "gini"]:
            for d in range(1, 9):
                params = {
                    "criterion": crit,
                    "max_depth": d,
                    "random_state": rnd,
                }
                f = DecisionTreeClassifier(**params)
                f.fit(X_train, y_train)
                vali_acc = f.score(X_vali, y_vali)
                result = ExperimentResult(vali_acc, params, f)
                performances.append(result)
    return max(performances, key=lambda result: result.vali_acc)


# Added "gini"
def consider_random_forest():
    print("Consider Random Forest.")
    performances: List[ExperimentResult] = []
    # Random Forest
    for rnd in range(3):
        for crit in ["entropy", "gini"]:
            for d in range(4, 9):
                params = {
                    "criterion": crit,
                    "max_depth": d,
                    "random_state": rnd,
                }
                f = RandomForestClassifier(**params)
                f.fit(X_train, y_train)
                vali_acc = f.score(X_vali, y_vali)
                result = ExperimentResult(vali_acc, params, f)
                performances.append(result)
    return max(performances, key=lambda result: result.vali_acc)


# Random state seems to like "2", so I have kept it unchanged.
# added penalty options (like l1)
# added max_iter options
#   Very interesting, the max_iter seems to like 1000, although it has higher
#   options, so this model probably converges before 1000.
def consider_perceptron() -> ExperimentResult:
    print("Consider Perceptron.")
    performances: List[ExperimentResult] = []
    for rnd in range(3):
        for penalty in ["l2", "l1", "elasticnet"]:
            for maxIt in range(1000, 5001, 1000):
                params = {
                    "random_state": rnd,
                    "penalty": penalty,
                    "max_iter": maxIt,
                }
                f = Perceptron(**params)
                f.fit(X_train, y_train)
                vali_acc = f.score(X_vali, y_vali)
                # print(maxIt)
                # print(vali_acc)
                result = ExperimentResult(vali_acc, params, f)
                performances.append(result)

    return max(performances, key=lambda result: result.vali_acc)


# Random state seems to like "0", so I have kept it unchanged.
# added penalty options (like l2)
# added solver options
#   Very interesting, when solver is newton-cg, it produces very accurate result
# I tried adding the max-iter options, but there were too many warnings, so I
# removed it.
# Netwon CG as the solver produce a HUGE improvements over other method,
# This is more significant than any other things, VERY curious to know why.
# Added try except so it continues when solver option doenst work with penalty option
def consider_logistic_regression() -> ExperimentResult:
    print("Consider Logistic Regression.")
    performances: List[ExperimentResult] = []
    for rnd in range(3):
        for penalty in ["l2", "l1", "elasticnet", "none"]:
            for solver in ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]:
                try:
                    params = {
                        "random_state": rnd,
                        "penalty": penalty,
                        "max_iter": 1000,
                        "C": 1.0,
                        "solver": solver,
                    }
                    f = LogisticRegression(**params)
                    f.fit(X_train, y_train)
                    vali_acc = f.score(X_vali, y_vali)
                    result = ExperimentResult(vali_acc, params, f)
                    performances.append(result)
                except:
                    continue

    return max(performances, key=lambda result: result.vali_acc)


# Random state seems to like "2", so I have kept it unchanged.
# added solver options
# Added layer size options, but it seems to like one single hidden layer with 32 units
def consider_neural_net() -> ExperimentResult:
    print("Consider Multi-Layer Perceptron.")
    performances: List[ExperimentResult] = []
    for rnd in range(3):
        for solver in ["sgd", "lbfgs", "adam"]:
            for size in range(1, 4):
                layerlist = []
                for s in range(0, size):
                    layerlist.append(32)
                print(layerlist)
                params = {
                    "hidden_layer_sizes": tuple(layerlist),
                    "random_state": rnd,
                    "solver": solver,
                    "max_iter": 10000,
                    "alpha": 0.0001,
                }
                f = MLPClassifier(**params)
                f.fit(X_train, y_train)
                vali_acc = f.score(X_vali, y_vali)
                result = ExperimentResult(vali_acc, params, f)
                performances.append(result)

    return max(performances, key=lambda result: result.vali_acc)


logit = consider_logistic_regression()
perceptron = consider_perceptron()
dtree = consider_decision_trees()
rforest = consider_random_forest()
mlp = consider_neural_net()

print("Best Logistic Regression", logit)
print("Best Perceptron", perceptron)
print("Best DTree", dtree)
print("Best RForest", rforest)
print("Best MLP", mlp)

#%% Plot Results

# Helper method to make a series of box-plots from a dictionary:
simple_boxplot(
    {
        "Logistic Regression": bootstrap_accuracy(logit.model, X_vali, y_vali),
        "Perceptron": bootstrap_accuracy(perceptron.model, X_vali, y_vali),
        "Decision Tree": bootstrap_accuracy(dtree.model, X_vali, y_vali),
        "RandomForest": bootstrap_accuracy(rforest.model, X_vali, y_vali),
        "MLP/NN": bootstrap_accuracy(mlp.model, X_vali, y_vali),
    },
    title="Validation Accuracy",
    xlabel="Model",
    ylabel="Accuracy",
    save="model-cmp.png",
)

# TODO("1. Understand consider_decision_trees; I have 'tuned' it.")
# TODO("2. Find appropriate max_iter settings to stop warning messages.")
# TODO(
#     "3. Pick a model: {perceptron, logistic regression, neural_network} and optimize it!"
# )
