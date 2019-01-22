import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

def loadData():
  train = pd.read_csv("titanic/train.csv")
  print(train.shape)

  NAs = pd.concat([train.isnull().sum()], axis=1, keys=['Train'])
  print(NAs[NAs.sum(axis=1) > 0])

  train['Age'] = train['Age'].fillna(train['Age'].mean())
  train['Embarked'].fillna(train['Embarked'].mode()[0])

  train['Pclass'] = train['Pclass'].apply(str)

  for col in train.dtypes[train.dtypes == 'object'].index:
     for_dummy = train.pop(col)
     train = pd.concat([train, pd.get_dummies(for_dummy, prefix=col)], axis=1)

  labels = train.pop('Survived')

  x_train, x_test, y_train, y_test = train_test_split(train, labels, test_size=0.25)
  return x_train, x_test, y_train, y_test

def calculateAUC(x_train, x_test, y_train, y_test):
  model = DecisionTreeClassifier()
  model.fit(x_train, y_train)
  y_pred = model.predict(x_test)
  false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
  roc_auc = auc(false_positive_rate, true_positive_rate)
  print(roc_auc)

# The first parameter to tune is max_depth. This indicates how deep the tree can be.
# The deeper the tree, the more splits it has and it captures more information about the data.
# We fit a decision tree with depths ranging from 1 to 32
def tuneMaxDepth(x_train, x_test, y_train, y_test):
  max_depths = np.linspace(1, 32, 32, endpoint=True)
  train_results = []
  test_results = []
  for max_depth in max_depths:
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = model.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
  from matplotlib.legend_handler import HandlerLine2D
  line1, = plt.plot(max_depths, train_results, 'b', label ="Train AUC")
  line2, = plt.plot(max_depths, test_results, 'r', label ="Test AUC")
  plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
  plt.ylabel('AUC score')
  plt.xlabel('Tree depth')
  plt.show()



if __name__ == "__main__":
  x_train, x_test, y_train, y_test = loadData()
  tuneNEstimators(x_train, x_test, y_train, y_test)