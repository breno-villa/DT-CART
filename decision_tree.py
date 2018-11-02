import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def entropy_criterion(data, labels):

  classes = np.unique(labels)
  
  s = 0
  for c in classes:
    p = np.mean(labels == c)
    s -= p * np.log(p)
    
  return s
  

def gini_criterion(data, labels):

  classes = np.unique(labels)
  
  s = 0
  for c in classes:
    p = np.mean(labels == c)
    s += p * (1 - p)
    
  return s


def find_cut_point(data, labels, impurity_criterion = gini_criterion):

  n_samples, n_features = data.shape
  max_info_gain = np.iinfo(np.int32).min
  feat_id = 0
  best_threshold = 0
  H_parent = impurity_criterion(data, labels)

  for j in range(n_features):

    values = np.unique(data[:, j])
    
    for i in range(values.shape[0] - 1):
      threshold = (values[i] + values[i + 1]) / 2.
      mask = data[:, j] <= threshold
      info_gain = H_parent \
                  - (mask.sum() * impurity_criterion(data[mask], labels[mask]) \
                  + (~mask).sum() * impurity_criterion(data[~mask], labels[~mask])) \
                  / float(n_samples)

      if max_info_gain < info_gain:
        best_threshold = threshold
        feat_id = j
        max_info_gain = info_gain
        
  return feat_id, best_threshold


def stopping_criterion(n_classes, depth, max_depth):

  return (max_depth is not None and max_depth == depth) or (n_classes == 1)

def build_tree(data, labels, tree, depth = 1):
    classes, counts = np.unique(labels, return_counts=True)

    n_classes = classes.shape[0]

    if not stopping_criterion(n_classes, depth, tree.max_depth):
        node = Node()


        feature, threshold = find_cut_point(data, labels, 
                                            tree.impurity_criterion)
        mask = data[:, feature] <= threshold        
        left = build_tree(data[mask], labels[mask], tree, depth + 1)
        right = build_tree(data[~mask], labels[~mask], tree, depth + 1)
   
        return Node(feature=feature, threshold=threshold, left=left, right=right)

    values = np.zeros(tree.n_classes)
    values[classes] = counts
    return Node(is_leaf=True, counts=values)


class Node(object):
  """Node"""
  def __init__(self, feature=None, threshold=None,
                     is_leaf=None, counts=None, left=None, right=None):
    super(Node, self).__init__()
    self.threshold = threshold
    self.is_leaf = is_leaf
    self.counts = counts
    self.left = left
    self.right = right
    self.feature = feature
    

class DecisionTreeClassifier(object):
  """DecisionTreeClassifier

  Parameters
  ----------
  max_depth:

  impurity_criterion:

  """
  def __init__(self, max_depth, impurity_criterion = gini_criterion):
    super(DecisionTreeClassifier, self).__init__()
    self.max_depth = max_depth
    self.impurity_criterion = impurity_criterion

  def recursive_predict(self, node, X):

    if node.is_leaf:
      return np.zeros(X.shape[0]) + np.argmax(node.counts)

    mask = X[:, node.feature] <= node.threshold

    y_pred = np.zeros(X.shape[0])
    if mask.sum() > 0:
      y_pred[mask] = self.recursive_predict(node.left, X[mask])

    if (~mask).sum() > 0:
      y_pred[~mask] = self.recursive_predict(node.right, X[~mask])

    return y_pred

  def fit(self, X, y):
    self.classes = np.unique(y)
    self.n_classes = self.classes.shape[0]

    self.root = build_tree(X, y, self)

    

    return self

  def predict(self, X):
    return self.recursive_predict(self.root, X)

  def showTree(self):
    self.preordem(self.root)
    return self

  def preordem(self, node):
    print(node.feature)
    print(node.is_leaf)
    print(node.counts)
    print(node.left)
    print(node.right)
    print(node.threshold)
    print("    ")
    if node.is_leaf:
      return
    self.preordem(node.left)
    self.preordem(node.right)
    



if __name__ == '__main__':
  X = np.array([[1,1], [1,0], [0,1], [0,0]])
  y = np.array([0, 1, 1, 0])

  

  data_mush = pd.read_csv('C:\glass_modificado.csv', sep=',', header=0)# https://www.kaggle.com/uciml/glass
  print("Data set: ", len(data_mush))



  X = data_mush.values[:, 0:8]
  y = data_mush.values[:,9]


  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.13, random_state=42)

  dt = DecisionTreeClassifier(max_depth=None, impurity_criterion = entropy_criterion)
  #dt = DecisionTreeClassifier(max_depth=None, impurity_criterion = gini_criterion)
  

  by_train = y_train.astype('int')
  by_test = y_test.astype('int')
  bX_train = X_train.astype('int')
  bX_test = X_test.astype('int')

  #y_pred = dt.fit(bX_train, by_train).showTree().predict(bX_test)
  y_pred = dt.fit(X_train, by_train).predict(X_test)

  print(np.mean(y_pred == by_test))

  tam = len(y_pred)

  for val in range(0, tam):
    if y_pred[val] != by_test[val]:
      print(y_pred[val])
      print(by_test[val])
      print("---------")

