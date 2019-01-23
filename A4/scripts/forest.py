import pickle
import pandas as pd
import numpy as np
import scipy.spatial.distance as dist

from random import sample, randint
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score
from collections import Counter


train_data_loc = '../data/train-data.txt'
test_data_loc = '../data/test-data.txt'

train_df = pd.read_csv(train_data_loc, header=None, delim_whitespace=True)
test_df = pd.read_csv(test_data_loc, header=None, delim_whitespace=True)

train_data = np.array(np.array(train_df)[:,2:], dtype=int)
train_label = np.array(np.array(train_df)[:,1].T, dtype=int)
train_label.resize((train_label.shape[0], 1))
test_data = np.array(np.array(test_df)[:,2:], dtype=int)
test_label = np.array(np.array(test_df)[:,1].T, dtype=int)
test_label.resize((test_label.shape[0], 1))

print(train_data.shape, test_data.shape)
print(train_label.shape, test_label.shape)


# idea and partial code adapted from Google Developers
# https://www.youtube.com/watch?v=LDRbO9a6XPU

def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    c = Counter()
    for row in rows:
        c[row[-1]] += 1
    return c


def partition(rows, question):
    """Partitions a dataset.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    """Calculate the Gini Impurity for a list of rows.
    """
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = float(counts[lbl]) / len(rows)
        impurity -= prob_of_lbl**2
    return impurity



def info_gain(left, right, current_uncertainty):
    """Information Gain.

    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def find_best_split(rows):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    best_gain = float('-inf')
    best_question = None
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1

    for col in range(n_features):
        values = set([row[col] for row in rows])  # unique values in the column
        for val in values:
            question = Question(col, val)

            true_rows, false_rows = partition(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            current_gain = info_gain(true_rows, false_rows, current_uncertainty)
            if current_gain >= best_gain:
                best_gain, best_question = current_gain, question

    return best_gain, best_question


class Question:
    """A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        return val >= self.value



class Leaf:
    """A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows):
        self.predictions = class_counts(rows)
        
    def get_element(self):
        return list(self.predictions.keys())[0]


class Decision_Node:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

        
class MyDecisionTreeClassifier:
    
    def __init__(self):
        self.my_tree = Decision_Node(None, None, None)
    
    def build_tree(rows):
        """Builds the tree.

        Rules of recursion: 
        1) Believe that it works.
        2) Start by checking for the base case (no further information gain).
        3) Prepare for giant stack traces.
        """
        gain, question = find_best_split(rows)
        if gain == 0 or not question:
            return Leaf(rows)

        true_rows, false_rows = partition(rows, question)
        true_branch = build_tree(true_rows)
        false_branch = build_tree(false_rows)
        return Decision_Node(question, true_branch, false_branch)

    
    def fit(self, train_data, train_label):
        training_data = np.concatenate((train_data, train_label), axis=1)
        self.my_tree = build_tree(training_data)
    
    def classify(self, row, node):
        """See the 'rules of recursion' above."""

        # Base case: we've reached a leaf
        if isinstance(node, Leaf):
            return node.get_element()

        # Decide whether to follow the true-branch or the false-branch.
        # Compare the feature / value stored in the node,
        # to the example we're considering.
        if node.question.match(row):
            return self.classify(row, node.true_branch)
        else:
            return self.classify(row, node.false_branch)

    def predict(self, test_data, test_label):
        testing_data = np.concatenate((test_data, test_label), axis=1)
        return np.array([self.classify(row, self.my_tree) for row in testing_data])

    
def dump_tree(tree_id, rowids, colids, clf):
    tree = {
        'rowids': rowids,
        'colids': colids,
        'clf': clf
    }
    f = open('../models/forest_model_test.txt', 'wb')
    f.write(pickle.dumps(tree))
    f.close()
    return tree
    
def do_mdtc(tree_id, selected_rows, selected_cols):
    clf = MyDecisionTreeClassifier()
    clf.fit(train_data[selected_rows,:][:, selected_cols], train_label[selected_rows,:])
    pred = clf.predict(test_data[:,selected_cols], test_label)
    score = accuracy_score(test_label, pred)
    tree = dump_tree(tree_id, selected_rows, selected_cols, clf)
    return pred, tree




f = open('../models/forest model.txt', 'rb')
models = pickle.loads(f.read())
my_models = models


res = []
for model in my_models:
    clf = model['clf']
    rowids, colids = model['rowids'], model['colids']
    pred = clf.predict(test_data[:,colids], test_label)
    res.append(pred)

my = np.array(res)
final = []
for i in range(my.shape[1]):
    c = Counter(my[:,i].ravel())
    val = c.most_common(n=1)[0][0]
    final.append(val)
pred = np.array(final)
score = accuracy_score(pred, test_label)

print(score)
