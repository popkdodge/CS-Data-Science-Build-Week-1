"""
Before we code a decision tree we know that random forrest
is consist of many decision trees hences "forrest".
because forrest have trees.

At first glance decision trees look like binary search trees 
how ever instead of spliting the search by greater than or less
than you would make a prediction on where to split the data
until you have almost a commplete split of all the data .

we would decide using "Entropy equation"

We calculate the entrophy split the data nd crate entrophy of the child

Entrophy(parent) - weighted average * Entrophy(Children)


Training the algrorithm is building the tree.

The tree would start at the top node at  each node select the best split base on the 
information gain.

A greedy way to find the best place to split we could loop over all the feature / thresholds
and test all the possible feature values .

However a better way is to cherry pick the best split of features and thresholds inprove 
upon it.

We could apply some methood to prevent over fitting such as limiting the maximum depth of The
decision tress or having it stop until it reaches a certian sample thresholds.

Once the algorithim is trained we could use the tree to predict by taking in the information
and traverse the tree recursively(because it would be easiest to code and generally more efficient)

AT each node the best split feature of the testing will go left or right depending on
the thresholds that is decided in the training phase. Until we reach the end and return the class lables.
"""

import numpy as np 
from collections import Counter

def entropy(y):
    hist = np.bincount(y) # calcuate the occurance of class lables
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

class Node:
    def __init__(self, feature=None, thresholds=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.thresholds = thresholds
        self.right = right
        self.left = left
        self.value = value 
    
    def is_leaf_node(self):
        return self.value is not None

class Decision_Tree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        # Training
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1]) 
        #make sure it input the correct data shape
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth = 0):
        n_sample, n_features = X.shape
        n_lables = len(np.unique(y))

        # This determine when to stop growing the tree 
        if (depth >= self.max_depth
            or n_lables == 1
            or n_sample < self.min_samples_split):
            leaf_value = self._most_common_lable(y)
            return Node(value = leaf_value)
        
        feature_ids = np.random.choice(n_features, self.n_feats, replace=False)

        # greedy search
        best_Feat, best_Threshold = self._best_criteria(X, y, feature_ids)
        left_ids, right_ids = self._split(X[:, best_Feat], best_Threshold)
        left = self._grow_tree(X[left_ids, :], y[left_ids], depth+1)
        right = self._grow_tree(X[right_ids, :], y[right_ids], depth+1)
        return Node(best_Feat, best_Threshold, left, right)

        
    def predict(self, X):
        # This would traverse the tree.
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.thresholds:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_lable(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def _best_criteria(self, X, y, feature_ids):
        best_gain = -1
        split_id, split_threshold = None, None
        for feature_id in feature_ids:
            X_column = X[:, feature_id]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_id = feature_id
                    split_threshold = threshold

        return split_id, split_threshold

    def _information_gain(self, y, X_column, split_threshold):
        # parent Entrophy
        parent_entrophy = entropy(y)
        # generate split
        left_ids, right_ids = self._split(X_column, split_threshold)

        if len(left_ids) == 0 or len(right_ids) == 0:
            return 0
        # weighted avg chile Entrophy
        n = len(y)
        n_left, n_right = len(left_ids), len(right_ids)
        entropy_l, entropy_r = entropy(y[left_ids]), entropy(y[right_ids])
        child_entropy = (n_left/n) * entropy_l + (n_right/n) * entropy_r
        # calculate the child entrophy

        information_gain = parent_entrophy - child_entropy

        return information_gain

    def _split(self, X_column, split_threshold):
        left_ids = np.argwhere(X_column <= split_threshold).flatten()
        right_ids = np.argwhere(X_column > split_threshold).flatten()
        return left_ids, right_ids