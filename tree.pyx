'''
Stuff.
'''

import cmath
import math
from collections import Counter

import numpy as np
cimport numpy as np
cimport cython


# Data struct used to store data points on each node.
cdef class Node:
    cdef int category_id
    cdef int feature_id
    cdef Node left
    cdef Node right


cdef class Tree:
    cdef int max_depth,num_features
    cdef public Node root

    def __init__(self,max_depth):
        self.max_depth = max_depth

    def fit(self,X,y):
        self.num_features = X.shape[1]
        print(f'fitted tree with {self.num_features} features')

        self.root = _grow_tree(X,y,max_depth=self.max_depth,depth=0)

    def predict(self,X):
        return predict_tree(X,self.root)

@cython.profile(True)
cdef predict_tree(np.ndarray[np.int64_t,ndim=2] X,Node root):
    ''' Simple function to loop through the observations in the X array.'''
    cdef int len_array,i, prediction
    len_array = X.shape[0]
    predictions = list()
    for i in range(len_array):
        prediction = _traverse_tree(X[i].reshape(1,-1),root)
        predictions.append(prediction)
    return np.array(predictions)

@cython.profile(True)
cdef _traverse_tree(np.ndarray[np.int64_t,ndim=2] X,Node node):
    ''' Recursively traverses the tree until a leaf node is reached.'''
    if node.category_id!=-1:
        return node.category_id
    
    if X[:,node.feature_id] == 0:
        return _traverse_tree(X, node.left)
    else:
        return _traverse_tree(X, node.right)


@cython.profile(True)
def _grow_tree(X, y, int max_depth, int depth):
    '''
    Grows the tree doing the following:
    - 1. Selects a random subset of features
    - 2. loops through the features to calc the feature with the highest entropy gain
    - 3. splits the data by the best feature and records the feature used in the node
    - 4. if criteria are met, it doesn't split the data and records the most common class in the node to create the leaf
    '''
    n_samples, n_feats = X.shape
    n_labels = len(np.unique(y))

    # check the stopping criteria
    if (depth>=max_depth or n_labels==1):
        node = Node()
        node.category_id = _most_common_label(y)
        node.feature_id = -1
        return node
    
    # find the best split in the data
    feat_idxs = np.random.choice(n_feats, n_feats, replace=False)


    cdef double best_gain
    cdef int best_feature_idx
    best_gain = -1.0
    best_feature_idx = -1
    for feat_idx in feat_idxs:

        # check feature has variance
        X_col = X[:,feat_idx].ravel()
        no_unique_vals = np.unique(X_col)
        if (len(no_unique_vals)==1):
            continue

        # if it has variance then check the information gain
        gain = calculate_information_gain(X_col.ravel(), y.ravel())
        if (gain>best_gain):
            best_gain=gain
            best_feature_idx = feat_idx
    
    # if there are no vars with variance then return as a leaf
    if best_feature_idx==-1:
        #print('creating node with best_feature_idx==-1!')
        leaf_node = Node()
        leaf_node.category_id = _most_common_label(y)
        leaf_node.feature_id = -1
        return leaf_node
    
    left_idxs, right_idxs = split_data(X[:, best_feature_idx])

    split_node = Node()
    split_node.left = _grow_tree(X[left_idxs, :], y[left_idxs],max_depth, depth+1)
    split_node.right = _grow_tree(X[right_idxs, :], y[right_idxs],max_depth, depth+1)
    split_node.feature_id = best_feature_idx
    split_node.category_id = -1
    return split_node


@cython.profile(True)
def _most_common_label(y):
    counter = Counter(y)
    return counter.most_common(1)[0][0]


@cython.profile(True)
def _best_feature(X, y, feat_idxs):
    best_gain = -1
    best_feature_idx = None

    y = y.ravel()
    
    for feat_idx in feat_idxs:
        X_column = X[:, feat_idx].ravel()
        gain = calculate_information_gain(X_column,y)
        if (gain > best_gain):
            best_gain = gain
            best_feature_idx = feat_idx

    return best_feature_idx


@cython.profile(True)
def calculate_information_gain(np.ndarray[np.int64_t,ndim=1] X, np.ndarray[np.int64_t,ndim=1] y):
    ''' Splits by the binary feature and calculates information gain.  '''
    cdef int N, N_L, N_R, i
    cdef int outcome, outcome_L, outcome_R,N_minus_outcome
    N = X.shape[0]
    N_L = 0
    N_R = 0
    outcome = 0
    outcome_L = 0
    outcome_R = 0
    for i in range(N):
        if X[i]==0:
            N_L+=1

            if y[i]==1:
                outcome+=1
                outcome_L+=1
        else:
            N_R+=1

            if y[i]==1:
                outcome+=1
                outcome_R+=1

    N_minus_outcome = N - outcome

    cdef double left_entropy,right_entropy,parent_entropy, information_gain

    parent_entropy = entropy(N,outcome)
    left_entropy = entropy(N_L,outcome_L)
    right_entropy = entropy(N_R,outcome_R)

    cdef double child_entropy, N_L_f, N_R_f, N_f
    N_L_f = <double>N_L
    N_f = <double>N
    N_R_f = <double>N_R
    child_entropy = (N_L_f/N_f)*left_entropy + (N_R_f/N_f)*right_entropy
    information_gain = parent_entropy - child_entropy

    return information_gain


@cython.profile(True)
cdef double entropy(int N,int N_1):

    cdef int N_0
    N_0 = N-N_1

    # Casting to double so 
    cdef double N_0_f,N_1_f,prob_0,prob_1
    N_0_f = <double>N_0
    N_1_f = <double>N_1

    prob_0 = N_0_f/(N_0_f+N_1_f)
    prob_1 = N_1_f/(N_0_f+N_1_f)

    cdef double enp0,enp1,enptot
    if (prob_0==0.0):
        enp0 = 0.0
    else:
        enp0 = prob_0*math.log(prob_0)

    if (prob_1==0.0):
        enp1 = 0.0
    else:
        enp1 = prob_1*math.log(prob_1)
    enptot = enp0 + enp1

    return -enptot


@cython.profile(True)
def split_data(best_feature_X_col):
    left_idxs = (best_feature_X_col==0).flatten().nonzero()[0]
    right_idxs = (best_feature_X_col==1).flatten().nonzero()[0]
    return left_idxs, right_idxs


@cython.profile(True)
cdef void describe_node(Node node):
    if node.category_id!=-1:
        print('\tLeaf with value: ',node.category_id, 'node',node)
    else:
        print('Split node with feature:',node.feature_id)
        describe_node(node.left)
        describe_node(node.right)


@cython.profile(True)
def describe_tree(Tree tree):
    cdef Node root_node
    root_node = tree.root
    describe_node(root_node)