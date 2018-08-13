# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 11:26:07 2018

@author: LISUZA5
"""
import pickle
import os 

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
import pandas as pd



os.getcwd()

# use pickle to reload the saved rf model
filename="O:\\GAML Technology\\_Analytics\\Model\\POC RF 13AA\\Output\\Heekyung\\final model\\threshold=0.0085 n_tree=2000n_feature72rf_finalized_model.sav"
saved_model = pickle.load(open(filename,'rb'))

saved_model
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features=72, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=2000, n_jobs=1,
            oob_score=False, random_state=10, verbose=0, warm_start=False)

type(saved_model)
sklearn.ensemble.forest.RandomForestClassifier

# Extract info from attributes
saved_model.estimators_[0]
type(saved_model.estimators_[0])
sklearn.tree.tree.DecisionTreeClassifier

t0 = saved_model.estimators_[0]

type(t0.tree_)
sklearn.tree._tree.Tree

# The decision estimator has an attribute called tree_  which stores the entire
# tree structure and allows access to low level attributes. The binary tree
# tree_ is represented as a number of parallel arrays. The i-th element of each
# array holds information about the node `i`. Node 0 is the tree's root. NOTE:
# Some of the arrays only apply to either leaves or split nodes, resp. In this
# case the values of nodes of the other type are arbitrary!
#
# Among those arrays, we have:
#   - left_child, id of the left child of the node
#   - right_child, id of the right child of the node
#   - feature, feature used for splitting the node
#   - threshold, threshold value at the node
#

# Using those arrays, we can parse the tree structure



t0.tree_.feature.shape
(3691,)

t0.tree_.threshold.shape

t0.TREE_UNDEFINED

t0.tree_

t0.tree_.feature.max()
# 239
t0.tree_.feature[t0.tree_.feature>=0].min()
# 0

t0.tree_.node_count
3691

t0.tree_.children_left.shape
t0.tree_.children_right


np.array(t0.tree_.feature).describe()

# A single underscore in front of a variable name (prefix) is a hint 
# that a variable is meant for internal use only. A double underscore prefix 
# causes the Python interpreter to rewrite the variable name in order to avoid 
# naming conflicts in subclasses. Double underscores are also called "dunders" in Python







def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    #print ("def tree({}):".format(", ".join(feature_names)))

    print ("data final;")
    print ("set test;")
    print("if 0=0 then do;")
    #for j,tree in enumerate 
    
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            #print ("{}if {} <= {}:".format(indent, name, threshold))
            #while depth<tree_.max_depth
            print ("{}if {} <= {} then do;".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            #print ("{}else:  # if {} > {}".format(indent, name, threshold))
            print ("{}else  if {} > {} then do;".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
            #if depth <= tree_.max_depth:
            print("{}end;".format(indent))
        else:
            #print ("{}result= {}".format(indent, tree_.value[node]))
            
            # This is how to create an INDICATOR!
            out=1 if tree_.value[node][0,tree_.max_n_classes-1]/(tree_.value[node].sum())>=0.5 else 0
            print ("{}pred= {}; end;".format(indent, out))
            #print("{}end;".format(indent))
        #print("{}end;".format(indent))
    recurse(0, 1)
    #print(" ",end='\r',flush=True)

tree_to_code(rf.estimators_[0],list(X_test))

print ('hello'),
sys.stdout.flush()
print ('\rhell '),
sys.stdout.flush()
...
print ('\rhel '),
sys.stdout.flush()

import sys
print("FAILED...")
sys.stdout.write("\033[F") #back to previous line
sys.stdout.write("\033[K") #clear line
print("SUCCESS!")

def tree_to_code1(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    #print ("def tree({}):".format(", ".join(feature_names)))

    print ("data final;")
    print ("set test;")

    #for j,tree in enumerate 
    
    def recurse(node, depth):
        print("if 0=0 then do;")
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            #print ("{}if {} <= {}:".format(indent, name, threshold))
            #while depth<tree_.max_depth
            print ("{}if {} <= {} then do;".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            #print ("{}else:  # if {} > {}".format(indent, name, threshold))
            print ("{}else  if {} > {} then do;".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
            #if depth <= tree_.max_depth:
            print("{}end;".format(indent))
        else:
            #print ("{}result= {}".format(indent, tree_.value[node]))
            
            # This is how to create an INDICATOR!
            out=1 if tree_.value[node][0,tree_.max_n_classes-1]/(tree_.value[node].sum())>=0.5 else 0
            print ("{}pred= {}; end;".format(indent, out))
            #print("{}end;".format(indent))
        #print("{}end;".format(indent))
    recurse(0, 1)
    #print(" ",end='\r',flush=True)

tree_to_code(rf.estimators_[0],list(X_test))




##=======================
## Revise to work for RF
##======================= 
    
def tree_to_code2(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    #print ("def tree({}):".format(", ".join(feature_names)))

    #print ("data final;")
    #print ("set test;")

    #for j,tree in enumerate 
    
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            #print ("{}if {} <= {}:".format(indent, name, threshold))
            #while depth<tree_.max_depth
            print ("{}if {} <= {} then do;".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            #print ("{}else:  # if {} > {}".format(indent, name, threshold))
            print ("{}else  if {} > {} then do;".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            #print ("{}result= {}".format(indent, tree_.value[node]))
            
            # This is how to create an INDICATOR!
            out=1 if tree_.value[node][0,tree_.max_n_classes-1]/(tree_.value[node].sum())>=0.5 else 0
            print ("{}pred= {}; end;".format(indent, out))
            
    recurse(0, 1)
    print("{}end;".format(indent))
    
    
for i in range(0,10):
    print(i)
    
for j in range(0,rf.n_estimators):
    tree_to_code2(rf.estimators_[j], list(X_test))

dir(rf)

tree_to_code(rf.estimators_[1],list(X_test))

len(rf.estimators_)
10

dt.tree_.value[1].sum()
dt.tree_.value[1].shape

dt.tree_.value[1][0,1]
dt.tree_.max_n_classes


## Try on a RF model from a small dataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import datasets
from sklearn.tree import _tree
from sklearn.model_selection import train_test_split

type(make_classification)
# function

iris = datasets.load_iris()
type(iris)
sklearn.utils.Bunch
dat = iris.data


X, y = make_classification(n_samples=1000, n_features=5,
                           n_informative=3, n_redundant=0,
                           random_state=0,shuffle=False)

type(X)
X_df = pd.DataFrame(X)
type(X_df)

# Apply train test split to generate 
#   X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split(
        X_df,y,test_size=0.2,random_state=0)

pd.DataFrame(X).shape

list(pd.DataFrame(X))

# Define a RF classifier
clf=RandomForestClassifier(max_depth=3, random_state=0)
rf = clf.fit(X_train,y_train) 
clf.fit(X_test,y_test) 


pd.Series(y_train).value_counts() # needs to be applied to series
0    418
1    382

#Apply trees in the forest to X, return leaf indices.
clf.apply(X_train)
clf.apply(X_train).shape
(800, 10)

pd.Series(clf.predict(X_train)).value_counts()
1    472
0    328


# clf and clf.fit are both randomforest classifiers!

type(clf.fit(X_train,y_train))
sklearn.ensemble.forest.RandomForestClassifier
type(clf)
sklearn.ensemble.forest.RandomForestClassifier

type(clf.predict_proba(X_test))

len(rf.estimators_)
# 10 trees

type(rf.estimators_[1])
sklearn.tree.tree.DecisionTreeClassifier

rf.estimators_[1].tree_.threshold
array([ 2.36480999, -0.12044846, -2.        , -2.        , -1.20683062,
       -2.        , -2.        ])
rf.estimators_[1].tree_.node_count
7
rf.estimators_[1].tree_.children_left
array([ 1,  2, -1, -1,  5, -1, -1], dtype=int64)
rf.estimators_[1].tree_.children_right
array([ 4,  3, -1, -1,  6, -1, -1], dtype=int64)

rf.estimators_[1].tree_.value

rf.estimators_[1].tree_.feature

dir(rf.estimators_[1].tree_)
['__class__',
 '__delattr__',
 '__dir__',
 '__doc__',
 '__eq__',
 '__format__',
 '__ge__',
 '__getattribute__',
 '__getstate__',
 '__gt__',
 '__hash__',
 '__init__',
 '__init_subclass__',
 '__le__',
 '__lt__',
 '__ne__',
 '__new__',
 '__pyx_vtable__',
 '__reduce__',
 '__reduce_ex__',
 '__repr__',
 '__setattr__',
 '__setstate__',
 '__sizeof__',
 '__str__',
 '__subclasshook__',
 'apply',
 'capacity',
 'children_left',
 'children_right',
 'compute_feature_importances',
 'decision_path',
 'feature',
 'impurity',
 'max_depth',
 'max_n_classes',
 'n_classes',
 'n_features',
 'n_node_samples',
 'n_outputs',
 'node_count',
 'predict',
 'threshold',
 'value',
 'weighted_n_node_samples']

rf.estimators_[1].tree_.__class__
rf.estimators_[1].tree_.__dir__
rf.estimators_[1].tree_.__sizeof__
rf.estimators_[1].tree_.impurity
rf.estimators_[1].tree_.max_depth
rf.estimators_[1].tree_.apply()
rf.estimators_[1].tree_.decision_path(np.array(X_test))
rf.estimators_[1].tree_.decision_path(np.array(X_test).astype(np.float32)).todense()
rf.estimators_[1].tree_.decision_path(np.array(X_test).astype(np.float32)).shape
(200, 7)
# indicator for whether the point passed through for each node

rf.estimators_[1].tree_.n_features
5

rf.estimators_[1].tree_.n_node_samples
array([497, 482, 134, 348,  15,  12,   3], dtype=int64)

rf.estimators_[1].tree_.node_count
array([ 2,  1, -2, -2,  0, -2, -2], dtype=int64)

tree.export_graphviz(rf.estimators_[1],out_file='tree.dot')
import os
os.getcwd()

# After generating this file, copy and paste the content to
# http://webgraphviz.com/
# for visualization


type(np.array(X_test))
np.array(X_test).shape

np.array(X_test).astype(np.float32)
rf.feature_importances_ 

X_train.dtype.names
# no names...
list(X_train)

tree_to_code(rf.estimators_[1],list(X_train))

    
test_data = pd.read_csv('O:\\GAML Technology\\_Analytics\\Model\\POC RF 13AA\\Data\\dat_91_94_final.csv')
features = list(test_data)

tree_to_code(t0.tree_,list())










import numpy as np
from sklearn import datasets
from sklearn import tree

# Load iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Build decision tree classifier
dt = tree.DecisionTreeClassifier(criterion='entropy')
dt.fit(X, y)


from sklearn.tree import _tree

tree_to_code(dt, list(iris.feature_names))









