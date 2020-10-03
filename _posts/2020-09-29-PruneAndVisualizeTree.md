---
title: "Pruning and Visualizing sklearn DecisionTreeClassifiers"
author: Markus Loecher
layout: post
permalink: PruningVisualizeTrees-Blog
tags: 
- Python-bloggers
output:
  md_document: 
    variant: markdown_github
    preserve_yaml: TRUE
  html_document: default
  toc: yes
  word_document: default
  pdf_document: default
---


This post serves two purposes:
1. It illustrates and compares three different methods of visualizing `DecisionTreeClassifiers` from sklearn.
2. It shows a simple quick way of manually pruning selected nodes from the tree.


```python
from dtreeviz.trees import *
from IPython.display import SVG  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.datasets import load_iris
import copy

#for plotting
import matplotlib.pyplot as plt
from sklearn import tree

import graphviz 

```

### dtreeviz
We are using the wonderful tree visualization library `dtreeviz` :
https://github.com/parrt/dtreeviz


```python
def ViewSVG(viz):
    from IPython.display import SVG  
    fname= viz.save_svg() 
    return SVG(fname)

```


```python
clf1 = tree.DecisionTreeClassifier(max_depth=3)  # limit depth of tree
iris = load_iris()
clf1.fit(iris.data, iris.target)

viz1 = dtreeviz(clf1, 
               iris.data, 
               iris.target,
               target_name='variety',
              feature_names=iris.feature_names, 
               class_names=["setosa", "versicolor", "virginica"]  # need class_names for classifier
              )  
ViewSVG(viz1)            

```




![svg](/assets/PruneAndVisualizeTree/output_4_0.svg)



We now selectively prune the last two children which belong to parent node #6:


```python
clf2 = copy.deepcopy(clf1)
#prune the tree
clf2.tree_.children_left[6] = -1
clf2.tree_.children_right[6]  = -1

viz2 = dtreeviz(clf2, 
               iris.data, 
               iris.target,
               target_name='variety',
              feature_names=iris.feature_names, 
               class_names=["setosa", "versicolor", "virginica"]  # need class_names for classifier
              )  
ViewSVG(viz2)            

```




![svg](/assets/PruneAndVisualizeTree/output_6_0.svg)



### Using `plot_tree` also works:


```python
plt.rcParams["figure.figsize"]=10,8

tmp=tree.plot_tree(clf1) 
```


![png](/assets/PruneAndVisualizeTree/output_8_0.png)



```python
plt.rcParams["figure.figsize"]=8,6

tmp=tree.plot_tree(clf2) 
```


![png](/assets/PruneAndVisualizeTree/output_9_0.png)


### Graphviz


```python
plt.rcParams["figure.figsize"]=5,5
dot_data = tree.export_graphviz(clf1, out_file=None, 
                    feature_names=iris.feature_names,  
                    class_names=iris.target_names,  
                    filled=True, rounded=True,  
                    special_characters=True)
graph = graphviz.Source(dot_data) 
graph
```




![svg](/assets/PruneAndVisualizeTree/output_11_0.svg)




```python
dot_data = tree.export_graphviz(clf2, out_file=None, 
                    feature_names=iris.feature_names,  
                    class_names=iris.target_names,  
                    filled=True, rounded=True,  
                    special_characters=True)
graph = graphviz.Source(dot_data) 
graph
```




![svg](/assets/PruneAndVisualizeTree/output_12_0.svg)


