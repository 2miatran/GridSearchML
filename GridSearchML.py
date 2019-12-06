import pandas as pd
# A small package to summarize data
from summary_statistics import suggesting_column_types, statistics_for_cat_data, statistics_for_cont_data

#!pip install mglearn
### Loadding Tools and Packages
##Basics
import pandas as pd
import numpy as np
import sys, random
import math
try:
    import cPickle as pickle
except:
    import pickle
import string
import re
import os
import time
from tqdm import tqdm
from termcolor import colored, cprint


## ML and Stats 
from sklearn.feature_selection import SelectFromModel
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import sklearn.metrics as m
import sklearn.linear_model  as lm
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.tree import export_graphviz
import statsmodels.api as sm
#import statsmodels.formula.api as sm
import patsy
from scipy import stats

from sklearn.metrics import make_scorer
def RMSE(y_true,y_pred):
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    return sqrt(mean_squared_error(y_true,y_pred))    


# A small package to summarize data
from summary_statistics import suggesting_column_types, statistics_for_cat_data, statistics_for_cont_data
#import mglearn
from sklearn.metrics import confusion_matrix


## Visualization

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import cm
#%matplotlib inline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode()

from IPython.display import HTML
import seaborn as sns
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import make_scorer


### ML model parts
# k-nearest neighbor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# Linear model
from sklearn.linear_model import (LogisticRegression, Ridge, Lasso)
# Tree based method
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              GradientBoostingRegressor, RandomForestRegressor)
from sklearn import tree

# Support vector machine
from sklearn.svm import SVC, LinearSVC, SVR

# neural network
from sklearn.neural_network import MLPClassifier, MLPRegressor

def RMSE(y_true,y_pred):
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    return sqrt(mean_squared_error(y_true,y_pred))    

from sklearn.model_selection import train_test_split


### some selfhelp utils functions:
def scale_data(data):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    #print(scaler)
    scaler.fit(data)
    return scaler.transform(data)


def plot_feature_importances(model, coef, feature_list, ax):
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    feature_list = [str(x) for x in feature_list]
    n_features = len(feature_list)

    sns.barplot(x = feature_list, y = coef, ax= ax)

    ax.set_xticks(np.arange(n_features))
    ax.set_xticklabels(feature_list)
    ax.tick_params(labelrotation=45)
    ax.set_ylabel("Feature importance")
    ax.set_title(model)


def run_model_selection(X_train, y_train, X_test, y_test, selectionmodel, trainmodel,threshold = "median"):
    select = SelectFromModel(selectionmodel, threshold=threshold)
    # selects all features that have an importance measure greater than the threshold
    # threshold="median": half of the features will be selected
    select.fit(X_train, y_train)
    X_train_l1 = select.transform(X_train)
    print("X_train.shape: {}".format(X_train.shape))
    print("X_train_l1.shape: {}".format(X_train_l1.shape))

    mask = select.get_support()
    # visualize the mask. black is True, white is False
    plt.matshow(mask.reshape(1, -1), cmap='gray_r')
    plt.xlabel("Sample index")
    plt.yticks(())

    #transform then apply LogisticRegression compared with not transform:

    score_before_variable_selection = trainmodel.fit(X_train, y_train).score(X_test, y_test)
    cprint("Test score before variable selections: {:.3f}".format(score_before_variable_selection), 'green', attrs=['bold'])

    X_test_l1 = select.transform(X_test)
    score_after_variable_selection = trainmodel.fit(X_train_l1, y_train).score(X_test_l1, y_test)
    cprint("Test score after variable selections: {:.3f}".format(score_after_variable_selection), 'green', attrs=['bold'])

    
def tnse_visuallization(df, stratified_by, column_to_drop, 
                        title ='Data visuallization using t-SNE', n_iter = 5000, perplexity = 30, init ='pca'):
    tnse_data = df.copy(deep=True)
    n_stratified = df[stratified_by].nunique()
    cmap = ListedColormap(sns.color_palette(palette= 'husl' , n_colors=n_stratified))
    fig = plt.figure(figsize=(15,5))
    fig.tight_layout() 
    fig.suptitle(title, fontsize=12)

    projections = ['rectilinear', '3d']
    n_dims = [2,3]
    i = 1
    for projection, n_dim, i in zip(projections, n_dims, range(len(n_dims))):

      tsne = TSNE(n_components=n_dim, init=init,random_state=0, perplexity= perplexity, n_iter=n_iter)
      tsne_results = tsne.fit_transform(tnse_data.drop(columns=column_to_drop))
      
      ax = fig.add_subplot(1, 2, i+1, projection = projection)
      ax.scatter(*zip(*tsne_results), c=tnse_data[stratified_by], cmap = cmap)

    
def plot_GridSearchResult(results, scoring, param_of_interest, lims, figsize):
    'support up to 2 parameters'
    param1, param2 =  ['param_' + param for param in param_of_interest]
    results[param2].fillna(0, inplace = True)
    results[param2] = results[param2].astype(int)
    #plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.Accent(np.linspace(0,1,5))))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = figsize)
    fig.suptitle("GridSearchCV evaluating using multiple scorers simultaneously", fontsize=16, color = 'C0') 
    #fig.tight_layout()
    sns.set_style("whitegrid") #darkgrid
    #cmap = ListedColormap(sns.color_palette(palette= 'husl' , n_colors=2))
    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results[param1], dtype=float)

    for scorer, i, ax in zip(sorted(scoring), range(len(scoring.keys())), (ax1, ax2)):

        sns.lineplot(x= param1,y='mean_train_%s' % (scorer), style =param2, data=results, ax = ax, label = 'Train')#, palette='husl')
        sns.lineplot(x= param1,y='mean_test_%s' % (scorer), style = param2, data=results, ax = ax, label = 'Test')#, palette='husl')
        
        best_x, best_y_train, best_y_test = results[abs(results['rank_test_%s' % (scorer)])==1][[param1, 'mean_train_%s' % (scorer),'mean_test_%s' % (scorer)]].values[0]

        ax.set_ylabel(scorer)
        xlim, ylim = lims[scorer][0], lims[scorer][1]
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.axvline(best_x, ls='--', color = 'C2')

    ax1.get_legend().remove()
    ax2.legend(bbox_to_anchor=(1.05, 0), loc='lower left', borderaxespad=0.)
    

    
def model_tuning(model_dict, X_train, y_train, X_test, y_test, cv = 5, 
                 scoring = 'roc_auc', scoring_name ='Results', refit = True, print_confuse = False):
    from sklearn.model_selection import GridSearchCV

    col_list = ['Model', 'Best Parameters', 'Train '+scoring_name, 'Test '+scoring_name]
    df_result = [] # store result for each grid search as list of df
    model_list = {} #append model for plotting later
    bestgridsearch = {}
    gridsearchresult ={}
    pred_values = {}
    for model_name in model_dict.keys():

        print('='*20,'Train and evaluate on', model_name, '='*20)
        model = model_dict[model_name]['model']
        param = model_dict[model_name]['params']
        
        X_train_, X_test_ = X_train, X_test # so that we dont modify on current data
        
        if model_name in ['MLP', 'Linear SVM', 'Logistic Regression']: #scale data if model are MLP or Log
            #Preprocessing data: scalingfor data between0 and 1'
            X_train_ = scale_data(X_train)
            X_test_ = scale_data(X_test)
        
        # define gridsearch object    
        grid_clf = GridSearchCV(model, param, scoring = scoring, cv = cv, return_train_score=True, refit=refit)
        bestgridsearch[model_name] = grid_clf #to access bestmodel outside
        
        grid_clf.fit(X_train_, y_train)
        gridsearchresult[model_name] = grid_clf.cv_results_
        

        # fit on entire training set
        train_score = grid_clf.score(X_train_,y_train) 
        
        #Evaluate values based on new parameters
        pred_values[model_name] = grid_clf.predict(X_test_)
        test_score = grid_clf.score(X_test_,y_test)

        if print_confuse: print(confusion_matrix(y_test, grid_clf.predict(X_test_)))
        ## append results, for this, only use accuracy
        df = pd.DataFrame([[model_name, grid_clf.best_params_, train_score, test_score]], columns=col_list)
        df_result.append(df)
        
        ## append model for plotting
        if model_name in ['Gradient Boosting Classifier' , 'Random Forest']:
            model_list[model_name] = grid_clf.best_estimator_.feature_importances_

    summary_table = pd.concat(df_result)
    return bestgridsearch, gridsearchresult, summary_table, model_list, pred_values
