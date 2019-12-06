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


classifier_model_hyperparameter = {
    
    "K-Nearest Neighbors": 
            {'model': KNeighborsClassifier(),
                 'params': 
                            {
                            'n_neighbors': [1, 5, 10],
                            'leaf_size': [4, 10]
                            }
                           
            },
    
    "Logistic Regression": 
            {'model': LogisticRegression(),
             'params' : 
                            {
                             'C': [0.01,0.1,1,10,100],
                             'penalty': ['l1','l2'],
                             'solver' : ['liblinear']   
                            }
                           
            },
             
    "Linear SVM": 
            {'model': SVC(),
             'params': 
                            {
                             'C': [1, 10, 100, 1000],
                             'gamma': [0.1, 0.01, 0.001, 0.0001],
                             'kernel': ['linear', 'rbf'],
                             'random_state': [117]
                            }
                           
            },
    "Gradient Boosting Classifier": 
            {'model': GradientBoostingClassifier(),
             'params': 
                            {
                             'learning_rate': [0.05, 0.1],
                             'n_estimators' :[50, 100, 200],
                             'max_depth':[3,None],
                             'random_state': [117]
                            }
                           
            },
    "Decision Tree":
            {'model': tree.DecisionTreeClassifier(),
             'params': 
                            {
                             'max_depth':[3, 4, 5, 10, None],
                             'random_state': [117]
                            }
                             
            },
    "Random Forest": 
            {'model': RandomForestClassifier(),
             'params': 
                            {
                              'n_estimators': [2, 4,8],
                              'random_state': [117]
                            }
                              
             
            },

     "MLP": 
            {'model': MLPClassifier(),
             'params': 
                            {
                              'solver': ['lbfgs' , 'adam', 'sgd'],
                              'activation': ['relu'], 
                              'alpha': [0.001, 0.01, 0.1, 0.3],  
                              #'hidden_layer_sizes': [[2],[5],[10],[10, 10], [10, 20],[20, 20]],
                                'hidden_layer_sizes': [[10, 20],[40, 20], [100, 80], [200, 100], [100, 200]],
                               'random_state': [117],
                               'max_iter': [200] 
                            }
                              
             
            }

}


### routine code to do manual hyperparameter tuning on different models
regressor_model_hyperparameter = {
    
    "K-Nearest Neighbors": 
            {'model': KNeighborsRegressor(),
                 'params': 
                            {
                            'n_neighbors': [1, 5],
                            'leaf_size': [4, 10],
                
                            }
                           
            },
    
    "Ridge Regression": 
            {'model': Ridge(),
             'params' : 
                            {
                             'alpha': [0.01,0.1,1,10,100]
                            }
                           
            },
    
    "Lasso Regression": 
            {'model': Lasso(),
             'params' : 
                            {
                             'alpha': [0.01,0.1,1,10,100]
                            }
                           
            },
             
    "Linear SVR": 
            {'model': SVR(),
             'params': 
                            {
                             'C': [1, 10, 100, 1000],
                             'gamma': [0.001, 0.0001],
                             'kernel': ['linear', 'rbf']
                            }
                           
            },
    "Gradient Boosting Regressor": 
            {'model': GradientBoostingRegressor(),
             'params': 
                            {
                             'learning_rate': [0.05, 0.1],
                             'n_estimators' :[50, 100, 200],
                             'max_depth':[3,1,None],
                             'random_state': [711]   
                            }
                           
            },
    "Decision Tree":
            {'model': tree.DecisionTreeRegressor(),
             'params': 
                            {
                             'max_depth':[3,5,10, 100, None],
                             'random_state': [711]   
                            }
                             
            },
    "Random Forest": 
            {'model': RandomForestRegressor(),
             'params': 
                            {
                              'n_estimators': [10, 20, 30, 50, 100, 200],
                               'max_depth': [None, 5, 10, 20],
                               'max_features': ['sqrt', 'auto'],
                               'random_state': [711] 
                            }
                                         
            },
     "MLP": 
            {'model': MLPRegressor(),
             'params': 
                            {
                              'solver': ['lbfgs' , 'adam', 'sgd'],
                              'alpha': [0.0001, 0.001, 0.01, 0.1, 0.3],  
                              'hidden_layer_sizes': [[2],[5],[10],[10, 10], [10, 20], [100, 200]],
                               'random_state': [711] 
                            }
                              
             
            }


}