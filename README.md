# GridSearchML
Automatic model tuning using GridSearch,  summarize results, t-sne visuallization. And many more!

## Using t-distributed Stochastic Neighbor Embedding (t-SNE) - a tool to visualize high-dimensional data. 
Plotting data in 2D and 3D. Input your dataframe, with options to remove/ignore a list of columns, and color the data based on a column (your label column)
![T-SNE 2D and 3D](https://github.com/2miatran/GridSearchML/blob/master/t-sne%202D%20and%203D.jpg)

## GridSearch outputing best train, test score and model parameters for each algorithm in the hyperparameter dictionary
The model_tuning in GridSearchML allow you to import a dictionary. I have 2 standard gridsearch for classification and regression problems for you to modify as you wish in hyperparameter.py.
For each combination of parameters, model_tuning will do cross-validation with cv folds, using the scoring criteria from scoring parameter.
You can also print out confuse for the best models. 
![Dictionary of Hyperparameter](https://github.com/2miatran/GridSearchML/blob/master/Dictionary%20of%20Hyperparameter.jpg){:height="700px" width="400px"}
#### Parameters:
- dictionary_of_hyperparameter, X_train, y_train, X_test, y_test
- Optional: cv (default is 3), scoring, print_confuse

#### Returns:
- bestgridsearch: storing the best gridsearch model for each ML algorithm in the dictionary. For ex. You can access model by bestgridsearch['Logistic Regression'] and then make a prediction on new dataset.
- gridsearchresult: storing the .cv_results_ for each ML algorithm in the dictionary
- summary_table: as in picture
- model_list: a dictionary of {model:coef} for accessing importance features. Currently, I am only doing this for Random Forest and Gradient Boosting
- pred_values: prediction results for the test set for each best model. You can access this for further exploration/evaluation. 
![GridSearch](https://github.com/2miatran/GridSearchML/blob/master/GridSearchResults.jpg)

## Accessing results from model_tuning.
For example, we can use results from model_list, which contains {model:coef}, to plot important features. 
![Image description](https://github.com/2miatran/GridSearchML/blob/master/plot_feature_importance.jpg)
