import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20,5) # Set standard output figure size
#import seaborn as sns

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import make_scorer

from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

# Import Classifiers
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier

import lightgbm as lgb
import shap # Calculate feature importance for tree-based algorithms

# Import SKOPT modules for HPO
from skopt import forest_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_evaluations, plot_objective
from skopt import dump, load

from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


seed = 42
print('seed = ', seed)

print('libraries loaded')
##############################################################################################################################
# Define helper function
def df_info(df):
    print('-------------------------------------------shape----------------------------------------------------------------')
    print(df.shape)
    print('-------------------------------------head() and tail(1)---------------------------------------------------------')
    display(df.head(), df.tail(1))
    print('------------------------------------------nunique()-------------------------------------------------------------')
    print(df.nunique())
    print('-------------------------------------describe().round()---------------------------------------------------------')
    print(df.describe().round())
    print('--------------------------------------------info()--------------------------------------------------------------')
    print(df.info())
    print('-------------------------------------------isnull()-------------------------------------------------------------')
    print(df.isnull().sum())
    print('--------------------------------------------isna()--------------------------------------------------------------')
    print(df.isna().sum())
    print('-----------------------------------------duplicated()-----------------------------------------------------------')
    print(len(df[df.duplicated()]))
    print('----------------------------------------------------------------------------------------------------------------') 

# calculate area under precision-recall curve: auprc
def auprc(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)

# Define auprc scorer 
auprc_score = make_scorer(auprc, greater_is_better = True, needs_proba=True)

# Define auprc loss function
def auprc_loss(y_pred, data):    
    y_true = data.get_label()
    lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_pred)
    lr_auc = auc(lr_recall, lr_precision)
    return 'auprc', lr_auc, True #Bigger is better should be set to True

#-------------------------------------------------------------------------------------------------------------------------------------
# Single plot for test
def auprc_plot(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba) 
    plt.figure(figsize=(5,5)) #set figure size        
    plt.plot(recall, precision, marker='.', color='k', linewidth=1)
    plt.fill_between(recall, precision, 0, alpha=0.5)
    plt.fill_between(recall, precision, 1, alpha=0.5, color='r')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')    
    plt.show()

# # Two plots for train and test
# def plot_auprc(model, X_train, y_train, X_test, y_test):
#     model_name = model.__class__.__name__
#     print('Training', model_name, '...')
#     model.fit(X_train, y_train)
#     y_train_pred_proba = model.predict_proba(X_train)[:,1]
#     y_test_pred_proba = model.predict_proba(X_test)[:,1]
#     #print(model_name, ': train_auprc =', auprc(y_train, y_train_pred_proba), '; test auprc =', auprc(y_test, y_test_pred_proba))
    
#     precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_pred_proba)  
#     precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred_proba) 
    
#     fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(11,5))
#     ax1.set_title('train_auprc = ' + str(auprc(y_train, y_train_pred_proba)))
#     ax1.plot(recall_train, precision_train, marker='.', color='k', linewidth=1)
#     ax1.fill_between(recall_train, precision_train, 0, alpha=0.5)
#     ax1.fill_between(recall_train, precision_train, 1, alpha=0.5, color='r')
#     ax1.axis([0, 1, 0, 1])
#     ax1.set_xlabel('Recall')
#     ax1.set_ylabel('Precision') 
    
#     ax2.set_title('test auprc = ' + str(auprc(y_test, y_test_pred_proba)))
#     ax2.plot(recall_test, precision_test, marker='.', color='k', linewidth=1)
#     ax2.fill_between(recall_test, precision_test, 0, alpha=0.5)
#     ax2.fill_between(recall_test, precision_test, 1, alpha=0.5, color='r')
#     ax2.axis([0, 1, 0, 1])
#     ax2.set_xlabel('Recall')
#     ax2.set_ylabel('Precision')  
    
#     fig.suptitle(model_name)
#     plt.show()

#One plot train test overlaying each other
def plot_auprc(model, X_train, y_train, X_test, y_test):
    model_name = model.__class__.__name__
    print('Training', model_name, '...')
    
    model.fit(X_train, y_train)
    y_train_pred_proba = model.predict_proba(X_train)[:,1]
    y_test_pred_proba = model.predict_proba(X_test)[:,1]
    #print(model_name, ': train_auprc =', auprc(y_train, y_train_pred_proba), '; test auprc =', auprc(y_test, y_test_pred_proba))
    
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_pred_proba)  
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred_proba) 
    
    plt.figure(figsize=(5,5))
    title = model_name + ': auprc train, test = ' + str(round(auprc(y_train, y_train_pred_proba),4)) + ', ' + str(round(auprc(y_test, y_test_pred_proba),4))
    plt.title(title)
    plt.plot(recall_train, precision_train, marker='.', color='b', linewidth=1, label='train')
    plt.fill_between(recall_train, precision_train, 0, alpha=0.25, color='b')
    plt.fill_between(recall_train, precision_train, 1, alpha=0.25, color='r')
    
    plt.plot(recall_test, precision_test, marker='.', color='r', linewidth=1, label='test')
    plt.fill_between(recall_test, precision_test, 0, alpha=0.25, color='b')
    plt.fill_between(recall_test, precision_test, 1, alpha=0.25, color='r')
    
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision') 
    plt.legend()
    plt.show()   
#-------------------------------------------------------------------------------------------------------------------------------------------------
##Tests
#model = LogisticRegression()
#model.fit(X_train, y_train)
#y_test_pred_proba = model.predict_proba(X_test)[:,1]
#print(auprc(y_test, y_test_pred_proba))
#plot_auprc(y_test, y_test_pred_proba)
#plot_auprc(model, X_train, y_train, X_test, y_test)

# Define custom metric optimization function
def LGBM_custom_metric(X_train, X_test, y_train, y_test, loss, num_boost_round, early_stopping_rounds, lambda_l2, verbose_eval):
    
    lgb_train_data = lgb.Dataset(X_train, label=y_train)
    lgb_test_data = lgb.Dataset(X_test, label=y_test)
#_________________________________________________________________________
    
    params = {'objective':        'binary',      
              'lambda_l2':        lambda_l2,              
              }
    
    evals_result = {}  # to record eval results for plotting
    model = lgb.train(params,
                      lgb_train_data,
                      valid_sets=[lgb_train_data, lgb_test_data],
                      valid_names=['train','test'],                      
                      num_boost_round = num_boost_round, 
                      early_stopping_rounds=early_stopping_rounds,
                      feval=loss,
                      evals_result=evals_result,  
                      verbose_eval=verbose_eval)

# Define custom metric optimization function
def LGBM_custom_metric_cv(X_train, X_test, y_train, y_test, loss, nfold, num_boost_round, early_stopping_rounds, lambda_l2, verbose_eval):
    
    lgb_train_data = lgb.Dataset(X_train, label=y_train)
    lgb_test_data = lgb.Dataset(X_test, label=y_test)
#_________________________________________________________________________
    
    params = {'objective':        'binary',             
              'lambda_l2':        lambda_l2,
              'seed':             42
             }
    
    evals_result = {}  # to record eval results for plotting
    eval_hist = lgb.cv(
                   params,
                   lgb_train_data,
                   num_boost_round = num_boost_round,
                   nfold=nfold,
                   early_stopping_rounds=early_stopping_rounds,
                   verbose_eval=verbose_eval,
                   feval=loss,
                   return_cvbooster=True
                       )
    
    return eval_hist

def plot_from_eval_hist(eval_hist, X_train, X_test, y_train, y_test):
    cvbooster = eval_hist['cvbooster']
    best_iteration = cvbooster.best_iteration

    y_train_pred_cv = cvbooster.predict(X_train, num_iteration=best_iteration)
    y_train_pred = np.mean(y_train_pred_cv, axis=0)
    y_test_pred_cv = cvbooster.predict(X_test, num_iteration=best_iteration)
    y_test_pred = np.mean(y_test_pred_cv, axis=0)

    print('auprc train = ', auprc(y_train, y_train_pred))
    auprc_plot(y_train,y_train_pred)
    print('auprc test = ', auprc(y_test,y_test_pred))
    auprc_plot(y_test,y_test_pred)
    

#HPO##############################################################################

def load_best_parameters(model):
    model_name = model.__class__.__name__
    try:
        res = load(r'output/models/'+model_name)
        param_names = res.param_names
        param_values = res.x
        best_pparameters_dict = dict(zip(param_names, param_values))
        model.set_params(**best_pparameters_dict)
        print(model_name, 'optimized parameters:', best_pparameters_dict)  
    except:
        print(model_name, 'parameters were not previously optimized')
    return model

# using external cv
def find_best_hyperparameters(model, X, y, dynamic_params_space, scoring, plot, nfold, **HPO_params):
    
    # filter these warnings - they are not consistent, arise even for float features
    from warnings import filterwarnings
    # simplefilter("ignore", UserWarning)
    filterwarnings("ignore", message="The objective has been evaluated at this point before", category=UserWarning)
  
    # Get model name
    model_name = model.__class__.__name__
    
    # Get dynamic parameters names: 
    @use_named_args(dynamic_params_space)
    def get_params_names(**dynamic_params):
        return list(dynamic_params.keys())    
    param_names = get_params_names(dynamic_params_space)
        
    # Define an objective function
    @use_named_args(dynamic_params_space)
    def objective(**dynamic_params):
        #model.set_params(**static_params)
        model.set_params(**dynamic_params) 
        cv = StratifiedKFold(n_splits=nfold, random_state=seed, shuffle=True)
        scores = cross_validate(model, X, y, cv=cv, scoring = scoring, n_jobs=-1)
        val_score = np.mean(scores['test_score'])
        return -val_score
    
    print(model_name, 'model training...')
    # Load previously trained results and get starting point (x0) as best model from previous run
    try:
        res = load(r'output/models/'+model_name)
        x0 = res.x       
    # If not trained before -> no initial point provided
    except:
        x0 = None
    
    res = forest_minimize(objective, dynamic_params_space, x0 = x0, **HPO_params)
    
    # add attribute - parameters names to the res
    res.param_names = param_names

    print('Optimized parameters:    ', res.param_names)
    print('Previous best parameters:', x0)
    print('Current  best parameters:', res.x)
    print('Best score:', -res.fun)
    
    # Saved optimization result  
    dump(res, r'output/models/'+model_name, store_objective=False)
        
    if plot == True:
        plt.figure(figsize=(5,2))
        plot_convergence(res)
        try:
            # plot_objective would not work if only one parameter was searched for
            plot_objective(res)
        except:
            pass
    plt.show()

# using external cv
def find_best_hyperparameters_sampling(model, X, y, dynamic_params_space, scoring, plot, nfold, **HPO_params):
    
    # filter these warnings - they are not consistent, arise even for float features
    from warnings import filterwarnings
    # simplefilter("ignore", UserWarning)
    filterwarnings("ignore", message="The objective has been evaluated at this point before", category=UserWarning)
  
    # Get model name
    model_name = model.__class__.__name__
    
    # Get dynamic parameters names: 
    @use_named_args(dynamic_params_space)
    def get_params_names(**dynamic_params):
        return list(dynamic_params.keys())    
    param_names = get_params_names(dynamic_params_space)
        
    # Define an objective function
    @use_named_args(dynamic_params_space)
    def objective(**dynamic_params):
        #model.set_params(**static_params)
        
        #--------------------thats the sampling part added--------------------------
        alpha_over = dynamic_params.pop('alpha_over')
        k_neighbors = dynamic_params.pop('k_neighbors')
        over = SMOTE(random_state=42, sampling_strategy=alpha_over, k_neighbors=k_neighbors)
        #----------------------------------------------------------------------------
        
        #print(dynamic_params)
        model.set_params(**dynamic_params) 
        
        #alpha_under = 0.01
        #under = RandomUnderSampler(random_state=42, sampling_strategy=alpha_under)
                
        pipeline = make_pipeline(over, model)
        
        cv = StratifiedKFold(n_splits=nfold, random_state=seed, shuffle=True)
        scores = cross_validate(pipeline, X, y, cv=cv, scoring = scoring, n_jobs=-1)
        val_score = np.mean(scores['test_score'])
        return -val_score
    
    print(model_name, 'model training...')
    # Load previously trained results and get starting point (x0) as best model from previous run
    try:
        res = load(r'output/models/'+model_name)
        x0 = res.x       
    # If not trained before -> no initial point provided
    except:
        x0 = None
    
    #print(dynamic_params_space)
    res = forest_minimize(objective, dynamic_params_space, x0 = x0, **HPO_params)
    
    # add attribute - parameters names to the res
    res.param_names = param_names
    print(param_names)

    print('Optimized parameters:    ', res.param_names)
    print('Previous best parameters:', x0)
    print('Current  best parameters:', res.x)
    print('Best score:', -res.fun)
    
    # Saved optimization result  
    dump(res, r'output/models/'+model_name, store_objective=False)
        
    if plot == True:
        plt.figure(figsize=(5,2))
        plot_convergence(res)
        try:
            # plot_objective would not work if only one parameter was searched for
            plot_objective(res)
        except:
            pass
    plt.show()

def load_best_parameters_sampling(model):
    model_name = model.__class__.__name__
    try:
        res = load(r'output/models/'+model_name)
        param_names = res.param_names
        param_values = res.x
        best_pparameters_dict = dict(zip(param_names, param_values))
        
        # --remove sampling parameters--
        alpha_over = best_pparameters_dict.pop('alpha_over')
        k_neighbors = best_pparameters_dict.pop('k_neighbors')
        #--------------------------------
        
        model.set_params(**best_pparameters_dict)
        print(model_name, 'optimized parameters:', best_pparameters_dict)  
    except:
        print(model_name, 'parameters were not previously optimized')
    return model, alpha_over, k_neighbors

def plot_shap_feature_importance(model, X, y):
    model.fit(X,y)
    shap_values = shap.TreeExplainer(model).shap_values(X)
    try:
        #this should work for classification tasks
        shap.summary_plot(shap_values, X, plot_type='bar', class_names=model.classes_)
    except:
        # this should work for regression tasks
        shap.summary_plot(shap_values, X, plot_type='bar')
    plt.show()


# # Find best hyperparameters using internal lightgbm cv function
# def find_best_hyperparameters_cv(model, X_train, y_train, dynamic_params_space, plot=True, nfold=5, **HPO_params):
    
#     # filter these warnings - they are not consistent, arise even for float features
#     from warnings import filterwarnings
#     filterwarnings("ignore", message="The objective has been evaluated at this point before", category=UserWarning)

#     # Get model name
#     model_name = model.__class__.__name__
    
#     lgb_train_data = lgb.Dataset(X_train, label=y_train)
    
#     # Get dynamic parameters names: 
#     @use_named_args(dynamic_params_space)
#     def get_params_names(**dynamic_params):
#         return list(dynamic_params.keys())    
#     param_names = get_params_names(dynamic_params_space)
        
#     # Define an objective function
#     @use_named_args(dynamic_params_space)
#     def objective(**dynamic_params):
#         #model.set_params(**static_params)
#         model.set_params(**dynamic_params)
        
#         #pipeline = make_pipeline(scaler, model)
#         #cv = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
            
#         evals_result = {}  # to record eval results for plotting
#         eval_hist = lgb.cv(
#                        dynamic_params,
#                        lgb_train_data,
#                        num_boost_round = 100,
#                        nfold=nfold,
#                        early_stopping_rounds=10,
#                        verbose_eval=-1,
#                        feval=auprc_loss,
#                        return_cvbooster=True
#                            )
#         val_score = eval_hist['auprc-mean'][-1]
#         return -val_score
    
#     print(model_name, 'model training...')
#     # Load previously trained results and get starting point (x0) as best model from previous run
#     try:
#         res = load(r'output/models/'+model_name)
#         x0 = res.x       
#     # If not trained before -> no initial point provided
#     except:
#         x0 = None
    
#     res = forest_minimize(objective, dynamic_params_space, x0 = x0, **HPO_params)
    
#     # add attribute - parameters names to the res
#     res.param_names = param_names

#     print('Optimized parameters:    ', res.param_names)
#     print('Previous best parameters:', x0)
#     print('Current  best parameters:', res.x)
#     print('Best score:', -res.fun)
    
#     # Saved optimization result  
#     dump(res, r'output/models/'+model_name, store_objective=False)
        
#     if plot == True:
#         plt.figure(figsize=(5,2))
#         plot_convergence(res)
#         try:
#             # plot_objective would not work if only one parameter was searched for
#             plot_objective(res)
#         except:
#             pass
#     plt.show()
    
#     return res


# HPO_params = {'n_calls': 300, 'n_random_starts': 20, 'random_state': 42}

# model = LGBMClassifier(verbose = -1, feature_pre_filter=False)

# dynamic_params_space  = [Integer(7, 2047, name='num_leaves'),
#                          Integer(2, 127, name='max_depth'),
#                          #Integer(63, 127, name='min_data_in_leaf'),
#                          #Integer(63, 2047, name='max_bin'),
#                          Real(1e-3, 1e3, prior='log-uniform', name='lambda_l2'),
#                          Real(1e-3, 1, prior='log-uniform', name='learning_rate')]

# res = find_best_hyperparameters_cv(model,  X_train, y_train, dynamic_params_space, nfold= 5, plot = True, **HPO_params)