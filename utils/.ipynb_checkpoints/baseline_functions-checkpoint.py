import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from utils.fairness_functions import compute_fairness
from utils.model_selection import nested_cross_validate
from interpret.glassbox import ExplainableBoostingClassifier




def EBM(X,Y, learning_rate=None, depth=None,estimators=None, holdout_split=None, seed=None):
    ### model & parameters
    ebm = ExplainableBoostingClassifier(random_state=seed)
    c_grid = {"n_estimators": estimators, 
              "max_tree_splits": depth, 
              "learning_rate": learning_rate, 
              "holdout_split": holdout_split}
    
    c_grid = {k: v for k, v in c_grid.items() if v is not None}
    
    holdout_auc, best_param, auc_diffs, fairness_overview = nested_cross_validate(X=X,
                                                                                  Y=Y,
                                                                                  estimator=ebm,
                                                                                  c_grid=c_grid,
                                                                                  seed=seed)
    return {'best_param': best_param,
            'holdout_test_auc': holdout_auc,
            'auc_diffs': auc_diffs,
            'fairness_overview': fairness_overview}
    

def XGB(X,Y,
        learning_rate=None, depth=None, estimators=None, gamma=None, child_weight=None, subsample=None,
        seed=None):

    ### model & parameters
    xgboost = xgb.XGBClassifier(random_state=seed)
    c_grid = {"learning_rate": learning_rate,
              "max_depth": depth,
              "n_estimators": estimators,
              "gamma": gamma,
              "min_child_weight": child_weight,
              "subsample": subsample}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}

    holdout_auc, best_param, auc_diffs, fairness_overview = nested_cross_validate(X=X,
                                                                                  Y=Y,
                                                                                  estimator=xgboost,
                                                                                  c_grid=c_grid,
                                                                                  seed=seed)
    return {'best_param': best_param,
            'holdout_test_auc': holdout_auc,
            'auc_diffs': auc_diffs,
            'fairness_overview': fairness_overview}


def RF(X, Y,
       depth=None, estimators=None, impurity=None,
       seed=None):

    ### model & parameters
    rf = RandomForestClassifier(bootstrap=True, random_state=seed)
    c_grid = {"n_estimators": estimators,
              "max_depth": depth,
              "min_impurity_decrease": impurity}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}

    holdout_auc, best_param, auc_diffs, fairness_overview = nested_cross_validate(X=X,
                                                                                  Y=Y,
                                                                                  estimator=rf,
                                                                                  c_grid=c_grid, 
                                                                                  seed=seed)
    return {'best_param': best_param,
            'holdout_test_auc': holdout_auc,
            'auc_diffs': auc_diffs,
            'fairness_overview': fairness_overview}


def CART(X, Y,
         depth=None, split=None, impurity=None,
         seed=None):

    ### model & parameters
    cart = DecisionTreeClassifier(random_state=seed)
    c_grid = {"max_depth": depth,
              "min_samples_split": split,
              "min_impurity_decrease": impurity}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}

    holdout_auc, best_param, auc_diffs, fairness_overview = nested_cross_validate(X=X,
                                                                                  Y=Y,
                                                                                  estimator=cart,
                                                                                  c_grid=c_grid,
                                                                                  seed=seed)

    return {'best_param': best_param,
            'holdout_test_auc': holdout_auc,
            'auc_diffs': auc_diffs,
            'fairness_overview': fairness_overview}


def LinearSVM(X, Y,
              C, 
              seed=None):
    
    ### model & parameters
    svm = LinearSVC(dual=False, max_iter=2e6, random_state=seed)
    c_grid = {"C": C}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}
    index = 'svm'
    
    holdout_auc, best_param, auc_diffs, fairness_overview = nested_cross_validate(X=X,
                                                                                  Y=Y,
                                                                                  estimator=svm,
                                                                                  c_grid=c_grid,
                                                                                  seed=seed, 
                                                                                  index = index)
    return {'best_param': best_param,
            'holdout_test_auc': holdout_auc,
            'auc_diffs': auc_diffs,
            'fairness_overview': fairness_overview}

def Lasso(X, Y,
          C,
          seed=None):
    
    ### model & parameters
    lasso = LogisticRegression(class_weight='balanced',
                               solver='liblinear', 
                               random_state=seed, 
                               penalty = 'l1')
    c_grid = {"C": C}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}

    holdout_auc, best_param, auc_diffs, fairness_overview = nested_cross_validate(X=X, 
                                                                                  Y=Y,
                                                                                  estimator=lasso,
                                                                                  c_grid=c_grid,                                 
                                                                                  seed=seed)
    return {'best_param': best_param,
            'holdout_test_auc': holdout_auc,
            'auc_diffs': auc_diffs,
            'fairness_overview': fairness_overview}


def Logistic(X, Y,
             C,
             seed=None):
    
    ### model & parameters
    lr = LogisticRegression(class_weight='balanced',
                            solver='liblinear', 
                            random_state=seed, 
                            penalty = 'l2')
    c_grid = {"C": C}
    c_grid = {k: v for k, v in c_grid.items() if v is not None}

    holdout_auc, best_param, auc_diffs, fairness_overview = nested_cross_validate(X=X, 
                                                                                  Y=Y,
                                                                                  estimator=lr,
                                                                                  c_grid=c_grid,                                 
                                                                                  seed=seed)
    return {'best_param': best_param,
            'holdout_test_auc': holdout_auc,
            'auc_diffs': auc_diffs,
            'fairness_overview': fairness_overview}
