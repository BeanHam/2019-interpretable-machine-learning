import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import roc_auc_score


## GAM -- generalized additive model
def EBM(KY_x, KY_y, FL_x, FL_y, learning_rate, depth, estimators, holdout_split, seed):
    
    ### model & parameters
    gam = ExplainableBoostingClassifier(random_state=seed)
    cross_validation = KFold(n_splits=5, shuffle=True, random_state=seed)
    c_grid = {"n_estimators": estimators, 
              "max_tree_splits": depth, 
              "learning_rate": learning_rate, 
              'holdout_split': holdout_split} 
    
    ### nested cross validation
    clf = GridSearchCV(estimator=gam, 
                       param_grid=c_grid, 
                       scoring='roc_auc',
                       cv=cross_validation, 
                       return_train_score=True).fit(FL_x,FL_y)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    gam = clf.fit(FL_x, FL_y)
    KY_score = roc_auc_score(KY_y, gam.predict_proba(KY_x)[:,1])
    
    return {'best_auc':best_auc, 
            'auc_diff':auc_diff, 
            'best_param':best_param, 
            'KY_score':KY_score}



### CART
def CART(KY_x, KY_y, FL_x, FL_y, depth, split, impurity, seed):
    
    ### model & parameters
    cart = DecisionTreeClassifier(random_state=seed)
    cross_validation = KFold(n_splits=5,shuffle=True, random_state=seed)
    c_grid = {"max_depth": depth, 
              "min_samples_split": split, 
              "min_impurity_decrease": impurity}
    
    ### nested cross validation
    clf = GridSearchCV(estimator=cart, 
                       param_grid=c_grid, 
                       scoring='roc_auc',
                       cv=cross_validation, 
                       return_train_score=True).fit(FL_x,FL_y)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    cart = clf.fit(FL_x, FL_y)
    KY_score = roc_auc_score(KY_y, cart.predict_proba(KY_x)[:,1])
    return {'best_auc':best_auc, 
            'auc_diff':auc_diff, 
            'best_param':best_param, 
            'KY_score':KY_score}