import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import roc_auc_score


## EBM ###
def EBM(KY_x, KY_y, FL_x, FL_y, learning_rate, depth, estimators, seed):

    KY_validation = []
    KY_score = []
    FL_score = []
    auc_diff = []
    best_param = []
    KY_x = KY_x.drop(['person_id'], axis=1)
    FL_x = FL_x.drop(['person_id'], axis=1)
    
    ### model & parameters
    gam = ExplainableBoostingClassifier(random_state=seed)
    c_grid = {"n_estimators": estimators, 
              "max_tree_splits": depth, 
              "learning_rate": learning_rate} 

    ## cross validation set up 
    outer_cv = KFold(n_splits=5,shuffle=True, random_state=seed)
    inner_cv = KFold(n_splits=5,shuffle=True, random_state=seed)
    
    for outer_train, outer_test in outer_cv.split(KY_x, KY_y):
        
        ## split FL data -- only use 4 folds
        outer_train_x, outer_train_y = KY_x.iloc[outer_train], KY_y[outer_train]
        outer_test_x, outer_test_y = KY_x.iloc[outer_test], KY_y[outer_test]
        
        ### cross validation on 4 folds
        clf = GridSearchCV(estimator=gam, 
                           param_grid=c_grid, 
                           scoring='roc_auc',
                           cv=inner_cv,
                           return_train_score=True).fit(outer_train_x, outer_train_y)
        
        train_score = clf.cv_results_['mean_train_score']
        test_score = clf.cv_results_['mean_test_score']
        
        ## save results
        KY_validation.append(clf.best_score_)
        auc_diff.append(train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_)
        best_param.append(clf.best_params_)
        
        ## best model
        FL_score.append(roc_auc_score(FL_y, clf.predict_proba(FL_x)[:,1]))
        KY_score.append(roc_auc_score(outer_test_y, clf.predict_proba(outer_test_x)[:,1]))
    
    return {'auc_diff':auc_diff, 
            'best_param':best_param, 
            'KY_validation': KY_validation,
            'KY_score': KY_score,
            'FL_score':FL_score}



### CART
def CART(KY_x, KY_y, FL_x, FL_y, depth, seed):
 
    KY_validation = []
    KY_score = []
    FL_score = []
    auc_diff = []
    best_param = []
    KY_x = KY_x.drop(['person_id'], axis=1)
    FL_x = FL_x.drop(['person_id'], axis=1)
    
    ### model & parameters
    cart = DecisionTreeClassifier(random_state=seed)
    c_grid = {"max_depth": depth}
    
    ## cross validation set up 
    outer_cv = KFold(n_splits=5,shuffle=True, random_state=seed)
    inner_cv = KFold(n_splits=5,shuffle=True, random_state=seed)
    
    for outer_train, outer_test in outer_cv.split(KY_x, KY_y):
        
        ## split FL data -- only use 4 folds
        outer_train_x, outer_train_y = KY_x.iloc[outer_train], KY_y[outer_train]
        outer_test_x, outer_test_y = KY_x.iloc[outer_test], KY_y[outer_test]
        
        ### cross validation on 4 folds
        clf = GridSearchCV(estimator=cart, 
                           param_grid=c_grid, 
                           scoring='roc_auc',
                           cv=inner_cv,
                           return_train_score=True).fit(outer_train_x, outer_train_y)
        
        train_score = clf.cv_results_['mean_train_score']
        test_score = clf.cv_results_['mean_test_score']
        
        ## save results
        KY_validation.append(clf.best_score_)
        auc_diff.append(train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_)
        best_param.append(clf.best_params_)
        
        ## best model
        FL_score.append(roc_auc_score(FL_y, clf.predict_proba(FL_x)[:,1])) 
        KY_score.append(roc_auc_score(outer_test_y, clf.predict_proba(outer_test_x)[:,1]))
    
    return {'auc_diff':auc_diff, 
            'best_param':best_param, 
            'KY_validation': KY_validation,
            'KY_score': KY_score,
            'FL_score':FL_score}