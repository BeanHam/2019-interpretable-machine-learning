import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV


### XGBoost
def XGB(KY_x, KY_y, FL_x, FL_y, learning_rate, depth, estimators, gamma, child_weight, subsample, seed):
    
    KY_score = [] ## KY test
    FL_score = [] ## FL test
    FL_validation = [] ## FL validation
    auc_diff = []
    best_param = []
    KY_x = KY_x.drop(['person_id'], axis=1)
    FL_x = FL_x.drop(['person_id'], axis=1)
    
    ### model & parameters
    xgboost = xgb.XGBClassifier(random_state=seed)
    
    ## grid search
    c_grid = {"learning_rate": learning_rate, 
              "max_depth": depth, 
              "n_estimators": estimators, 
              "gamma": gamma, 
              "min_child_weight": child_weight, 
              "subsample": subsample}
    
    ## cross validation set up 
    outer_cv = KFold(n_splits=5,shuffle=True, random_state=seed)
    inner_cv = KFold(n_splits=5,shuffle=True, random_state=seed)
    
    for outer_train, outer_test in outer_cv.split(FL_x, FL_y):
        
        ## split FL data -- only use 4 folds
        outer_train_x, outer_train_y = FL_x.iloc[outer_train], FL_y[outer_train]
        outer_test_x, outer_test_y = FL_x.iloc[outer_test], FL_y[outer_test]
        
        ### cross validation on 4 folds
        clf = GridSearchCV(estimator=xgboost, 
                           param_grid=c_grid, 
                           scoring='roc_auc',
                           cv=inner_cv,
                           return_train_score=True).fit(outer_train_x, outer_train_y)
        
        train_score = clf.cv_results_['mean_train_score']
        test_score = clf.cv_results_['mean_test_score']
        
        ## save results
        FL_validation.append(clf.best_score_)
        auc_diff.append(train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_)
        best_param.append(clf.best_params_)
        
        ## best model
        FL_score.append(roc_auc_score(outer_test_y, clf.predict_proba(outer_test_x)[:,1])) 
        KY_score.append(roc_auc_score(KY_y, clf.predict_proba(KY_x)[:,1])) 
        
    return {'auc_diff':auc_diff, 
            'best_param':best_param, 
            'FL_validation': FL_validation,
            'KY_score':KY_score,
            'FL_score':FL_score}



### Random Forest
def RF(KY_x, KY_y, FL_x, FL_y, depth, estimators,impurity, seed):

    KY_score = []
    FL_validation = []
    auc_diff = []
    best_param = []
    KY_x = KY_x.drop(['person_id'], axis=1)
    FL_x = FL_x.drop(['person_id'], axis=1)
    
    ### model & parameters
    rf = RandomForestClassifier(bootstrap=True, random_state=seed)
    c_grid = {"n_estimators": estimators, 
              "max_depth": depth, 
              "min_impurity_decrease": impurity}
 
    ## cross validation set up 
    outer_cv = KFold(n_splits=5,shuffle=True, random_state=seed)
    inner_cv = KFold(n_splits=5,shuffle=True, random_state=seed)
    
    for outer_train, outer_test in outer_cv.split(FL_x, FL_y):
        
        ## split FL data -- only use 4 folds
        outer_train_x, outer_train_y = FL_x.iloc[outer_train], FL_y[outer_train]
        outer_test_x, outer_test_y = FL_x.iloc[outer_test], FL_y[outer_test]

        ### cross validation on 4 folds
        clf = GridSearchCV(estimator=rf, 
                           param_grid=c_grid, 
                           scoring='roc_auc',
                           cv=inner_cv,
                           return_train_score=True).fit(outer_train_x, outer_train_y)
        
        train_score = clf.cv_results_['mean_train_score']
        test_score = clf.cv_results_['mean_test_score']
        
        ## save results
        FL_validation.append(clf.best_score_)
        auc_diff.append(train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_)
        best_param.append(clf.best_params_)
        
        ## best model
        FL_score.append(roc_auc_score(outer_test_y, clf.predict_proba(outer_test_x)[:,1]))
        KY_score.append(roc_auc_score(KY_y, clf.predict_proba(KY_x)[:,1]))

    return {'auc_diff':auc_diff, 
            'best_param':best_param, 
            'FL_validation': FL_validation,
            'KY_score':KY_score, 
            'FL_score':FL_score}

### Linear SVM
def LinearSVM(KY_x, KY_y, FL_x, FL_y, C, seed):

    KY_score = []
    FL_validation = []
    auc_diff = []
    best_param = []
    KY_x = KY_x.drop(['person_id'], axis=1)
    FL_x = FL_x.drop(['person_id'], axis=1)
    
    ### model & parameters
    svm = LinearSVC(dual=False, max_iter=1e8, random_state=seed)
    c_grid = {"C": C}

    ## cross validation set up 
    outer_cv = KFold(n_splits=5,shuffle=True, random_state=seed)
    inner_cv = KFold(n_splits=5,shuffle=True, random_state=seed)
    
    
    for outer_train, outer_test in outer_cv.split(FL_x, FL_y):
        
        ## split FL data -- only use 4 folds
        outer_train_x, outer_train_y = FL_x.iloc[outer_train], FL_y[outer_train]
        outer_test_x, outer_test_y = FL_x.iloc[outer_test], FL_y[outer_test]

        ### cross validation on 4 folds
        clf = GridSearchCV(estimator=svm, 
                           param_grid=c_grid, 
                           scoring='roc_auc',
                           cv=inner_cv,
                           return_train_score=True).fit(outer_train_x, outer_train_y)
        
        train_score = clf.cv_results_['mean_train_score']
        test_score = clf.cv_results_['mean_test_score']
        
        ## save results
        FL_validation.append(clf.best_score_)
        auc_diff.append(train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_)
        best_param.append(clf.best_params_)
        
        ## best model
        best_model = CalibratedClassifierCV(clf, cv=5).fit(outer_train_x, outer_train_y)
        FL_score.append(roc_auc_score(outer_test_y, best_model.predict_proba(outer_test_x)[:,1]))
        KY_score.append(roc_auc_score(KY_y, best_model.predict_proba(KY_x)[:,1]))  
        
    return {'auc_diff':auc_diff, 
            'best_param':best_param, 
            'FL_validation': FL_validation,
            'KY_score':KY_score,
            'FL_score':FL_score}


### Lasso
def Lasso(KY_x, KY_y, FL_x, FL_y, C, seed):
    
    KY_score = []
    FL_validation = []
    auc_diff = []
    best_param = []
    KY_x = KY_x.drop(['person_id'], axis=1)
    FL_x = FL_x.drop(['person_id'], axis=1)
    
    ### model & parameters
    lasso = LogisticRegression(class_weight = 'balanced', solver='liblinear', random_state=seed, penalty='l1')
    c_grid = {"C": C}
    
    ## cross validation set up 
    outer_cv = KFold(n_splits=5,shuffle=True, random_state=seed)
    inner_cv = KFold(n_splits=5,shuffle=True, random_state=seed)    
    
    for outer_train, outer_test in outer_cv.split(FL_x, FL_y):
        
        ## split FL data -- only use 4 folds
        outer_train_x, outer_train_y = FL_x.iloc[outer_train], FL_y[outer_train]
        outer_test_x, outer_test_y = FL_x.iloc[outer_test], FL_y[outer_test]

        ### cross validation on 4 folds
        clf = GridSearchCV(estimator=lasso, 
                           param_grid=c_grid, 
                           scoring='roc_auc',
                           cv=inner_cv,
                           return_train_score=True).fit(outer_train_x, outer_train_y)
        
        train_score = clf.cv_results_['mean_train_score']
        test_score = clf.cv_results_['mean_test_score']
        
        ## save results
        FL_validation.append(clf.best_score_)
        auc_diff.append(train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_)
        best_param.append(clf.best_params_)
        
        ## best model
        FL_score.append(roc_auc_score(outer_test_y, clf.predict_proba(outer_test_x)[:,1]))
        KY_score.append(roc_auc_score(KY_y, clf.predict_proba(KY_x)[:,1])) 

    return {'auc_diff':auc_diff, 
            'best_param':best_param, 
            'FL_validation': FL_validation,
            'KY_score':KY_score, 
            'FL_score':FL_score}

### Logistic
def Logistic(KY_x, KY_y, FL_x, FL_y, C, seed):
    
    KY_score = []
    FL_validation = []
    auc_diff = []
    best_param = []
    KY_x = KY_x.drop(['person_id'], axis=1)
    FL_x = FL_x.drop(['person_id'], axis=1)
    
    ### model & parameters
    lr = LogisticRegression(class_weight = 'balanced', solver='liblinear', random_state=seed)
    c_grid = {"C": C}
    
    ## cross validation set up 
    outer_cv = KFold(n_splits=5,shuffle=True, random_state=seed)
    inner_cv = KFold(n_splits=5,shuffle=True, random_state=seed)
    
    for outer_train, outer_test in outer_cv.split(FL_x, FL_y):
        
        ## split FL data -- only use 4 folds
        outer_train_x, outer_train_y = FL_x.iloc[outer_train], FL_y[outer_train]
        outer_test_x, outer_test_y = FL_x.iloc[outer_test], FL_y[outer_test]

        ### cross validation on 4 folds
        clf = GridSearchCV(estimator=lr, 
                           param_grid=c_grid, 
                           scoring='roc_auc',
                           cv=inner_cv,
                           return_train_score=True).fit(outer_train_x, outer_train_y)
        
        train_score = clf.cv_results_['mean_train_score']
        test_score = clf.cv_results_['mean_test_score']
        
        ## save results
        FL_validation.append(clf.best_score_)
        auc_diff.append(train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_)
        best_param.append(clf.best_params_)
        
        ## best model
        FL_score.append(roc_auc_score(outer_test_y, clf.predict_proba(outer_test_x)[:,1]))
        KY_score.append(roc_auc_score(KY_y, clf.predict_proba(KY_x)[:,1]))
    
    return {'auc_diff':auc_diff, 
            'best_param':best_param, 
            'FL_validation': FL_validation,
            'KY_score':KY_score,
            'FL_score':FL_score}
