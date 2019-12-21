from sklearn.model_selection import KFold, GridSearchCV
import numpy as np
from sklearn.metrics import roc_auc_score
from utils.fairness_functions import compute_fairness
from sklearn.calibration import CalibratedClassifierCV

def cross_validate(X, Y, estimator, c_grid, seed):
    """Performs cross validation and selects a model given X and Y dataframes, 
    an estimator, a dictionary of parameters, and a random seed. 
    """
    # settings 
    n_splits = 5
    scoring = 'roc_auc'

    cross_validation = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    clf = GridSearchCV(estimator=estimator, param_grid=c_grid, scoring=scoring,
                       cv=cross_validation, return_train_score=True).fit(X, Y)
    mean_train_score = clf.cv_results_['mean_train_score']
    mean_test_score = clf.cv_results_['mean_test_score']
    test_std = clf.cv_results_['std_test_score']

    # scores
    best_auc = clf.best_score_
    best_std = test_std[np.where(mean_test_score == clf.best_score_)[0][0]]
    best_param = clf.best_params_
    auc_diff = mean_train_score[np.where(mean_test_score == clf.best_score_)[
        0][0]] - clf.best_score_

    return mean_train_score, mean_test_score, test_std, best_auc, best_std, best_param, auc_diff



def nested_cross_validate(X, Y, estimator, c_grid, seed, index = None):
    
    ## outer cv
    train_outer = []
    test_outer = []
    outer_cv = KFold(n_splits=5, random_state=seed, shuffle=True)
    
    ## 5 sets of train & test index
    for train, test in outer_cv.split(X, Y):
        train_outer.append(train)
        test_outer.append(test)
        
    ## storing lists
    holdout_auc = []
    best_params = []
    auc_diffs = []
    fairness_overviews = []

    ## inner cv
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    for i in range(len(train_outer)):
        
        ## subset train & test sets in inner loop
        train_x, test_x = X.iloc[train_outer[i]], X.iloc[test_outer[i]]
        train_y, test_y = Y[train_outer[i]], Y[test_outer[i]]
        
        ## holdout test with "race" for fairness
        holdout_with_attrs = test_x.copy()
        
        ## remove unused feature in modeling
        train_x = train_x.drop(['person_id', 'screening_date', 'race'], axis=1)
        test_x = test_x.drop(['person_id', 'screening_date', 'race'], axis=1)
        
        ## GridSearch: inner CV
        clf = GridSearchCV(estimator=estimator, param_grid=c_grid, scoring='roc_auc',
                           cv=inner_cv, return_train_score=True).fit(train_x, train_y)

        ## best parameter & scores
        mean_train_score = clf.cv_results_['mean_train_score']
        mean_test_score = clf.cv_results_['mean_test_score']        
        best_param = clf.best_params_
        auc_diffs.append(mean_train_score[np.where(mean_test_score == clf.best_score_)[0][0]] - clf.best_score_)

        ## train model on best param
        if index == 'svm':
            best_model = CalibratedClassifierCV(clf, cv=5)
            best_model.fit(train_x, train_y)
            prob = best_model.predict_proba(test_x)[:, 1]
            holdout_pred = best_model.predict(test_x)
        else:
            best_model = clf.fit(train_x, train_y)
            prob = best_model.predict_proba(test_x)[:, 1]
            holdout_pred = best_model.predict(test_x)

        ## fairness 
        holdout_fairness_overview = compute_fairness(df=holdout_with_attrs,
                                                     preds=holdout_pred,
                                                     labels=test_y)
        fairness_overviews.append(holdout_fairness_overview)

        ## store results
        holdout_auc.append(roc_auc_score(test_y, prob))
        best_params.append(best_param)

    return holdout_auc, best_params, auc_diffs, fairness_overviews
