import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_auc_score


def stump_cv(KY_x, KY_y,FL_x, FL_y, columns, c_grid, seed):

    FL_score = []
    KY_validation = []
    auc_diff = []
    best_param = []
    KY_x = KY_x.drop(['person_id'], axis=1)
    FL_x = FL_x.drop(['person_id'], axis=1)
    
    ## cross validation set up 
    outer_cv = KFold(n_splits=5,shuffle=True, random_state=seed)
    inner_cv = KFold(n_splits=5,shuffle=True, random_state=seed)
    
    ## estimator
    lasso = LogisticRegression(class_weight = 'balanced', solver='liblinear', random_state=seed, penalty='l1')
    
    for outer_train, outer_test in outer_cv.split(KY_x, KY_y):
        
        ## split KY data -- only use 4 folds
        outer_train_x, outer_train_y = KY_x.iloc[outer_train], KY_y[outer_train]
        outer_test_x, outer_test_y = KY_x.iloc[outer_test], KY_y[outer_test]
        
        ### cross validation on 4 folds
        clf = GridSearchCV(estimator=lasso, 
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
        best_model = LogisticRegression(class_weight = 'balanced', 
                                        solver='liblinear', 
                                        random_state=seed, 
                                        penalty='l1', 
                                        C=clf.best_params_['C']).fit(outer_train_x, outer_train_y)
        coefs = best_model.coef_[best_model.coef_ != 0]
        features = columns[best_model.coef_[0] != 0].tolist()
        intercept = best_model.intercept_[0]
        
        ## dictionary
        lasso_dict = {}
        for i in range(len(features)):
            lasso_dict.update({features[i]: coefs[i]})
        
        ## prediction on test set -- FL_X
        prob = 0
        for k in features:
            test_values = FL_x[k]*(lasso_dict[k])
            prob += test_values
        holdout_prob = np.exp(prob)/(1+np.exp(prob)) 
        FL_score.append(roc_auc_score(FL_y, holdout_prob))     
        
    return {'best_params': best_param,
            'dictionary': lasso_dict,
            'auc_diff': auc_diff,
            'KY_validation':KY_validation,
            'FL_score': FL_score}


def stump_model(KY_x, KY_y,FL_x, FL_y, c, columns, seed):

    ## preprocess
    FL_x = FL_x.drop(['person_id'], axis=1)
    KY_x = KY_x.drop(['person_id'], axis=1)
    
    ## estimator
    lasso = LogisticRegression(class_weight = 'balanced', 
                               solver='liblinear', 
                               random_state=seed, 
                               penalty='l1', 
                               C = c).fit(KY_x, KY_y)
    coefs = lasso.coef_[lasso.coef_ != 0]
    features = columns[lasso.coef_[0] != 0].tolist()
    intercept = lasso.intercept_[0]
     
    ## dictionary
    lasso_dict = {}
    for i in range(len(features)):
        lasso_dict.update({features[i]: coefs[i]})
    
    ## prediction on test set
    prob = 0
    for k in features:
        test_values = FL_x[k]*(lasso_dict[k])
        prob += test_values
    
    holdout_prob = np.exp(prob)/(1+np.exp(prob))
    test_auc = roc_auc_score(FL_y, holdout_prob)
    
    return {'coefs': coefs, 
            'features': features, 
            'intercept': intercept, 
            'dictionary': lasso_dict, 
            'test_auc': test_auc}