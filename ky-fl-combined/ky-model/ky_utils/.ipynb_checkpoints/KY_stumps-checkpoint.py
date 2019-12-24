import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_auc_score


def stump_cv(KY_x, KY_y,FL_x, FL_y, columns, c_grid, seed):
    
    ## estimator
    lasso = LogisticRegression(class_weight = 'balanced', solver='liblinear', random_state=seed, penalty='l1')
    cross_validation = KFold(n_splits=5, random_state=seed, shuffle=True)
    clf = GridSearchCV(estimator=lasso, 
                       param_grid=c_grid, 
                       scoring='roc_auc',
                       cv=cross_validation,
                       return_train_score=True).fit(KY_x, KY_y)

    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    ### best model
    best_model = LogisticRegression(class_weight = 'balanced', 
                                    solver='liblinear', 
                                    random_state=seed, 
                                    penalty='l1', 
                                    C=best_param['C']).fit(KY_x, KY_y)
    coefs = best_model.coef_[best_model.coef_ != 0]
    features = columns[best_model.coef_[0] != 0].tolist()
    intercept = best_model.intercept_[0]
    length = len(coefs)
       
    ## dictionary
    lasso_dict_rounding = {}
    for i in range(len(features)):
        lasso_dict_rounding.update({features[i]: coefs[i]})
        
    ## prediction on test set
    prob = 0
    for k in features:
        test_values = FL_x[k]*(lasso_dict_rounding[k])
        prob += test_values
    holdout_prob = np.exp(prob)/(1+np.exp(prob))
    FL_score = roc_auc_score(FL_y, holdout_prob)
        
    return {'best_auc': best_auc,
            'best_params': best_param,
            'number of coefs': length,
            'dictionary': lasso_dict_rounding,
            'auc_diffs': auc_diff,
            'FL_score': FL_score}
