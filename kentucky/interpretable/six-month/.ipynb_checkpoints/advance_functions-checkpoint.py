### AdaBoost -- one-depth decision tree
#def Adaboost(x, y, learning_rate, estimators, seed):
#    
#    import numpy as np
#    from sklearn.model_selection import KFold, GridSearchCV
#    from sklearn.ensemble import AdaBoostClassifier
#    
#    ### model & parameters
#    ada = AdaBoostClassifier(random_state=seed)
#    cross_validation = KFold(n_splits=5, shuffle=True, random_state=seed)
#    c_grid = {"n_estimators": estimators, 
#              "learning_rate": learning_rate}  
#    
#    ### nested cross validation
#    clf = GridSearchCV(estimator=ada, param_grid=c_grid, scoring='roc_auc',cv=cross_validation, return_train_score=True).fit(x,y)
#    train_score = clf.cv_results_['mean_train_score']
#    test_score = clf.cv_results_['mean_test_score']
#    test_std = clf.cv_results_['std_test_score']
#    
#    ### scores
#    best_auc = clf.best_score_
#    best_std = test_std[np.where(test_score == clf.best_score_)[0][0]]
#    best_param = clf.best_params_
#    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
#    
#    return best_auc, best_std, auc_diff, best_param



## GAM -- generalized additive model
def EBM(train_x, train_y, test_x, test_y, learning_rate, depth, estimators, seed):
    
    import numpy as np
    from sklearn.model_selection import KFold, GridSearchCV
    from interpret.glassbox import ExplainableBoostingClassifier
    from sklearn.metrics import roc_auc_score
    
    ### extract race & gender
    train_gender = train_x['Gender'].values
    train_race = train_x['Race'].values
    test_gender = test_x['Gender'].values
    test_race = test_x['Race'].values
    
    ### process train_x & test_x
    train_x = train_x.drop(['Race'], axis=1).values
    test_x = test_x.drop(['Race'], axis=1).values 
    
    ### model & parameters
    gam = ExplainableBoostingClassifier(random_state=seed)
    cross_validation = KFold(n_splits=5, shuffle=True, random_state=seed)
    c_grid = {"n_estimators": estimators, 
              "max_tree_splits": depth, 
              "learning_rate": learning_rate} 
    
    ### nested cross validation
    clf = GridSearchCV(estimator=gam, param_grid=c_grid, scoring='roc_auc',
                       cv=cross_validation, return_train_score=True).fit(train_x, train_y)
    train_score = clf.cv_results_['mean_train_score']
    test_score = clf.cv_results_['mean_test_score']
    test_std = clf.cv_results_['std_test_score']
    
    ### scores
    best_auc = clf.best_score_
    best_std = test_std[np.where(test_score == clf.best_score_)[0][0]]
    best_param = clf.best_params_
    auc_diff = train_score[np.where(test_score == clf.best_score_)[0][0]] - clf.best_score_
    
    gam = ExplainableBoostingClassifier(random_state=seed, 
                                        n_estimators = best_param['n_estimators'], 
                                        max_tree_splits = best_param['max_tree_splits'], 
                                        learning_rate = best_param['learning_rate']).fit(train_x, train_y)
    
    holdout_prob = gam.predict_proba(test_x)[:,1]
    holdout_pred = gam.predict(test_x)
    holdout_auc = roc_auc_score(test_y, holdout_prob) 
    
    return {'best_param': best_param, 
            'best_validation_auc': best_auc, 
            'best_validation_std': best_std, 
            'best_validation_auc_diff': auc_diff, 
            'holdout_test_proba': holdout_prob, 
            'holdout_test_pred': holdout_pred, 
            'holdout_test_auc': holdout_auc} 
