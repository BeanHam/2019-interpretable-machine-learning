import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from utils.fairness_functions import compute_confusion_matrix_stats, compute_calibration_fairness, conditional_balance_positive_negative, fairness_in_auc, balance_positive_negative
from sklearn.calibration import CalibratedClassifierCV



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
    best_params = []
    train_auc = []
    validation_auc = []
    auc_diffs = []
    
    holdout_with_attr_test = []
    holdout_prediction = []
    holdout_probability = []
    holdout_y = []
    holdout_auc = []
    
    confusion_matrix_rets = []
    calibrations = []
    race_auc = []
    condition_pn = []
    no_condition_pn = []
    
    ## inner cv
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    
    for i in range(len(train_outer)):
        
        ## subset train & test sets in inner loop
        train_x, test_x = X.iloc[train_outer[i]], X.iloc[test_outer[i]]
        train_y, test_y = Y[train_outer[i]], Y[test_outer[i]]
        
        ## holdout test with "race" for fairness
        holdout_with_attrs = test_x.copy()
        
        ## remove unused feature in modeling
        train_x = train_x.drop(['person_id', 'screening_date', 'race'], axis=1).values
        test_x = test_x.drop(['person_id', 'screening_date', 'race'], axis=1).values
        
        ## GridSearch: inner CV
        clf = GridSearchCV(estimator=estimator, 
                           param_grid=c_grid, 
                           scoring='roc_auc',
                           cv=inner_cv, 
                           return_train_score=True).fit(train_x, train_y)

        ## best parameter & scores
        mean_train_score = clf.cv_results_['mean_train_score']
        mean_test_score = clf.cv_results_['mean_test_score']        
        best_param = clf.best_params_
        train_auc.append(mean_train_score[np.where(mean_test_score == clf.best_score_)[0][0]])
        validation_auc.append(clf.best_score_)
        auc_diffs.append(mean_train_score[np.where(mean_test_score == clf.best_score_)[0][0]] - clf.best_score_)

        ## train model on best param
        if index == 'svm':
            best_model = CalibratedClassifierCV(clf, cv=5)
            best_model.fit(train_x, train_y)
            prob = best_model.predict_proba(test_x)[:, 1]
            holdout_pred = best_model.predict(test_x)
        else:
            #best_model = clf.fit(train_x, train_y)
            #prob = best_model.predict_proba(test_x)[:, 1]
            #holdout_pred = best_model.predict(test_x)
            prob = clf.predict_proba(test_x)[:, 1]
            holdout_pred = clf.predict(test_x)
        
        ########################################################################
        ## confusion matrix stats
        confusion_matrix_fairness = compute_confusion_matrix_stats(df=holdout_with_attrs,
                                                                   preds=holdout_pred,
                                                                   labels=test_y, protected_variables=["sex", "race"])
        cf_final = confusion_matrix_fairness.assign(fold_num = [i]*confusion_matrix_fairness['Attribute'].count())
        confusion_matrix_rets.append(cf_final)
        
        ########################################################################
        ## calibration
        calibration = compute_calibration_fairness(df=holdout_with_attrs, 
                                                   probs=prob, 
                                                   labels=test_y, 
                                                   protected_variables=["sex", "race"])
        calibration_final = calibration.assign(fold_num = [i]*calibration['Attribute'].count())
        calibrations.append(calibration_final)
        
        ########################################################################
        ## race auc
        try:
            race_auc_matrix = fairness_in_auc(df = holdout_with_attrs,
                                              probs = prob, 
                                              labels = test_y)
            race_auc_matrix_final = race_auc_matrix.assign(fold_num = [i]*race_auc_matrix['Attribute'].count())
            race_auc.append(race_auc_matrix_final)
        except:
            pass
        
        ########################################################################
        ## ebm_pn
        no_condition_pn_matrix = balance_positive_negative(df = holdout_with_attrs,
                                                           probs = prob, 
                                                           labels = test_y)
        no_condition_pn_matrix_final = no_condition_pn_matrix.assign(fold_num = [i]*no_condition_pn_matrix['Attribute'].count())
        no_condition_pn.append(no_condition_pn_matrix_final)
        
        ########################################################################
        ## ebm_condition_pn
        condition_pn_matrix = conditional_balance_positive_negative(df = holdout_with_attrs,
                                                                    probs = prob, 
                                                                    labels = test_y)
        condition_pn_matrix_final = condition_pn_matrix.assign(fold_num = [i]*condition_pn_matrix['Attribute'].count())
        condition_pn.append(condition_pn_matrix_final)      
        
        ########################################################################
        ## store results
        holdout_with_attr_test.append(holdout_with_attrs)
        holdout_probability.append(prob)
        holdout_prediction.append(holdout_pred)
        holdout_y.append(test_y)
        holdout_auc.append(roc_auc_score(test_y, prob))
        best_params.append(best_param)

    ## confusion matrix
    confusion_df = pd.concat(confusion_matrix_rets, ignore_index=True)
    confusion_df.sort_values(["Attribute", "Attribute Value"], inplace=True)
    confusion_df = confusion_df.reset_index(drop=True)
    
    ## calibration matrix
    calibration_df = pd.concat(calibrations, ignore_index=True)
    calibration_df.sort_values(["Attribute", "Lower Limit Score", "Upper Limit Score"], inplace=True)
    calibration_df = calibration_df.reset_index(drop=True)
    
    ## race_auc
    race_auc_df = []
    try:
        race_auc_df = pd.concat(race_auc, ignore_index=True)
        race_auc_df.sort_values(["fold_num", "Attribute"], inplace=True)
        race_auc_df = race_auc_df.reset_index(drop=True)
    except:
        pass
    
    ## no_condition_pn
    no_condition_pn_df = pd.concat(no_condition_pn, ignore_index=True)
    no_condition_pn_df.sort_values(["fold_num", "Attribute"], inplace=True)
    no_condition_pn_df = no_condition_pn_df.reset_index(drop=True)
    
    ## condition_pn
    condition_pn_df = pd.concat(condition_pn, ignore_index=True)
    condition_pn_df.sort_values(["fold_num", "Attribute"], inplace=True)
    condition_pn_df = condition_pn_df.reset_index(drop=True)
    
    return {'best_param': best_params,
            'train_auc': train_auc,
            'validation_auc': validation_auc,
            'auc_diffs': auc_diffs,
            'holdout_test_auc': holdout_auc,
            'holdout_with_attrs_test': holdout_with_attr_test,
            'holdout_proba': holdout_probability,
            'holdout_pred': holdout_prediction,
            'holdout_y': holdout_y,
            'confusion_matrix_stats': confusion_df, 
            'calibration_stats': calibration_df, 
            'race_auc': race_auc_df, 
            'condition_pn': condition_pn_df, 
            'no_condition_pn': no_condition_pn_df}

