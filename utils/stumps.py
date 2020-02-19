import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from utils.fairness_functions import compute_confusion_matrix_stats, compute_calibration_fairness, conditional_balance_positive_negative, \
                                     fairness_in_auc, balance_positive_negative


def stump_cv(X, Y, columns, c_grid, seed):
    
    ## estimator
    lasso = LogisticRegression(class_weight = 'balanced', solver='liblinear', random_state=seed, penalty='l1')
    
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
    train_auc = []
    validation_auc = []
    best_params = []
    auc_diffs = []
    holdout_with_attrs_test = []
    holdout_probability = []
    holdout_prediction = []
    holdout_y = []
    
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
        holdout_with_attrs = holdout_with_attrs.rename(columns = {'sex1':'sex'})
        
        ## remove unused feature in modeling
        train_x = train_x.drop(['person_id', 'screening_date', 'race', 'age_at_current_charge', 'p_charges'], axis=1)
        test_x = test_x.drop(['person_id', 'screening_date', 'race', 'age_at_current_charge', 'p_charges'], axis=1)
        
        ## GridSearch: inner CV
        clf = GridSearchCV(estimator=lasso, 
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
        
        ## run model with best parameter
        best_model = LogisticRegression(class_weight = 'balanced', 
                                        solver='liblinear', 
                                        random_state=seed, 
                                        penalty='l1', 
                                        C=best_param['C']).fit(train_x, train_y)
        coefs = best_model.coef_[best_model.coef_ != 0]
        features = columns[best_model.coef_[0] != 0].tolist()
        intercept = best_model.intercept_[0]
        
        ## dictionary
        lasso_dict = {}
        for j in range(len(features)):
            lasso_dict.update({features[j]: coefs[j]})
        
        ## prediction on test set
        prob = 0
        for k in features:
            test_values = test_x[k]*(lasso_dict[k])
            prob += test_values
        test_prob = np.exp(prob)/(1+np.exp(prob))
        test_pred = (test_prob > 0.5)
        
        ########################
        ## confusion matrix
        confusion_matrix_fairness = compute_confusion_matrix_stats(df=holdout_with_attrs,
                                                                   preds=test_pred,
                                                                   labels=test_y, 
                                                                   protected_variables=["sex", "race"])
        cf_final = confusion_matrix_fairness.assign(fold_num = [i]*confusion_matrix_fairness['Attribute'].count())
        confusion_matrix_rets.append(cf_final)
        
        ########################
        ## calibration matrix
        calibration = compute_calibration_fairness(df=holdout_with_attrs, 
                                                   probs=test_prob, 
                                                   labels=test_y, 
                                                   protected_variables=["sex", "race"])
        calibration_final = calibration.assign(fold_num = [i]*calibration['Attribute'].count())
        calibrations.append(calibration_final)

        
        ########################
        ## race auc
        try:
            race_auc_matrix = fairness_in_auc(df = holdout_with_attrs,
                                              probs = test_prob, 
                                              labels = test_y)
            race_auc_matrix_final = race_auc_matrix.assign(fold_num = [i]*race_auc_matrix['Attribute'].count())
            race_auc.append(race_auc_matrix_final)
        except:
            pass
        
        ########################
        ## ebm_pn
        no_condition_pn_matrix = balance_positive_negative(df = holdout_with_attrs,
                                                           probs = test_prob, 
                                                           labels = test_y)
        no_condition_pn_matrix_final = no_condition_pn_matrix.assign(fold_num = [i]*no_condition_pn_matrix['Attribute'].count())
        no_condition_pn.append(no_condition_pn_matrix_final)
        
        ########################
        ## ebm_condition_pn
        condition_pn_matrix = conditional_balance_positive_negative(df = holdout_with_attrs,
                                                                    probs = test_prob, 
                                                                    labels = test_y)
        condition_pn_matrix_final = condition_pn_matrix.assign(fold_num = [i]*condition_pn_matrix['Attribute'].count())
        condition_pn.append(condition_pn_matrix_final)      
        
        ########################
        ## store results
        holdout_auc.append(roc_auc_score(test_y, test_prob))
        best_params.append(best_param)
        holdout_with_attrs_test.append(holdout_with_attrs)
        holdout_probability.append(test_prob)
        holdout_prediction.append(test_pred)
        holdout_y.append(test_y)
    
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
    
    return {'best_params': best_params,
            'holdout_test_auc': holdout_auc,
            'auc_diffs': auc_diffs,
            'holdout_with_attrs_test': holdout_with_attrs_test,
            'holdout_proba': holdout_probability,
            'holdout_pred': holdout_prediction,
            'holdout_y': holdout_y,
            'confusion_matrix_stats': confusion_df, 
            'calibration_stats': calibration_df,             
            'race_auc': race_auc_df, 
            'condition_pn': condition_pn_df, 
            'no_condition_pn': no_condition_pn_df}



def stump_model(X_train, Y_train, X_test, Y_test, c, columns, seed):
    
    ## remove unused feature in modeling
    X_train = X_train.drop(['person_id', 'screening_date', 'race'], axis=1)
    X_test = X_test.drop(['person_id', 'screening_date', 'race'], axis=1)
    
    ## estimator
    lasso = LogisticRegression(class_weight = 'balanced', 
                               solver='liblinear', 
                               random_state=seed, 
                               penalty='l1', 
                               C = c).fit(X_train, Y_train)
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
        test_values = X_test[k]*(lasso_dict[k])
        prob += test_values
    
    holdout_prob = np.exp(prob)/(1+np.exp(prob))
    test_auc = roc_auc_score(Y_test, holdout_prob)
    
    return {'coefs': coefs, 
            'features': features, 
            'intercept': intercept, 
            'dictionary': lasso_dict, 
            'test_auc': test_auc}


def latex_stump_table(coefs, features, intercept, dictionary):
    print('\\begin{{tabular}}{{|l|r|r|}} \\hline')
    for i in range(len(dictionary)):
        sign = '+' if dictionary[features[i]] >= 0 else '-'
        print('{index}.'.format(index = i+1), features[i], '&',np.abs(dictionary[features[i]]), '&', sign+'...', '\\\\ \\hline')
    print('{}.'.format(len(dictionary)+1), 'Intercept', '&', round(intercept, 3), '&', sign+'...', '\\\\ \\hline')
    print('\\textbf{{ADD POINTS FROM ROWS 1 TO {length}}}  &  \\textbf{{SCORE}} & = ..... \\\\ \\hline'
              .format(length=len(dictionary)+1))
    print('\\multicolumn{{3}}{{l}}{{Pr(Y = 1) = exp(score/100) / (1 + exp(score/100))}} \\\\ \\hline')   

    

def stump_plots(features, coefs, indicator):
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    def stump_visulization(label, sub_features, features, coefs):
        cutoffs = []
        cutoff_values = []        
        cutoff_prep = []
        cutoff_values_prep = []
        
        ## select features
        if (label == 'age_at_current_charge'):
            
            ## sanity check
            if len(sub_features) == 1:
                cutoffs.append(int(sub_features[0][sub_features[0].find(label)+len(label):]))
                cutoff_values.append( coefs[np.where(np.array(features) == sub_features[0])[0][0]] )
                
                ## prepare values
                cutoff_prep.append(np.linspace(18, cutoffs[0]+0.5, 1000))
                cutoff_prep.append(np.linspace(cutoffs[0]+0.5, 70, 1000))
                cutoff_values_prep.append(np.repeat(cutoff_values[0], 1000))
                cutoff_values_prep.append(np.repeat(0, 1000))
                
                plt.figure(figsize=(4,3))
                plt.scatter(cutoff_prep, cutoff_values_prep, s=0.05)
                plt.title(label, fontsize=14)
                plt.ylabel('Contribution', fontsize=14)
                plt.show()
            else:
                for j in sub_features:
                    cutoff_values.append(coefs[np.where(np.array(features) == j)[0][0]])
                    cutoffs.append(int(j[j.find(label)+len(label):])) 
                
                cutoffs.insert(0,18)
                cutoffs.append(70)
                cutoff_values.append(0)
                
                ## prepare cutoff values for plots
                for n in range(len(cutoffs)-1):
                    cutoff_prep.append(np.linspace(cutoffs[n]+0.5, cutoffs[n+1]+0.5, 1000))
                    cutoff_values_prep.append(np.repeat(np.sum(cutoff_values[n:]), 1000)) 
                    
                ## visulization
                unique = np.unique(cutoff_values_prep)[::-1]
                unique_len = len(unique)
                
                plt.figure(figsize=(4,3))
                plt.scatter(cutoff_prep, cutoff_values_prep, s=0.05)
                #for m in range(1,unique_len):
                #    plt.vlines(x=cutoffs[m]-0.5, ymin=unique[m], ymax=unique[m-1], colors = "C0", linestyles='dashed')
                plt.title(label, fontsize=14)
                plt.ylabel('Contribution', fontsize=14)
                plt.show()
        else:
            ## sanity check
            if len(sub_features) == 1:
                cutoffs.append(int(sub_features[0][sub_features[0].find(label)+len(label):]))
                cutoff_values.append(coefs[np.where(np.array(features) == sub_features[0])[0][0]])
                
                ## prepare values
                cutoff_prep.append(np.linspace(-0.5, cutoffs[0]-0.5, 1000))
                cutoff_prep.append(np.linspace(cutoffs[0]-0.5, cutoffs[0]+0.5, 1000))
                cutoff_values_prep.append(np.repeat(0, 1000))
                cutoff_values_prep.append(np.repeat(cutoff_values[0], 1000))
                
                plt.figure(figsize=(4,3))
                plt.scatter(cutoff_prep, cutoff_values_prep, s=0.05)
                plt.title(label, fontsize=14)
                plt.ylabel('Contribution', fontsize=14)
                plt.show()     
            else:
                for j in sub_features:
                    cutoff_values.append(coefs[np.where(np.array(features) == j)[0][0]])
                    cutoffs.append(int(j[j.find(label)+len(label):])) 
    
                ## prepare cutoff values for plots
                cutoff_prep = []
                cutoff_values_prep = []
                
                for n in range(len(cutoffs)-1):
                    cutoff_prep.append(np.linspace(cutoffs[n]-0.5, cutoffs[n+1]-0.5, 1000))
                    cutoff_values_prep.append(np.repeat(np.sum(cutoff_values[:n+1]), 1000))    
                cutoff_prep.append(np.linspace(cutoffs[-1]-0.5, cutoffs[-1]+0.5, 1000))
                cutoff_values_prep.append(np.repeat(np.sum(cutoff_values), 1000))   
                
                ## visualization
                unique = np.unique(cutoff_values_prep)
                unique_len = len(unique)
                plt.figure(figsize=(4,3))
                plt.scatter(cutoff_prep, cutoff_values_prep, s=0.05)
                plt.title(label, fontsize=14)
                plt.ylabel('Contribution', fontsize=14)
                plt.show()  
                
    if indicator == 'FL':
        labels = ['sex', 'age_at_current_charge', 'p_arrest', 'p_charges', 'p_violence', 'p_felony', 'p_incarceration',
                  'p_misdemeanor', 'p_property','p_murder', 'p_assault', 'p_sex_offense', 'p_weapon', 'p_felprop_viol', 
                  'p_felassault', 'p_misdeassult', 'p_traffic', 'p_drug', 'p_dui', 'p_stalking', 'p_voyeurism', 'p_fraud', 
                  'p_stealing', 'p_trespass', 'ADE', 'Treatment', 'p_incarceration' 'p_fta_two_year', 'fta_two_year_plus', 
                  'p_pending_charge', 'p_probation', 'six_month', 'one_year', 'three_year', 'five_year', 'current_violence', 
                  'current_pending_charge', 'current_violence20', 'age_at_first_charge']
    else: 
        labels = ['sex', 'age_at_current_charge', 'p_arrest', 'p_charges', 'p_violence', 'p_felony', 'p_incarceration',
                  'p_misdemeanor', 'p_property','p_murder', 'p_assault', 'p_sex_offense', 'p_weapon', 'p_felprop_viol', 
                  'p_felassault', 'p_misdeassult', 'p_traffic', 'p_drug', 'p_dui', 'p_stalking', 'p_voyeurism', 'p_fraud', 
                  'p_stealing', 'p_trespass', 'ADE', 'Treatment', 'p_incarceration' 'p_fta_two_year', 'p_fta_two_year_plus', 
                  'p_pending_charge', 'p_probation', 'six_month', 'one_year', 'three_year', 'five_year', 'current_violence', 
                  'current_pending_charge', 'current_violence20']
    
    for i in labels:
        if i == 'p_fta_two_year':
            sub_features = np.array(np.array(features)[[i in k for k in features]])[:-1]
        else:
            sub_features = np.array(np.array(features)[[i in k for k in features]])
        if len(sub_features) == 0:
            continue
        stump_visulization(i, sub_features, features, coefs)
    
 
    