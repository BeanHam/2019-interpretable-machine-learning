from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import utils.fairness_functions as fairness
import pandas as pd

def risktool(X, Y, label, seed=816):
    
    ## set up cross validation
    cv = KFold(n_splits=5,shuffle=True,random_state=seed)
    auc_summary = []
    race_auc = []
    condition_pn = []
    no_condition_pn = []
    
    i = 0
    for train, test in cv.split(X, Y):
        
        train_x, train_y = X.iloc[train], Y[train]
        test_x, test_y = X.iloc[test], Y[test]
        holdout_with_attrs = test_x.copy()
        
        ################################
        ## auc
        auc_summary.append(roc_auc_score(test_y, test_x[label].values))
        
        ################################
        ## race_auc
        try:
            fair_race_auc = fairness.fairness_in_auc(df = holdout_with_attrs,
                                                     probs = test_x[label],
                                                     labels = test_y)
            fair_race_auc_final = fair_race_auc.assign(fold_num = [i]*fair_race_auc['Attribute'].count())
            race_auc.append(fair_race_auc_final)
        except:
            pass
        
        ################################
        ## condition pn
        fair_condition_pn = fairness.conditional_balance_positive_negative(df = holdout_with_attrs,
                                                                           probs = test_x[label],
                                                                           labels = test_y)
        fair_condition_pn_final = fair_condition_pn.assign(fold_num = [i]*fair_condition_pn['Attribute'].count())
        condition_pn.append(fair_condition_pn_final)
        
        ################################
        ## no condition pn
        fair_no_condition_pn = fairness.balance_positive_negative(df = holdout_with_attrs,
                                                                    probs = test_x[label],
                                                                    labels = test_y)
        fair_no_condition_pn_final = fair_no_condition_pn.assign(fold_num = [i]*fair_no_condition_pn['Attribute'].count())
        no_condition_pn.append(fair_no_condition_pn_final)
        
        i += 1
        
    ## race_auc
    race_auc_summary = []
    try:
        race_auc_summary = pd.concat(race_auc, ignore_index=True)
        race_auc_summary.sort_values(["fold_num", "Attribute"], inplace=True)
        race_auc_summary = race_auc_summary.reset_index(drop=True)
    except:
        pass
    
    ## condition_pn
    condition_pn_summary = pd.concat(condition_pn, ignore_index=True)
    condition_pn_summary.sort_values(["fold_num", "Attribute"], inplace=True)
    condition_pn_summary = condition_pn_summary.reset_index(drop=True)
    
    ## no_condition_pn
    no_condition_pn_summary = pd.concat(no_condition_pn, ignore_index=True)
    no_condition_pn_summary.sort_values(["fold_num", "Attribute"], inplace=True)
    no_condition_pn_summary = no_condition_pn_summary.reset_index(drop=True)
    
    return {'auc': auc_summary,
            'race_auc': race_auc_summary,
            'condition_pn': condition_pn_summary,
            'no_condition_pn': no_condition_pn_summary}




