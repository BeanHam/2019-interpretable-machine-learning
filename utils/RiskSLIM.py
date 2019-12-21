import numpy as np
import pandas as pd

from utils.fairness_functions import compute_confusion_matrix_stats, compute_calibration_fairness, conditional_balance_positive_negative, \
                                     fairness_in_auc, balance_positive_negative
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle

from pprint import pprint
from riskslim.helper_functions import load_data_from_csv, print_model
from riskslim.setup_functions import get_conservative_offset
from riskslim.coefficient_set import CoefficientSet
from riskslim.lattice_cpa import run_lattice_cpa
from riskslim.lattice_cpa import setup_lattice_cpa, finish_lattice_cpa



def riskslim_prediction(X, feature_name, model_info):
    
    """
    @parameters
    
    X: test input features (np.array)
    feature_name: feature names
    model_info: output from RiskSLIM model
    
    """
    
    ## initialize parameters
    dictionary = {}
    prob = np.zeros(len(X))
    scores = np.zeros(len(X))
    
    ## prepare statistics
    subtraction_score = model_info['solution'][0]
    coefs = model_info['solution'][1:]
    index = np.where(coefs != 0)[0]
    
    nonzero_coefs = coefs[index]
    features = feature_name[index]
    X_sub = X[:,index]
    
    ## build dictionaries
    for i in range(len(features)):
        single_feature = features[i]
        coef = nonzero_coefs[i]
        dictionary.update({single_feature: coef})
        
    ## calculate probability
    for i in range(len(X_sub)):
        summation = 0
        for j in range(len(features)):
            a = X_sub[i,j]
            summation += dictionary[features[j]] * a
        scores[i] = summation
    
    prob = 1/(1+np.exp(-(scores + subtraction_score)))
    
    return prob



def riskslim_accuracy(X, Y, feature_name, model_info, threshold=0.5):
    
    prob = riskslim_prediction(X, feature_name, model_info)
    pred = np.mean((prob > threshold) == Y)
    
    return pred



###############################################################################################################################
###########################################    unconstrained RiskSLIM    ######################################################
###############################################################################################################################

def risk_slim(data, max_coefficient, max_L0_value, c0_value, max_offset, max_runtime = 120, w_pos = 1):
    
    """
    @parameters:
    
    max_coefficient:  value of largest/smallest coefficient
    max_L0_value:     maximum model size (set as float(inf))
    max_offset:       maximum value of offset parameter (optional)
    c0_value:         L0-penalty parameter such that c0_value > 0; larger values -> 
                      sparser models; we set to a small value (1e-6) so that we get a model with max_L0_value terms
    max_runtime:      max algorithm running time
    w_pos:            relative weight on examples with y = +1; w_neg = 1.00 (optional)
    
    """
    
    # create coefficient set and set the value of the offset parameter
    coef_set = CoefficientSet(variable_names = data['variable_names'], lb = -max_coefficient, ub = max_coefficient, sign = 0)
    conservative_offset = get_conservative_offset(data, coef_set, max_L0_value)
    max_offset = min(max_offset, conservative_offset)
    coef_set['(Intercept)'].ub = max_offset
    coef_set['(Intercept)'].lb = -max_offset

    constraints = {
        'L0_min': 0,
        'L0_max': max_L0_value,
        'coef_set':coef_set,
    }
    
    # Set parameters
    settings = {
        # Problem Parameters
        'c0_value': c0_value,
        'w_pos': w_pos,

        # LCPA Settings
        'max_runtime': max_runtime,                         # max runtime for LCPA
        'max_tolerance': np.finfo('float').eps,             # tolerance to stop LCPA (set to 0 to return provably optimal solution)
        'display_cplex_progress': True,                     # print CPLEX progress on screen
        'loss_computation': 'lookup',                       # how to compute the loss function ('normal','fast','lookup')
        
        # LCPA Improvements
        'round_flag': False,                                # round continuous solutions with SeqRd
        'polish_flag': False,                               # polish integer feasible solutions with DCD
        'chained_updates_flag': False,                      # use chained updates
        'add_cuts_at_heuristic_solutions': True,            # add cuts at integer feasible solutions found using polishing/rounding
        
        # Initialization
        'initialization_flag': True,                        # use initialization procedure
        'init_max_runtime': 300.0,                          # max time to run CPA in initialization procedure
        'init_max_coefficient_gap': 0.49,

        # CPLEX Solver Parameters
        'cplex_randomseed': 0,                              # random seed
        'cplex_mipemphasis': 0,                             # cplex MIP strategy
    }
    

    # train model using lattice_cpa
    model_info, mip_info, lcpa_info = run_lattice_cpa(data, constraints, settings)
        
    return model_info, mip_info, lcpa_info


def risk_nested_cv(X, 
                   Y,
                   indicator,
                   y_label, 
                   max_coef,
                   max_coef_number,
                   max_runtime,
                   max_offset,
                   c,
                   seed):

    ## set up data
    sample_weights = np.repeat(1, len(Y))

    ## set up cross validation
    #outer_cv = KFold(n_splits=5, random_state=seed, shuffle=True)
    #inner_cv = KFold(n_splits=5, random_state=seed, shuffle=True)
    
    outer_cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    inner_cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    
    train_auc = []
    validation_auc = []
    test_auc = []
    holdout_with_attrs_test = []
    holdout_probability = []
    holdout_prediction = []
    holdout_y = []
    
    confusion_matrix_rets = []
    calibrations = []
    race_auc = []
    condition_pn = []
    no_condition_pn = []
    
    i = 0
    for outer_train, outer_test in outer_cv.split(X, Y):
        
        outer_train_x, outer_train_y = X.iloc[outer_train], Y[outer_train]
        outer_test_x, outer_test_y = X.iloc[outer_test], Y[outer_test]
        outer_train_sample_weight, outer_test_sample_weight = sample_weights[outer_train], sample_weights[outer_test]
        
        ## holdout test
        holdout_with_attrs = outer_test_x.copy().drop(['(Intercept)'], axis=1)
        holdout_with_attrs = holdout_with_attrs.rename(columns = {'sex1': 'sex'})
        
        ## remove unused feature in modeling
        if indicator == 1:
            outer_train_x = outer_train_x.drop(['person_id', 'screening_date', 'race', 'age_at_current_charge', 'p_charges'], axis=1)
            outer_test_x = outer_test_x.drop(['person_id', 'screening_date', 'race', 'age_at_current_charge', 'p_charges'], axis=1)
        else:
            outer_train_x = outer_train_x.drop(['person_id', 'screening_date', 'race','sex1','age_at_current_charge', 'p_charges'], axis=1)
            outer_test_x = outer_test_x.drop(['person_id', 'screening_date', 'race', 'sex1', 'age_at_current_charge', 'p_charges'], axis=1)
            
        cols = outer_train_x.columns.tolist()
        
        
        ## inner cross validation
        for inner_train, validation in inner_cv.split(outer_train_x, outer_train_y):
            
            ## subset train data & store test data
            inner_train_x, inner_train_y = outer_train_x.iloc[inner_train].values, outer_train_y[inner_train]
            validation_x, validation_y = outer_train_x.iloc[validation].values, outer_train_y[validation]
            inner_train_sample_weight = outer_train_sample_weight[inner_train]
            validation_sample_weight = outer_train_sample_weight[validation]
            inner_train_y = inner_train_y.reshape(-1,1)
       
            ## create new data dictionary
            new_train_data = {
                'X': inner_train_x,
                'Y': inner_train_y,
                'variable_names': cols,
                'outcome_name': y_label,
                'sample_weights': inner_train_sample_weight
            }
                
            ## fit the model
            model_info, mip_info, lcpa_info = risk_slim(new_train_data, 
                                                        max_coefficient=max_coef, 
                                                        max_L0_value=max_coef_number, 
                                                        c0_value=c, 
                                                        max_runtime=max_runtime, 
                                                        max_offset = max_offset)
            
            ## check validation auc
            validation_x = validation_x[:,1:] ## remove the first column, which is "intercept"
            validation_y[validation_y == -1] = 0 ## change -1 to 0
            validation_prob = riskslim_prediction(validation_x, np.array(cols), model_info)
            validation_auc.append(roc_auc_score(validation_y, validation_prob))
        
        ## outer loop
        outer_train_x = outer_train_x.values
        outer_test_x = outer_test_x.values
        outer_train_y = outer_train_y.reshape(-1,1)
        new_train_data = {
            'X': outer_train_x,
            'Y': outer_train_y,
            'variable_names': cols,
            'outcome_name': y_label,
            'sample_weights': outer_train_sample_weight
        }
                
        ## fit the model
        model_info, mip_info, lcpa_info = risk_slim(new_train_data, 
                                                    max_coefficient=max_coef, 
                                                    max_L0_value=max_coef_number, 
                                                    c0_value=c, 
                                                    max_runtime=max_runtime, 
                                                    max_offset = max_offset)
        print_model(model_info['solution'], new_train_data)  
        
        ## change data format
        outer_train_x, outer_test_x = outer_train_x[:,1:], outer_test_x[:,1:] ## remove the first column, which is "intercept"
        outer_train_y[outer_train_y == -1] = 0 ## change -1 to 0
        outer_test_y[outer_test_y == -1] = 0 ## change -1 to 0
        
        ## probability & accuracy
        outer_train_prob = riskslim_prediction(outer_train_x, np.array(cols), model_info).reshape(-1,1)
        outer_test_prob = riskslim_prediction(outer_test_x, np.array(cols), model_info)
        outer_test_pred = (outer_test_prob > 0.5)
        
        ########################
        ## AUC
        train_auc.append(roc_auc_score(outer_train_y, outer_train_prob))
        test_auc.append(roc_auc_score(outer_test_y, outer_test_prob))        
        
        ########################
        ## confusion matrix
        confusion_matrix_fairness = compute_confusion_matrix_stats(df=holdout_with_attrs,
                                                                   preds= outer_test_pred,
                                                                   labels= outer_test_y, 
                                                                   protected_variables=["sex", "race"])
        cf_final = confusion_matrix_fairness.assign(fold_num = [i]*confusion_matrix_fairness['Attribute'].count())
        confusion_matrix_rets.append(cf_final)
        
        ########################
        ## calibration matrix
        calibration = compute_calibration_fairness(df=holdout_with_attrs, 
                                                   probs=outer_test_prob, 
                                                   labels=outer_test_y, 
                                                   protected_variables=["sex", "race"])
        calibration_final = calibration.assign(fold_num = [i]*calibration['Attribute'].count())
        calibrations.append(calibration_final)
        
        ########################
        ## race auc
        try:
            race_auc_matrix = fairness_in_auc(df = holdout_with_attrs,
                                              probs = outer_test_prob, 
                                              labels = outer_test_y)
            race_auc_matrix_final = race_auc_matrix.assign(fold_num = [i]*race_auc_matrix['Attribute'].count())
            race_auc.append(race_auc_matrix_final)
        except:
            pass
        
        ########################
        ## ebm_pn
        no_condition_pn_matrix = balance_positive_negative(df = holdout_with_attrs,
                                                                 probs = outer_test_prob, 
                                                                  labels = outer_test_y)
        no_condition_pn_matrix_final = no_condition_pn_matrix.assign(fold_num = [i]*no_condition_pn_matrix['Attribute'].count())
        no_condition_pn.append(no_condition_pn_matrix_final)
        
        ########################
        ## ebm_condition_pn
        condition_pn_matrix = conditional_balance_positive_negative(df = holdout_with_attrs,
                                                                             probs = outer_test_prob, 
                                                                             labels = outer_test_y)
        condition_pn_matrix_final = condition_pn_matrix.assign(fold_num = [i]*condition_pn_matrix['Attribute'].count())
        condition_pn.append(condition_pn_matrix_final)   
        
        ########################
        ## store results
        holdout_with_attrs_test.append(holdout_with_attrs)
        holdout_probability.append(outer_test_prob)
        holdout_prediction.append(outer_test_pred)
        holdout_y.append(outer_test_y)
        
        i += 1
        
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
    
    return {'train_auc': train_auc,
            'validation_auc': validation_auc,
            'test_auc': test_auc, 
            'holdout_with_attrs_test': holdout_with_attrs_test,
            'holdout_proba': holdout_probability,
            'holdout_pred': holdout_prediction,
            'holdout_y': holdout_y,
            'confusion_matrix_stats': confusion_df, 
            'calibration_stats': calibration_df,             
            'race_auc': race_auc_df, 
            'condition_pn': condition_pn_df, 
            'no_condition_pn': no_condition_pn_df}



###############################################################################################################################
###########################################     constrained RiskSLIM     ######################################################
###############################################################################################################################

    
def risk_slim_constrain(data, max_coefficient, max_L0_value, c0_value, max_offset, max_runtime = 120, w_pos = 1):
    
    """
    @parameters:
    
    max_coefficient:  value of largest/smallest coefficient
    max_L0_value:     maximum model size (set as float(inf))
    max_offset:       maximum value of offset parameter (optional)
    c0_value:         L0-penalty parameter such that c0_value > 0; larger values -> 
                      sparser models; we set to a small value (1e-6) so that we get a model with max_L0_value terms
    max_runtime:      max algorithm running time
    w_pos:            relative weight on examples with y = +1; w_neg = 1.00 (optional)
    
    """
    
    # create coefficient set and set the value of the offset parameter
    coef_set = CoefficientSet(variable_names = data['variable_names'], lb = 0, ub = max_coefficient, sign = 0)
    conservative_offset = get_conservative_offset(data, coef_set, max_L0_value)
    max_offset = min(max_offset, conservative_offset)
    coef_set['(Intercept)'].ub = max_offset
    coef_set['(Intercept)'].lb = -max_offset

    constraints = {
        'L0_min': 0,
        'L0_max': max_L0_value,
        'coef_set':coef_set,
    }
    
    # Set parameters
    settings = {
        # Problem Parameters
        'c0_value': c0_value,
        'w_pos': w_pos,

        # LCPA Settings
        'max_runtime': max_runtime,                         # max runtime for LCPA
        'max_tolerance': np.finfo('float').eps,             # tolerance to stop LCPA (set to 0 to return provably optimal solution)
        'display_cplex_progress': True,                     # print CPLEX progress on screen
        'loss_computation': 'lookup',                       # how to compute the loss function ('normal','fast','lookup')
        
        # LCPA Improvements
        'round_flag': False,                                # round continuous solutions with SeqRd
        'polish_flag': False,                               # polish integer feasible solutions with DCD
        'chained_updates_flag': False,                      # use chained updates
        'add_cuts_at_heuristic_solutions': True,            # add cuts at integer feasible solutions found using polishing/rounding
        
        # Initialization
        'initialization_flag': True,                        # use initialization procedure
        'init_max_runtime': 300.0,                          # max time to run CPA in initialization procedure
        'init_max_coefficient_gap': 0.49,

        # CPLEX Solver Parameters
        'cplex_randomseed': 0,                              # random seed
        'cplex_mipemphasis': 0,                             # cplex MIP strategy
    }
    

    # train model using lattice_cpa
    model_info, mip_info, lcpa_info = run_lattice_cpa(data, constraints, settings)
        
    return model_info, mip_info, lcpa_info


def risk_nested_cv_constrain(X, 
                             Y,
                             indicator,
                             y_label, 
                             max_coef,
                             max_coef_number,
                             max_runtime,
                             max_offset,
                             c,
                             seed):

    ## set up data
    sample_weights = np.repeat(1, len(Y))

    ## set up cross validation
    outer_cv = KFold(n_splits=5, random_state=seed, shuffle=True)
    inner_cv = KFold(n_splits=5, random_state=seed, shuffle=True)
    
    train_auc = []
    validation_auc = []
    test_auc = []
    holdout_with_attrs_test = []
    holdout_probability = []
    holdout_prediction = []
    holdout_y = []
    
    confusion_matrix_rets = []
    calibrations = []
    race_auc = []
    condition_pn = []
    no_condition_pn = []
    
    i = 0
    for outer_train, outer_test in outer_cv.split(X, Y):
        
        outer_train_x, outer_train_y = X.iloc[outer_train], Y[outer_train]
        outer_test_x, outer_test_y = X.iloc[outer_test], Y[outer_test]
        outer_train_sample_weight, outer_test_sample_weight = sample_weights[outer_train], sample_weights[outer_test]
        
        ## holdout test
        holdout_with_attrs = outer_test_x.copy().drop(['(Intercept)'], axis=1)
        holdout_with_attrs = holdout_with_attrs.rename(columns = {'sex1': 'sex'})
        
        ## remove unused feature in modeling
        if indicator == 1:
            outer_train_x = outer_train_x.drop(['person_id', 'screening_date', 'race', 'age_at_current_charge', 'p_charges'], axis=1)
            outer_test_x = outer_test_x.drop(['person_id', 'screening_date', 'race', 'age_at_current_charge', 'p_charges'], axis=1)
        else:
            outer_train_x = outer_train_x.drop(['person_id', 'screening_date', 'race','sex1','age_at_current_charge', 'p_charges'], axis=1)
            outer_test_x = outer_test_x.drop(['person_id', 'screening_date', 'race', 'sex1', 'age_at_current_charge', 'p_charges'], axis=1)
            
        cols = outer_train_x.columns.tolist()
        
        
        ## inner cross validation
        for inner_train, validation in inner_cv.split(outer_train_x, outer_train_y):
            
            ## subset train data & store test data
            inner_train_x, inner_train_y = outer_train_x.iloc[inner_train].values, outer_train_y[inner_train]
            validation_x, validation_y = outer_train_x.iloc[validation].values, outer_train_y[validation]
            inner_train_sample_weight = outer_train_sample_weight[inner_train]
            validation_sample_weight = outer_train_sample_weight[validation]
            inner_train_y = inner_train_y.reshape(-1,1)
       
            ## create new data dictionary
            new_train_data = {
                'X': inner_train_x,
                'Y': inner_train_y,
                'variable_names': cols,
                'outcome_name': y_label,
                'sample_weights': inner_train_sample_weight
            }
                
            ## fit the model
            model_info, mip_info, lcpa_info = risk_slim_constrain(new_train_data, 
                                                                  max_coefficient=max_coef, 
                                                                  max_L0_value=max_coef_number, 
                                                                  c0_value=c, 
                                                                  max_runtime=max_runtime, 
                                                                  max_offset = max_offset)
            
            ## check validation auc
            validation_x = validation_x[:,1:] ## remove the first column, which is "intercept"
            validation_y[validation_y == -1] = 0 ## change -1 to 0
            validation_prob = riskslim_prediction(validation_x, np.array(cols), model_info)
            validation_auc.append(roc_auc_score(validation_y, validation_prob))
        
        ## outer loop
        outer_train_x = outer_train_x.values
        outer_test_x = outer_test_x.values
        outer_train_y = outer_train_y.reshape(-1,1)
        new_train_data = {
            'X': outer_train_x,
            'Y': outer_train_y,
            'variable_names': cols,
            'outcome_name': y_label,
            'sample_weights': outer_train_sample_weight
        }
                
        ## fit the model
        
        model_info, mip_info, lcpa_info = risk_slim_constrain(new_train_data, 
                                                              max_coefficient=max_coef, 
                                                              max_L0_value=max_coef_number, 
                                                              c0_value=c, 
                                                              max_runtime=max_runtime, 
                                                              max_offset = max_offset)
        print_model(model_info['solution'], new_train_data)  

        
        ## change data format
        outer_train_x, outer_test_x = outer_train_x[:,1:], outer_test_x[:,1:] ## remove the first column, which is "intercept"
        outer_train_y[outer_train_y == -1] = 0 ## change -1 to 0
        outer_test_y[outer_test_y == -1] = 0 ## change -1 to 0
        
        ## probability & accuracy
        outer_train_prob = riskslim_prediction(outer_train_x, np.array(cols), model_info).reshape(-1,1)
        outer_test_prob = riskslim_prediction(outer_test_x, np.array(cols), model_info)
        outer_test_pred = (outer_test_prob > 0.5)
        
        ########################
        ## AUC
        train_auc.append(roc_auc_score(outer_train_y, outer_train_prob))
        test_auc.append(roc_auc_score(outer_test_y, outer_test_prob))        
        
        ########################
        ## confusion matrix
        confusion_matrix_fairness = compute_confusion_matrix_stats(df=holdout_with_attrs,
                                                                   preds= outer_test_pred,
                                                                   labels= outer_test_y, 
                                                                   protected_variables=["sex", "race"])
        cf_final = confusion_matrix_fairness.assign(fold_num = [i]*confusion_matrix_fairness['Attribute'].count())
        confusion_matrix_rets.append(cf_final)
        
        ########################
        ## calibration matrix
        calibration = compute_calibration_fairness(df=holdout_with_attrs, 
                                                   probs=outer_test_prob, 
                                                   labels=outer_test_y, 
                                                   protected_variables=["sex", "race"])
        calibration_final = calibration.assign(fold_num = [i]*calibration['Attribute'].count())
        calibrations.append(calibration_final)
        
        ########################
        ## race auc
        try:
            race_auc_matrix = fairness_in_auc(df = holdout_with_attrs,
                                              probs = outer_test_prob, 
                                              labels = outer_test_y)
            race_auc_matrix_final = race_auc_matrix.assign(fold_num = [i]*race_auc_matrix['Attribute'].count())
            race_auc.append(race_auc_matrix_final)
        except:
            pass
        
        ########################
        ## ebm_pn
        no_condition_pn_matrix = balance_positive_negative(df = holdout_with_attrs,
                                                           probs = outer_test_prob, 
                                                           labels = outer_test_y)
        no_condition_pn_matrix_final = no_condition_pn_matrix.assign(fold_num = [i]*no_condition_pn_matrix['Attribute'].count())
        no_condition_pn.append(no_condition_pn_matrix_final)
        
        ########################
        ## ebm_condition_pn
        condition_pn_matrix = conditional_balance_positive_negative(df = holdout_with_attrs,
                                                                    probs = outer_test_prob, 
                                                                    labels = outer_test_y)
        condition_pn_matrix_final = condition_pn_matrix.assign(fold_num = [i]*condition_pn_matrix['Attribute'].count())
        condition_pn.append(condition_pn_matrix_final)   
        
        ########################
        ## store results
        holdout_with_attrs_test.append(holdout_with_attrs)
        holdout_probability.append(outer_test_prob)
        holdout_prediction.append(outer_test_pred)
        holdout_y.append(outer_test_y)
        
        i += 1
        
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
    
    return {'train_auc': train_auc,
            'validation_auc': validation_auc,
            'test_auc': test_auc, 
            'holdout_with_attrs_test': holdout_with_attrs_test,
            'holdout_proba': holdout_probability,
            'holdout_pred': holdout_prediction,
            'holdout_y': holdout_y,
            'confusion_matrix_stats': confusion_df, 
            'calibration_stats': calibration_df,             
            'race_auc': race_auc_df, 
            'condition_pn': condition_pn_df, 
            'no_condition_pn': no_condition_pn_df}


def risk_cv_constrain(X, 
                      Y,
                      indicator,
                      y_label, 
                      max_coef,
                      max_coef_number,
                      max_runtime,
                      max_offset,
                      c,
                      seed):

    ## set up data
    Y = Y.reshape(-1,1)
    sample_weights = np.repeat(1, len(Y))

    ## set up cross validation
    cv = KFold(n_splits=5, random_state=seed, shuffle=True)
    train_auc = []
    validation_auc = []
    
    i = 0
    for train, validation in cv.split(X, Y):
    
        ## subset train data & store test data
        train_x, train_y = X.iloc[train], Y[train]
        validation_x, validation_y = X.iloc[validation], Y[validation]
        sample_weights_train, sample_weights_validation = sample_weights[train], sample_weights[validation]
        
        ## holdout test with "race" for fairness
        holdout_with_attrs = validation_x.copy().drop(['(Intercept)'], axis=1)
        holdout_with_attrs = holdout_with_attrs.rename(columns = {'sex1': 'sex'})
        
        ## remove unused feature in modeling
        if indicator == 1:
            train_x = train_x.drop(['person_id', 'screening_date', 'race', 'age_at_current_charge', 'p_charges'], axis=1)
            validation_x = validation_x.drop(['person_id', 'screening_date', 'race', 'age_at_current_charge', 'p_charges'], axis=1).values
        else:
            train_x = train_x.drop(['person_id', 'screening_date', 'race','sex1','age_at_current_charge', 'p_charges'], axis=1)
            validation_x = validation_x.drop(['person_id', 'screening_date', 'race', 'sex1', 
                                              'age_at_current_charge', 'p_charges'], axis=1).values

        cols = train_x.columns.tolist()
        train_x = train_x.values
        
        ## create new data dictionary
        new_train_data = {
            'X': train_x,
            'Y': train_y,
            'variable_names': cols,
            'outcome_name': y_label,
            'sample_weights': sample_weights_train
        }
            
        ## fit the model
        model_info, mip_info, lcpa_info = risk_slim_constrain(new_train_data, 
                                                              max_coefficient=max_coef, 
                                                              max_L0_value=max_coef_number, 
                                                              max_offset=max_offset,
                                                              c0_value=c, 
                                                              max_runtime=max_runtime)
        print_model(model_info['solution'], new_train_data)
        
        ## change data format
        train_x, validation_x = train_x[:,1:], validation_x[:,1:] ## remove the first column, which is "intercept"
        train_y[train_y == -1] = 0 ## change -1 to 0
        validation_y[validation_y == -1] = 0 ## change -1 to 0
        
        ## probability & accuracy
        train_prob = riskslim_prediction(train_x, np.array(cols), model_info).reshape(-1,1)
        validation_prob = riskslim_prediction(validation_x, np.array(cols), model_info).reshape(-1,1)
        validation_pred = (validation_prob > 0.5)
        
        ## AUC
        train_auc.append(roc_auc_score(train_y, train_prob))
        validation_auc.append(roc_auc_score(validation_y, validation_prob))
        i += 1
    
    return {'train_auc': train_auc, 
            'validation_auc': validation_auc}




def risk_cv(X, 
            Y,
            indicator,
            y_label, 
            max_coef,
            max_coef_number,
            max_runtime,
            max_offset,
            c,
            seed):

    ## set up data
    Y = Y.reshape(-1,1)
    sample_weights = np.repeat(1, len(Y))

    ## set up cross validation
    cv = KFold(n_splits=5, random_state=seed, shuffle=True)
    train_auc = []
    validation_auc = []
    
    i = 0
    for train, validation in cv.split(X, Y):
    
        ## subset train data & store test data
        train_x, train_y = X.iloc[train], Y[train]
        validation_x, validation_y = X.iloc[validation], Y[validation]
        sample_weights_train, sample_weights_validation = sample_weights[train], sample_weights[validation]
        
        ## holdout test with "race" for fairness
        holdout_with_attrs = validation_x.copy().drop(['(Intercept)'], axis=1)
        holdout_with_attrs = holdout_with_attrs.rename(columns = {'sex1': 'sex'})
        
        ## remove unused feature in modeling
        if indicator == 1:
            train_x = train_x.drop(['person_id', 'screening_date', 'race', 'age_at_current_charge', 'p_charges'], axis=1)
            validation_x = validation_x.drop(['person_id', 'screening_date', 'race', 'age_at_current_charge', 'p_charges'], axis=1).values
        else:
            train_x = train_x.drop(['person_id', 'screening_date', 'race','sex1','age_at_current_charge', 'p_charges'], axis=1)
            validation_x = validation_x.drop(['person_id', 'screening_date', 'race', 'sex1', 
                                              'age_at_current_charge', 'p_charges'], axis=1).values

        cols = train_x.columns.tolist()
        train_x = train_x.values
        
        ## create new data dictionary
        new_train_data = {
            'X': train_x,
            'Y': train_y,
            'variable_names': cols,
            'outcome_name': y_label,
            'sample_weights': sample_weights_train
        }
            
        ## fit the model
        model_info, mip_info, lcpa_info = risk_slim(new_train_data, 
                                                    max_coefficient=max_coef, 
                                                    max_L0_value=max_coef_number, 
                                                    max_offset=max_offset,
                                                    c0_value=c, 
                                                    max_runtime=max_runtime)
        print_model(model_info['solution'], new_train_data)
        
        ## change data format
        train_x, validation_x = train_x[:,1:], validation_x[:,1:] ## remove the first column, which is "intercept"
        train_y[train_y == -1] = 0 ## change -1 to 0
        validation_y[validation_y == -1] = 0 ## change -1 to 0
        
        ## probability & accuracy
        train_prob = riskslim_prediction(train_x, np.array(cols), model_info).reshape(-1,1)
        validation_prob = riskslim_prediction(validation_x, np.array(cols), model_info).reshape(-1,1)
        validation_pred = (validation_prob > 0.5)
        
        ## AUC
        train_auc.append(roc_auc_score(train_y, train_prob))
        validation_auc.append(roc_auc_score(validation_y, validation_prob))
        i += 1
    
    return {'train_auc': train_auc, 
            'validation_auc': validation_auc}
