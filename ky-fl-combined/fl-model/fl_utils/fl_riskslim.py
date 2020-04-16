import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle

from pprint import pprint
from riskslim.helper_functions import load_data_from_csv, print_model
from riskslim.setup_functions import get_conservative_offset
from riskslim.coefficient_set import CoefficientSet
from riskslim.lattice_cpa import run_lattice_cpa
from riskslim.lattice_cpa import setup_lattice_cpa, finish_lattice_cpa



def risk_slim(data, max_coefficient, max_L0_value, c0_value, max_offset, max_runtime = 120,w_pos = 1):
    
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
    #coef_set = CoefficientSet(variable_names = data['variable_names'], lb = -max_coefficient, ub = max_coefficient, sign = 0)
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



def risk_cv(KY_x, KY_y, FL_x, FL_y, 
            y_label, 
            max_coef, 
            max_coef_number, 
            max_offset,
            max_runtime, 
            c, 
            seed):
    
    KY_score = []
    FL_score = []
    FL_validation = []
    
    ## set up basic values
    cols = FL_x.columns.tolist()
    sample_weights = np.repeat(1, len(FL_y))
    KY_x = KY_x.values
    KY_y[KY_y == -1] = 0 ## change -1 to 0
    
    ## cross validation set up 
    outer_cv = KFold(n_splits=5,shuffle=True, random_state=seed)
    inner_cv = KFold(n_splits=5,shuffle=True, random_state=seed)    
    
    for outer_train, test in outer_cv.split(FL_x, FL_y):
        
        ## split train & test
        outer_train_x, outer_train_y = FL_x.iloc[outer_train], FL_y[outer_train]
        outer_test_x, outer_test_y = FL_x.iloc[test], FL_y[test]
        outer_train_sample_weights = sample_weights[outer_train]
        
        ## inner loop
        for inner_train, validation in inner_cv.split(outer_train_x, outer_train_y):
            
            ## split inner train & validation
            inner_train_x, inner_train_y = outer_train_x.iloc[inner_train].values, outer_train_y[inner_train]
            validation_x, validation_y = outer_train_x.iloc[validation].values, outer_train_y[validation]
            inner_train_sample_weights = outer_train_sample_weights[inner_train]
            validation_sample_weights = outer_train_sample_weights[validation]
            inner_train_y = inner_train_y.reshape(-1,1)
            
            ## new data
            new_train_data = {
                'X': inner_train_x,
                'Y': inner_train_y,
                'variable_names': cols,
                'outcome_name': y_label,
                'sample_weights': inner_train_sample_weights
            }
            
            ## modeling
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
            FL_validation.append(roc_auc_score(validation_y, validation_prob))     
        
        ## outer loop
        outer_train_x = outer_train_x.values
        outer_train_y = outer_train_y.reshape(-1,1)
        
        ## new data
        new_train_data = {
            'X': outer_train_x,
            'Y': outer_train_y,
            'variable_names': cols,
            'outcome_name': y_label,
            'sample_weights': outer_train_sample_weights
        }   
        
        ## fit the model
        model_info, mip_info, lcpa_info = risk_slim(new_train_data, 
                                                    max_coefficient=max_coef, 
                                                    max_L0_value=max_coef_number, 
                                                    c0_value=c, 
                                                    max_runtime=max_runtime, 
                                                    max_offset = max_offset)
        print_model(model_info['solution'], new_train_data)          
    
        ## KY test
        KY_prob = riskslim_prediction(KY_x, np.array(cols), model_info).reshape(-1,1)
        KY_score.append(roc_auc_score(KY_y, KY_prob))     
        
        ## FL_test
        outer_test_x = outer_test_x.values[:, 1:]
        outer_test_y[outer_test_y == -1] = 0 ## change -1 to 0
        FL_prob = riskslim_prediction(outer_test_x, np.array(cols), model_info).reshape(-1,1)
        FL_score.append(roc_auc_score(outer_test_y, FL_prob))     
        
    return {'KY_score': KY_score,
            'FL_score': FL_score,
            'FL_validation': FL_validation}

    