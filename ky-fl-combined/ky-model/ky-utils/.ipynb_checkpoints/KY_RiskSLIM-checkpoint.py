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

    cols = KY_x.columns.tolist()
    sample_weights = np.repeat(1, len(KY_y))
    FL_x = FL_x.values
    FL_y = FL_y.reshape(-1,1)
    FL_y[FL_y == -1] = 0 ## change -1 to 0
    
    ## set up cross validation
    cv = KFold(n_splits=5, random_state=seed, shuffle=True)
    test_auc = []
    
    for train, test in cv.split(KY_x, KY_y):
        
        train_x, train_y = KY_x.iloc[train], KY_y[train]
        test_x, test_y = KY_x.iloc[test], KY_y[test]
        sample_weights_train = sample_weights[train]
        
        ## set up data
        train_x = train_x.values
        train_y = train_y.reshape(-1,1)
        
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
                                                    max_offset = max_offset,
                                                    c0_value=c, 
                                                    max_runtime=max_runtime)
        print_model(model_info['solution'], new_train_data)
        
        ## change data format
        test_x = test_x.values
        test_y = test_y.reshape(-1,1)
        test_x = test_x[:,1:]## remove the first column, which is "intercept"
        test_y[test_y == -1] = 0 ## change -1 to 0
    
        ## probability & accuracy
        test_prob = riskslim_prediction(test_x, np.array(cols), model_info).reshape(-1,1)
        test_auc.append(roc_auc_score(test_y, test_prob))
        
    ### build model on the whole FL data
    KY_x = KY_x.values
    KY_y = KY_y.reshape(-1,1)
    new_train_data = {
        'X': KY_x,
        'Y': KY_y,
        'variable_names': cols,
        'outcome_name': y_label,
        'sample_weights': sample_weights
    }
    model_info, mip_info, lcpa_info = risk_slim(new_train_data, 
                                                max_coefficient=max_coef, 
                                                max_L0_value=max_coef_number, 
                                                max_offset = max_offset,
                                                c0_value=c, 
                                                max_runtime=max_runtime)
    print_model(model_info['solution'], new_train_data)
    
    ## KY score
    FL_prob = riskslim_prediction(FL_x, np.array(cols), model_info).reshape(-1,1)
    FL_score = roc_auc_score(FL_y, FL_prob)
        
    return {'FL_score': FL_score,
            'KY_train_score': test_auc}

    