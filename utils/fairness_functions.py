import pandas as pd 
import numpy as np 
from sklearn.metrics import confusion_matrix, roc_auc_score
from utils.load_settings import load_settings

settings = load_settings()
decoders = settings["decoders"]


def compute_confusion_matrix_stats(df, preds, labels, protected_variables):
    df.loc[:, "score"] = preds
    df.loc[:, "label_value"] = labels
    df['entity_id'] = df['person_id'].map(str) + " " + df["screening_date"].map(str)
    df = df[["entity_id", 
             "sex", 
             "race", 
             "score", 
             "label_value"]]
    # decode numeric encodings for cat var
    for decoder_name, decoder_dict in decoders.items():
        df = df.replace({decoder_name: decoder_dict})
    
    rows = []
    for var in protected_variables:
        variable_summary = {}
        for value in df[var].unique():
            predictions = df["score"][df[var]==value]
            labels = df["label_value"][df[var]==value]
            tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0,1]).ravel()
            # predictive parity
            ppv = tp / (tp + fp) if (tp+fp) != 0 else 0
            # false positive error rate balance
            fpr = fp / (fp + tn) if (fp+tn) !=0 else 0
            # false negative error rate balance
            fnr = fn / (fn + tp) if (fn+tp) != 0 else 0
            # equalized odds
            
            # conditional use accuracy equality
            
            # overall accuracy equality
            acc = (tp + tn)/ (tp + tn + fp + fn)
            
            # treatment equality
            ratio = fn / fp if fp is not 0 else 0
            
            # positive predictive value
            npv = tn / (tn + fn) if (tn+fn) is not 0 else 0
            # negative predictive value
            

            rows.append({
                "Attribute": var,
                "Attribute Value": value,
                "PPV": ppv,
                "NPV": npv,
                "FPR": fpr,
                "FNR": fnr,
                "Accuracy": acc,
                "Treatment Equality": ratio,
                "Individuals Evaluated On": len(labels)        
            })
    return pd.DataFrame(rows)


def compute_calibration_fairness(df, probs, labels, protected_variables):
    df.loc[:, "score"] = probs
    df.loc[:, "label_value"] = labels
    df['entity_id'] = df['person_id'].map(str) + " " + df["screening_date"].map(str)
    df = df[["entity_id", 
             "sex", 
             "race", 
             "score", 
             "label_value"]]
    # decode numeric encodings for cat var
    for decoder_name, decoder_dict in decoders.items():
        df = df.replace({decoder_name: decoder_dict})    
        
    rows = []
    for var in protected_variables:
        for value in df[var].unique():
            for window in [(0, 0.1), (0.1, 0.2), (0.2, 0.3), 
                           (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), 
                           (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1)]:
                lo = window[0]
                hi = window[1]
                predictions = df["score"][(df[var]==value) & (df["score"] >= lo) & (df["score"] < hi)]
                labels = df["label_value"][(df[var]==value) & (df["score"] >= lo) & (df["score"] < hi)]
                
                prob = labels.sum() / len(labels)     

                rows.append({
                    "Attribute": var,
                    "Attribute Value": value,
                    "Lower Limit Score": lo,
                    "Upper Limit Score": hi,
                    "Conditional Frequency": prob,
                    "Individuals Evaluated On": len(labels)        
                })
    return pd.DataFrame(rows)



def conditional_balance_positive_negative(df, probs, labels):
    df.loc[:, "score"] = probs
    df.loc[:, "label_value"] = labels
    df['entity_id'] = df['person_id'].map(str) + " " + df["screening_date"].map(str)
    df = df[["entity_id", 
             "sex", 
             "race", 
             "score",
             "p_charges",
             "age_at_current_charge",
             "label_value"]]
    # decode numeric encodings for cat var
    for decoder_name, decoder_dict in decoders.items():
        df = df.replace({decoder_name: decoder_dict})    
        
    rows = []
    for race in df["race"].unique():
        for age in df["age_at_current_charge"][df["race"]==race].unique():
            for label in df["label_value"][(df["race"]==race) & (df["age_at_current_charge"]==age)].unique():
                for priors in [0, 1, 2, 3, 4]:
                    if priors != 4:
                        scores = df["score"][(df["race"]==race) & (df["age_at_current_charge"]==age) & (df["label_value"]==label) & (df["p_charges"]==priors)]
                        if len(scores):
                            expectation = scores.sum() / len(scores)
                            rows.append({
                                #"Race": race,
                                "Attribute": "race",
                                "Attribute Value": race,
                                "Age": age,
                                "Prior": priors, 
                                "Label": label,
                                "Expected Score": expectation
                            })
                    else:
                        scores = df["score"][(df["race"]==race) & (df["age_at_current_charge"]==age) & (df["label_value"]==label) & (df["p_charges"]>=priors)]
                        if len(scores):
                            expectation = scores.sum() / len(scores)
                            rows.append({
                                #"Race": race,
                                "Attribute": "race",
                                "Attribute Value": race,
                                "Age": age,
                                "Prior": priors, 
                                "Label": label,
                                "Expected Score": expectation
                            })

    for sex in df["sex"].unique():
        for age in df["age_at_current_charge"][df["sex"]==sex].unique():
            for label in df["label_value"][(df["sex"]==sex) & (df["age_at_current_charge"]==age)].unique():
                for priors in [0, 1, 2, 3, 4]:
                    if priors != 4:
                        scores = df["score"][(df["sex"]==sex) & (df["age_at_current_charge"]==age) & (df["label_value"]==label) & (df["p_charges"]==priors)]
                        if len(scores):
                            expectation = scores.sum() / len(scores)
                            rows.append({
                                "Attribute": "sex",
                                "Attribute Value": sex,
                                "Age": age,
                                "Prior": priors, 
                                "Label": label,
                                "Expected Score": expectation
                            })
                    else:
                        scores = df["score"][(df["sex"]==sex) & (df["age_at_current_charge"]==age) & (df["label_value"]==label) & (df["p_charges"]>=priors)]
                        if len(scores):
                            expectation = scores.sum() / len(scores)
                            rows.append({
                                "Attribute": "sex",
                                "Attribute Value": sex,
                                "Age": age,
                                "Prior": priors, 
                                "Label": label,
                                "Expected Score": expectation
                            })
    return pd.DataFrame(rows)



def fairness_in_auc(df, probs, labels):
    df.loc[:, "score"] = probs
    df.loc[:, "label_value"] = labels
    df['entity_id'] = df['person_id'].map(str) + " " + df["screening_date"].map(str)
    df = df[["entity_id", 
             "sex", 
             "race", 
             "score",
             "p_charges",
             "age_at_current_charge",
             "label_value"]]
    # decode numeric encodings for cat var
    for decoder_name, decoder_dict in decoders.items():
        df = df.replace({decoder_name: decoder_dict})    
        
    rows = []
    
    for race in df["race"].unique():
        probs = df["score"][df["race"]==race]
        labels = df["label_value"][df["race"]==race]
        auc = roc_auc_score(labels, probs)
        rows.append({
            "Attribute": "race",
            "Attribute Value": race,
            "AUC": auc
        })
        
    for sex in df["sex"].unique():
        probs = df["score"][df["sex"]==sex]
        labels = df["label_value"][df["sex"]==sex]
        auc = roc_auc_score(labels, probs)
        rows.append({
            "Attribute": "sex",
            "Attribute Value": sex,
            "AUC": auc
        })

    return pd.DataFrame(rows)




def balance_positive_negative(df, probs, labels):
    df.loc[:, "score"] = probs
    df.loc[:, "label_value"] = labels
    df['entity_id'] = df['person_id'].map(str) + " " + df["screening_date"].map(str)
    df = df[["entity_id", 
             "sex", 
             "race", 
             "score",
             "p_charges",
             "age_at_current_charge",
             "label_value"]]
    # decode numeric encodings for cat var
    for decoder_name, decoder_dict in decoders.items():
        df = df.replace({decoder_name: decoder_dict})    
        
    rows = []
    for race in df["race"].unique():
        for outcome in df["label_value"][df["race"]==race].unique():
            scores = df["score"][(df["race"]==race)&(df["label_value"]==outcome)]
            rows.append({
                "Attribute": "race",
                "Attribute Value": race,              
                "Label": outcome,
                "Expected Score": scores.mean()
            })
    for sex in df["sex"].unique():
        for outcome in df["label_value"][df["sex"]==sex].unique():
            scores = df["score"][(df["sex"]==sex)&(df["label_value"]==outcome)]
            rows.append({
                "Attribute": "sex",
                "Attribute Value": sex,              
                "Label": outcome,
                "Expected Score": scores.mean()
            })
    return pd.DataFrame(rows)



def compute_calibration_discrete_score(long_df:pd.DataFrame, 
                                        problem_name:str, 
                                        score_name:str) -> (pd.DataFrame, pd.DataFrame):
    """Returns dataframes of calibration values for discrete-valued score
    
    Keyword arguments: 
        long_df -- 
        problem_name -- 
        score_name -- 
    Returns:
        calib -- dataframe with the calibration values over all groups 
        calib_grps -- dataframe with the calibration values for each sensitive grp
    """
    # compute calibration overall
    calib = (long_df[[score_name, problem_name]]
                       .groupby(score_name)
                       .agg(['sum', 'size'])
                       .reset_index())

    calib.columns = [score_name, 'n_inds_recid', 'total_inds']
    calib["P(Y = 1 | Score = score)"] =  calib['n_inds_recid'] / calib['total_inds']
    
    # compute calibration for sensitive groups
    calib_grps = (long_df[[score_name, problem_name, 'Attribute Value']]
                           .groupby([score_name, 'Attribute Value'])
                           .agg(['sum', 'size'])
                           .reset_index())

    calib_grps.columns = [score_name, 'Attribute Value', 'n_inds_recid', 'total_inds']
    calib_grps["P(Y = 1 | Score = score, Attr = attr)"] =  calib_grps['n_inds_recid'] / calib_grps['total_inds']
    
    return calib, calib_grps



def parse_calibration_matrix(calibration_matrix: pd.DataFrame, 
                             problem_name:str, 
                             score_name:str):
    
    calibration_matrix[score_name] = (calibration_matrix["Lower Limit Score"].astype(str) + "-" 
                                     + calibration_matrix["Upper Limit Score"].astype(str))
    calibration_matrix.drop(columns = ['Lower Limit Score', 'Upper Limit Score'], inplace=True)
    calibration_matrix.rename(columns={'Conditional Frequency': "P(Y = 1 | Score = score, Attr = attr, Fold = fold)"}, 
                              inplace=True)
    
    # compute calibration by sensitive attribute (average out the fold)
    # filter entries where # inds evaluated on is 0
    calib_grps = (calibration_matrix[calibration_matrix["Individuals Evaluated On"] != 0]
                            .groupby([score_name, 'Attribute', 'Attribute Value']).apply(lambda x: np.average(x["P(Y = 1 | Score = score, Attr = attr, Fold = fold)"], 
                                                                         weights=x["Individuals Evaluated On"]))
                           .reset_index()
                           .rename(columns={0: "P(Y = 1 | Score = score, Attr = attr)"}))

    # put back the groups where the # inds eval on is 0
    num_inds = (calibration_matrix[[score_name, 'Attribute', 'Attribute Value', 'Individuals Evaluated On']]
                .groupby([score_name, 'Attribute', 'Attribute Value'])
                .sum()
                .reset_index())

    calib_grps = calib_grps.merge(num_inds, 
                                  on=[score_name, 'Attribute', 'Attribute Value'],
                                  how='right')

    # compute overall calib 
    calib = (calib_grps[calib_grps["Individuals Evaluated On"] != 0].groupby(score_name)
                       .apply(lambda x: np.average(x["P(Y = 1 | Score = score, Attr = attr)"], 
                                                   weights=x["Individuals Evaluated On"]))
                       .reset_index()
                       .rename(columns={0: 'P(Y = 1 | Score = score)'}))

    # num individuals w/ sensitive attrs per fold 
    num_inds = (calib_grps[[score_name, 'Individuals Evaluated On']]
                .groupby([score_name])
                .sum()
                .reset_index())

    calib = calib.merge(num_inds, 
                      on=[score_name],
                      how='right')

    return calib, calib_grps


def compute_eq_odds_arnold_nvca(long_df:pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """Returns dataframes of equalized odds values for the binary Arnold NVCA. 
    The problem_name is violent_two_year, the score_name is arnold_nvca.
    
    Keyword arguments: 
        long_df -- 
    Returns:
        eq_odds -- dataframe with the eq_odds values over all groups 
        eq_odds_grps -- dataframe with the eq_odds values for each sensitive grp
    """
    long_df = long_df.replace({"arnold_nvca": {"No": 0, "Yes": 1}})
    
    # compute eq odds overall
    eq_odds = (long_df[["arnold_nvca", "violent_two_year"]]
                       .groupby("violent_two_year")
                       .agg(['sum', 'size'])
                       .reset_index())

    eq_odds.columns = ["violent_two_year", 'n_inds_recid', 'total_inds']
    eq_odds["P(Score = Yes | Y = i)"] =  eq_odds['n_inds_recid'] / eq_odds['total_inds']
    
    # compute eq odds for sensitive groups
    eq_odds_grps = (long_df[["arnold_nvca", "violent_two_year", 'Attribute Value']]
                           .groupby(["violent_two_year", 'Attribute Value'])
                           .agg(['sum', 'size'])
                           .reset_index())

    eq_odds_grps.columns = ["violent_two_year", 'Attribute Value', 'n_inds_recid', 'total_inds']
    eq_odds_grps["P(Score = Yes | Y = i, Attr = attr)"] =  eq_odds_grps['n_inds_recid'] / eq_odds_grps['total_inds']
    
    return eq_odds, eq_odds_grps


def reshape_general_violent_cond_auc_summaries(general_auc:pd.DataFrame,
                                               general_model_name:str,
                                               violent_auc:pd.DataFrame,
                                               violent_model_name:str
                                               ):
    """General and Violent AUC dfs both have the following columns: 
    "Attribute", "Attribute Value", "AUC", "fold_num"
    """
    # general summary dataframe
    general_summary = (general_auc
                         .drop(columns=["fold_num"])
                         .groupby(["Attribute", "Attribute Value"])
                         .agg('mean')
                         .reset_index())
    general_summary['Label'] = 'general\_two\_year'

    # violent summary dataframe
    violent_summary = (violent_auc
                        .drop(columns=["fold_num"])
                        .groupby(["Attribute", "Attribute Value"])
                        .agg('mean')
                        .reset_index())

    violent_summary['Label'] = 'violent\_two\_year'

    # combine the two 
    df = pd.concat([general_summary, violent_summary], axis=0)

    # compute ranges
    max_min = (df[["Attribute", "Label", "AUC"]]
               .groupby(["Attribute", "Label"])
               .agg(['max', 'min']))
    max_min['range'] = max_min[('AUC', 'max')] - max_min[('AUC', 'min')]
    
    range_df = (max_min.range.reset_index()
                    .pivot(index='Label', 
                      columns='Attribute',
                        values='range')
                    .reset_index()
                    .rename(columns={"race": "race_range",
                                     "sex": "sex_range"}))
    range_df.columns.name = None
    
    # long to wide
    wide_df = (df.pivot(index="Label",
                        columns='Attribute Value', 
                        values=['AUC'])
                 .reset_index())

    # clean up column names
    wide_df.columns = wide_df.columns.droplevel(0)
    wide_df.columns.name = None
    wide_df.rename(columns={"": "Label"}, inplace=True)
    wide_df["Model"] = [general_model_name, violent_model_name]
    
    # merge wide and range
    wide_df = wide_df.merge(range_df, on="Label", how="inner")

    return wide_df