import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from utils.plotting_helpers import safe_save_plt


def prob_recid_conditioned_sensitive_attr(df:pd.DataFrame, 
                                          attribute:str, 
                                          dataset_name:str,
                                          save_path:None):
    """
    Bar plot of conditional base rates of recidivism problem
    """

    # cast df from long to wide with each attribute being a different column 
    wide_df = (df[df["Attribute"] == attribute]
                .pivot(index='label', 
                       columns='Attribute Value', 
                       values=[ 'P(Y = 1 | Attr = attr)']))
    
    # get a list of unique columns
    attribute_values = df[df["Attribute"] == attribute ]["Attribute Value"].unique()
    
    # set width of bar
    barWidth = 0.15

    # set height of bar
    bars = {attribute_value: {"bar": None, "pos": None} for attribute_value in attribute_values}
    for attribute_value in attribute_values:
        bars[attribute_value]["bar"] = wide_df[('P(Y = 1 | Attr = attr)', attribute_value)]
        bar_len = len(bars[attribute_value]["bar"])

    # Set position of bar on X axis
    for i, (bar_name, bar_dict) in enumerate(bars.items()):
        if i == 0:
            bar_dict["pos"] = np.arange(len(bar_dict["bar"]))
        else: 
            prev_bar_pos = bars[prev_bar_name]["pos"]
            bar_dict["pos"] = [x + barWidth for x in prev_bar_pos]
        prev_bar_name = bar_name 

    # Make the plot
    plt.figure(figsize=(10, 5))
    plt.style.use('ggplot')
    
    colors = ['cornflowerblue', 'lightslategrey', 'lightskyblue', 'steelblue']
    for i, (bar_name, bar_dict) in enumerate(bars.items()):
        plt.bar(bar_dict["pos"], bar_dict["bar"], color=colors[i], width=barWidth, edgecolor='white', label=bar_name)

    # Add xticks on the middle of the group bars
    plt.xlabel('Prediction Problem', fontweight='bold')
    plt.xticks([r + barWidth for r in range(bar_len)], wide_df.index, rotation=45)

    # Limit y axis to 0,1 
    plt.ylim(0,1)
    plt.ylabel('P(Y = 1 | Attr = attr)', fontweight='bold')

    # Create legend, add title, & show/save graphic
    plt.legend()
    plt.title(f'Probability of recidivism (conditioned on {attribute}) is not the same for \nany prediction problem on {dataset_name}')
    
    if save_path is not None: 
        safe_save_plt(plt, save_path)
    plt.show()
    plt.close()
    return


def plot_calibration_for_score_on_problem(calib: pd.DataFrame, 
                                          calib_grps: pd.DataFrame,
                                          problem_name: str,
                                          score_name: str,
                                          region: str,
                                          xtick_labels=None,
                                          rotate_xticks=False,
                                          include_legend=True,
                                          save_path=None):
    """Plots calibration for risk scores which are NONBINARY AND DISCRETE 
    (i.e. Arnold NCA, COMPAS)
    Keyword Arguments:
        calib: df with columns [score_name, P(Y = 1 | Score = score)]; should contain the probability of 
                recidivism for each score. 
        calib_grps: df with columns [score_name, Attribute Value, P(Y = 1 | Score = score, Attr=attr)]; 
                                should contain the prob of recidivism for each sensitive group and score 
        problem_name: variable name of the prediction problem for this plot 
        score_name: 
    """
    
    plt.figure(figsize=(8, 5.5))
    plt.style.use('ggplot')

    # calibration reference line doesn't make sense here
    # overall calibration 
    plt.plot(calib[score_name], 
             calib['P(Y = 1 | Score = score)'], 
             color='black', marker='o', 
             linewidth=1, markersize=4,
             label='All individuals')

    # group level calibration 
    colors=['red', 'green', 'orange', 'maroon', 'royalblue', 'mediumslateblue']
    for i, (name, group_df) in enumerate(calib_grps.groupby("Attribute Value")):
        if name == 'female' or name == 'male':
            plt.plot(group_df[score_name], 
                     group_df['P(Y = 1 | Score = score, Attr = attr)'], 
                     color=colors[i], marker='o', linewidth=1, markersize=4,
                     linestyle='--',
                     label=name)
        else:
            plt.plot(group_df[score_name], 
                     group_df['P(Y = 1 | Score = score, Attr = attr)'], 
                     color=colors[i], marker='o', linewidth=1, markersize=4,
                     label=name)

    # axes settings
    if xtick_labels is not None:
        plt.xticks(np.arange(len(xtick_labels)), xtick_labels)
    plt.xlabel(f"{score_name} score", fontsize=25)

    plt.ylim(0,1)
    plt.ylabel('P(Y = 1 | Score = score, \nAttr = attr)', fontsize=25)

    # Create legend, add title, format & show/save graphic
    if include_legend:
        plt.legend(fontsize=20, ncol=2, framealpha=0.3)
    plt.title(f'Calib. of {score_name} on \n{problem_name} in {region}', fontsize=25)

    if rotate_xticks:
        plt.tick_params(axis="x", labelsize=20, rotation=25)
    else:
        plt.tick_params(axis="x", labelsize=20)

    plt.tick_params(axis="y", labelsize=25)

    if save_path is not None: 
        safe_save_plt(plt, save_path)
        
    plt.show()
    plt.close()
    return


def plot_binary_calib_arnold_nvca(calib:pd.DataFrame, 
                                  calib_grps:pd.DataFrame, 
                                  region_name:str,
                                  save_path=None):
    
    """Binary calib is equivalent to conditional use accuracy equality. 
    Takes output of compute_calibration_discrete_score() as input
    This fcn is only used for the scaled Arnold NVCA, because the other
    Arnold scores are not binary
    """
    # process calib_grps, calib
    calib_grps_wide = (calib_grps.replace({"Attribute Value": {"African-American": "Afr-Am.",
                                                               "Caucasian": "Cauc.",
                                                               "Hispanic": "Hisp.",
                                                               "Other": "Other \n(race)"
                                                              }})
                                 .pivot(index='Attribute Value', 
                                        columns="arnold_nvca", 
                                        values=[ 'P(Y = 1 | Score = score, Attr = attr)']))

    calib["Attribute Value"] = "All"
    calib_wide = (calib.pivot(index='Attribute Value',
                        columns="arnold_nvca", 
                        values=[ 'P(Y = 1 | Score = score)'])
                  .rename({"P(Y = 1 | Score = score)": "P(Y = 1 | Score = score, Attr = attr)"}, 
                         axis=1)
            )

    wide_df = pd.concat([calib_wide, calib_grps_wide])
    
    # make plot
    plt.figure(figsize=(10, 7))
    plt.style.use('ggplot')

        # set width of bar
    barWidth = 0.15

    # set height of bar
    score_values = ["No", "Yes"]
    bars = {score_value: {"bar": None, "pos": None} for score_value in score_values}
    for score_value in score_values:
        bars[score_value]["bar"] = wide_df[('P(Y = 1 | Score = score, Attr = attr)', score_value)]
        bar_len = len(bars[score_value]["bar"])

    # Set position of bar on X axis
    for i, (bar_name, bar_dict) in enumerate(bars.items()):
        if i == 0:
            bar_dict["pos"] = np.arange(len(bar_dict["bar"]))
        else: 
            prev_bar_pos = bars[prev_bar_name]["pos"]
            bar_dict["pos"] = [x + barWidth for x in prev_bar_pos]
        prev_bar_name = bar_name 

    colors = ['cornflowerblue', 'lightslategrey', 'lightskyblue', 'steelblue']
    for i, (bar_name, bar_dict) in enumerate(bars.items()):
        plt.bar(bar_dict["pos"], 
                bar_dict["bar"], 
                color=colors[i], 
                width=barWidth, 
                edgecolor='white', 
                label=f"P(Y=1 | score={bar_name}, Attr=attr")

    # Add xticks on the middle of the group bars
    plt.xlabel('Sensitive Attribute', fontsize=25)
    plt.xticks([r + barWidth for r in range(bar_len)], wide_df.index, rotation=45, fontsize=20)

    plt.ylim(0,1)
    plt.ylabel('P(Y = 1 | Score = score, \nAttr = attr)\n', fontsize=25)

    # Create legend, add title, format & show/save graphic
    plt.title(f"Calibration (Cond. Use Acc. Eq.) of \narnold_nvca on violent_two_year in {region_name}", fontsize=30)
    plt.legend(fontsize=20)

    if save_path is not None: 
        safe_save_plt(plt, save_path)

    plt.show()
    plt.close()


def plot_eq_odds_arnold_nvca(eq_odds:pd.DataFrame, 
                             eq_odds_grps:pd.DataFrame, 
                             region_name:str,
                             save_path=None):
    
    """Binary BPC/BNC is equivalent to equalized odds. 
    Takes output of compute_eq_odds_arnold_nvca() as input
    This fcn is only used for the scaled Arnold NVCA, because the other
    Arnold scores are not binary
    """
    # process calib_grps, calib
    eq_odds_grps_wide = (eq_odds_grps.replace({"Attribute Value": {"African-American": "Afr-Am.",
                                                               "Caucasian": "Cauc.",
                                                               "Hispanic": "Hisp.",
                                                               "Other": "Other \n(race)"
                                                              }})
                                 .pivot(index='Attribute Value', 
                                        columns="violent_two_year", 
                                        values=[ 'P(Score = Yes | Y = i, Attr = attr)']))

    eq_odds["Attribute Value"] = "All"
    eq_odds_wide = (eq_odds.pivot(index='Attribute Value',
                                  columns="violent_two_year", 
                                  values=[ 'P(Score = Yes | Y = i)'])
                  .rename({"P(Score = Yes | Y = i)": "P(Score = Yes | Y = i, Attr = attr)"}, 
                          axis=1)
            )

    wide_df = pd.concat([eq_odds_wide, eq_odds_grps_wide])
    
    # make plot
    plt.figure(figsize=(10, 7))
    plt.style.use('ggplot')

        # set width of bar
    barWidth = 0.15

    # set height of bar
    label_values = [0, 1]
    bars = {label_value: {"bar": None, "pos": None} for label_value in label_values}
    for label_value in label_values:
        bars[label_value]["bar"] = wide_df[('P(Score = Yes | Y = i, Attr = attr)', label_value)]
        bar_len = len(bars[label_value]["bar"])

    # Set position of bar on X axis
    for i, (bar_name, bar_dict) in enumerate(bars.items()):
        if i == 0:
            bar_dict["pos"] = np.arange(len(bar_dict["bar"]))
        else: 
            prev_bar_pos = bars[prev_bar_name]["pos"]
            bar_dict["pos"] = [x + barWidth for x in prev_bar_pos]
        prev_bar_name = bar_name 

    colors = ['cornflowerblue', 'lightslategrey', 'lightskyblue', 'steelblue']
    
    for i, (bar_name, bar_dict) in enumerate(bars.items()):
        plt.bar(bar_dict["pos"], 
                bar_dict["bar"], 
                color=colors[i], 
                width=barWidth, 
                edgecolor='white', 
                label=f"P(Score=Yes | Y={bar_name}, Attr=attr)")

    # Add xticks on the middle of the group bars
    plt.xlabel('Sensitive Attribute', fontsize=25)
    plt.xticks([r + barWidth for r in range(bar_len)], wide_df.index, rotation=45, fontsize=20)

    plt.ylim(0,1)
    plt.ylabel('P(Score = Yes | Y = i, \nAttr = attr)\n', fontsize=25)

    # Create legend, add title, format & show/save graphic
    plt.title(f"BPC/BNC (Eq. odds) of arnold_nvca \non violent_two_year in {region_name}", fontsize=30)
    plt.legend(fontsize=20)

    if save_path is not None: 
        safe_save_plt(plt, save_path)

    plt.show()
    plt.close()


def plot_fpr_fnr(fpr_fnr_summary: pd.DataFrame, 
                 problem_name:str, 
                 model_name:str,
                 model_performance:str,
                 base_rate:str=None,
                 save_path = None):
    """Bar plot equalized odds for interpretable models
    """
    plt.style.use('ggplot')
    for attribute, df in summary.groupby("Attribute"):
        fig, ax = plt.subplots()    

        width = 0.35
        x = np.arange(len(df['Attribute Value']))
        ax.bar(x - width/2, df["FPR"], width, label = 'FPR')
        ax.bar(x + width/2, df["FNR"], width, label = 'FNR')
                
        # formatting
        if base_rate is not None: 
            ax.set_title(f"FPR and FNR for {model_name} on {problem_name} \n AUC (std): {model_performance}, Base rate: {base_rate}")
        else: 
            ax.set_title(f"FPR and FNR for {model_name} on {problem_name} \n AUC (std): {model_performance}")

        ax.set_xlabel(f"Atttribute: {attribute}")
        ax.set_xticks(x)
        ax.set_xticklabels(list(df['Attribute Value']))
        ax.legend(loc='best')

        if save_path is not None: 
            safe_save_plt(plt, save_path)
        plt.show()
        plt.close()
