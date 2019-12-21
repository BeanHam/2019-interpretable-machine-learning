import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from utils.plotting_helpers import safe_save_plt


def plot_calibration_for_score_on_problem(calib: pd.DataFrame, 
                                          calib_grps: pd.DataFrame,
                                          problem_name: str,
                                          score_name: str,
                                          region: str,
                                          xtick_labels=None,
                                          save_path=None):
    """Plots calibration for risk scores which are not probabilities
    Keyword Arguments:
        calib: df with columns [score_name, P(Y = 1 | Score = score)]; should contain the probability of 
                recidivism for each score. 
        calib_grps: df with columns [score_name, Attribute Value, P(Y = 1 | Score = score, Attr=attr)]; 
                                should contain the prob of recidivism for each sensitive group and score 
        problem_name: variable name of the prediction problem for this plot 
        score_name: 
    """
    
    plt.figure(figsize=(10, 7))
    plt.style.use('ggplot')

    # calibration reference line doesn't make sense here
    # overall calibration 
    plt.plot(calib[score_name], 
             calib['P(Y = 1 | Score = score)'], 
             color='black', marker='o', 
             linewidth=1, markersize=2,
             label='All individuals')

    # group level calibration 
    colors=['red', 'green', 'orange', 'maroon', 'royalblue', 'mediumslateblue']
    for i, (name, group_df) in enumerate(calib_grps.groupby("Attribute Value")):
        if name == 'female' or name == 'male':
            plt.plot(group_df[score_name], 
                     group_df['P(Y = 1 | Score = score, Attr = attr)'], 
                     color=colors[i], marker='o', linewidth=1, markersize=2,
                     linestyle='--',
                     label=name)
        else:
            plt.plot(group_df[score_name], 
                     group_df['P(Y = 1 | Score = score, Attr = attr)'], 
                     color=colors[i], marker='o', linewidth=1, markersize=2,
                     label=name)

    # axes settings
    if xtick_labels is not None:
        plt.xticks(np.arange(len(xtick_labels)), xtick_labels)
    plt.xlabel(f"\n{score_name} score", fontsize=20)

    plt.ylim(0,1)
    plt.ylabel('P(Y = 1 | Score = score, Attr = attr)\n', fontsize=20)

    # Create legend, add title, & show/save graphic
    plt.legend(fontsize=14)
    score_name_formatted = score_name.capitalize()
    plt.title(f'Calibration of {score_name_formatted} on \n{problem_name} in {region}', fontsize=24)

    if save_path is not None: 
        safe_save_plt(plt, save_path)
        
    plt.show()
    plt.close()
    return
    