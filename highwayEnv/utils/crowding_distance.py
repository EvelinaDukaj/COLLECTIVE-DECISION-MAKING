# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 23:59:19 2020

@author: PC_Nt
"""

import pandas as pd
import pickle 
import pandas as pd
import csv
import numpy as np
import random as rn
import math as math

id = 'id'
hypervolume = 'hypervolume'
first_risk = 'first_risk'

def calculate_crowding(scores):
    # Crowding is based on a vector for each individual
    # All dimension is normalised between low and high. For any one dimension, all
    # solutions are sorted in order low to high. Crowding for solution x
    # for that score is the difference between the next highest and next
    # lowest score. Total crowding value sums all crowding for all scores

    population_size = len(scores[:, 0])
    number_of_scores = len(scores[0, :])

    # create crowding matrix of population (row) and score (column)
    crowding_matrix = np.zeros((population_size, number_of_scores))

    # normalise scores (ptp is max-min)
    print(scores.ptp(0))
    normed_scores = (scores - scores.min(0)) / scores.ptp(0)

    # calculate crowding distance for each score in turn
    for col in range(number_of_scores):
        crowding = np.zeros(population_size)

        # end points have maximum crowding
        crowding[0] = 1
        crowding[population_size - 1] = 1

        # Sort each score (to calculate crowding between adjacent scores)
        sorted_scores = np.sort(normed_scores[:, col])

        sorted_scores_index = np.argsort(
            normed_scores[:, col])

        # Calculate crowding distance for each individual
        crowding[1:population_size - 1] = \
            (sorted_scores[2:population_size] -
             sorted_scores[0:population_size - 2])

        # resort to orginal order (two steps)
        re_sort_order = np.argsort(sorted_scores_index)
        sorted_crowding = crowding[re_sort_order]

        # Record crowding distances
        crowding_matrix[:, col] = sorted_crowding

    # Sum crowding distances of each score
    crowding_distances = np.sum(crowding_matrix, axis=1)

    return crowding_distances


def reduce_by_crowding(input_df, 
                       threshold,
                       id, 
                       hypervolume, 
                       first_risk,
                       input_df_columns):
    
    id_list = input_df[id].unique()
    
    buildup_df = pd.DataFrame(columns=input_df_columns)
    del buildup_df[id]
    buildup_df['crowding_distance'] = 'crowding_distance'
    buildup_df[id] = id
    
    
    for current_id in id_list:
        
        temp_df = input_df[input_df[id] == current_id]
        del temp_df[id]
        
        temp_original_df = temp_df.copy()
        temp_df = temp_df[[hypervolume, first_risk]].copy()
        
        temp_df=temp_df.to_numpy()
        
        crowding_distances = calculate_crowding(temp_df)
        crowding_distances = pd.DataFrame(crowding_distances)        
        
        crowding_distances.index = temp_original_df.index
        temp_original_df['crowding_distance'] = crowding_distances
        temp_original_df = temp_original_df.sort_values(by=['crowding_distance'], ascending=False)
        
        number_to_select = math.ceil(temp_original_df.shape[0] * threshold)
        temp_original_df = temp_original_df.head(number_to_select)
        temp_original_df[id] = current_id
        
        buildup_df = buildup_df.append(temp_original_df)
    
    return buildup_df


def start_crowding_distance(risk_tol, threshold_tol, hv_tol, sigma):
    pickle_in = open(rf'C:\test\Archives\{risk_tol}_{threshold_tol}_{hv_tol}_{sigma}.pkl',"rb")
    example_dict = pickle.load(pickle_in)
    pickle_in.close()
    print(example_dict)
    df = pd.DataFrame.from_dict(data=example_dict, orient='index')
    df.transpose().to_csv(rf'C:\test\Archives\CentralDB_50terminals{risk_tol}_{threshold_tol}_{hv_tol}_{sigma}.csv')
    #for id in range(0,9):
    input_df=pd.read_csv(rf'C:\test\Archives\CentralDB_50terminals{risk_tol}_{threshold_tol}_{hv_tol}_{sigma}.csv')

    input_df_columns = input_df.columns
    threshold = 0.9
    import matplotlib.pyplot as plt
    output_df = reduce_by_crowding(input_df,
                                   threshold,
                                   id,
                                   hypervolume,
                                   first_risk,
                                   input_df_columns)
    # dropping the column corresponding to crowding distance and resetting the index to the right value

    output_df.index = range(len(output_df.index))
    del output_df['Unnamed: 0']

    pickle_file_cd = open(rf'C:\test\Archives\{risk_tol}_{threshold_tol}_{hv_tol}_{sigma}_cd.pkl', 'wb')
    pickle.dump(output_df, pickle_file_cd)
    pickle_file_cd.close()


    output_df.plot(x='hypervolume', y='first_risk', style='+')
    plt.savefig(rf'C:\test\Archives\CentralDB_50terminals{risk_tol}_{threshold_tol}_{hv_tol}_{sigma}_AfterCrowding_with_{threshold}Th.png', dpi=400)



