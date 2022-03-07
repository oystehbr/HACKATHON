import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import lightgbm as lgb
import optuna
import sklearn
import torch
import torchvision
import seaborn as sns
from sklearn.linear_model import LinearRegression
from IPython import embed
from collections import Counter

def write_data():
    # TOFAST FOR YOU
    # HAAAAAAAALLO
    pass

def read_data():
    
    prediction_input = pd.read_parquet("../data/prediction_input.parquet")

    df2 = pd.read_parquet("../data/input_dataset-2.parquet")

    df2 = df2[df2["Bolt_1_Tensile"].isna() == False]
    # n_rows = df2.shape[0]
    # df2["dates"] = df2.index
    # df2.index = np.arange(n_rows)
    # use the necessary parameters for the prediction
    df2 = df2.drop(['Bolt_1_Steel tmp',
        'Bolt_1_Torsion', 
        'Bolt_2_Torsion',
        'Bolt_3_Torsion',
        'Bolt_4_Torsion',
        'Bolt_5_Torsion',
        'Bolt_6_Torsion',
        'lower_bearing_vib_vrt',
        'turbine_bearing_vib_vrt',], 
        axis = 1
        )

    

    # plotting the correlation matrix, of the data of prediction
    corr_df2 = df2.corr()
    fig, ax = plt.subplots(figsize=(18,12))
    sns.heatmap(corr_df2, annot=True)
    plt.savefig('../output/correlation_df2_new.png')
    # plt.show()
    plt.close()

    return df2, prediction_input


def add_time_dataframe(df2, prediction_input):
    # make a column for time from 0, ..., -> seconds from the beggining of dataset 
    df2_time_array2 = df2.index.values.astype(float)
    df2_time_array1 = df2_time_array2 - df2_time_array2[0]
    df2_time_array = df2_time_array1/df2_time_array1[1]
    df2['time'] = df2_time_array


    df_bolt_time = df2[['Bolt_1_Tensile', 'Bolt_2_Tensile', 'Bolt_3_Tensile', 'Bolt_4_Tensile', 'Bolt_5_Tensile', 'Bolt_6_Tensile', 'time']]


    corr_df_bolt_time = df_bolt_time.corr()
    fig, ax = plt.subplots(figsize=(18,12))
    sns.heatmap(corr_df_bolt_time, annot=True)
    plt.savefig(f'../output/correlation_bolt_time')
    # plt.show()
    plt.close()
    embed()

    corr_df2 = df2.corr()
    fig, ax = plt.subplots(figsize=(18,12))
    sns.heatmap(corr_df2, annot=True)
    plt.savefig('../output/correlation_df2_add_time.png')
    # plt.show()
    plt.close()

    # same for the prediction_input
    prediction_input_time_array2 = prediction_input.index.values.astype(float)
    prediction_input_time_array1 = prediction_input_time_array2 - prediction_input_time_array2[0]
    prediction_input_time_array = prediction_input_time_array1/prediction_input_time_array1[1]
    prediction_input['time'] = prediction_input_time_array


def add_cumulative_openings(df2, prediction_input):
    df2_cumulative_openings = np.cumsum(df2['mode'] == 'start')
    df2['cum_openings'] = df2_cumulative_openings

    corr_df2 = df2.corr()
    fig, ax = plt.subplots(figsize=(18,12))
    sns.heatmap(corr_df2, annot=True)
    plt.savefig('../output/correlation_df2_add_cum_opening.png')
    # plt.show()
    plt.close()


def vein_opening_cumulative(df2, prediction_input):
    df2_vein_changing = df2['Turbine_Guide Vane Opening'] - df2['Turbine_Guide Vane Opening'].shift(-1)
    df2['vein_changing'] = np.cumsum((df2_vein_changing)**(1/3))

    corr_df2 = df2.corr()
    fig, ax = plt.subplots(figsize=(18,12))
    sns.heatmap(corr_df2, annot=True)
    plt.savefig('../output/correlation_df2_add_cum_vein.png')
    # plt.show()
    plt.close()


def time_switch(df2, prediction_input):

    for i in range(0, 100000, 1000):
        df2['switch_bolt1'] = df2['Bolt_1_Tensile'].shift(-i)
        print(df2['switch_bolt1'].corr(df2['Turbine_Pressure Drafttube']))
    
    pass


def linear_regression(df2):

    
    df2['mode'] = np.float(df2.mode == 'start')
    y = df2[['Bolt_1_Tensile', 'Bolt_2_Tensile', 'Bolt_3_Tensile', 'Bolt_4_Tensile', 'Bolt_5_Tensile', 'Bolt_6_Tensile']].to_numpy()
    X = df2.drop(['Bolt_1_Tensile', 'Bolt_2_Tensile', 'Bolt_3_Tensile', 'Bolt_4_Tensile', 'Bolt_5_Tensile', 'Bolt_6_Tensile'], 
        axis = 1
        ).to_numpy()

    reg = LinearRegression().fit(X, y['Bolt_1_Tensile'])
    print(reg.score())
    reg.predict(np.array([[3, 5]]))


def Sort(sub_li):
      
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of 
    # sublist lambda has been used
    sub_li.sort(key = lambda x: x[0])
    return sub_li

def find_long_time_gaps(df2):

    df2['long_time_steps'] = df2['cum_openings']

    a = np.array(df2['long_time_steps'])
    b = Counter(a)
    c = b.most_common(26)

    sorted_val = Sort(c)
    print(sorted_val)
    for i in range(len(sorted_val) - 1):
        df2['long_time_steps'] = np.where(df2['long_time_steps'].between(sorted_val[i][0], sorted_val[i+1][0]), sorted_val[i+1][1]*i, df2['long_time_steps'])

    # df2['long_time_steps'] = df2['long_time_steps'].between(sorted_val[i+1][0], sorted_val[i+1][0]), sorted_val[i+1][1]*i, df2['long_time_steps'])
    return df2


def main():
    df2, prediction_input = read_data()

    add_time_dataframe(df2, prediction_input)

    add_cumulative_openings(df2, prediction_input)

    find_long_time_gaps(df2)

    linear_regression(df2)



    # time_switch(df2, prediction_input)

    # vein_opening_cumulative(df2, prediction_input)

if __name__ == '__main__':
    main()
