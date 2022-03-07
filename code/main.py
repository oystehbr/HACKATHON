import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import lightgbm as lgb
import optuna
import sklearn
import torch
import torchvision
import seaborn as sns
from IPython import embed


def write_data():
    # TOFAST FOR YOU
    # HAAAAAAAALLO
    pass


def read_data():
    
    prediction_input = pd.read_parquet("../data/prediction_input.parquet")
    df2 = pd.read_parquet("../data/input_dataset-2.parquet")

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
    plt.savefig('../output/correlation_df2.png')
    # plt.show()
    plt.close()
   
    embed()

    
    
    




def main():
    read_data()

if __name__ == '__main__':
    main()
