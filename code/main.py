import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import lightgbm as lgb
import optuna
import sklearn
import torch
import torchvision

from IPython import embed


def read_data():
    
    prediction_input = pd.read_parquet("../data/prediction_input.parquet")
    df2 = pd.read_parquet("../data/input_dataset-2.parquet")
    embed()
    
    

def write_data():
    # TOFAST FOR YOU
    # HAAAAAAAALLO
    pass



def main():

    read_data()
    a = 2
    b = 3
    abc = a + b

    print(abc)
    pass

if __name__ == '__main__':
    main()
