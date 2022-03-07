import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import embed

input = pd.read_parquet("data/prediction_input.parquet")
input = input[input["Turbine_Pressure Drafttube"].isna() == False]
df = pd.read_parquet("data/input_dataset-2.parquet")
df = df[df["Bolt_1_Tensile"].isna() == False]
n_rows = df.shape[0]
df["dates"] = df.index
df.index = np.arange(n_rows)
n_rows_input = input.shape[0]
input["dates"] = input.index
input.index = np.arange(n_rows_input)


def plot_normalized(inputs, min_time, max_time):
    for input in inputs:
        stad = np.std(df[input].iloc[min_time:max_time])
        plt.plot ((df[input].iloc[min_time:max_time] - np.mean(df[input].iloc[min_time:max_time]))/stad, label=input)
    plt.legend()
    plt.show()

def plot_normalized2(inputs, min_time, max_time):
    for input in inputs:
        stad = np.std(df[input].iloc[min_time:max_time])
        plt.plot ((df[input].iloc[min_time:max_time] - np.mean(df[input].iloc[min_time:max_time]))/stad, label=input)
    plt.legend()
    plt.show()

min_time = 0; max_time = n_rows
inputs=["Bolt_1_Tensile", "Bolt_2_Tensile", "Bolt_3_Tensile"]
plot_normalized(inputs, min_time, max_time)

