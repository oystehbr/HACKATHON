# TOBBB

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import lightgbm as lgb
import optuna
from IPython import embed


if __name__ == "__main__":
    # INSTALL pyarrow AND fastparquet
    input = pd.read_parquet("../data/prediction_input.parquet")
    df = pd.read_parquet("../data/input_dataset-2.parquet")
    embed()