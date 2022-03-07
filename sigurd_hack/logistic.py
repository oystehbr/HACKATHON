import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import embed
from sklearn.linear_model import LinearRegression
from collections import Counter

def find_long_time_gaps(df2):
    
    df2_cumulative_openings = np.cumsum(df2['mode'] == 'start')
    df2['cum_openings'] = df2_cumulative_openings
    df2['long_time_steps'] = df2['cum_openings']
    

    a = np.array(df2['long_time_steps'])
    b = Counter(a)
    c = b.most_common(26)

    sorted_val = Sort(c)
    for i in range(len(sorted_val) - 1):
        # time i works for class 1
        df2['long_time_steps'] = np.where(df2['long_time_steps'].between(sorted_val[i][0], sorted_val[i+1][0]), sorted_val[i+1][1]*i, df2['long_time_steps'])

    # df2['long_time_steps'] = df2['long_time_steps'].between(sorted_val[i+1][0], sorted_val[i+1][0]), sorted_val[i+1][1]*i, df2['long_time_steps'])
    return df2

def Sort(sub_li):
      
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of 
    # sublist lambda has been used
    sub_li.sort(key = lambda x: x[0])
    return sub_li


input = pd.read_parquet("../data/prediction_input.parquet")
input = input[input["Turbine_Pressure Drafttube"].isna() == False]
df = pd.read_parquet("../data/input_dataset-2.parquet")
df = df[df["Bolt_1_Tensile"].isna() == False]
df_time_array2 = df.index.values.astype(float)
df_time_array1 = df_time_array2 - df_time_array2[0]
df_time_array = df_time_array1/df_time_array1[1]
df['time'] = df_time_array
df['time_s'] = df_time_array**2
n_rows = df.shape[0]
df["dates"] = df.index
df.index = np.arange(n_rows)
n_rows_input = input.shape[0]
input["dates"] = input.index
input.index = np.arange(n_rows_input)
df = find_long_time_gaps(df)


split = int(n_rows*(1-0.23)) + 2
X_train = df[["Unit_4_Power", "Unit_4_Reactive Power", "Turbine_Guide Vane Opening", "Turbine_Pressure Drafttube", "Turbine_Pressure Spiral Casing", "Turbine_Rotational Speed", \
 "time", "time_s"]].iloc[:split]
X_test = df[["Unit_4_Power", "Unit_4_Reactive Power", "Turbine_Guide Vane Opening", "Turbine_Pressure Drafttube", "Turbine_Pressure Spiral Casing", "Turbine_Rotational Speed", \
 "time", "time_s"]].iloc[split:]
y_train = df["Bolt_1_Tensile"].iloc[:split]
y_test = df["Bolt_1_Tensile"].iloc[split:]

logreg = LinearRegression().fit(X_train, y_train)
pred_train = logreg.predict(X_train)
error_train = np.mean((pred_train - y_train)**2)

pred_test = logreg.predict(X_test)
error_test = np.mean((pred_test - y_test)**2)

plt.plot(pred_train, label = "pred"); plt.plot(y_train, label="label")
plt.legend(); plt.show()
plt.plot(pred_test, label="pred"); plt.plot(y_test.values, label="label")
plt.legend(); plt.show()
print(error_test)
print(logreg.coef_)


    

