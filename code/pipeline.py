import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import optuna
from IPython import embed
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from sklearn.linear_model import LinearRegression
from collections import Counter


class ModelPipeLine:
    """
    Class / pipeline for fitting and predicting.
    """
    
    def __init__(self, train_path="../data/input_dataset-2.parquet", 
                 test_path="../data/prediction_input.parquet", split_size=0.77, 
                 feature_selections=None, make_real_preds=False, y_transforms=None):
        """
        Initialiazes stuff. Read below :D
        
        In:
            train_path (str): Path to labeled training set.
            test_path (str): Path to unlabeled testing set (only nessecary if
                make_real_preds is True.
            split_size (float): If make_real_preds is False, this determines
                where the set is split horisontally.
            feature_selections (list of str): List of features to be included
                in the model. If None, some will be chosen for you.
            make_real_preds (boolean): If True, the model will use the 
                test-path as testing set. If False, it will split on the
                training set horisontally. 
            y_transform (list of int): Tranforms y linearly with the time.
                If not None, must be a list of factor for each bolt.
        """
        self.y_transforms = y_transforms
        self.make_real_preds = make_real_preds
        self.data = pd.read_parquet(train_path)
        if make_real_preds:
            preds = pd.read_parquet(test_path)
            self.prediction_data = preds
            self.data = pd.concat([self.data, preds])
        self.preprocess()
        self.feature_engineer()
        self.set_train_test(feature_selections, split_size)
    
    def preprocess(self):
        """
        Preprocesses data. This includes removing nans, moving dates from 
        indicies and replacing indicies with acutal 0, ... , n_rows indicies.
        """
        self.data = self.data[self.data["Unit_4_Power"].isna() == False] # Remove nans
        self.n_rows = self.data.shape[0]
        self.data["dates"] = self.data.index # Save dates
        self.data.index = np.arange(self.n_rows)
        self.data["mode"] = pd.Series(self.data["mode"], dtype="category") # From object to category
        
    def set_train_test(self, feature_selections, split_size):
        if feature_selections is None:
            # feature_selections = ['Unit_4_Power', 'Unit_4_Reactive Power', 'Turbine_Guide Vane Opening',
            #     'Turbine_Pressure Drafttube', 'Turbine_Pressure Spiral Casing',
            #     'Turbine_Rotational Speed', 'mode', 'time', 'long_time_steps', 'long_time_steps2']
            feature_selections = ['Unit_4_Power', 'Unit_4_Reactive Power', 'Turbine_Pressure Spiral Casing',
                                  'Turbine_Rotational Speed', 'time', 'long_time_steps2']
        input = self.data.loc[:, feature_selections]
        responses = self.data.loc[:, ['Bolt_1_Tensile', 'Bolt_2_Tensile', 'Bolt_3_Tensile', 
                              'Bolt_4_Tensile', 'Bolt_5_Tensile', 'Bolt_6_Tensile']]
        
        if self.y_transforms is not None:
            self.time_slopes = []
            for i in range(len(self.y_transforms)):
                slope = self.y_transforms[i]/self.data["time"].max()
                time_slope = slope * self.data["time"]
                responses.iloc[:, i] = responses.iloc[:, i] - time_slope
                self.time_slopes.append(time_slope)
                
        # Split:
        if self.make_real_preds:
            split_point = 1750000 # Size of training data
        else:
            split_point = np.floor(self.n_rows * split_size).astype(int)
        self.x_train = input[:split_point]
        self.x_test = input[split_point:]
        self.y_train = responses[:split_point]
        self.y_test = responses[split_point:]
        
    def feature_engineer(self):
        """
        Creates new features. self.preprocess() should be called first.
        
            time (float): Time in seconds since first observation.
            c_openings (int): Cumulative sum of how many times mode == "start".
            long_time_steps (int): Cumulative sum of when a mode has stopped
                doing start.
            long_time_steps2 (int): The same as long_time_steps2, but the 
                cumulative steps are multiplied with the amount of times 
                mode was equal to start.
        """
        # embed()
        time_array = self.data["dates"].values.astype(float) # Convert to float
        time_array = time_array - time_array[0] # Start at 0
        self.data['time'] = time_array/time_array[1] # Standarize by second value
        
        self.data["c_openings"] = np.cumsum(self.data["mode"] == "start")
    
        # Not so pretty code:
        self.data['long_time_steps'] = self.data['c_openings']
        self.data['long_time_steps2'] = self.data['c_openings']
        a = np.array(self.data['long_time_steps'])
        b = Counter(a)
        if self.make_real_preds:
            counts = 30 # Number of turn on / off events
        else:
            counts = 26
        sorted_val = b.most_common(counts) # Hard coded
        sorted_val.sort(key = lambda x: x[0])
        for i in range(len(sorted_val) - 1):
            self.data['long_time_steps'] = np.where(self.data['long_time_steps'].between(sorted_val[i][0], sorted_val[i+1][0]), i, self.data['long_time_steps'])
        self.data['long_time_steps'] = self.data['long_time_steps'].mask(self.data['long_time_steps'] >= sorted_val[i+1][0], (i+1))
        for i in range(len(sorted_val) - 1):
            self.data['long_time_steps2'] = np.where(self.data['long_time_steps2'].between(sorted_val[i][0], sorted_val[i+1][0]), i*sorted_val[i+1][1], self.data['long_time_steps2'])
        
        # Shorter code for the same part, but it is not equal al the way
        # large_steps = np.zeros(self.n_rows)
        # large_steps2 = np.zeros(self.n_rows)
        # counter = 0
        # for i in range(len(c_array)-1):
        #     if (self.data["mode"].iloc[i] == "start"):
        #         if (self.data["mode"].iloc[i+1] == "operation"):
        #             large_steps[i:] += 1
        #             large_steps2[i:] += counter
        #             counter = 0
        #         else: 
        #             counter += 1            

    def fit_boosting(self, params_list=None, eval=False, models_indicies=[0, 1, 2, 3, 4, 5]):
        """
        Fits a Light Gradient Boosting model on the responses in models_indicies.
        """
        if params_list is None:
            params_list = [{}, {}, {}, {}, {}, {}]
        self.models_indicies = models_indicies
        self.models = []
        self.train_preds = []
        self.train_losses = []
        for i in models_indicies:
            model = lgb.LGBMRegressor(**params_list[i])
            if eval:
                model.fit(self.x_train, self.y_train.iloc[:, i], eval_set=(self.x_test, self.y_test.iloc[:, i]))
            else:
                model.fit(self.x_train, self.y_train.iloc[:, i])
            self.models.append(model)
            train_preds = model.predict(self.x_train)
            self.train_preds.append(train_preds)
            self.train_losses.append(mean_squared_error(self.y_train.iloc[:, i], train_preds))
    
    def fit_linear(self, models_indicies=[0, 1, 2, 3, 4, 5]):
        """
        Fits a LinearModel.
        """
        # Time has to be included in the model
        self.x_train["time_s"] = self.x_train["time"]**2
        self.x_test["time_s"] = self.x_test["time"]**2
        self.models_indicies = models_indicies
        self.models = []
        self.train_preds = []
        self.train_losses = []
        for i in models_indicies:
            model = LinearRegression()
            model.fit(self.x_train, self.y_train.iloc[:, i])
            # self.y_train.iloc[:, i]
            
            self.models.append(model)
            train_preds = model.predict(self.x_train)
            self.train_preds.append(train_preds)
            self.train_losses.append(mean_squared_error(self.y_train.iloc[:, i], train_preds))
        
        
    def predict(self):
        """
        Predicts on the test set and saves the loss, on every model.
        """
        self.test_preds = []
        self.test_losses = []
        for i in range(len(self.models)):
            model_index = self.models_indicies[i]
            test_pred = self.models[i].predict(self.x_test)
            self.test_preds.append(test_pred)
            if not self.make_real_preds:
                # embed()
                test_loss = mean_squared_error(self.y_test.iloc[:, model_index], test_pred)
                self.test_losses.append(test_loss)
            
    def plot_all(self):
        """
        Plots training and test predictions against real values.
        """
        x_values_train = np.arange(self.x_train.shape[0])
        x_values_test = np.arange(self.x_test.shape[0])
        for i in range(len(self.models)):
            model_index = self.models_indicies[i]
            scaling = 0 # For transforms
            if self.y_transforms is not None: # Transform back
                scaling = self.time_slopes[model_index]
                scaling = scaling[:self.x_train.shape[0]]
                
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(x_values_train, self.train_preds[i] + scaling, label="Predictions Training")
            ax[0].plot(x_values_train, self.y_train.iloc[:, model_index] + scaling, label="Labels Training")
            ax[0].legend()
            plt.title(f"Bolt{model_index}")
            # plt.show()
            
            if self.y_transforms is not None: # Transform back
                scaling = self.time_slopes[model_index]
                scaling = scaling[self.x_train.shape[0]:]
                
            ax[1].plot(x_values_test, self.test_preds[i] + scaling, label="Predictions Testing")
            if not self.make_real_preds:
                ax[1].plot(x_values_test, self.y_test.iloc[:, model_index] + scaling, label="Labels Testing")
            ax[1].legend()
            # plt.title(f"Bolt{model_index}")
            plt.show()
        
if __name__ == "__main__":
    mpp_linear = ModelPipeLine(make_real_preds=False)
    mpp_linear.fit_linear(models_indicies=[0])
    mpp_linear.predict()
    # embed()
    
    # params = {
    #     "n_estimators": 100, 
    #     "learning_rate": 0.05,
    #     'objective': 'regression',
    #     'metric': 'mse'}
    # params_list = [params for _ in range(6)]
    # mpp = ModelPipeLine(make_real_preds=True, y_transforms=[42, 26.5, 21, 9, 10, 21])
    # mpp.fit_boosting(params_list=params_list, models_indicies=[0, 1, 2, 3, 4, 5], eval=False)
    # mpp.predict()
    
    params = {
        "n_estimators": 500, 
        "learning_rate": 0.01,
        'objective': 'regression',
        'metric': 'mse', 
        "early_stopping_round": 5}
    params_list = [params for _ in range(6)]
    mpp = ModelPipeLine(make_real_preds=False, y_transforms=[42, 26.5, 21, 9, 10, 21])
    mpp.fit_boosting(params_list=params_list, models_indicies=[0, 1, 2, 3, 4, 5], eval=True)
    mpp.predict()
    embed()
        
        