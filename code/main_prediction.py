import pipeline
import pandas as pd
from IPython import embed
import numpy as np
import csv


def write_csv_prediction(test_preds, dates, scales=None):
    """
    Write the prediction to a csv file (along the timepoints)
    :param test_preds:  
        test_preds[i] is the prediction of the (i+1)-th bolt, as an array, ordered after the timepoints
    :param dates:   
        corresponding datepoints to the prediction
    """
    if scales is None:
        scales = [0, 0, 0, 0, 0, 0]
        
    df_prediction = pd.DataFrame()
    df_prediction['timepoints'] = dates
    df_prediction = df_prediction.set_index('timepoints')
    
    # embed()
    for bolt_no, bolt_data in enumerate(test_preds):
        df_prediction[f'Bolt_{bolt_no+1}_Tensile'] = np.array(bolt_data + scales[bolt_no])
    
    df_prediction.to_csv('output/prediction3.csv')
    
    
        

def main():

    # params = {
    #     "n_estimators": 100, 
    #     "learning_rate": 0.05,
    #     'objective': 'regression',
    #     'metric': 'mse'}
    params = {
        "n_estimators": 280, 
        "learning_rate": 0.01,
        'objective': 'regression',
        # "early_stopping_round": 5,
        'metric': 'mse'} 
    params_list = [params for _ in range(6)]


    mpp = pipeline.ModelPipeLine(make_real_preds=True, y_transforms=[42, 26.5, 21, 9, 10, 21])
    mpp.fit_boosting(params_list=params_list, models_indicies=[0, 1, 2, 3, 4, 5], eval=False)
    mpp.predict()
    
    scales = []
    if mpp.y_transforms is not None: # Transform back
        for i in range(len(mpp.models)):
            model_index = mpp.models_indicies[i]
            scaling = mpp.time_slopes[model_index]
            scaling = scaling[mpp.x_train.shape[0]:]
            scales.append(scaling)
        
    write_csv_prediction(mpp.test_preds, mpp.prediction_data.index, scales)
    embed()

    # params = {
    #     "n_estimators": 280, 
    #     "learning_rate": 0.01,
    #     'objective': 'regression',
    #     # "early_stopping_round": 5,
    #     'metric': 'mse'} 
    # params_list = [params for _ in range(6)]
    # mpp = pipeline.ModelPipeLine(make_real_preds=False, y_transforms=[42, 26.5, 21, 9, 10, 21])
    # mpp.fit_boosting(params_list=params_list, models_indicies=[0, 1, 2, 3, 4, 5], eval=True)
    # mpp.predict()
    # mpp.plot_all()
    # 

if __name__ == '__main__':
    main()