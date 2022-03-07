import pipeline
import pandas as pd
from IPython import embed
import csv


def write_csv_prediction(test_preds, dates):
    """
    Write the prediction to a csv file (along the timepoints)

    :param test_preds:  
        test_preds[i] is the prediction of the (i+1)-th bolt, as an array, ordered after the timepoints
    :param dates:   
        corresponding datepoints to the prediction

    """

    df_prediction = pd.DataFrame()
    df_prediction['timepoints'] = dates
    df_prediction = df_prediction.set_index('timepoints')

    for bolt_no, bolt_data in enumerate(test_preds):
        df_prediction[f'Bolt_{bolt_no+1}_Tensile'] = bolt_data
    
    df_prediction.to_csv('../output/prediction.csv')
    
    
        

def main():

    # params = {
    #     "n_estimators": 100, 
    #     "learning_rate": 0.05,
    #     'objective': 'regression',
    #     'metric': 'mse'}
    # params_list = [params for _ in range(6)]


    # mpp = pipeline.ModelPipeLine(make_real_preds=True, y_transforms=[42, 26.5, 21, 9, 10, 21])
    # mpp.fit_boosting(params_list=params_list, models_indicies=[0, 1, 2, 3, 4, 5], eval=False)
    # mpp.predict()

    # write_csv_prediction(mpp.test_preds, mpp.prediction_data.index)


    params = {
        "n_estimators": 500, 
        "learning_rate": 0.01,
        'objective': 'regression',
        'metric': 'mse', 
        "early_stopping_round": 5}
    params_list = [params for _ in range(6)]
    mpp = pipeline.ModelPipeLine(make_real_preds=False, y_transforms=[42, 26.5, 21, 9, 10, 21])
    mpp.fit_boosting(params_list=params_list, models_indicies=[0, 1, 2, 3, 4, 5], eval=True)
    mpp.predict()
    mpp.plot_all()


if __name__ == '__main__':
    main()