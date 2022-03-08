# HACKATHON
CLOPEN tries to beat other teams in hackathon competition.

![](https://media.giphy.com/media/VekcnHOwOI5So/giphy.gif)

## TEAM MEMBERS:
### Sigurd Holmsen 🐻‍❄️
**Bachelor's degree:** Mathematics and Economics

**Study:** Computational Science with Applied Mathematics and Risk Analysis

### Tobias Opsahl 🐼
**Bachelor's degree:** Mathematics with Informatics

**Study:** Data Science with Applied Japanese

### Øystein Høistad Bruce 🐮
**Bachelor's degree:** Mathematics and Economics

**Study:** Computational Science with Applied Mathematics and Risk Analysis


# How to fit the models:

In the file code/pipline.py, there is a pretty generic class ModelsPipeLine. Here one can fit both a linear model, a boosting model, and with different parameters and testing sets. The doc-string should be pretty well documented, but here is a breif overview. Change the path in the constructor arguments so that the path goes to the correct dataset. If make_real_preds is set to True, the model will predict on the real test set (without labels), if False, it will split the training and evaluate the test error. The features to be used is also given as an argument, but both arguments have descent keywordarguments. The constructor reads, preprocesses and splits. After that, call fit_boosting() or fit_linear() to fit the model. The argument model_indicies marks which bolts the model fits. There are one model for each bolt. For example, if model_indicies is [0, 1, 2, 3, 4, 5], one model is fitted for each bolt. if it is [1, 5], only the second and sixth bolt gets a model. After fitting, call predict() and plot_all() to see the results. Be careful as for the boostingmodel, the real predictions should be plussed with the time_slope, which is done in main_predictions.py and and plot_all(). The file main_predictions.py prints the predictions to csv files.


## The code will exclusively _not_ be written in notebooks. Why?

![](Annoyed_Cat.png)
