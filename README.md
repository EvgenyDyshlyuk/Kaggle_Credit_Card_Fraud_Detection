# Kaggle-Credit_Card_Fraud_Detection
**This is a personal ML project for Credit Card Fraud Detection task from a [Kaggle competition](https://www.kaggle.com/mlg-ulb/creditcardfraud)**  

## About the data:
The datasets contains transactions made by credit cards in September 2013 by european cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

## The project consists of 5 parts:
**[1_EDA.ipynb](https://github.com/EvgenyDyshlyuk/Kaggle_Credit_Card_Fraud_Detection/blob/master/1_EDA.ipynb)**

- Unstratified train/test split was done at 70/30 (target split 74/26).
- Time was shuffled so we order by time to restore the original dataset.
- Some nice data visualizations can be found [here](https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets)
- Shapley feature importance [(read why it's good)](https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf) for an LGBMClassifier trained on the dataset is below in Fig.1.  
![Shap](https://github.com/EvgenyDyshlyuk/Kaggle_Credit_Card_Fraud_Detection/blob/master/figures/SHAP.png)  
***Fig.1. Shapley Feature Importance***

**[2_Fit_Classifiers.ipynb](https://github.com/EvgenyDyshlyuk/Kaggle_Credit_Card_Fraud_Detection/blob/master/2_Fit_Classifiers.ipynb)**
- A number of sklearn classifiers + lgbm classifier fit to dataset as a quick solution
- Setting parameter class_weight = "balanced" improves performance on all classifiers except RandomTrees and ExtraTrees which show best results with and without this parameter used (see Fig.2. below).

![Class_weight](https://github.com/EvgenyDyshlyuk/Kaggle_Credit_Card_Fraud_Detection/blob/master/figures/Class_weight.png)   
***Fig.2.RandomForest and ExtraTrees classifiers trained without (left) and with (right) "class_weigth" parameter set to balanced***

**[3_Metric.ipynb](https://github.com/EvgenyDyshlyuk/Kaggle_Credit_Card_Fraud_Detection/blob/master/3_Metric.ipynb)**

**[4_HPO.ipynb](https://github.com/EvgenyDyshlyuk/Kaggle_Credit_Card_Fraud_Detection/blob/master/4_HPO.ipynb)**

![HPO](https://github.com/EvgenyDyshlyuk/Kaggle_Credit_Card_Fraud_Detection/blob/master/figures/HPO.png)   
***Fig.3.Hyperparameters optimization space in addition to model hyperparameters includes 2 SMOTE upsampling parameters: alpha_over and k_neighbour***

**[5_Ensemble.ipynb](https://github.com/EvgenyDyshlyuk/Kaggle_Credit_Card_Fraud_Detection/blob/master/5_Ensemble.ipynb)**
