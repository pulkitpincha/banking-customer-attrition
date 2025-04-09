# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 05:59:02 2023

@author: stimp
"""

#importing libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import scipy.stats
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

#loading the data
df = pd.read_csv("Datasets/BankChurners.csv")
df.head()

#removing NB flagged data (causes bias during prediction)
df = df.drop(['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'], axis=1)
df.info()

#checking for unknown data
df[df == 'Unknown'].count()

#distribution of target variable (attrition)
df['Attrition_Flag'].value_counts(normalize=True)
num = df.groupby(['Attrition_Flag'])['CLIENTNUM'].count().to_frame().reset_index()
num

#plotting target variable distribution
labels = num.Attrition_Flag
values = num.CLIENTNUM

fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0.2])])
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.90,
    xanchor="left",
    x=0.20))

#target variable -> numeric
df['Attrition_Flag'] = df.Attrition_Flag.map({'Attrited Customer': 1,
                                               'Existing Customer': 0})
df.head()

##exploratory data analysis
#categorical variables
#plotting categorical variable distribution
cat_var = ['Gender', 'Income_Category', 'Marital_Status', 'Card_Category', 'Education_Level']
crosstab_list = []
for c in cat_var:
    plt.figure(figsize = (20,35))
    plt.subplot(6,2,1)
    sns.countplot(x = c, palette = 'Set2', data = df)
    plt.show()
    
    cross_tab = pd.crosstab(index=df[c],
                    columns=df['Attrition_Flag'])
    crosstab_list.append(cross_tab)
    cross_tab_prop = pd.crosstab(index=df[c], columns=df['Attrition_Flag'], normalize="index")
    cross_tab_prop

#checking if it is significant
chi2_list = []
p_list = []
dof_list = []
expected_f_list = []

for c in crosstab_list:
    chi2_stat, p, dof, expected_f = scipy.stats.chi2_contingency(c)
    chi2_list.append(chi2_stat)
    p_list.append(p)
    dof_list.append(dof)
    expected_f_list.append(expected_f)

for item1, item2, item3, item4, item5 in zip(cat_var, chi2_list, p_list, dof_list, expected_f_list):
    print(f'Variable: {item1} \nChi2: {item2} \nP Value: {item3} \nDOF: {item4} \nExpected Frequency:\n {item5}\n')

#numeric variables
fig = plt.figure(figsize = (20,20))
ax = fig.gca()
plt.tight_layout()
df.loc[:, ~df.columns.isin(['CLIENTNUM', "Attrition_Flag_Int"])].hist(ax = ax)
fig.show()

#correlation matrix
correlation = df.loc[:, ~df.columns.isin(['CLIENTNUM'])].corr().round(2)
plt.figure(figsize = (14,7))
sns.heatmap(correlation, annot = True, cmap = 'RdBu', vmin=-1, vmax=1)
plt.show()

#box-plot
num_var = ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count',
           'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
           'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']

for n in num_var:
    plt.figure(figsize = (20,35))
    plt.subplot(6,2,1)
    sns.boxplot(x='Attrition_Flag', y=n, data=df)
    plt.show()

##selecting variables for models
df.columns
numeric = ['Dependent_count', 'Total_Relationship_Count', 'Months_Inactive_12_mon',
           'Contacts_Count_12_mon', 'Total_Trans_Ct', 'Avg_Utilization_Ratio']
categorical = ['Gender', 'Income_Category']

##data wrangling
#categorical
Income_Category_enc = df['Income_Category']
Income_Category_enc.unique()

#mapping unknown values to 0/less than 40k (most repeated value)
df['Income_Category'] =  df['Income_Category'].map({'Unknown':1,
                                                'Less than $40K':1,
                                                '$40K - $60K':2,
                                                '$60K - $80K':3,
                                                '$80K - $120K':4, 
                                                '$120K +':5})
#onehot encoding variable gender
categoricals = ['Gender']

enc = OneHotEncoder(drop='first')
X = df[categoricals]
enc.fit(X)
enc.categories_

dummies = enc.transform(X).toarray()
dummies
dummies.shape
dummies_df = pd.DataFrame(dummies)
dummies_df

col_names = [categoricals[i] + '_' + enc.categories_[i] for i in range(len(categoricals)) ]
col_names
col_names_drop_first = [sublist[i] for sublist in col_names for i in range(len(sublist)) if i != 0]
col_names_drop_first
dummies_df.columns = col_names_drop_first
dummies_df

#numeric
numeric.append('Income_Category')
X = df[numeric]

#standardizing
scaler = StandardScaler()
scaler.fit(X)

std_numerical_data = scaler.transform(X)
std_df = pd.DataFrame(std_numerical_data)
std_df.columns = [i + '_std' for i in numeric]
std_df

#concat (combining dataframes)
df_prep = pd.concat([dummies_df, std_df], axis = 1)
df_prep

X = df_prep #independent variables
y = df.Attrition_Flag #dependant/target variable

##splitting data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state = 12)

CM = []
AC = []
PS = []
RS = []
F1 = []

##naive bayes model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
y_pred

#confusion matrix
CM.append(confusion_matrix(y_test, y_pred))
AC.append(accuracy_score(y_test, y_pred)) #higher the better (overall accuracy)
PS.append(precision_score(y_test, y_pred)) #higher the better (positive prediction accuracy / [TP/(TP + FP)])
RS.append(recall_score(y_test, y_pred)) #[TP/(TP + FN)]
F1.append(f1_score(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='.0f')
plt.title("Naive Bayes")
plt.ylabel('Real')
plt.xlabel('Prediction');

##logistic regression
logistic_regression = LogisticRegression(penalty='none') 
logistic_regression.fit(X_train, y_train);

y_pred = logistic_regression.predict(X_test)
y_pred

#confusion matrix
CM.append(confusion_matrix(y_test, y_pred))
AC.append(accuracy_score(y_test, y_pred))
PS.append(precision_score(y_test, y_pred))
RS.append(recall_score(y_test, y_pred))
F1.append(f1_score(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='.0f')
plt.title("Logistic Regression")
plt.ylabel('Real')
plt.xlabel('Prediction');

##KNN model
#function to check different KNN neighbours
def scores_knn(X, y, start,stop,step):
    
    scores_list = []
    
    for i in range(start,stop,step):
        
        model = KNeighborsClassifier(n_neighbors=i)

        kf = KFold(n_splits=10, shuffle=True, random_state=10)
        cv_scores = cross_val_score(model, X, y, cv=kf)

        dict_row_score = {'mean_score':np.mean(cv_scores),'score_std':np.std(cv_scores),'n_neighbours':i}

        scores_list.append(dict_row_score)
    
    df_scores = pd.DataFrame(scores_list)
    
    df_scores['lower_bound'] = df_scores['mean_score'] - df_scores['score_std']
    df_scores['upper_bound'] = df_scores['mean_score'] + df_scores['score_std']
    
    return df_scores

#1 to 20 neighbours
df_scores= scores_knn(X_train, y_train, 1, 21, 1)

#plotting the scores
plt.plot(df_scores['n_neighbours'], df_scores['lower_bound'], color='r')
plt.plot(df_scores['n_neighbours'], df_scores['mean_score'], color='b')
plt.plot(df_scores['n_neighbours'], df_scores['upper_bound'], color='r')
plt.title("N-neighbours")
plt.ylim(0.7, 1);

df_scores.loc[df_scores.mean_score == df_scores.mean_score.max()]

best_k = df_scores.loc[df_scores.mean_score == df_scores.mean_score.max(),'n_neighbours'].values[0]
best_k

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#confusion matrix
CM.append(confusion_matrix(y_test, y_pred))
AC.append(accuracy_score(y_test, y_pred)) #higher the better (overall accuracy)
PS.append(precision_score(y_test, y_pred)) #higher the better (positive prediction accuracy / [TP/(TP + FP)])
RS.append(recall_score(y_test, y_pred)) #[TP/(TP + FN)]
F1.append(f1_score(y_test, y_pred))

##ANN
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

#compiling model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#training
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

#evaluation on test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

#prediction on test set
y_pred = model.predict(X_test)

#predictions to binary
y_pred_binary = (y_pred > 0.5).astype(int)

#confusion matrix
CM.append(confusion_matrix(y_test, y_pred_binary))
AC.append(accuracy_score(y_test, y_pred_binary)) #higher the better (overall accuracy)
PS.append(precision_score(y_test, y_pred_binary)) #higher the better (positive prediction accuracy / [TP/(TP + FP)])
RS.append(recall_score(y_test, y_pred_binary)) #[TP/(TP + FN)]
F1.append(f1_score(y_test, y_pred_binary))

models = ['Naive Bayes', 'Logistic Regression', 'K-Nearest Neighbour', 'Artificial Neural Network']

for item1, item2, item3, item4, item5, item6 in zip(models, CM, AC, PS, RS, F1):
    print(f'Model: {item1} \nConfusion Matrix: \n{item2} \nAccuracy: {item3} \nPrecision: {item4} \nRecall: {item5} \nF1 Score: {item6}\n')
