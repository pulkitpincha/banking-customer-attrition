# banking-customer-attrition
Dataset: https://www.kaggle.com/datasets/thedevastator/predicting-credit-card-customer-attrition-with-m

Analyzing variables that contribute to the attrition of customers at banks as well as seeing if we can use them to predict future attrition. 

# Research Questions:
1) Can customer demographics such as gender, marital status, education level, and income 
category be used to predict which customers are more likely to churn?
2) Is it possible to predict the likelihood of customer churning in the future by analyzing their 
spending behaviour leading up to churning?
3) Can a classifier be created to predict which customers are more susceptible to attrition based 
on their credit score, credit limit, utilization ratio, and other spending behaviour metrics over 
time, as a potential early warning system for predicting attrition?

# Banking Customer Attrition
- Importing the dataset and removing the Naïve Bayes flagged columns as they would cause bias during predictive modelling.
- Converting our target variable ‘Attrition_Flag’ into numeric (0,1).
- Plotting all the categorical feature distributions.
- Checking the significance of the features by calculating chi2, p-value, DOF, and expected frequency:

![image](https://github.com/user-attachments/assets/bdfdc102-336b-4057-9508-47c5d19c3505)

From this we can see that the features ‘Marital_Status’, ‘Card_Category’, and ‘Education_Level’ are not significant as they have p-values of greater than 0.05. Hence, we drop these features later on.

After this we plot a corelation matrix for the numeric features to further help our feature selection:
![image](https://github.com/user-attachments/assets/d2106a64-7f57-49b0-9843-5fee445f3111)

- After doing so we drop the features that have high corelation with one another.
- Now we map the unknown values in the ‘Income_Category’ feature to the lowest and label encode the feature.
- We then make two separate dataframes, one of dummy variables of our categorical features, and the second of our numeric features after standardizing them.
- We then combine them and split it into train and test datasets.

## Running different classification models to see which gives us the best results:
![image](https://github.com/user-attachments/assets/3227508c-b7c5-4506-a8ad-0ce6deb498e7)

## What are we looking for?
1) High Accuracy Score: It measures overall correctness of the model. [(TP+TN)/n]
2) High Precision Score: Measure of the accuracy of the positive predictions. [TP/(TP+FP)]
3) High Recall Score: Measures the proportion of true positives out of all the actual positives. [TP/(TP+FN)]
4) High F1 Score: Shows the balance between precision and recall. [(2×Precision×Recall)/(Precision + Recall)]

# Conclusion
Each use case has a different order of priorities of these 4 measures. For our scenario, our order of priority is as follows: Recall -> Precision -> F1 -> Accuracy. This is because our industry is one where customer acquisition costs are high and preventing customer churn is prioritized over efficient resource allocation, influencing a higher emphasis on recall.

However, here we can see that our Artificial Neural Network model excels further in all 4 measures than the other models, and hence we don’t need the order of prioritizing to choose our best model. The ANN model is built with two hidden layers and is run through three stages of ‘RELU’ activations followed by a ‘Sigmoid’ activation which is used commonly for predicting categorical features. The hidden layers also have dropout functions to make the model more regularized and to reduce overfitting.


