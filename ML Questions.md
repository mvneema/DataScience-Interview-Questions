# Data Science Interview Questions:

## Machine Learning Question:

### 1.What are the assumptions required for linear regression? What if some of these assumptions are violated?
There are four assumptions associated with a linear regression model:

**Linearity:** The relationship between X and the mean of Y is linear.
**Homoscedasticity:** The variance of the residual is the same for any value of X.
**Independence:** Observations are independent of each other.
**Normality:** For any fixed value of X, Y is normally distributed.
Extreme violations of these assumptions will make the results redundant. Small violations of these assumptions will result in a greater bias or variance of the estimate.

### 2. What is collinearity? What is multicollinearity? How do you deal with it?
Collinearity is a linear association between two predictors. Multicollinearity is a situation where two or more predictors are highly linearly related.
This can be problematic because it undermines the statistical significance of an independent variable. 
While it may not necessarily have a large impact on the model’s accuracy, it affects the variance of the prediction and reduces the quality of the interpretation of the independent variables.
You could use the Variance Inflation Factors (VIF) to determine if there is any multicollinearity between independent variables — a standard benchmark is that if the VIF is greater than 5 then multicollinearity exists.

### 3. What are the drawbacks of a linear model?
There are a couple of drawbacks of a linear model: A linear model holds some strong assumptions that may not be true in application. 
- It assumes a linear relationship, multivariate normality, no or little multicollinearity, no auto-correlation, and homoscedasticity.
- A linear model can’t be used for discrete or binary outcomes.
- You can’t vary the model flexibility of a linear model.

### 4. What is overfitting?
Overfitting is an error where the model ‘fits’ the data too well, resulting in a model with high variance and low bias. As a consequence, an overfit model will inaccurately predict new data points even though it has a high accuracy on the training data.

### 5. Describe decision trees, SVMs, and random forests. Talk about their advantage and disadvantages.
Decision Trees: a tree-like model used to model decisions based on one or more conditions.
Pros: easy to implement, intuitive, handles missing values
Cons: high variance, inaccurate
Support Vector Machines: a classification technique that finds a hyperplane or a boundary between the two classes of data that maximizes the margin between the two classes. There are many planes that can separate the two classes, but only one plane can maximize the margin or distance between the classes.
Pros: accurate in high dimensionality
Cons: prone to over-fitting, does not directly provide probability estimates
Random Forests: an ensemble learning technique that builds off of decision trees. Random forests involve creating multiple decision trees using bootstrapped datasets of the original data and randomly selecting a subset of variables at each step of the decision tree. The model then selects the mode of all of the predictions of each decision tree.
Pros: can achieve higher accuracy, handle missing values, feature scaling not required, can determine feature importance.
Cons: black box, computationally intensive.

### 6. Dimensionality reduction is the process of reducing the number of features in a dataset. This is important mainly in the case when you want to reduce variance in your model (overfitting).
Wikipedia states four advantages of dimensionality reduction (see here):
It reduces the time and storage space required
Removal of multi-collinearity improves the interpretation of the parameters of the machine learning model
It becomes easier to visualize the data when reduced to very low dimensions such as 2D or 3D
It avoids the curse of dimensionality.

### 7. What is boosting?
Boosting is an ensemble method to improve a model by reducing its bias and variance, ultimately converting weak learners to strong learners. The general idea is to train a weak learner and sequentially iterate and improve the model by learning from the previous learner. 


### 8. What is the meaning of ACF and PACF?
To understand ACF and PACF, you first need to know what autocorrelation or serial correlation is. Autocorrelation looks at the degree of similarity between a given time series and a lagged version of itself.
Therefore, the autocorrelation function (ACF) is a tool that is used to find patterns in the data, specifically in terms of correlations between points separated by various time lags. For example, ACF(0)=1 means that all data points are perfectly correlated with themselves and ACF(1)=0.9 means that the correlation between one point and the next one is 0.9.
The PACF is short for partial autocorrelation function. Quoting a text from StackExchange, “It can be thought as the correlation between two points that are separated by some number of periods n, but with the effect of the intervening correlations removed.” For example. If T1 is directly correlated with T2 and T2 is directly correlated with T3, it would appear that T1 is correlated with T3. PACF will remove the intervening correlation with T2.

### 9. What is the bias-variance tradeoff?
The bias of an estimator is the difference between the expected value and true value. A model with a high bias tends to be oversimplified and results in underfitting. Variance represents the model’s sensitivity to the data and the noise. A model with high variance results in overfitting.
Therefore, the bias-variance tradeoff is a property of machine learning models in which lower variance results in higher bias and vice versa. Generally, an optimal balance of the two can be found in which error is minimized.
        
### 10. How does XGBoost handle the bias-variance tradeoff?

XGBoost is an ensemble Machine Learning algorithm that leverages the gradient boosting algorithm. In essence, XGBoost is like a bagging and boosting technique on steroids. Therefore, you can say that XGBoost handles bias and variance similar to that of any boosting technique. Boosting is an ensemble meta-algorithm that reduces both bias and variance by takes a weighted average of many weak models. By focusing on weak predictions and iterating through models, the error (thus the bias) is reduced. Similarly, because it takes a weighted average of many weak models, the final model has a lower variance than each of the weaker models themselves.

### 11. What is a random forest? Why is Naive Bayes better?
Random forests are an ensemble learning technique that builds off of decision trees. Random forests involve creating multiple decision trees using bootstrapped datasets of the original data and randomly selecting a subset of variables at each step of the decision tree. The model then selects the mode of all of the predictions of each decision tree. By relying on a “majority wins” model, it reduces the risk of error from an individual tree.

For example, if we created one decision tree, the third one, it would predict 0. But if we relied on the mode of all 4 decision trees, the predicted value would be 1. This is the power of random forests.
Random forests offer several other benefits including strong performance, can model non-linear boundaries, no cross-validation needed, and gives feature importance.
Naive Bayes is better in the sense that it is easy to train and understand the process and results. A random forest can seem like a black box. Therefore, a Naive Bayes algorithm may be better in terms of implementation and understanding. However, in terms of performance, a random forest is typically stronger because it is an ensemble technique.

### 12. Why is Rectified Linear Unit a good activation function?
The Rectified Linear Unit, also known as the ReLU function, is known to be a better activation function than the sigmoid function and the tanh function because it performs gradient descent faster. Notice in the image to the left that when x (or z) is very large, the slope is very small, which slows gradient descent significantly. This, however, is not the case for the ReLU function.

### 13. What is the use of regularization? What are the differences between L1 and L2 regularization?
Both L1 and L2 regularization are methods used to reduce the overfitting of training data. Least Squares minimizes the sum of the squared residuals, which can result in low bias but high variance.

### L1 vs L2 Regularization
L2 Regularization, also called ridge regression, minimizes the sum of the squared residuals plus lambda times the slope squared. This additional term is called the Ridge Regression Penalty. This increases the bias of the model, making the fit worse on the training data, but also decreases the variance.
If you take the ridge regression penalty and replace it with the absolute value of the slope, then you get Lasso regression or L1 regularization.
L2 is less robust but has a stable solution and always one solution. L1 is more robust but has an unstable solution and can possibly have multiple solutions.

### 14. What is the difference between online and batch learning?
Batch learning, also known as offline learning, is when you learn over groups of patterns. This is the type of learning that most people are familiar with, where you source a dataset and build a model on the whole dataset at once.
Online learning, on the other hand, is an approach that ingests data one observation at a time. Online learning is data-efficient because the data is no longer required once it is consumed, which technically means that you don’t have to store your data.
### 15. How would you handle NULLs when querying a data set? Are there any other ways?
There are a number of ways to handle null values including the following:
* You can omit rows with null values altogether
* You can replace null values with measures of central tendency (mean, median, mode) or replace it with a new category (eg. ‘None’)
* You can predict the null values based on other variables. For example, if a row has a null value for weight, but it has a value for height, you can replace the null value with the average weight for that given height.
* Lastly, you can leave the null values if you are using a machine learning model that automatically deals with null values.
### 16. How do you prevent overfitting and complexity of a model?
For those who don’t know, overfitting is a modeling error when a function fits the data too closely, resulting in high levels of error when new data is introduced to the model.
There are a number of ways that you can prevent overfitting of a model:
### Cross-validation:###
Cross-validation is a technique used to assess how well a model performs on a new independent dataset. The simplest example of cross-validation is when you split your data into two groups: training data and testing data, where you use the training data to build the model and the testing data to test the model.
### Regularization:###
Overfitting occurs when models have higher degree polynomials. Thus, regularization reduces overfitting by penalizing higher degree polynomials.
Reduce the number of features: You can also reduce overfitting by simply reducing the number of input features. You can do this by manually removing features, or you can use a technique, called Principal Component Analysis, which projects higher dimensional data (eg. 3 dimensions) to a smaller space (eg. 2 dimensions).
Ensemble Learning Techniques: Ensemble techniques take many weak learners and converts them into a strong learner through bagging and boosting. Through bagging and boosting, these techniques tend to overfit less than their alternative counterparts.
