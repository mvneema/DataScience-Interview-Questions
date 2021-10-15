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

