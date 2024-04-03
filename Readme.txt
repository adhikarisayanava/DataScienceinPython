	• Data Wrangling : process of transforming raw data into more useful format.
	• Exploratory Data Analysis(EDA) : is a process of analysing data to gain valuable insights as statistical summary and visualizations.
	• Pandas is a powerful opensource data analysis tool in python.
	• OneHotEncoder : it is a process in data preprocessing to convert categorical data into a format that works better with machine learning algorithms
	• Scaling : is an important step to take prior to training of machine learning models to ensure that features are within the same scale. Scikit-Learn offers several tools to perform feature scaling.
		○ Normalization : conducted to make feature values range from 0 to 1.
		○ Standardization(Z-score) : conducted to transform data to have mean of 0 and standard deviation of 1,but can have any upper and lower values.
	• Data Visualization
		○ Matplotlib:
			§ Comprehensive library for creating static, animated and interactive visualizations in Python.
			§ Works great with Pandas dataFrames.
		○ Seaborn:
			§ Is a data visualization library that sits on top of matplotlib
	• Regression
		○ Works by estimating a continuous dependent variable Y from a list of independent input variables X(Linear regression for example predicting weather forecast).
		○ Perfect regression model shall have small bias(error) and small variability(difference in fits between training and testing dataset).
		○ Used in real-life applications like financial forecasting, weather analysis/forecast.
		○ 
		○ XGBoost is a supervised learning algorithm that implements gradient boosted tree algorithms. It works by building a series of models from training data, each model is built to correct the mistakes made by the previous model(Boosting). Models
		are added sequentially until no further improvements can be made. 
			§ Extremely fast
			§ Good memory utilization
			§ Robust
			§ Does not need scaling
			§ It works for both regression and classification
			§ Example of ensemble learning
		
	• Classification
		○ KPIs : accuracy, precision, recall(sensitivity), ROC(Receiver Operating Characteristic)(AUC) 
		○ Types:
			§ Logistic regression : is used to predict  binary output with 2 possible values(0 or 1)(pass/fail, win/lose, healthy/sick)
			§ Support Vector Machine : technique that helps in deciding the best boundary that differentiates between various categories of data.
			§ Random Forest : It creates a set of decision trees from randomly selected subsets of training set. It then combines votes from decision trees to decide the final class of test object. For large data, it produces highly accurate predictions.
			§ K-nearest neighbour(KNN) :  works by finding the most similar data points in the training data, and attempt to make an educated guess based on their classifications.
			§ Naïve Bayes:  based on probability * likelihood
	• AutoGluon 
		○ It allows for quick prototyping of AI/ML models(both classification and regression) using few simple lines of code.
		○ Open source library from AWS Sagemaker Autopilot
		○ It works with text, image and tabular datasets.

	• Model Optimization
		○ Parameter : values that are obtained by the training process such as slope and Y-intercept or network weights and biases.
		○ Hyperparameters : values set prior to the training process such as learning rate. Learning rate is a hyperparameter that represents the size of the steps taken which indicates how aggressive you'd like to update the parameters.
		○ Strategies:
			§ Grid Search :
				□ Performs exhaustive search over a specified list of parameters.
				□ You provide the algorithm with the hyperparameters you'd like to experiment with and the values you want to try out.
				□ Works great if  the number of combinations are limited.
				
			§ Randomized Search:
				□ Preferred compared to Grid Search, when the search space is large.
				□ It works by evaluating a select few numbers of random combinations.
				
			§ Bayesian Optimization:
				□ It overcomes the drawbacks of random search algorithms by exploring search spaces in a more efficient manner.
				□ If a region in the search space appears to be promising(resulted in a small error), this region should be explored more which increases the chances of achieving better performance.
		○ Ridge(L2) Regression : works by attempting at increasing the (penalty)bias(error) to improve variance and to overcome overfitting. All features are maintained but weighted accordingly. No features are allowed to go to zero.
		○ Lasso(L1) Regression : similar to ridge regression, but instead of squaring the slope, the absolute value of the slope is added as a penalty term. Used to perform feature selection so some features are allowed to go to zero.
	• Least Sum of Squares
		○ Linear Regression : y= b+mx
		○ Least sum of squares is used to obtain the coefficients m & b(y-intercept).
	• Scikit-Learn
		○ Free machine learning library developed for python.
		○ It offers algorithms for classification, regression and clustering.
		○ Can be used efficiently in data preprocessing.

******** To see Histogram outputs, we need to add the extension "Jupyter" to VS Code IDE.