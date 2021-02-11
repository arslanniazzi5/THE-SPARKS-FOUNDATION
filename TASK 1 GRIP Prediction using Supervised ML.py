#!/usr/bin/env python
# coding: utf-8

# # Author: Arslan Haider Khan 
# 
# # TASK 1: Prediction using Supervised ML
# 
# Problem Statement:Predict the percentage of a student based on the number of study hours and calculate the predicted score if a student studies for 9.25 hours/day.
# 
# Solution :For predicting the student's score based on the number of hours' studied, I have used Linear Regression.In this regression task I will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables. 
# 
# Dataset  : https://bit.ly/3kXTdox
# 

# # Step 1: Reading and Understanding the Data
# 
# Let's start with the following steps:
# 
# 1: Importing data using the pandas library
# 2: Understanding the structure of the data

# In[1]:


# Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[2]:


# Read data from the given url
url ="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
data= pd.read_csv(url)
print("Data loaded successfully!")
data.head()


# Let's inspect the various aspects of our dataframe

# In[3]:


data.describe()


# In[4]:


data.info()


# In[5]:


data.shape


# In[6]:


#checking for null values
data.isnull().sum()


# # Step 2: Visualising the Data
# 
# Let's plot our data on a graph to look closely at the dataset given and try to find the relationship between the data.

# In[7]:


#Plot of Scores Distribution
data.plot(x='Hours',y='Scores',style='o')
plt.title('Hours Vs Percentage Score')
plt.xlabel('Hours studied')
plt.ylabel('Percentage Score')
plt.show()


# Inference: From the above graph, we can infer that there is a positive linear relation between the number of study hours and Percentage Score.

# # Step 3: Data Preprocessing
# 
# I first assign the feature variable, Hours, in this case, to the variable X and the response variable, Score, to the variable y.The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).

# In[8]:


# Data is divided into "attributes" and "labels"
X = data.iloc[:, :-1].values
y= data.iloc[:, 1].values


# # Step 4: Training the Algorithm
# 
# After splitting the data into training and testing sets,finally it's the time to train our algorithm.

# In[9]:


#Splitting of data into training and testing sets
X_train, X_test, y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)
regressor= LinearRegression()
regressor.fit(X_train.reshape(-1,1), y_train)
print("Training Completed")


# # Plotting the Line of Regression

# In[10]:


#Visualizing the best fit line of Regression.
line= regressor.coef_*X + regressor.intercept_
plt.scatter(X,y)
plt.plot(X,line, color='black')
plt.show()


# # Predicting the Data
# 
# Now that we have fitted a regression line on our train dataset, it's time to make some predictions on the test data. For this, we first need to add a constant to the X_test data like we did for X_train and then we can simply go on and predict the y values corresponding to X_test using the predict attribute of the fitted regression line.

# In[11]:



print(X_test) #Testing data
y_pred= regressor.predict(X_test) #Model predictions


# # Step 5: Comparison of Actual result vs Predicted result

# In[12]:


df = pd.DataFrame({'Actual Result': y_test, 'Predicted Result': y_pred})  
df


# # Step 6: Estimating Training & Test Score

# In[13]:



# Training score
print("Training Score: ", regressor.score(X_train,y_train))
# Test Score
print("Test Score: ", regressor.score(X_test,y_test))


# In[14]:


df.plot(kind='bar', figsize=(5,5))
plt.grid(linewidth='0.5', color='blue')
plt.grid(linewidth='0.5', color='red')


# In[15]:




# Testing with own Data
Hours = 9.25
test= np.array([Hours]).reshape(-1,1)
prediction = regressor.predict(test)
print("Number Of Hours = {}".format(Hours))
print("Predicted Score= {}".format(prediction[0]))


# # Step 7 (Final Step): Evaluating the model
# 
# The final step is to evaluate the performance of algorithm. This step is quite important to compare how well different algorithms perform on a particular dataset. we have chosen the mean square error. Also, there are many such metrics which we can choose.

# In[16]:


# Let's calculate different errors to compare the model performance and predict accuracy.
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
print("Mean Absolute Error: ",mean_absolute_error(y_test,y_pred))
print("Mean Squared Error: ",mean_squared_error(y_test,y_pred))
print("R2 score: ", r2_score(y_test,y_pred))


# # Conclusion:
# 
# After Analysing the dataset we got Predicted score around 93 based on the number of study hours i.e., 9.25 hrs/day.
# 
# Thank you :)
