#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libraries
import pandas as pd
import numpy as np 
import sklearn
from sklearn.neural_network import MLPClassifier


# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score


# In[2]:


#Importing the dataset and verifying it's details
data= pd.read_excel("dataset.xlsx")
print(data.shape)
data.describe().transpose()


# In[3]:


#Removing url and urlid columns as it was added for out convenience only
extra_column1 = ['urlid'] 
extra_column2 = ['url'] 
#Removing Data labels from X set 
target_column = ['label'] 
predictors = list(set(list(data.columns))-set(extra_column1)-set(extra_column2)-set(target_column))
#Normalzing the parameters
data[predictors] = data[predictors]/data[predictors].max()
data.describe().transpose()


# In[78]:


#Defining Training and test sets
X = data[predictors].values
y = data[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.250, random_state=42)
print(X_train.shape); print(X_test.shape)


# In[79]:


#Training the model. A neural network architecture with 3 hidden layers is created (A relatively smaller Newural network due to less data available) 

from sklearn.neural_network import MLPClassifier

model =  MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, max_iter=200, shuffle=True, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.fit(X_train,y_train)

predict_train = model.predict(X_train)
predict_test = model.predict(X_test)


# In[80]:


#Analysing F score, recall and precision for training data
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_train,predict_train))
print(classification_report(y_train,predict_train))

#Accuracy comes out to 97%


# In[81]:


#Analysing F score, recall and precision for test data
print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))

#There is slight reduction in precision and recall


# In[82]:


training_accuracy = model.score(X_train, y_train)
print("Training Accuracy: ", training_accuracy)


# In[83]:


test_accuracy = model.score(X_test, y_test)
print("Training Accuracy: ", test_accuracy)


# In[84]:


def getIndexes(dfObj, value):
    ''' Get index positions of value in dataframe i.e. dfObj.'''
    listOfPos = list()
    # Get bool dataframe with True at positions where the given value exists
    result = dfObj.isin([value])
    # Get list of columns that contains the value
    seriesObj = result.any()
    columnNames = list(seriesObj[seriesObj == True].index)
    # Iterate over list of columns and fetch the rows indexes where value exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            listOfPos.append((row, col))
    # Return a list of tuples indicating the positions of value in the dataframe
    return listOfPos


# In[85]:


X_url = data[extra_column2]

X_url_train, X_url_test, y_train, y_test = train_test_split(X_url, y, test_size=0.250, random_state=42)

#Show it after running on a single data


# In[86]:


# make a prediction
x = getIndexes(X_url_train, "support.apple.com")[0][0]
ynew = model.predict(X_train[[x]])


# In[87]:


#Result of prediction

if(ynew==0):
    print("Prediction: Non-Malicious")
else:
    print("Prediction: Malicious")
    

print("The prediction made by the model was:",(y_train[x]==ynew))


# In[15]:





# In[47]:





# In[23]:





# In[40]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




