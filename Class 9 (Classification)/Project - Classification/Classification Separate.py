#!/usr/bin/env python
# coding: utf-8

# In[249]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os


# In[250]:


os.getcwd()


# In[251]:


loan_train  = pd.read_csv('/Users/umangpandya/Downloads/Class 9 - Codes/Project - Classification/Python_Module_Day_15.2_Credit_Risk_Train_data_XTRAIN.csv')


# In[252]:


loan_test = pd.read_csv('/Users/umangpandya/Downloads/Class 9 - Codes/Project - Classification/Python_Module_Day_15.4_Credit_Risk_Validate_data_XTEST.csv')


# In[253]:


loan_predict = pd.read_csv('/Users/umangpandya/Downloads/Class 9 - Codes/Project - Classification/Python_Module_Day_15.3_Credit_Risk_Test_data.csv')


# In[254]:


loan_train = pd.DataFrame(loan_train)
loan_train.head()


# In[255]:


loan_train.isnull().sum()


# # Data Pre-Processing

# In[256]:


from statistics import mode
from statistics import mean
from statistics import median


# In[257]:


#filling NAs
loan_train['Gender'].fillna(mode(loan_train['Gender']),inplace = True,axis = 0)
loan_train['Married'].fillna(mode(loan_train['Married']),inplace = True,axis = 0)
loan_train['Dependents'].fillna(mode(loan_train['Dependents']),inplace = True,axis = 0)
loan_train['Self_Employed'].fillna(mode(loan_train['Self_Employed']),inplace = True,axis = 0)
loan_train['Loan_Amount_Term'].fillna(mode(loan_train['Loan_Amount_Term']),inplace = True,axis = 0)
loan_train['Credit_History'].fillna(mode(loan_train['Credit_History']),inplace = True,axis = 0)
loan_train['LoanAmount'].fillna((loan_train['LoanAmount'].mean()),inplace = True,axis = 0)


# In[258]:


loan_train.isnull().sum()


# In[259]:


loan_train.head()


# In[260]:


#Dependents has '+3' which is wierdly converting the whole series in to str need to convert that in to 3
loan_train['Dependents'].replace({'3+':3},inplace = True)


# In[261]:


#Encoding Categorical Data
loan_train['Gender'] = loan_train['Gender'].astype('category')
loan_train['Gender'] = loan_train['Gender'].cat.codes
loan_train['Married'] = loan_train['Married'].astype('category')
loan_train['Married'] = loan_train['Married'].cat.codes
loan_train['Education'] = loan_train['Education'].astype('category')
loan_train['Education'] = loan_train['Education'].cat.codes
loan_train['Self_Employed'] = loan_train['Self_Employed'].astype('category')
loan_train['Self_Employed'] = loan_train['Self_Employed'].cat.codes
loan_train['Property_Area'] = loan_train['Property_Area'].astype('category')
loan_train['Property_Area'] = loan_train['Property_Area'].cat.codes
loan_train['Loan_Status'] = loan_train['Loan_Status'].astype('category')
loan_train['Loan_Status'] = loan_train['Loan_Status'].cat.codes


# In[262]:


loan_train.drop('Loan_ID',axis = 1,inplace = True)


# In[263]:


loan_train.dtypes


# In[264]:


loan_train.Dependents = loan_train.Dependents.astype('int')
loan_train.Credit_History = loan_train.Credit_History.astype('int')
loan_train.Loan_Amount_Term = loan_train.Loan_Amount_Term.astype('int')


# In[265]:


loan_train.dtypes


# In[266]:


#Pre-Processing finished on Train Data
loan_train


# In[267]:


#Loan_test Data Pre-processing
loan_test.isnull().sum()


# In[268]:


#filling NAs
loan_test.Gender = loan_test.Gender.fillna(mode(loan_test.Gender),axis = 0)
loan_test.Dependents = loan_test.Dependents.fillna(mode(loan_test.Dependents),axis = 0)
loan_test.Self_Employed = loan_test.Self_Employed.fillna(mode(loan_test.Self_Employed),axis = 0)
loan_test.LoanAmount = loan_test.LoanAmount.fillna(loan_test.LoanAmount.mean(),axis = 0)
loan_test.Loan_Amount_Term = loan_test.Loan_Amount_Term.fillna(mode(loan_test.Loan_Amount_Term),axis = 0)
loan_test.Credit_History = loan_test.Credit_History.fillna(mode(loan_test.Credit_History),axis = 0)



# In[269]:


loan_test.isnull().sum()


# In[270]:


loan_test.rename(columns = {'outcome' : 'Loan_Status'}, inplace = True)


# In[271]:


#Encoding Categorical Data
loan_test['Gender'] = loan_test['Gender'].astype('category')
loan_test['Gender'] = loan_test['Gender'].cat.codes
loan_test['Married'] = loan_test['Married'].astype('category')
loan_test['Married'] = loan_test['Married'].cat.codes
loan_test['Education'] = loan_test['Education'].astype('category')
loan_test['Education'] = loan_test['Education'].cat.codes
loan_test['Self_Employed'] = loan_test['Self_Employed'].astype('category')
loan_test['Self_Employed'] = loan_test['Self_Employed'].cat.codes
loan_test['Property_Area'] = loan_test['Property_Area'].astype('category')
loan_test['Property_Area'] = loan_test['Property_Area'].cat.codes
loan_test['Loan_Status'] = loan_test['Loan_Status'].astype('category')
loan_test['Loan_Status'] = loan_test['Loan_Status'].cat.codes


# In[272]:


loan_test['Dependents'].replace({'3+':3},inplace = True)


# In[273]:


loan_test.drop('Loan_ID', axis = 1,inplace = True)


# In[274]:


loan_test.dtypes


# In[275]:


loan_test.Dependents = loan_test.Dependents.astype('int')
loan_test.Credit_History = loan_test.Credit_History.astype('int')
loan_test.Loan_Amount_Term = loan_test.Loan_Amount_Term.astype('int')
loan_test.CoapplicantIncome = loan_test.CoapplicantIncome.astype('float')


# In[276]:


#Pre-Processing finished on Test Data
loan_test


# In[277]:


#Pre-processing on predict data
loan_predict.head()


# In[278]:


loan_predict.isnull().sum()


# In[279]:


loan_predict.Gender.fillna(mode(loan_predict.Gender),inplace = True, axis = 0)
loan_predict.Dependents.fillna(mode(loan_predict.Dependents),inplace = True, axis = 0)
loan_predict.Self_Employed.fillna(mode(loan_predict.Self_Employed),inplace = True, axis = 0)
loan_predict.LoanAmount.fillna((loan_predict.LoanAmount).mean(),inplace = True, axis = 0)
loan_predict.Loan_Amount_Term.fillna(mode(loan_predict.Loan_Amount_Term),inplace = True, axis = 0)
loan_predict.Credit_History.fillna(mode(loan_predict.Credit_History),inplace = True, axis = 0)


# In[280]:


loan_predict.head()


# In[281]:


loan_predict.head()


# In[282]:


loan_predict.isnull().sum()


# In[283]:


loan_predict['Dependents'].replace({'3+':3},inplace = True)


# In[284]:


#Encoding Categorical Data
loan_predict['Gender'] = loan_predict['Gender'].astype('category')
loan_predict['Gender'] = loan_predict['Gender'].cat.codes
loan_predict['Married'] = loan_predict['Married'].astype('category')
loan_predict['Married'] = loan_predict['Married'].cat.codes
loan_predict['Education'] = loan_predict['Education'].astype('category')
loan_predict['Education'] = loan_predict['Education'].cat.codes
loan_predict['Self_Employed'] = loan_predict['Self_Employed'].astype('category')
loan_predict['Self_Employed'] = loan_predict['Self_Employed'].cat.codes
loan_predict['Property_Area'] = loan_predict['Property_Area'].astype('category')
loan_predict['Property_Area'] = loan_predict['Property_Area'].cat.codes


# In[285]:


loan_predict.dtypes


# In[286]:


loan_predict.Dependents = loan_predict.Dependents.astype('int')
loan_predict.Credit_History = loan_predict.Credit_History.astype('int')
loan_predict.Loan_Amount_Term = loan_predict.Loan_Amount_Term.astype('int')
loan_predict.CoapplicantIncome = loan_predict.CoapplicantIncome.astype('float')


# In[287]:


loan_predict.dtypes


# In[288]:


#Pre-Processing finished on Predict Data
loan_predict


# In[289]:


loan_train.head()


# In[290]:


loan_test.head()


# In[291]:


loan_predict


# In[292]:


x_train = loan_train.iloc[:,:11].values
y_train = loan_train.iloc[:,11].values
x_test = loan_test.iloc[:,:11].values
y_test = loan_test.iloc[:,11].values


# # Logistic Regression
# 

# In[293]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression() 
lr.fit(x_train,y_train)


# In[294]:


y_pred = lr.predict(x_test)
y_pred


# In[295]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# In[296]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# # Naive Bayes

# In[297]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)


# In[298]:


y_pred = nb.predict(x_test)
y_pred


# In[299]:


confusion_matrix(y_test,y_pred)


# In[300]:


accuracy_score(y_test,y_pred)


# # K - Nearest Neighbours

# In[301]:


from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
knc.fit(x_train, y_train)


# In[302]:


y_pred = knc.predict(x_test)
y_pred


# In[303]:


confusion_matrix(y_test,y_pred)


# In[304]:


accuracy_score(y_test,y_pred)


# # Support Vector Machine

# In[305]:


#Linear
from sklearn.svm import SVC
svl = SVC (kernel = 'linear')
# types of kernels are linear, sigmoid, Radial Basis Function(rbf), Polynomial
svl.fit(x_train, y_train)


# In[306]:


y_pred = svl.predict(x_test)
y_pred


# In[307]:


confusion_matrix(y_test,y_pred)


# In[308]:


accuracy_score(y_test,y_pred)


# In[309]:


#Sigmoid
from sklearn.svm import SVC
svs = SVC (kernel = 'sigmoid')
# types of kernels are linear, sigmoid, Radial Basis Function(rbf), Polynomial
svs.fit(x_train, y_train)


# In[310]:


y_pred = svs.predict(x_test)
y_pred


# In[311]:


confusion_matrix(y_test,y_pred)


# In[312]:


accuracy_score(y_test,y_pred)


# In[313]:


#rbf
from sklearn.svm import SVC
svr = SVC (kernel = 'rbf')
# types of kernels are linear, sigmoid, Radial Basis Function(rbf), Polynomial
svr.fit(x_train, y_train)


# In[314]:


y_pred = svr.predict(x_test)
y_pred


# In[315]:


confusion_matrix(y_test,y_pred)


# In[316]:


accuracy_score(y_test,y_pred)


# In[317]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = 'entropy')
dt.fit(x_train, y_train)


# In[318]:


y_pred = dt.predict(x_test)
y_pred


# In[319]:


confusion_matrix(y_test,y_pred)


# In[320]:


accuracy_score(y_test,y_pred)


# In[321]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 10,criterion = 'entropy')
rfc.fit(x_train,y_train)


# In[322]:


y_pred = rfc.predict(x_test)


# In[323]:


y_pred


# In[324]:


confusion_matrix(y_test,y_pred)


# In[325]:


accuracy_score(y_test,y_pred)


# In[326]:


#Since we know now that the logistic model 
#is well accurate we will apply prediction to loan_predict again re-arranging x_test


# In[327]:


loan_predict.head()


# In[328]:


x_test = loan_predict.iloc[:,1:].values


# # Logistic Regression model for Prediction

# In[329]:


y_pred = lr.predict(x_test)
y_pred


# In[330]:


Predictions = pd.DataFrame(y_pred)


# In[331]:


#concatenating the predicted data in loan_predict data which is processed
loan_predict = pd.concat([loan_predict,Predictions],axis = 1)


# In[332]:


loan_predict.rename({0:'Predicted_Loan_Status'},axis=1,inplace=True)


# In[347]:


loan_predict.head()


# In[334]:


#Or rather putting Back to original file to share it with team
loan_predict = pd.read_csv('/Users/umangpandya/Downloads/Class 9 - Codes/Project - Classification/Python_Module_Day_15.3_Credit_Risk_Test_data.csv')


# In[335]:


loan_predict = pd.concat([loan_predict,Predictions],axis = 1)


# In[336]:


loan_predict.rename({0:'Predicted_Loan_Status'},axis = 1,inplace=True)


# In[337]:


loan_predict['Predicted_Loan_Status'].replace({0:'N',1:'Y'},inplace = True)


# In[346]:


loan_predict.head()

