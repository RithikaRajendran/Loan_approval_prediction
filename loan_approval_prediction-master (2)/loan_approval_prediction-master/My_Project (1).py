#!/usr/bin/env python
# coding: utf-8

# ### Problem Statement: Loan Approval Prediction Problem
# 

# In[225]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.ensemble import AdaBoostClassifier , GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 

from sklearn.metrics import classification_report , accuracy_score , confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

import warnings
warnings.filterwarnings("ignore")


# In[226]:


data = pd.read_csv("loan_prediction.csv")
data.head(5)


# In[227]:


data.shape


# In[228]:


data.dtypes


# In[229]:


sns.countplot(x="Gender",hue="Loan_Status",data=data)


# In[230]:


sns.countplot(x="Married",hue="Loan_Status",data=data)


# In[231]:


correlation_mat = data.corr()


# In[232]:


sns.heatmap(correlation_mat,annot=True,linewidths=.5,cmap="YlGnBu")


# ### There is a positive correlation between ApplicantIncome and LoanAmount, CoapplicantIncome and LoanAmount.

# In[233]:


sns.pairplot(data)
plt.show()


# In[234]:


data.describe()


# In[235]:


data.info()


# In[236]:


data.isnull().sum()


# In[237]:


plt.figure(figsize=(10,6))
sns.heatmap(data.isnull(),yticklabels=False)


# Prepare data for model training i.e. removing ouliers , filling null values 

# In[238]:


print(data["Gender"].value_counts())
print(data["Married"].value_counts())
print(data["Self_Employed"].value_counts())
print(data["Dependents"].value_counts())
print(data["Credit_History"].value_counts())
print(data["Loan_Amount_Term"].value_counts())


# In[239]:


data["Gender"].fillna(data["Gender"].mode()[0],inplace=True)
data["Married"].fillna(data["Married"].mode()[0],inplace=True)
data["Self_Employed"].fillna(data["Self_Employed"].mode()[0],inplace=True)
data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].mode()[0],inplace=True)
data["Dependents"].fillna(data["Dependents"].mode()[0],inplace=True)
data["Credit_History"].fillna(data["Credit_History"].mode()[0],inplace=True)

data["Dependents"] = data["Dependents"].replace('3+',int(3))
data["Dependents"] = data["Dependents"].replace('1',int(1))
data["Dependents"] = data["Dependents"].replace('2',int(2))
data["Dependents"] = data["Dependents"].replace('0',int(0))

data["LoanAmount"].fillna(data["LoanAmount"].median(),inplace=True)

print(data.isnull().sum())

plt.figure(figsize=(10,6))
sns.heatmap(data.isnull())


# In[240]:


data.head(5)


# In[241]:


data["Gender"] = le.fit_transform(data["Gender"])
data["Married"] = le.fit_transform(data["Married"])
data["Education"] = le.fit_transform(data["Education"])
data["Self_Employed"] = le.fit_transform(data["Self_Employed"])
data["Property_Area"] = le.fit_transform(data["Property_Area"])
data["Loan_Status"] = le.fit_transform(data["Loan_Status"])

data.head(5)


# In[242]:


X = data.drop(["Loan_Status","Loan_ID"],axis=1)
y = data["Loan_Status"]


# In[243]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# ## Logistic Regression

# In[244]:


model=LogisticRegression(solver="liblinear")


# In[245]:


model.fit(X_train,y_train)


# In[246]:


model.score(X_train,y_train)


# In[247]:


model.score(X_test,y_test)


# ## Decision Tree

# In[248]:


dtree=DecisionTreeClassifier(criterion="gini")
dtree.fit(X_train,y_train)


# In[249]:


dtree.score(X_train,y_train)


# In[250]:


dtree.score(X_test,y_test)


# In[251]:


dTreeR = DecisionTreeClassifier(criterion = 'gini', max_depth = 3, random_state=0)
dTreeR.fit(X_train, y_train)
print(dTreeR.score(X_train, y_train))


# In[252]:


y_predict = dTreeR.predict(X_test)


# In[253]:


accuracy_score(y_test,y_predict)


# In[254]:


print(dTreeR.score(X_test, y_test))


# In[255]:


from sklearn import metrics


# In[256]:


cm=metrics.confusion_matrix(y_test, y_predict,labels=[0, 1])

df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True ,fmt='g')


# ## Bagging Classifier

# In[257]:


from sklearn.ensemble import BaggingClassifier
bgcl = BaggingClassifier( n_estimators=150,base_estimator=dTreeR,random_state=0)
bgcl = bgcl.fit(X_train,y_train)
y_predict = bgcl.predict(X_test)
print(bgcl.score(X_test,y_test))


# ## Confusion_Matrix

# In[258]:


from sklearn import metrics
cm=metrics.confusion_matrix(y_test, y_predict,labels=[0, 1])

df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True ,fmt='g')


# ## AdaBoost Classifier

# In[259]:


from sklearn.ensemble import AdaBoostClassifier
abcl = AdaBoostClassifier(n_estimators = 120,random_state=0)
abcl = abcl.fit(X_train, y_train)
y_predict = abcl.predict(X_test)
print(abcl.score(X_test, y_test))


# ## GradientBoosting Classifier

# In[260]:


from sklearn.ensemble import GradientBoostingClassifier
gbcl = GradientBoostingClassifier(n_estimators = 200,random_state=0)
gbcl = gbcl.fit(X_train, y_train)
y_predict = gbcl.predict(X_test)
print(gbcl.score(X_test, y_test))


# In[261]:


cm=metrics.confusion_matrix(y_test, y_predict,labels=[0, 1])

df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True ,fmt='g')


# ## RandomForest Classifier

# In[262]:


from sklearn.ensemble import RandomForestClassifier
rfcl = RandomForestClassifier(n_estimators = 160, random_state=0,max_features=3)
rfcl = rfcl.fit(X_train, y_train)


# In[263]:


y_predict = rfcl.predict(X_test)
print(rfcl.score(X_test, y_test))


# In[264]:


cm=metrics.confusion_matrix(y_test, y_predict,labels=[0, 1])

df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                  columns = [i for i in ["No","Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True ,fmt='g')


# ## Feature Importance

# In[265]:


importances = pd.Series(rfcl.feature_importances_, index=X.columns)

importances.plot(kind='barh', figsize=(12,8))


# In[266]:


X = train.drop('Loan_Status', 1)
y = train.Loan_Status


# ## Logistic Regression

# In[267]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
regressor = LogisticRegression()


# In[268]:


regressor.fit(X, y)
y_pred=regressor.predict(x_test)

accuracy_score(y_test,y_pred)


# ## RandomForest Classifier

# In[269]:


from sklearn.ensemble import RandomForestClassifier
rfcl = RandomForestClassifier(n_estimators = 160, random_state=0,max_features=3)
rfcl = rfcl.fit(X_train, y_train)


# In[270]:


y_predict = rfcl.predict(X_test)
accuracy_score(y_test,y_predict)


# ## GradientBoosting Classifier

# In[271]:


from sklearn.ensemble import GradientBoostingClassifier
gbcl = GradientBoostingClassifier(n_estimators = 200,random_state=0)
gbcl = gbcl.fit(X_train, y_train)


# In[272]:


y_predict1 = gbcl.predict(X_test)
accuracy_score(y_test,y_predict1)


# ## AdaBoost Classifier

# In[273]:


from sklearn.ensemble import AdaBoostClassifier
abcl = AdaBoostClassifier(n_estimators = 120,random_state=0)
abcl = abcl.fit(X_train, y_train)


# In[274]:


y_predict2 = abcl.predict(X_test)
accuracy_score(y_test,y_predict2)


# ## Bagging Classifier

# In[275]:


from sklearn.ensemble import BaggingClassifier
bgcl = BaggingClassifier( n_estimators=150,base_estimator=dTreeR,random_state=0)
bgcl = bgcl.fit(X_train,y_train)


# In[276]:


y_predict3 = bgcl.predict(X_test)
accuracy_score(y_test,y_predict3)


# ## DecisionTree Classifier

# In[277]:


dtree=DecisionTreeClassifier(criterion="gini")
dtree.fit(X_train,y_train)


# In[278]:


y_predict4 = dtree.predict(X_test)


# In[279]:


accuracy_score(y_test,y_predict4)


# In[ ]:




