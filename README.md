# Titanic-Prediction-dataset
#Prediction of survival of passengers
#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train=pd.read_csv(r"C:\Users\Lenovo\OneDrive\Desktop\Titanic dataset\train.csv")
test=pd.read_csv(r"C:\Users\Lenovo\OneDrive\Desktop\Titanic dataset\test.csv")
train.head()
test.head()
train.shape
train.columns
train['Sex'].value_counts()
#visualization
train['Died']=1-train['Survived']
train.groupby('Sex').agg('sum')[['Survived','Died']].plot(kind='bar',figsize=(10,5),stacked=True)
figure=plt.figure(figsize=(16,7))
plt.hist([train[train['Survived']==1]['Fare'],train[train['Survived']==0]['Fare']],stacked=True, bins=50,label=['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()
#cleaning
data1=train.drop(['Name','Ticket','Cabin','PassengerId','Died'],axis=1)
data1.head(5)
data1.isnull().sum()
#conversion of categorical data to numerical
data1.Sex=data1.Sex.map({'female':0,'male':1})
data1.Embarked=data1.Embarked.map({'S':0,'C':0,'Q':2,'nan':'NAN'})
data1.head()
mean_age_men=data1[data1['Sex']==1]['Age'].mean()
mean_age_women=data1[data1['Sex']==0]['Age'].mean()
data1.loc[(data1.Age.isnull())&(data1['Sex']==0),'Age']=mean_age_women
data1.loc[(data1.Age.isnull())&(data1['Sex']==1),'Age']=mean_age_men
data1.isnull().sum()
data1.dropna(inplace=True)
data1.isnull().sum()
#scaling
data1.Age=(data1.Age-min(data1.Age))/(max(data1.Age)-min(data1.Age))
data1.Fare=(data1.Fare-min(data1.Fare))/(max(data1.Fare)-min(data1.Fare))
data1.describe()
#train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data1.drop(['Survived'],axis=1),data1.Survived,test_size=0.2,random_state=0,stratify=data1.Survived)
from sklearn.linear_model import LogisticRegression
lrmod=LogisticRegression()
lrmod.fit(x_train,y_train)
from sklearn.metrics import accuracy_score
y_predict=lrmod.predict(x_test)
accuracy_score(y_test,y_predict)
#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predict)
sns.heatmap(cm,annot=True)
test.head()
data2=test.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
data2
#Converting the categorical features 'Sex' and 'Embarked' into numerical values 
data2.Sex=data2.Sex.map({'female':0, 'male':1})
data2.Embarked=data2.Embarked.map({'S':0, 'C':1, 'Q':2,'nan':'NaN'})
data2.head()
data2.isnull().sum()
#calculating mean age
mean_age_men2=data2[data2['Sex']==1]['Age'].mean()
mean_age_women2=data2[data2['Sex']==0]['Age'].mean()
# replacing null value with mean
data2.loc[(data2.Age.isnull()) & (data2['Sex']==0),'Age']=mean_age_women2
data2.loc[(data2.Age.isnull()) & (data2['Sex']==1),'Age']=mean_age_men2
data2['Fare']=data2['Fare'].fillna(data2['Fare'].mean())
data2.isnull().sum()
#Standardizing independent features
data2.Age = (data2.Age-min(data2.Age))/(max(data2.Age)-min(data2.Age))
data2.Fare = (data2.Fare-min(data2.Fare))/(max(data2.Fare)-min(data2.Fare))
data2.describe()
prediction = lrmod.predict(data2)
prediction
submission = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": prediction})
submission.to_csv('submission.csv', index=False)
prediction_df = pd.read_csv('submission.csv')
sns.countplot(x='Survived', data=prediction_df)
