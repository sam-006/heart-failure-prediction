import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
df=pd.read_csv("C:\\Users\\SAMPATH\\heart.csv")
print(df)
print(df["HeartDisease"].value_counts())#data is balanced
#now let us check fo null values and plot some graphs
sns.lineplot(x="Age",y="Cholesterol",data=df)
print(df[df.isna()].sum())
#no null values
#now let us fix categorical columns
df=df.replace(["M","F"],[1,0])#binary encoding
def one_hot_encode(df,column):#one hot encoding

    dummies=pd.get_dummies(df[column],prefix=column)
    df=pd.concat([df,dummies],axis=1)
    df=df.drop(column,axis=1)
    return df
df=one_hot_encode(df,"RestingECG")
df=one_hot_encode(df,"ExerciseAngina")
df=one_hot_encode(df,"ST_Slope")
df=one_hot_encode(df,"ChestPainType")
#categorical columns has been fixed
X=df.drop("HeartDisease",axis=1)
y=df["HeartDisease"]
print(X) 
#now let us use support vector machine for the classification and 
#grid serach cv for best parameter fit
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
scaler=StandardScaler()
scaled_X_train=scaler.fit_transform(X_train)
scaled_X_test=scaler.transform(X_test)
print(scaled_X_test)
svm=SVC()
param_grid={"C":[0.01,0.1,1],"kernel":["linear","rbf"]}
grid=GridSearchCV(svm, param_grid)
grid.fit(scaled_X_train,y_train)
print(grid.best_params_)
from sklearn.metrics import confusion_matrix,classification_report
grid_pred = grid.predict(scaled_X_test)
print(confusion_matrix(y_test,grid_pred))
print(classification_report(y_test,grid_pred))






