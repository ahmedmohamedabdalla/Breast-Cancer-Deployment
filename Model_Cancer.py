import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

data_cancer=pd.read_csv('data.csv')
print(data_cancer.head(n=3))


print(data_cancer.shape)
# This Code Useed To Show Names Of Columns In Data
print("Name of Column:")
for i in data_cancer.columns:
   print(i)

# I check null and duplicated values in Data
null_value=data_cancer.isna().sum()
duplicated_value=data_cancer.duplicated().sum()

# I Create Data frame it include value of null and duplicted in Data
print(null_value)
print(duplicated_value)



print(data_cancer['diagnosis'].value_counts())

la=LabelEncoder()
data_cancer['diagnosis']=la.fit_transform(data_cancer['diagnosis'])


print(data_cancer['diagnosis'].value_counts())


sns.heatmap(data_cancer.corr(),annot=True)

features=['radius_mean','perimeter_mean','area_mean',
          'compactness_mean','concavity_mean','concave points_mean',
          'radius_worst','perimeter_worst','area_worst',
          'concavity_worst','concave points_worst','compactness_worst']
X=data_cancer[features].values
y=data_cancer['diagnosis'].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=123,shuffle=y)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)


Model=RandomForestClassifier(n_estimators=400,max_depth=500,min_samples_split=2)
Model.fit(X_train, y_train)

print('The Score of model Training is = {x}%'.format(x=Model.score(X_train,y_train).round(5)))
print('The Score of model Testing is = {y}%'.format(y=Model.score(X_train,y_train).round(5)))

y_pred=Model.predict(X_test)
a=accuracy_score(y_test,y_pred)
print(a)

cm=confusion_matrix(y_test,y_pred)
print(cm)


sns.heatmap(cm,annot=True)

pre=Model.predict([[]])
if pre==0:
    print('Benign')
else:
    print('Malignant')    


import pickle
#pickle.dump(Model,open('model.pkl','wb'))









