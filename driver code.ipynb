#Note: This program was written and run on google colaboratory.


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/content/drive/MyDrive/diabetes_null.csv')
df.head()

df.isnull().sum()
df.info()
df1  = df.fillna(method='ffill')

df1.head()

df1.isnull().sum()

df1['Outcome'].value_counts()       #dataset shows class imbalance 

df1.shape

from sklearn.utils import resample

df1_out_0 = df1[df1['Outcome']==0]
df1_out_1 = df1[df1['Outcome']==1]

len(df1_out_0)
df1_out_1_upsampled = resample(df1_out_1, replace = True, n_samples =len(df1_out_0) - len(df1_out_1) )
df1_out_1_upsampled.shape
df1_out_1_new = pd.concat([df1_out_1, df1_out_1_upsampled])
df1_out_1_new.shape                #class imbalance has been fixed


#------

df_new = pd.concat([df1_out_1_new, df1_out_0], ignore_index = True)       #ignore index jumbles the data which allows the model to learn better during training and testing.
df_new.shape
df_new
df_new.isnull().sum()
df_new.columns

sns.boxplot(x = 'Pregnancies', data = df_new)
#no outliers

sns.boxplot(x ='Glucose', data = df_new)
sns.boxplot(x = 'BloodPressure', data =df_new)
sns.boxplot(x = 'SkinThickness', data =df_new)
sns.boxplot(x = 'Insulin', data =df_new)
sns.boxplot(x = 'BMI', data =df_new)
sns.boxplot(x = 'DiabetesPedigreeFunction', data =df_new)
sns.boxplot(x = 'Age', data =df_new)

def remove_outliers(dfs,field):
  q1,q3 = np.percentile(dfs[field],[25,75])
  iqr = q3 - q1
  lb = q1 - 1.5*iqr
  ub = q3 + 1.5*iqr
  n_outliers = dfs[(dfs[field]>=lb)&(dfs[field]<=ub)]
  return n_outliers

n_outliers = remove_outliers(df_new, 'Glucose')

plt.subplot(221)
plt.title('Before removal')
sns.boxplot(df_new['Glucose'])

plt.subplot(222)
plt.title('After removal')
sns.boxplot(n_outliers['Glucose'])

n_outliers = remove_outliers(df_new, 'Age')

plt.subplot(221)
plt.title('Before removal')
sns.boxplot(df_new['Age'])

plt.subplot(222)
plt.title('After removal')
sns.boxplot(n_outliers['Age'])

n_outliers = remove_outliers(df_new, 'Pregnancies')

plt.subplot(221)
plt.title('Before removal')
sns.boxplot(df_new['Pregnancies'])

plt.subplot(222)
plt.title('After removal')
sns.boxplot(n_outliers['Pregnancies'])

n_outliers = remove_outliers(df_new, 'BloodPressure')

plt.subplot(221)
plt.title('Before removal')
sns.boxplot(df_new['BloodPressure'])

plt.subplot(222)
plt.title('After removal')
sns.boxplot(n_outliers['BloodPressure'])

n_outliers = remove_outliers(df_new, 'BMI')

plt.subplot(221)
plt.title('Before removal')
sns.boxplot(df_new['BMI'])

plt.subplot(222)
plt.title('After removal')
sns.boxplot(n_outliers['BMI'])

n_outliers = remove_outliers(df_new, 'DiabetesPedigreeFunction')

plt.subplot(221)
plt.title('Before removal')
sns.boxplot(df_new['DiabetesPedigreeFunction'])

plt.subplot(222)
plt.title('After removal')
sns.boxplot(n_outliers['DiabetesPedigreeFunction'])

n_outliers = remove_outliers(df_new, 'SkinThickness')

plt.subplot(221)
plt.title('Before removal')
sns.boxplot(df_new['SkinThickness'])

plt.subplot(222)
plt.title('After removal')
sns.boxplot(n_outliers['SkinThickness'])

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

x = df_new.drop(['Outcome'],axis =1)
y = df_new['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)

model = SVC(C=1, kernel= 'poly')
model.fit(x_train, y_train)    
y_pred = model.predict(x_test)
accuracy_score(y_pred, y_test)

#-------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

lr = LogisticRegression(max_iter = 200)         
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)
accuracy_score(y_pred_lr,y_test)
print(classification_report(y_pred_lr, y_test))


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

dt_gini = DecisionTreeClassifier(criterion='gini')
dt_gini.fit(x_train,y_train)

plt.figure(figsize=(15,10))
tree.plot_tree(dt_gini, filled = True)
y_pred_gini = dt_gini.predict(x_test)
accuracy_score(y_pred_gini, y_test)
print(classification_report(y_pred_gini, y_test))

#--------

dt_entropy = DecisionTreeClassifier(criterion = 'entropy', max_depth =7)
dt_entropy.fit(x_train,y_train)
tree.plot_tree(dt_entropy, filled = True)
y_pred_ent = dt_entropy.predict(x_test)
accuracy_score(y_pred_ent, y_test)
print(classification_report(y_pred_ent, y_test))

#---------

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=20)
rfc.fit(x_train,y_train)
y_pred_rfc=rfc.predict(x_test)
accuracy_score(y_pred_rfc,y_test)
print(classification_report(y_pred_rfc,y_test))

#-------------------------
from sklearn.model_selection import train_test_split, GridSearchCV

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
param_grid = {'n_estimators' : [10,20,30,40], 'criterion':['gini','entropy','log_loss'], 'max_depth':[2,3,4,5,None]} #n_estimator is the number of trees considered in the model.
grid_rfc = GridSearchCV(rfc, param_grid = param_grid, verbose =1, cv = 5)
grid_rfc.fit(x_train, y_train)
grid_rfc.best_params_
grid_rfc.best_score_
