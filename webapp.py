import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import pickle
from sklearn.tree import BaseDecisionTree

df = pd.read_csv("creditcard.csv")
#Upsampling the Imbalanced Data
df_majority = df[df.Class==0]
df_minority = df[df.Class==1]

df_minority_upsampled = resample(df_minority, 
                                 replace=True,     
                                 n_samples=284315,    
                                 random_state=123)

df_upsampled = pd.concat([df_majority, df_minority_upsampled])

X = df_upsampled.iloc[:,0:30]
Y = df_upsampled.iloc[:,-1]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=42,test_size=0.2)
classifier = DecisionTreeClassifier()
classifier = classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)

#Upsampling the Imbalanced Data
df_majority = df[df.Class==0]
df_minority = df[df.Class==1]

df_minority_upsampled = resample(df_minority, 
                                 replace=True,     
                                 n_samples=284315,    
                                 random_state=123)

df_upsampled = pd.concat([df_majority, df_minority_upsampled])

X = df_upsampled.iloc[:,0:30]
X.columns
Y = df_upsampled.iloc[:,-1]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=42,test_size=0.2)
classifier = DecisionTreeClassifier()
classifier = classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
accuracy_score(Y_test,Y_pred)

#Plotting the Decision Tree
plt.figure(figsize=(30,20))
plot_tree(classifier,filled=True,feature_names=df_upsampled.columns)
plt.show()
df_importances = pd.DataFrame({"Features":X.columns,"Importances":classifier.feature_importances_})

pickle.dump(classifier, open('model.pkl', 'wb'))

