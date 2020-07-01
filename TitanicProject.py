import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datasetin = pd.read_csv("./Titanic.csv")
dataset = datasetin[['Embarked', 'Sex',	'SibSp', 'Parch', 'Pclass', 'Fare', 'Survived']]
dataset = pd.get_dummies(dataset, columns=["Embarked", "Sex", "Pclass", "SibSp", "Parch"])

X = dataset.drop(columns='Survived').values
Y = dataset['Survived'].values

from sklearn.preprocessing import StandardScaler
stdscale = StandardScaler()
X[:, 0:1] = stdscale.fit_transform(X[:, 0:1])

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred1=logreg.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred1)

from sklearn.metrics import accuracy_score
print()
print("Accuracy with Logistic Regression :")
print(accuracy_score(Y_test, Y_pred1))
print()

from sklearn.tree import DecisionTreeClassifier 
dtclassifier = DecisionTreeClassifier()
dtclassifier.fit(X_train, Y_train)

Y_pred2=dtclassifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred2)

from sklearn.metrics import accuracy_score
print()
print("Accuracy with Decision Tree Classifier :")
print(accuracy_score(Y_test, Y_pred2))
print()
