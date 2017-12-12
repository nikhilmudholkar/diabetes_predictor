import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PyPDF2
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV



df = pd.read_csv('diabetes.csv')
X = df.drop('diabetes',axis = 1).values
y = df['diabetes']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.4,random_state = 42)
knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
#print(y_pred)
y_pred_prob_knn = knn.predict_proba(X_test)[:,1]

fpr,tpr,thresholds = roc_curve(y_test, y_pred_prob_knn)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.savefig('Roc_curve_knn.png')


#parameter hypertuning for KNN algorithm
param_grid = {'n_neighbors': np.arange(1,50)}
knn_ht = KNeighborsClassifier()
knn_cv = GridSearchCV(knn_ht,param_grid,cv = 5)
knn_cv.fit(X_train,y_train)
print("the best parameter for KNN algorithm using paramter hypertuning is : " +str(knn_cv.best_params_))
print("the best score for KNN algorithm using parameter hypertuning is : "+str(knn_cv.best_score_))



print("##### classification report for KNN algorithm")
print("score : "+str(knn.score(X_test, y_test)))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred_log = logreg.predict(X_test)
print("AUC for knn: {}".format(roc_auc_score(y_test,y_pred_prob_knn)))
cv_auc = cross_val_score(knn,X,y, cv = 5,scoring = 'roc_auc')
print("AUC scores computed using 5-fold cross-validation for KNN algorithm: {}".format(cv_auc))
print("mean AUC score for KNN algorithm using 5 fold algorithm is: {}".format(np.mean(cv_auc)))


print("##### classification report for logistic regression")
print("score : "+str(logreg.score(X_test, y_test)))
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

y_pred_prob_log = logreg.predict_proba(X_test)[:,1]
fpr,tpr,thresholds = roc_curve(y_test, y_pred_prob_log)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.savefig('Roc_curve_logistic.png')


print("AUC for logistic regression: {}".format(roc_auc_score(y_test,y_pred_prob_log)))


cv_auc = cross_val_score(logreg,X,y, cv = 5,scoring = 'roc_auc')
print("AUC scores computed using 5-fold cross-validation for logistic regression: {}".format(cv_auc))
print("mean AUC score for logistic regression using 5 fold algorithm is: {}".format(np.mean(cv_auc)))


#parameter hypertuning for logitic regression
c_space = np.logspace(-5,8,15)
param_grid= {'C' : c_space, 'penalty':['l1','l2']}
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, param_grid, cv = 5)
logreg_cv.fit(X_train,y_train)

print("the best parameter for logistic regression using paramter hypertuning is : " +str(logreg_cv.best_params_))
print("the best score for logistic regression using parameter hypertuning is : "+str(logreg_cv.best_score_))







